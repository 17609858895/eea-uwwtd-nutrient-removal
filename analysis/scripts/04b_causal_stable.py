"""Stabilised real-data causal estimates.

Changes vs 04: 3x repeated cross-fitting; primary estimand = overlap-weighted effect
(weights h = e(1-e)), secondary = trimmed ATE on e in [0.05, 0.95]; policy subgroup
reported within common support. Same refuters, plus feature-set sensitivity.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import load, RES
from causal_core import (add_context, prepare, repeated_pseudo, weighted_se, cluster_se,
                         plr_complete_case, PLANT_X)

REPS = 3
rows = []
pollutants = sys.argv[1:] or ["N", "P"]
for P in pollutants:
    treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
    d0, ctx = add_context(load(P))
    X_cols = PLANT_X + [other] + ctx
    d = prepare(d0, treat, X_cols)
    cl = d.country.values

    def report(name, phi, e, pop="registry", mask=None, dd=None):
        dd = d if dd is None else dd
        c = dd.country.values
        h = e * (1 - e)
        m = np.ones(len(phi), bool) if mask is None else mask
        th, se = weighted_se(phi, h, c, m)
        tm = m & (e > 0.05) & (e < 0.95)
        th2, se2 = cluster_se(phi, c, tm)
        rows.append(dict(P=P, estimator=name, population=pop, n=int(m.sum()), n_support=int(tm.sum()),
                         ATO_log10=th, ATO_se=se, ATO_red_pct=(1 - 10 ** th) * 100,
                         ATO_red_lo=(1 - 10 ** (th + 1.96 * se)) * 100, ATO_red_hi=(1 - 10 ** (th - 1.96 * se)) * 100,
                         trim_log10=th2, trim_se=se2, trim_red_pct=(1 - 10 ** th2) * 100))
        print(P, name, pop, f"ATO {th:.3f}±{se:.3f}  trim {th2:.3f}±{se2:.3f}", flush=True)

    cc = d[d.R == 1].reset_index(drop=True)
    for nm, th in [("Naive difference (labelled)", cc.Y[cc["T"] == 1].mean() - cc.Y[cc["T"] == 0].mean()),
                   ("Treatment-class median (EF-style)", cc.Y[cc["T"] == 1].median() - cc.Y[cc["T"] == 0].median())]:
        rows.append(dict(P=P, estimator=nm, population="labelled", ATO_log10=th, ATO_red_pct=(1 - 10 ** th) * 100))
    th, se = plr_complete_case(d, X_cols)
    rows.append(dict(P=P, estimator="Complete-case DML-PLR", population="labelled", ATO_log10=th, ATO_se=se,
                     ATO_red_pct=(1 - 10 ** th) * 100, ATO_red_lo=(1 - 10 ** (th + 1.96 * se)) * 100,
                     ATO_red_hi=(1 - 10 ** (th - 1.96 * se)) * 100))
    phi, e, _ = repeated_pseudo(cc, X_cols, REPS, use_pi=False, use_src=False)
    report("Complete-case AIPW", phi, e, "labelled", dd=cc)
    phi, e, _ = repeated_pseudo(d, X_cols, REPS, use_pi=True, use_src=False)
    report("Missingness-aware AIPW (no provenance)", phi, e)
    phi, e, nu = repeated_pseudo(d, X_cols, REPS, use_pi=True, use_src=True)
    report("MP-DR (ours)", phi, e)
    pol = ((d.capacity_pe >= 10000) & (d["T"] == 0)).to_numpy()
    report("MP-DR (ours)", phi, e, "recast target (>=10k p.e., untreated)", pol)
    for lab, lo, hi in [("<2k", 0, 2e3), ("2k-10k", 2e3, 1e4), ("10k-150k", 1e4, 1.5e5), (">=150k", 1.5e5, 1e12)]:
        report("MP-DR (ours)", phi, e, f"size {lab}", ((d.capacity_pe >= lo) & (d.capacity_pe < hi)).to_numpy())
    d.assign(phi=phi, e=e, **{k: v for k, v in nu.items() if k != "e"}).to_parquet(
        RES / f"04b_causal_units_{P}.parquet", index=False)
    # refuters
    rng = np.random.default_rng(1)
    dp = d.copy(); dp["T"] = rng.permutation(dp["T"].to_numpy())
    phip, ep, _ = repeated_pseudo(dp, X_cols, REPS); report("Refute: placebo (permuted T)", phip, ep, dd=dp)
    dr = d.copy(); dr["rcc"] = rng.normal(size=len(dr))
    phir, er, _ = repeated_pseudo(dr, X_cols + ["rcc"], REPS); report("Refute: random common cause", phir, er, dd=dr)
    for s in range(5):
        ds = d.sample(frac=0.5, random_state=10 + s).reset_index(drop=True)
        phis, es, _ = repeated_pseudo(ds, X_cols, 1); report(f"Refute: 50% subset #{s + 1}", phis, es, dd=ds)
    Xn = [c for c in X_cols if c != "f_UV"] + [treat]
    dn = prepare(d0.copy(), "f_UV", Xn)
    phin, en, _ = repeated_pseudo(dn, Xn, REPS); report("Negative-control exposure: UV", phin, en, dd=dn)
    # feature-set sensitivity
    for nm, cols in [("Sensitivity: no country context", PLANT_X + [other]),
                     ("Sensitivity: + more-stringent flag", X_cols + ["trt_other"])]:
        phis, es, _ = repeated_pseudo(d, cols, REPS); report(nm, phis, es)

pd.DataFrame(rows).to_csv(RES / f"04b_causal_stable_{'_'.join(pollutants)}.csv", index=False)
