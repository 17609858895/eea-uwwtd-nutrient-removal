"""Final causal estimates with repeated cross-fitting and median aggregation.

Usage: python 11_final_estimates.py [lgbm|sl] [n_repeats]
  lgbm : LightGBM nuisances (main analysis, default 10 repeats)
  sl   : stacked super-learner nuisances (sensitivity, default 3 repeats)
Estimates are aggregated over repeats with the median rule of Chernozhukov et al. (2018):
theta = median(theta_s), se = median(sqrt(se_s^2 + (theta_s - theta)^2)).
Also runs the absolute-scale (kg p.e.^-1 yr^-1) MOSAIC-DR used for load-based policy values.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import load, RES
from causal_core import (add_context, prepare, fit_nuisances, pseudo_outcome, weighted_se, cluster_se,
                         plr_complete_case, PLANT_X, sl_clf, sl_reg, joint_support, labelled_country)

MODE = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
S = int(sys.argv[2]) if len(sys.argv) > 2 else (10 if MODE == "lgbm" else 3)
FN = dict(clf_fn=sl_clf, reg_fn=sl_reg) if MODE == "sl" else {}
EF = {"N": np.log10(0.20 / 0.45), "P": np.log10(0.10 / 0.40)}
SIZES = [("<2k", 0, 2e3), ("2k-10k", 2e3, 1e4), ("10k-150k", 1e4, 1.5e5), (">=150k", 1.5e5, 1e12)]


def populations(d, e, nu):
    lab = labelled_country(d)
    js = joint_support(e, nu["pi1"], nu["pi0"], lab)
    nolab = ~lab
    pops = {"registry": np.ones(len(d), bool), "joint support": js, "zero-label countries": nolab,
            "recast target (>=10k untreated) in support": js & (d.capacity_pe >= 1e4).to_numpy() & (d["T"] == 0).to_numpy()}
    for nm, lo, hi in SIZES:
        pops[f"size {nm} in support"] = js & (d.capacity_pe >= lo).to_numpy() & (d.capacity_pe < hi).to_numpy()
    return pops


def one_repeat(d, X_cols, s, scale):
    out = []
    dd = d.copy()
    if scale == "abs":
        dd["Y"] = 10 ** dd["Y"]
    cc = dd[dd.R == 1].reset_index(drop=True)
    variants = [("Complete-case AIPW", cc, dict(use_pi=False, use_src=False)),
                ("Missingness-aware AIPW", dd, dict(use_pi=True, use_src=False)),
                ("MOSAIC-DR", dd, dict(use_pi=True, use_src=True))]
    keep = None
    for name, data, kw in variants:
        if scale == "abs" and name != "MOSAIC-DR":
            continue
        nu = fit_nuisances(data, X_cols, seed=s, **kw, **FN)
        phi = pseudo_outcome(data, nu); e = nu["e"]; h = e * (1 - e)
        for pop, m in populations(data, e, nu).items():
            if m.sum() < 30:
                continue
            th_w, se_w = weighted_se(phi, h, data.country.values, m)
            th_u, se_u = cluster_se(phi, data.country.values, m)
            out.append(dict(estimator=name, population=pop, rep=s, n=int(m.sum()),
                            ato=th_w, ato_se=se_w, mean=th_u, mean_se=se_u))
        if name == "MOSAIC-DR":
            keep = dict(phi=phi, e=e, pi1=nu["pi1"], pi0=nu["pi0"], mu0=nu["mu0"], mu1=nu["mu1"], off=nu["off"])
    if scale == "log":
        th, se = plr_complete_case(dd, X_cols, seed=s)
        out.append(dict(estimator="Complete-case DML-PLR", population="labelled", rep=s, n=int(dd.R.sum()),
                        ato=th, ato_se=se, mean=th, mean_se=se))
    return out, keep


def median_agg(df):
    rows = []
    for (est, pop), g in df.groupby(["estimator", "population"], sort=False):
        r = dict(estimator=est, population=pop, n=int(g.n.median()), repeats=len(g))
        for k in ["ato", "mean"]:
            th = np.median(g[k]); se = np.median(np.sqrt(g[f"{k}_se"] ** 2 + (g[k] - th) ** 2))
            r.update({k: th, f"{k}_se": se, f"{k}_lo": th - 1.96 * se, f"{k}_hi": th + 1.96 * se})
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allrows = []
    for P in ["N", "P"]:
        treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
        d0, ctx = add_context(load(P))
        X_cols = PLANT_X + [other] + ctx
        d = prepare(d0, treat, X_cols)
        if MODE == "lgbm":
            # absolute-scale outcome: winsorise per-capita discharge at the 99.5th percentile
            cap = np.nanquantile(10 ** d.Y, 0.995)
            d_abs = d.copy(); d_abs["Y"] = np.log10(np.minimum(10 ** d.Y, cap))
            scales = [("log", d), ("abs", d_abs)]
        else:
            scales = [("log", d)]
        for scale, data in scales:
            res = Parallel(n_jobs=min(S, 10))(delayed(one_repeat)(data, X_cols, s, scale) for s in range(S))
            df = pd.DataFrame([r for out, _ in res for r in out])
            agg = median_agg(df).assign(P=P, scale=scale, nuisance=MODE)
            if scale == "log":
                for k in ["ato", "mean"]:
                    agg[f"{k}_red_pct"] = (1 - 10 ** agg[k]) * 100
                    agg[f"{k}_red_lo"] = (1 - 10 ** agg[f"{k}_hi"]) * 100
                    agg[f"{k}_red_hi"] = (1 - 10 ** agg[f"{k}_lo"]) * 100
            allrows.append(agg)
            df.assign(P=P, scale=scale).to_csv(RES / f"11_repeats_{MODE}_{P}_{scale}.csv", index=False)
            keeps = [k for _, k in res]
            units = data.assign(**{k: np.mean([kk[k] for kk in keeps], axis=0) for k in keeps[0]})
            units.to_parquet(RES / f"11_units_{MODE}_{P}_{scale}.parquet", index=False)
            print(agg.round(3).to_string(), flush=True)
    pd.concat(allrows).to_csv(RES / f"11_final_estimates_{MODE}.csv", index=False)
