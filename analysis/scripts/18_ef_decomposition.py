"""Where does the emission-factor gap come from? Arm-specific pass-through on the joint support.

Emission factors fix the fraction of influent load that leaves a plant: 0.45 (N) and 0.40 (P)
under secondary treatment, 0.20 and 0.10 with nutrient removal. The upgrade effect they imply is
the ratio of the two. Here the doubly robust arm scores of MOSAIC-DR estimate both potential
pass-through fractions, E[log10(out/in) | do(T=t)], on the plants with a plausible reported
influent load, so the gap between the estimated and the assumed effect can be split into a
baseline part (secondary plants pass through less than assumed) and a removal part (plants with
nutrient removal pass through more than assumed).

Outcome: log10 pass-through, out/in. Covariates: those of the main analysis plus log10 influent
per p.e. Estimand: overlap-weighted arm means on the joint support. Variants: all plants with a
plausible influent load, and plants whose influent load is measured.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import load, RES
from causal_core import add_context, prepare, fit_nuisances, weighted_se, PLANT_X, joint_support, labelled_country

LIM = {"N": (0.5, 30), "P": (0.05, 5)}           # plausible influent kg/p.e./yr
EF_PASS = {"N": (0.45, 0.20), "P": (0.40, 0.10)}  # secondary, with nutrient removal
REPS = 5


def arm_scores(d, nu):
    T = d["T"].to_numpy(); R = d["R"].to_numpy()
    Yt = np.where(R == 1, d["Y"].to_numpy() - nu["off"], 0.0)
    psi1 = nu["mu1"] + T * R / (nu["e"] * nu["pi1"]) * (Yt - nu["mu1"])
    psi0 = nu["mu0"] + (1 - T) * R / ((1 - nu["e"]) * nu["pi0"]) * (Yt - nu["mu0"])
    return psi1, psi0


rows = []
for P in ["N", "P"]:
    treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
    d0, ctx = add_context(load(P))
    lo, hi = LIM[P]
    d0 = d0[np.isfinite(d0.y_logq_in) & (d0.y_logq_in > np.log10(lo)) & (d0.y_logq_in < np.log10(hi))].copy()
    d0["y_pass"] = d0.y_logq_out - d0.y_logq_in
    X_cols = PLANT_X + [other] + ctx + ["y_logq_in"]
    for variant, keep in [("plausible influent", np.ones(len(d0), bool)),
                          ("measured influent", (d0.src_in == "M").to_numpy())]:
        d = prepare(d0[keep], treat, X_cols, y_col="y_pass", lo=1e-4, hi=1.5)
        p1s, p0s, nus = [], [], []
        for r in range(REPS):
            nu = fit_nuisances(d, X_cols, seed=r)
            p1, p0 = arm_scores(d, nu)
            p1s.append(p1); p0s.append(p0); nus.append(nu)
        psi1, psi0 = np.mean(p1s, axis=0), np.mean(p0s, axis=0)
        e = np.mean([n["e"] for n in nus], axis=0)
        pi1 = np.mean([n["pi1"] for n in nus], axis=0); pi0 = np.mean([n["pi0"] for n in nus], axis=0)
        h = e * (1 - e)
        js = joint_support(e, pi1, pi0, labelled_country(d))
        big = d.load_pe.to_numpy() >= 10_000
        for pop, m in [("joint support", js), ("joint support, >=10k p.e.", js & big)]:
            cl = d.country.values
            t1, s1 = weighted_se(psi1, h, cl, m)
            t0, s0 = weighted_se(psi0, h, cl, m)
            td, sd = weighted_se(psi1 - psi0, h, cl, m)
            obs = m & (d.R.to_numpy() == 1)
            naive1 = d.Y[obs & (d["T"] == 1)].mean(); naive0 = d.Y[obs & (d["T"] == 0)].mean()
            ef0, ef1 = EF_PASS[P]
            row = dict(P=P, variant=variant, population=pop, n=int(m.sum()), labelled=int(d.R[m].sum()),
                       pass0=10 ** t0, pass0_lo=10 ** (t0 - 1.96 * s0), pass0_hi=10 ** (t0 + 1.96 * s0),
                       pass1=10 ** t1, pass1_lo=10 ** (t1 - 1.96 * s1), pass1_hi=10 ** (t1 + 1.96 * s1),
                       reduction_pct=(1 - 10 ** td) * 100, red_lo=(1 - 10 ** (td + 1.96 * sd)) * 100,
                       red_hi=(1 - 10 ** (td - 1.96 * sd)) * 100,
                       naive_pass0=10 ** naive0, naive_pass1=10 ** naive1,
                       EF_pass0=ef0, EF_pass1=ef1, EF_reduction_pct=(1 - ef1 / ef0) * 100)
            # split of the log-scale gap (estimated - EF effect) into baseline and removal parts
            gap = td - np.log10(ef1 / ef0)
            row["gap_log10"] = gap
            row["gap_share_baseline"] = -(t0 - np.log10(ef0)) / gap if gap else np.nan
            row["gap_share_removal"] = (t1 - np.log10(ef1)) / gap if gap else np.nan
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "18_ef_decomposition.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())
