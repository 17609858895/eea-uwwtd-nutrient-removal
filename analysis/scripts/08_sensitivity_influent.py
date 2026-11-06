"""Sensitivity: does influent strength confound the effect?

Restrict to plants with a plausible reported influent load, adjust for log10 influent
per-capita load, and use the pass-through ratio log10(out/in) as the outcome, which
removes influent strength from the outcome by construction. Also reports E-values.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import load, RES
from causal_core import (add_context, prepare, repeated_pseudo, weighted_se, PLANT_X, joint_support, labelled_country)

LIM = {"N": (0.5, 30), "P": (0.05, 5)}          # plausible influent kg/p.e./yr
EF_PASS = {"N": np.log10(0.20 / 0.45), "P": np.log10(0.10 / 0.40)}


def evalue(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))


rows = []
for P in ["N", "P"]:
    treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
    d0, ctx = add_context(load(P))
    lo, hi = LIM[P]
    d0 = d0[np.isfinite(d0.y_logq_in) & (d0.y_logq_in > np.log10(lo)) & (d0.y_logq_in < np.log10(hi))].copy()
    d0["y_pass"] = d0.y_logq_out - d0.y_logq_in
    X_base = PLANT_X + [other] + ctx
    for outcome, X_cols, label in [("y_logq_out", X_base, "per-capita discharge, influent-reporting plants"),
                                   ("y_logq_out", X_base + ["y_logq_in"], "per-capita discharge + influent covariate"),
                                   ("y_pass", X_base + ["y_logq_in"], "pass-through log10(out/in) + influent covariate")]:
        d = prepare(d0, treat, X_cols, y_col=outcome, lo=1e-4 if outcome == "y_pass" else 0.005,
                    hi=1.5 if outcome == "y_pass" else 50)
        phi, e, nu = repeated_pseudo(d, X_cols, 3)
        h = e * (1 - e)
        js = joint_support(e, nu["pi1"], nu["pi0"], labelled_country(d))
        for pop, m in [("all", np.ones(len(d), bool)), ("joint support", js)]:
            th, se = weighted_se(phi, h, d.country.values, m)
            rows.append(dict(P=P, analysis=label, population=pop, n=int(m.sum()), labelled=int(d.R[m].sum()),
                             effect_log10=th, se=se, reduction_pct=(1 - 10 ** th) * 100,
                             red_lo=(1 - 10 ** (th + 1.96 * se)) * 100, red_hi=(1 - 10 ** (th - 1.96 * se)) * 100,
                             EF_effect_log10=EF_PASS[P], E_value=evalue(10 ** th),
                             E_value_CI=evalue(10 ** (th + 1.96 * se)) if th + 1.96 * se < 0 else 1.0))
        print(rows[-1], flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "08_sensitivity_influent.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string())
