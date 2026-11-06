"""Omitted-variable-bias sensitivity for the partially linear (overlap-weighted) effect.

Nonparametric extension of Cinelli & Hazlett (2020) by Chernozhukov et al. (2022):
|bias| <= sqrt(R2_Y * R2_D / (1 - R2_D)) * sd(Y residual | T, X) / sd(T residual | X),
where R2_Y and R2_D are partial R2 of the latent confounder with the outcome and treatment.
Fitted on labelled plants inside the joint support (overlap and observation positivity).
Benchmarks: confounders k times as strong as observed covariate groups.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from common import load, RES
from causal_core import add_context, prepare, clf, reg, PLANT_X

EF = {"N": np.log10(0.20 / 0.45), "P": np.log10(0.10 / 0.40)}
REPS = 3


def crossfit(X, T, Y, reps=REPS):
    """Return cross-fitted e(X), l(X)=E[Y|X], g(T,X)=E[Y|T,X], averaged over repeats."""
    n = len(T); e = np.zeros(n); l = np.zeros(n); g = np.zeros(n)
    for r in range(reps):
        for tr, te in KFold(5, shuffle=True, random_state=r).split(X):
            e[te] += clf(r).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1] / reps
            l[te] += reg(r).fit(X[tr], Y[tr]).predict(X[te]) / reps
            g[te] += reg(r).fit(np.c_[X[tr], T[tr]], Y[tr]).predict(np.c_[X[te], T[te]]) / reps
    return e, l, g


def bias(ry, rd, sdy, sdd):
    return np.sqrt(ry * rd / (1 - rd)) * sdy / sdd


def rv(delta, sdy, sdd):
    f2 = (delta * sdd / sdy) ** 2
    return (-f2 + np.sqrt(f2 ** 2 + 4 * f2)) / 2


summary, bench, grid = [], [], []
for P in ["N", "P"]:
    treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
    d0, ctx = add_context(load(P))
    X_cols = PLANT_X + [other] + ctx
    d = prepare(d0, treat, X_cols)
    ref = pd.read_parquet(RES / f"04b_causal_units_{P}.parquet")
    assert len(ref) == len(d)
    js = (ref.e > 0.05) & (ref.e < 0.95) & (ref.pi1 > 0.05) & (ref.pi0 > 0.05)
    sub = d[js.to_numpy() & (d.R == 1).to_numpy()].reset_index(drop=True)
    X = sub[X_cols].astype(float).to_numpy(); T = sub["T"].to_numpy().astype(float); Y = sub["Y"].to_numpy()
    e, l, g = crossfit(X, T, Y)
    Dt, Yt = T - e, Y - l
    theta = np.sum(Dt * Yt) / np.sum(Dt ** 2)
    sdy = np.std(Yt - theta * Dt); sdd = np.std(Dt)
    psi = (Yt - theta * Dt) * Dt / np.mean(Dt ** 2)
    se = np.sqrt((pd.Series(psi).groupby(sub.country.values).sum() ** 2).sum()) / len(sub)
    d_ef = abs(EF[P] - theta)
    summary.append(dict(P=P, n=len(sub), theta=theta, se=se, reduction_pct=(1 - 10 ** theta) * 100,
                        sd_y_resid=sdy, sd_t_resid=sdd, RV_null=rv(abs(theta), sdy, sdd),
                        RV_null_CI=rv(max(abs(theta) - 1.96 * se, 0), sdy, sdd),
                        RV_to_EF=rv(d_ef, sdy, sdd), EF=EF[P]))
    print(summary[-1], flush=True)
    var_d_full, var_y_full = np.var(Dt), np.var(Y - g)
    groups = {"geography (lat, lon)": ["lat", "lon"],
              "regulatory requirement": ["req_NTot", "req_PTot"],
              "country context": ctx,
              "plant size & load": ["log_load", "log_cap", "util"],
              "co-installed treatment": [other, "f_UV", "f_Chlorination", "f_Ozonation", "f_SandFiltration",
                                         "f_MicroFiltration", "f_Other"]}
    for gname, cols in groups.items():
        keep = [i for i, c in enumerate(X_cols) if c not in cols]
        e_g, _, g_g = crossfit(X[:, keep], T, Y, reps=1)
        rd = float(np.clip(1 - var_d_full / np.var(T - e_g), 0, 0.99))
        ry = float(np.clip(1 - var_y_full / np.var(Y - g_g), 0, 0.99))
        for k in [1, 2, 3]:
            b = bias(min(k * ry, 0.99), min(k * rd, 0.99), sdy, sdd)
            bench.append(dict(P=P, group=gname, k=k, R2_Y=ry, R2_D=rd, bias_bound=b,
                              theta_toward_EF=theta - b, red_toward_EF=(1 - 10 ** (theta - b)) * 100,
                              theta_toward_null=theta + b, red_toward_null=(1 - 10 ** (theta + b)) * 100))
        print(bench[-3], flush=True)
    for ry in np.linspace(0, 0.5, 51):
        for rd in np.linspace(0, 0.5, 51):
            b = bias(ry, rd, sdy, sdd)
            grid.append(dict(P=P, R2_Y=ry, R2_D=rd, bias=b, red_toward_EF=(1 - 10 ** (theta - b)) * 100,
                             red_toward_null=(1 - 10 ** (theta + b)) * 100))

pd.DataFrame(summary).to_csv(RES / "12_ovb_summary.csv", index=False)
pd.DataFrame(bench).to_csv(RES / "12_ovb_benchmarks.csv", index=False)
pd.DataFrame(grid).to_csv(RES / "12_ovb_grid.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(summary).round(3).to_string()); print(pd.DataFrame(bench).round(3).to_string())
