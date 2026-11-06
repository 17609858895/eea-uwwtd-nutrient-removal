"""Semi-synthetic benchmark with known effects.

Real covariates X, real treatment T, real outcome-observation pattern R and real label
provenance S are kept; only the outcome is simulated, so the true registry-level and
policy-level effects are known. Scenario B adds outcome-dependent (MNAR) reporting to
show where the estimator's assumption breaks.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import load, RES
from causal_core import (add_context, prepare, fit_nuisances, pseudo_outcome, cluster_se,
                         plr_complete_case, PLANT_X, reg, joint_support, labelled_country)

P, treat, other = "N", "f_NRemoval", "f_PRemoval"
d0, ctx = add_context(load(P))
X_cols = PLANT_X + [other] + ctx
base = prepare(d0, treat, X_cols)
X = base[X_cols].astype(float).to_numpy()
# smooth "true" control surface learned once from real labelled controls
ctrl = (base.R == 1) & (base["T"] == 0)
mu0_true = reg(0).fit(X[ctrl.to_numpy()], base.Y[ctrl]).predict(X)
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
tau_true = (-0.50 - 0.12 * z(base.log_load.to_numpy()) + 0.10 * z(base.util.to_numpy())
            + 0.08 * np.nan_to_num(z(base.lat.to_numpy())))
# label offsets in scenarios A and B (log10); the reported benchmark uses 0.02 for both
OFF = {"C": float(sys.argv[1]) if len(sys.argv) > 1 else 0.02,
       "E": float(sys.argv[2]) if len(sys.argv) > 2 else 0.02}
# common-support population fixed by the reference propensity from the real-data run (04b)
ref = pd.read_parquet(RES / "04b_causal_units_N.parquet")
assert len(ref) == len(base)
e_ref = ref.e.to_numpy()
# joint support: treatment overlap and outcome-observation positivity in both arms
sup = joint_support(e_ref, ref.pi1, ref.pi0, labelled_country(base))
pol = ((base.capacity_pe >= 10000) & (base["T"] == 0)).to_numpy() & sup
truth = dict(registry=tau_true[sup].mean(), policy=tau_true[pol].mean())


def one_rep(rep, scenario):
    rng = np.random.default_rng(100 + rep)
    d = base.copy()
    T = d["T"].to_numpy()
    eps = rng.normal(0, 0.22, len(d))
    ystar = mu0_true + tau_true * T + eps
    off = np.where(d.sC == 1, OFF["C"], np.where(d.sE == 1, OFF["E"], 0.0))
    if scenario == "C_provenance":
        # calculated/estimated labels of treated plants use optimistic default removal rates
        off = np.where((d.sC == 1) | (d.sE == 1), np.where(T == 1, -0.15, 0.05), 0.0)
    R = d.R.to_numpy().copy()
    if scenario == "B_MNAR":
        # untreated plants with poor (high) discharge withhold reports, beyond X and T
        keep = rng.random(len(d)) > 0.6 * (eps > 0.10) * (T == 0)
        R = R * keep
    d["R"] = R
    d["Y"] = np.where(R == 1, ystar + off, np.nan)
    cl = d.country.values
    out = []
    cc = d[d.R == 1]
    out.append(("Naive difference", cc.Y[cc["T"] == 1].mean() - cc.Y[cc["T"] == 0].mean(), np.nan, np.nan))
    out.append(("Treatment-class median (EF-style)", cc.Y[cc["T"] == 1].median() - cc.Y[cc["T"] == 0].median(), np.nan, np.nan))
    th, se = plr_complete_case(d, X_cols, seed=rep); out.append(("Complete-case DML-PLR", th, se, np.nan))
    ccm = (d.R == 1).to_numpy()
    dc = cc.reset_index(drop=True)
    nu_cc = fit_nuisances(dc, X_cols, seed=rep, use_pi=False, use_src=False)
    phi_cc = phi = pseudo_outcome(dc, nu_cc)
    th, se = cluster_se(phi, dc.country.values, sup[ccm])
    thp, _ = cluster_se(phi, dc.country.values, pol[ccm]) if pol[ccm].sum() > 0 else (np.nan, 0)
    out.append(("Complete-case AIPW", th, se, thp))
    phi = pseudo_outcome(d, fit_nuisances(d, X_cols, seed=rep, use_pi=True, use_src=False))
    th, se = cluster_se(phi, cl, sup); thp, _ = cluster_se(phi, cl, pol)
    out.append(("Missingness-aware AIPW", th, se, thp))
    nu = fit_nuisances(d, X_cols, seed=rep, use_pi=True, use_src=True)
    phi = pseudo_outcome(d, nu)
    th, se = cluster_se(phi, cl, sup); thp, _ = cluster_se(phi, cl, pol)
    out.append(("MP-DR (ours)", th, se, thp))
    # CATE quality (PEHE on the joint support): support-trained DR-learner vs complete-case DR-learner
    import lightgbm as lgb
    def cate_fit(Xa, ph, w):
        ph = np.clip(ph, *np.quantile(ph, [0.01, 0.99]))
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                 verbose=-1, n_jobs=4, random_state=rep).fit(Xa, ph, sample_weight=w)
    si = np.where(sup)[0]
    cate = cate_fit(X[si], phi[si], nu["e"][si] * (1 - nu["e"][si])).predict(X)
    Xcc = X[ccm]
    cate_cc = cate_fit(Xcc, phi_cc, nu_cc["e"] * (1 - nu_cc["e"])).predict(X)
    pehe = np.sqrt(np.mean((cate[sup] - tau_true[sup]) ** 2))
    pehe_cc = np.sqrt(np.mean((cate_cc[sup] - tau_true[sup]) ** 2))
    pehe_const = np.std(tau_true[sup])
    rows = []
    for name, th, se, thp in out:
        rows.append(dict(scenario=scenario, rep=rep, estimator=name, est=th, se=se,
                         bias=th - truth["registry"],
                         covered=float(abs(th - truth["registry"]) <= 1.96 * se) if np.isfinite(se) else np.nan,
                         policy_bias=(thp - truth["policy"]) if np.isfinite(thp) else np.nan,
                         pehe=pehe if name == "MP-DR (ours)" else (pehe_cc if name == "Complete-case AIPW" else np.nan),
                         pehe_constant=pehe_const))
    return rows


if __name__ == "__main__":
    nrep = 20
    jobs = [(r, s) for s in ["A_MAR", "B_MNAR", "C_provenance"] for r in range(nrep)]
    res = Parallel(n_jobs=10)(delayed(one_rep)(r, s) for r, s in jobs)
    res = pd.DataFrame([x for rr in res for x in rr])
    res.to_csv(RES / "05_semisynthetic_reps.csv", index=False)
    summ = res.groupby(["scenario", "estimator"]).agg(
        mean_bias=("bias", "mean"), rmse=("bias", lambda b: np.sqrt(np.mean(b ** 2))),
        coverage=("covered", "mean"), policy_bias=("policy_bias", "mean"), pehe=("pehe", "mean"),
        pehe_constant=("pehe_constant", "mean"), mean_se=("se", "mean"))
    summ.to_csv(RES / "05_semisynthetic_summary.csv")
    print("truth", truth, "offsets", OFF)
    print(summ.round(4).to_string())
