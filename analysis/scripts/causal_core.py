"""Missingness-aware, provenance-standardised doubly robust (MP-DR) estimation.

Target: tau = E[Y_M(1) - Y_M(0)] over the full registry population, where Y is
log10 per-capita discharge (kg p.e.^-1 yr^-1) and Y_M is the measurement-standardised
outcome. Outcomes are missing for about half the plants (R=0), with missingness that
depends on country and treatment; labels come from Measured/Calculated/Estimated sources.

Nuisances (cross-fitted):
  e(x)      = P(T=1 | X)
  pi(x, t)  = P(R=1 | X, T=t)
  mu_t(x,s) = E[Y | X, T=t, S=s, R=1]
Pseudo-outcome:
  Y~  = Y - [mu_T(x,S) - mu_T(x,M)]                       (provenance standardisation)
  phi = mu_1(x,M) - mu_0(x,M)
        + T R / (e pi_1) (Y~ - mu_1(x,M)) - (1-T) R / ((1-e) pi_0) (Y~ - mu_0(x,M))
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

CTX_SRC = ["f_NRemoval", "f_PRemoval", "f_UV", "f_Chlorination", "req_NTot", "req_PTot",
           "log_load", "util", "ww_per_pe", "f_Other"]
PLANT_X = ["log_load", "log_cap", "util", "lat", "lon", "age", "age_missing", "ww_per_pe",
           "req_NTot", "req_PTot", "f_UV", "f_Chlorination", "f_Ozonation", "f_SandFiltration",
           "f_MicroFiltration", "f_Other"]


def add_context(d):
    """Label-free country context computed from all registry plants (covariates only)."""
    g = d.groupby("country")
    ctx = g[CTX_SRC].mean().add_prefix("ctx_")
    ctx["ctx_n_plants"] = np.log10(g.size())
    ctx["ctx_wwmethod_M"] = g.ww_method.apply(lambda s: (s == "M").mean())
    return d.join(ctx, on="country"), list(ctx.columns)


def labelled_country(d):
    """True for plants in countries with at least one plausible discharge label (R=1)."""
    return d.country.isin(d.loc[d.R == 1, "country"].unique()).to_numpy()


def joint_support(e, pi1, pi0, labelled):
    """Joint support: treatment overlap, reporting positivity and a country with discharge labels.

    The label condition keeps plants in countries that report no discharge out of the support even
    when the reporting model extrapolates a probability above the 0.05 threshold to them.
    """
    e, pi1, pi0 = (np.asarray(v, float) for v in (e, pi1, pi0))
    return (e > 0.05) & (e < 0.95) & (pi1 > 0.05) & (pi0 > 0.05) & np.asarray(labelled, bool)


def clf(seed=0):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              verbose=-1, random_state=seed, n_jobs=4)


def reg(seed=0):
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             verbose=-1, random_state=seed, n_jobs=4)


def prepare(d, treat, X_cols, y_col="y_logq_out", lo=0.005, hi=50):
    d = d[d.trt_secondary == 1].copy()
    d = d[d[treat].notna()].copy()
    y = d[y_col].to_numpy()
    R = np.isfinite(y) & (y > np.log10(lo)) & (y < np.log10(hi))
    d["R"] = R.astype(int)
    d["T"] = d[treat].astype(int)
    d["sC"] = (d.src_out == "C").astype(float)
    d["sE"] = (d.src_out == "E").astype(float)
    d["Y"] = np.where(R, y, np.nan)
    return d.reset_index(drop=True)


def sl_clf(seed=0):
    """Compact stacked (super-learner) classifier for nuisance sensitivity analyses."""
    from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return StackingClassifier(
        [("logit", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(max_iter=2000))),
         ("et", make_pipeline(SimpleImputer(), ExtraTreesClassifier(300, min_samples_leaf=10, n_jobs=3, random_state=seed))),
         ("lgb", clf(seed)), ("hgb", HistGradientBoostingClassifier(random_state=seed))],
        final_estimator=LogisticRegression(max_iter=2000), cv=3, stack_method="predict_proba")


def sl_reg(seed=0):
    """Compact stacked (super-learner) regressor with a non-negative linear meta-learner."""
    from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return StackingRegressor(
        [("ridge", make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), RidgeCV(np.logspace(-2, 3, 11)))),
         ("et", make_pipeline(SimpleImputer(), ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.7, n_jobs=3, random_state=seed))),
         ("lgb", reg(seed)), ("hgb", HistGradientBoostingRegressor(random_state=seed))],
        final_estimator=LinearRegression(positive=True), cv=3)


def fit_nuisances(d, X_cols, k=5, seed=0, use_pi=True, use_src=True, eclip=0.02, piclip=0.02,
                  clf_fn=None, reg_fn=None):
    clf_fn = clf_fn or clf
    reg_fn = reg_fn or reg
    X = d[X_cols].astype(float).to_numpy()
    T = d["T"].to_numpy(); R = d["R"].to_numpy(); Y = d["Y"].to_numpy()
    S = d[["sC", "sE"]].to_numpy()
    n = len(d)
    e = np.zeros(n); pi1 = np.ones(n); pi0 = np.ones(n)
    mu1 = np.zeros(n); mu0 = np.zeros(n); off = np.zeros(n)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        e[te] = clf_fn(seed).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        if use_pi:
            XT = np.c_[X, T]
            m = clf_fn(seed).fit(XT[tr], R[tr])
            pi1[te] = m.predict_proba(np.c_[X[te], np.ones(len(te))])[:, 1]
            pi0[te] = m.predict_proba(np.c_[X[te], np.zeros(len(te))])[:, 1]
        for t, store in [(1, mu1), (0, mu0)]:
            idx = tr[(T[tr] == t) & (R[tr] == 1)]
            Z = np.c_[X, S] if use_src else X
            m = reg_fn(seed).fit(Z[idx], Y[idx])
            ZM = np.c_[X[te], np.zeros((len(te), 2))] if use_src else X[te]
            store[te] = m.predict(ZM)
            if use_src:
                obs = (T[te] == t) & (R[te] == 1)
                off[te[obs]] = m.predict(np.c_[X[te], S[te]][obs]) - store[te[obs]]
    e = np.clip(e, eclip, 1 - eclip)
    pi1 = np.clip(pi1, piclip, 1); pi0 = np.clip(pi0, piclip, 1)
    return dict(e=e, pi1=pi1, pi0=pi0, mu1=mu1, mu0=mu0, off=off)


def pseudo_outcome(d, nu):
    T = d["T"].to_numpy(); R = d["R"].to_numpy()
    Yt = np.where(R == 1, d["Y"].to_numpy() - nu["off"], 0.0)
    return (nu["mu1"] - nu["mu0"]
            + T * R / (nu["e"] * nu["pi1"]) * (Yt - nu["mu1"])
            - (1 - T) * R / ((1 - nu["e"]) * nu["pi0"]) * (Yt - nu["mu0"]))


def cluster_se(phi, clusters, mask=None):
    """Mean of the pseudo-outcome with a country-clustered standard error."""
    if mask is None:
        mask = np.ones(len(phi), bool)
    p = phi[mask]; c = np.asarray(clusters)[mask]
    th = p.mean()
    s = pd.Series(p - th).groupby(c).sum()
    return th, np.sqrt((s ** 2).sum()) / len(p)


def weighted_se(phi, w, clusters, mask=None):
    """Weighted mean of phi (e.g. overlap weights h=e(1-e)) with country-clustered SE."""
    if mask is None:
        mask = np.ones(len(phi), bool)
    p, w, c = phi[mask], w[mask], np.asarray(clusters)[mask]
    th = np.sum(w * p) / np.sum(w)
    s = pd.Series(w * (p - th)).groupby(c).sum()
    return th, np.sqrt((s ** 2).sum()) / np.sum(w)


def repeated_pseudo(d, X_cols, reps=3, **kw):
    """Average pseudo-outcomes and propensities over repeated cross-fitting splits."""
    phis, es, nus = [], [], []
    for r in range(reps):
        nu = fit_nuisances(d, X_cols, seed=r, **kw)
        phis.append(pseudo_outcome(d, nu)); es.append(nu["e"]); nus.append(nu)
    nu = {k: np.mean([n[k] for n in nus], axis=0) for k in nus[0]}
    return np.mean(phis, axis=0), np.mean(es, axis=0), nu


def plr_complete_case(d, X_cols, k=5, seed=0):
    """Partially linear DML on labelled plants only (the common practice)."""
    cc = d[d.R == 1].reset_index(drop=True)
    X = cc[X_cols].astype(float).to_numpy(); T = cc["T"].to_numpy(); Y = cc["Y"].to_numpy()
    rt = np.zeros(len(cc)); ry = np.zeros(len(cc))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        rt[te] = T[te] - clf(seed).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        ry[te] = Y[te] - reg(seed).fit(X[tr], Y[tr]).predict(X[te])
    th = np.sum(rt * ry) / np.sum(rt * rt)
    psi = (ry - th * rt) * rt / np.mean(rt * rt)
    s = pd.Series(psi).groupby(cc.country.values).sum()
    return th, np.sqrt((s ** 2).sum()) / len(cc)
