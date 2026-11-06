"""21-learner benchmark + super-learner stacking for per-capita N/P discharge.

Usage: python 06_benchmark20.py N LOCO   (pollutant: N|P; scheme: LOCO|random5)
Stacking is nested: meta-weights come from inner CV inside each outer training fold
(grouped by country for LOCO), so no outer-test information enters the stack.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
                                  HuberRegressor, RidgeCV)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from common import load, BASE_FEATS, r2, RES
from causal_core import add_context


def pre(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


class Subsampled(BaseEstimator, RegressorMixin):
    """Fit on a random subsample (for O(n^3) learners such as GP)."""
    def __init__(self, est=None, n=2000, seed=0):
        self.est, self.n, self.seed = est, n, seed

    def fit(self, X, y):
        idx = np.random.default_rng(self.seed).permutation(len(y))[: self.n]
        self.m_ = clone(self.est).fit(X[idx], y[idx]); return self

    def predict(self, X):
        return np.ravel(self.m_.predict(X))


class PLS(PLSRegression):
    def predict(self, X, copy=True):
        return np.ravel(super().predict(X))


MODELS = {
    "OLS": lambda: pre(LinearRegression()),
    "Ridge": lambda: pre(Ridge(alpha=10)),
    "Lasso": lambda: pre(Lasso(alpha=1e-3, max_iter=5000)),
    "ElasticNet": lambda: pre(ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000)),
    "BayesRidge": lambda: pre(BayesianRidge()),
    "Huber": lambda: pre(HuberRegressor(max_iter=500)),
    "PLS": lambda: pre(PLS(n_components=8)),
    "KNN": lambda: pre(KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "SVR": lambda: pre(SVR(C=1.0, epsilon=0.05, cache_size=1000)),
    "KRR-Nystroem": lambda: pre(make_pipeline(Nystroem(n_components=600, random_state=0), Ridge(alpha=1.0))),
    "GPR": lambda: pre(Subsampled(GaussianProcessRegressor(
        ConstantKernel() * RBF(length_scale=3.0) + WhiteKernel(), normalize_y=True), n=2000)),
    "DecisionTree": lambda: pre(DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=0)),
    "RandomForest": lambda: pre(RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, n_jobs=1, random_state=0)),
    "ExtraTrees": lambda: pre(ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.7, n_jobs=1, random_state=0)),
    "AdaBoost": lambda: pre(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.05, random_state=0)),
    "GBDT": lambda: pre(GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)),
    "HistGB": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=30, random_state=0),
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
                                        colsample_bytree=0.8, min_child_weight=5, n_jobs=1, random_state=0),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1, random_state=0),
    "CatBoost": lambda: cb.CatBoostRegressor(iterations=700, learning_rate=0.05, depth=6, verbose=0, thread_count=1, random_seed=0),
    "MLP": lambda: pre(MLPRegressor((128, 64), alpha=1e-3, early_stopping=True, max_iter=400, random_state=0)),
}


def fit_predict_all(Xtr, ytr, Xte):
    """Fit every learner; predictions are clipped to the training-label range +/- 0.5."""
    lo, hi = ytr.min() - 0.5, ytr.max() + 0.5
    out = {}
    for name, mk in MODELS.items():
        out[name] = np.clip(np.ravel(mk().fit(Xtr, ytr).predict(Xte)), lo, hi)
    return out


def mb_hybrid(dtr, dte, feats):
    """Mass-balance hybrid: log q_out = f_in(x) + log10(sigmoid(g(x))), pass-through bounded in (0,1)."""
    mk = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
    fin = np.isfinite(dtr.y_logq_in) & (dtr.y_logq_in > -1) & (dtr.y_logq_in < 1.7)
    m_in = mk().fit(dtr.loc[fin, feats].astype(float), dtr.y_logq_in[fin])
    ok = np.isfinite(dtr.y_logit_pass) & fin & np.isfinite(dtr.y_logq_out)
    m_g = mk().fit(dtr.loc[ok, feats].astype(float), dtr.y_logit_pass[ok])
    Xte = dte[feats].astype(float)
    g = m_g.predict(Xte)
    return m_in.predict(Xte) + np.log10(1 / (1 + np.exp(-g)))


def outer_fold(k, tr, te, lab, feats, scheme):
    t0 = time.time()
    Xall = lab[feats].astype(float).to_numpy()
    y = lab.y_logq_out.to_numpy()
    Xte = Xall[te].copy(); Xte[:, -2:] = 0.0          # standardise provenance to Measured
    base = fit_predict_all(Xall[tr], y[tr], Xte)
    # inner CV for stacking weights
    inner = (GroupKFold(5).split(tr, groups=lab.country.values[tr]) if scheme == "LOCO"
             else KFold(5, shuffle=True, random_state=k).split(tr))
    Z = np.zeros((len(tr), len(MODELS)))
    for itr, ite in inner:
        Xi = Xall[tr[ite]].copy(); Xi[:, -2:] = 0.0
        p = fit_predict_all(Xall[tr[itr]], y[tr[itr]], Xi)
        Z[ite] = np.column_stack([p[m] for m in MODELS])
    w, _ = nnls(Z, y[tr]); w = w / w.sum()
    P = np.column_stack([base[m] for m in MODELS])
    base["SuperLearner (NNLS)"] = P @ w
    meta = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                             verbose=-1, n_jobs=1).fit(Z, y[tr])
    base["Stacking (LightGBM meta)"] = meta.predict(P)
    base["MB-hybrid (LightGBM)"] = mb_hybrid(lab.iloc[tr], lab.iloc[te], feats[:-2])
    print(f"fold {k} done in {time.time() - t0:.0f}s", flush=True)
    return te, base, dict(zip(MODELS, w))


if __name__ == "__main__":
    P, scheme = sys.argv[1], sys.argv[2]
    d, ctx = add_context(load(P))
    lab = d[np.isfinite(d.y_logq_out)].copy()
    lab = lab[(lab.y_logq_out > np.log10(0.005)) & (lab.y_logq_out < np.log10(50))].reset_index(drop=True)
    lab["src_C"] = (lab.src_out == "C").astype(float)
    lab["src_E"] = (lab.src_out == "E").astype(float)
    # country-context features were dropped: under LOCO they act as country IDs and
    # degrade P transfer (results/03_transfer_ideas.csv)
    feats = BASE_FEATS + ["src_C", "src_E"]
    g = lab.country.values
    splits = (list(GroupKFold(n_splits=len(np.unique(g))).split(lab, groups=g)) if scheme == "LOCO"
              else list(KFold(5, shuffle=True, random_state=0).split(lab)))
    res = Parallel(n_jobs=min(len(splits), 28))(delayed(outer_fold)(k, tr, te, lab, feats, scheme)
                                                for k, (tr, te) in enumerate(splits))
    names = list(res[0][1].keys())
    oof = pd.DataFrame(np.nan, index=lab.index, columns=names)
    W = []
    for te, base, w in res:
        for m in names:
            oof.loc[te, m] = base[m]
        W.append(w)
    oof = pd.concat([lab[["uwwCode", "country", "src_out", "y_logq_out", "log_load"]], oof], axis=1)
    oof.to_parquet(RES / f"06_oof_{P}_{scheme}.parquet", index=False)
    pd.DataFrame(W).to_csv(RES / f"06_superlearner_weights_{P}_{scheme}.csv", index=False)
    rows = []
    y = lab.y_logq_out.to_numpy()
    for sub, mask in [("all", np.ones(len(y), bool)), ("Measured", (lab.src_out == "M").to_numpy())]:
        for m in names:
            p = oof[m].to_numpy()
            rows.append(dict(P=P, scheme=scheme, subset=sub, model=m, n=int(mask.sum()),
                             R2=r2(y[mask], p[mask]), RMSE=np.sqrt(np.mean((y[mask] - p[mask]) ** 2)),
                             MAE=np.mean(np.abs(y[mask] - p[mask])),
                             R2_totalload=r2(y[mask] + lab.log_load.values[mask], p[mask] + lab.log_load.values[mask])))
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"06_benchmark_{P}_{scheme}.csv", index=False)
    print(out[out.subset == "Measured"].sort_values("R2", ascending=False).round(3).to_string())
