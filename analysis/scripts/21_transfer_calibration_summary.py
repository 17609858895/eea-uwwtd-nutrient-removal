"""Summaries behind the component ablation, interval calibration and country-transfer panels.

Reads the semi-synthetic outputs of 05_semisynthetic.py and the benchmark outputs of
06_benchmark20.py; nothing is refitted. Writes
  results/21_semisynthetic_ablation.csv   absolute average-effect bias as components are added
  results/21_interval_calibration.csv     mean standard error / s.d. of replicate estimates
  results/21_transfer_loss.csv            R2 under random minus leave-country-out validation
  results/21_country_rmse.csv             RMSE of withheld-country predictions (measured labels)
"""
import numpy as np
import pandas as pd
from common import RES

SCENARIOS = ["A_MAR", "B_MNAR", "C_provenance"]
STEPS = [("Complete-case AIPW", "complete-case AIPW"), ("Missingness-aware AIPW", "+ reporting model"),
         ("MP-DR (ours)", "MOSAIC-DR")]
ESTIMATORS = ["Complete-case DML-PLR", "Complete-case AIPW", "Missingness-aware AIPW", "MP-DR (ours)"]
FAMILY = {"Linear": ["OLS", "Ridge", "Lasso", "ElasticNet", "BayesRidge", "Huber", "PLS"],
          "Kernel, k-NN": ["KNN", "SVR", "KRR-Nystroem", "GPR"],
          "Trees, bagging": ["DecisionTree", "RandomForest", "ExtraTrees", "AdaBoost"],
          "Boosting": ["GBDT", "HistGB", "XGBoost", "LightGBM", "CatBoost"],
          "Neural net": ["MLP"],
          "Ensembles": ["SuperLearner (NNLS)", "Stacking (LightGBM meta)", "MB-hybrid (LightGBM)"]}
HEAT = ["Lasso", "SVR", "KNN", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "MLP",
        "MB-hybrid (LightGBM)", "SuperLearner (NNLS)"]
MIN_LABELS = 30


def ablation():
    s = pd.read_csv(RES / "05_semisynthetic_summary.csv").set_index(["scenario", "estimator"])
    return pd.DataFrame([dict(scenario=sc, step=i, label=lab, estimator=est, abs_bias=abs(s.loc[(sc, est), "mean_bias"]))
                         for sc in SCENARIOS for i, (est, lab) in enumerate(STEPS)])


def calibration():
    r = pd.read_csv(RES / "05_semisynthetic_reps.csv")
    s = pd.read_csv(RES / "05_semisynthetic_summary.csv").set_index(["scenario", "estimator"])
    rows = []
    for est in ESTIMATORS:
        for sc in SCENARIOS:
            sd = r[(r.scenario == sc) & (r.estimator == est)].est.std()
            se = s.loc[(sc, est), "mean_se"]
            rows.append(dict(scenario=sc, estimator=est, mean_se=se, sd_estimates=sd, ratio=se / sd))
    return pd.DataFrame(rows)


def transfer_loss():
    rows = []
    for P in "NP":
        a = pd.read_csv(RES / f"06_benchmark_{P}_LOCO.csv"); b = pd.read_csv(RES / f"06_benchmark_{P}_random5.csv")
        a = a[a.subset == "Measured"].set_index("model").R2; b = b[b.subset == "Measured"].set_index("model").R2
        for fam, models in FAMILY.items():
            for m in models:
                if m in a.index and m in b.index:
                    rows.append(dict(P=P, family=fam, model=m, R2_random=b[m], R2_LOCO=a[m], loss=b[m] - a[m]))
    return pd.DataFrame(rows)


def country_rmse():
    rows = []
    for P in "NP":
        o = pd.read_parquet(RES / f"06_oof_{P}_LOCO.parquet")
        o = o[o.src_out == "M"]
        n = o.country.value_counts()
        for c in n[n >= MIN_LABELS].index:
            d = o[o.country == c]
            for m in HEAT:
                rows.append(dict(P=P, country=c, n_measured=int(n[c]), model=m,
                                 rmse=float(np.sqrt(np.mean((d[m] - d.y_logq_out) ** 2)))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ab, ca, tl, cr = ablation(), calibration(), transfer_loss(), country_rmse()
    ab.to_csv(RES / "21_semisynthetic_ablation.csv", index=False)
    ca.to_csv(RES / "21_interval_calibration.csv", index=False)
    tl.to_csv(RES / "21_transfer_loss.csv", index=False)
    cr.to_csv(RES / "21_country_rmse.csv", index=False)
    pd.set_option("display.width", 200)
    print(ab.pivot(index="label", columns="scenario", values="abs_bias").round(3))
    print("SE / s.d. ratio range:", round(ca.ratio.min(), 2), "-", round(ca.ratio.max(), 2))
    print(tl.groupby(["P", "family"]).loss.median().unstack(0).round(3))
    for P in "NP":
        M = cr[cr.P == P].pivot(index="country", columns="model", values="rmse")
        med = M.median(axis=1); spread = M.max(axis=1) - M.min(axis=1)
        print(P, "country median RMSE", round(med.min(), 2), "-", round(med.max(), 2),
              "| median within-country spread", round(spread.median(), 2),
              "| most countries won by one learner", int(M.idxmin(axis=1).value_counts().max()))
