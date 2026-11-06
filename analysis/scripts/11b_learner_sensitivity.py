"""Nuisance-learner sensitivity on common populations.

Compares LightGBM (10 splits) and super-learner (3 splits) MOSAIC-DR scores, both averaged over
splits, on the same populations: the joint support defined by LightGBM, all labelled countries and
the whole registry. Also reports the median reporting probability in countries without labels.
Run after 11_final_estimates.py in both modes.
"""
import numpy as np
import pandas as pd
from common import RES
from causal_core import weighted_se, joint_support, labelled_country

rows = []
for P in ["N", "P"]:
    lg = pd.read_parquet(RES / f"11_units_lgbm_{P}_log.parquet")
    sl = pd.read_parquet(RES / f"11_units_sl_{P}_log.parquet")
    assert (lg.uwwCode.values == sl.uwwCode.values).all()
    lab = labelled_country(lg)
    D = joint_support(lg.e, lg.pi1, lg.pi0, lab)
    for name, u in [("LightGBM", lg), ("Super learner", sl)]:
        e = u.e.values; h = e * (1 - e)
        for pop, m in [("joint support (LightGBM-defined)", D), ("labelled countries", lab),
                       ("registry", np.ones(len(u), bool))]:
            th, se = weighted_se(u.phi.values, h, u.country.values, m)
            rows.append(dict(P=P, nuisance=name, population=pop, n=int(m.sum()), effect=th, se=se,
                             red=(1 - 10 ** th) * 100, lo=(1 - 10 ** (th + 1.96 * se)) * 100,
                             hi=(1 - 10 ** (th - 1.96 * se)) * 100,
                             pi_zero_label_median=float(np.median(u.pi1[~lab]))))
r = pd.DataFrame(rows)
r.to_csv(RES / "11_learner_sensitivity.csv", index=False)
print(r.round(3).to_string())
