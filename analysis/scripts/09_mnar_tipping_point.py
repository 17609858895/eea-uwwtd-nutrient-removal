"""Selective-reporting (MNAR) tipping point by delta adjustment.

Unreported untreated plants are assumed to discharge 10^delta times as much as comparable reporting
plants. Adding delta to their log10 outcome shifts the overlap-weighted effect by approximately
-delta x (overlap-weighted mean non-reporting probability under no treatment); the tipping delta is
the shift that brings the effect to the emission-factor value. Uses the 3-split estimates of 04b.
"""
import numpy as np
import pandas as pd
from common import RES
from causal_core import weighted_se, joint_support, labelled_country

EF = {"N": np.log10(0.20 / 0.45), "P": np.log10(0.10 / 0.40)}

rows = []
for P in ["N", "P"]:
    u = pd.read_parquet(RES / f"04b_causal_units_{P}.parquet")
    e = u.e.values; h = e * (1 - e)
    js = joint_support(e, u.pi1, u.pi0, labelled_country(u))
    for pop, m in [("registry", np.ones(len(u), bool)), ("joint support", js)]:
        th, _ = weighted_se(u.phi.values, h, u.country.values, m)
        miss0 = np.average(1 - u.pi0.values[m], weights=h[m])
        d_tip = (th - EF[P]) / miss0
        rows.append(dict(P=P, population=pop, effect=th, EF=EF[P], mean_unreported_untreated=miss0,
                         delta_tip_log10=d_tip, factor=10 ** d_tip))
r = pd.DataFrame(rows)
r.to_csv(RES / "09_mnar_tipping_point.csv", index=False)
print(r.round(3).to_string())
