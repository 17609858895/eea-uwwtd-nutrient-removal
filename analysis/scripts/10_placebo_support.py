"""Placebo (permuted treatment) and real effects split by identification region.

Regions: joint support (treatment overlap and outcome-observation positivity), outside
support, and zero-label countries (no reported discharge at all, pure transport).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import load, RES
from causal_core import add_context, prepare, repeated_pseudo, weighted_se, PLANT_X, joint_support, labelled_country

rows = []
for P in ["N", "P"]:
    treat, other = ("f_NRemoval", "f_PRemoval") if P == "N" else ("f_PRemoval", "f_NRemoval")
    d0, ctx = add_context(load(P))
    X_cols = PLANT_X + [other] + ctx
    d = prepare(d0, treat, X_cols)
    lab = labelled_country(d)
    nolab = ~lab
    for kind in ["real", "placebo"]:
        for rep in range(3 if kind == "placebo" else 1):
            dd = d.copy()
            if kind == "placebo":
                dd["T"] = np.random.default_rng(rep).permutation(dd["T"].to_numpy())
            phi, e, nu = repeated_pseudo(dd, X_cols, 2)
            h = e * (1 - e)
            js = joint_support(e, nu["pi1"], nu["pi0"], lab)
            for region, m in [("all", np.ones(len(dd), bool)), ("joint support", js),
                              ("outside support", ~js & ~nolab), ("zero-label countries", nolab)]:
                th, se = weighted_se(phi, h, dd.country.values, m)
                rows.append(dict(P=P, kind=kind, rep=rep, region=region, n=int(m.sum()), effect=th, se=se))
                print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(RES / "10_placebo_by_support.csv", index=False)
