"""Event-study trajectories and plant locations for the three-round treatment histories.

Reuses the cleaning, linking and within-country contrast of 19_panel_check.py without
rewriting its outputs. For plants with secondary treatment and plausible discharge in all three
rounds (2018, 2020, 2022), the change in log10 per-capita discharge since 2018 is compared
between each treatment-history group and never-treated plants of the same country:
  011  removal added by 2020 and kept;  001  removal added by 2022;  111  removal throughout.

Outputs: results/20_panel_event.csv, results/20_panel_plants.parquet.
"""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("panel19", HERE / "19_panel_check.py")
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
RES = pc.RES
YEARS = (2018, 2020, 2022)


def histories(R, P):
    t = f"f_{P}Removal"
    cols = ["uwwCode", "country", "trt_secondary", t, f"{P}_y", f"{P}_src"]
    a, b, c = (R[y][cols].rename(columns=lambda k, y=y: k if k == "uwwCode" else f"{k}_{y}") for y in YEARS)
    m = a.merge(b, on="uwwCode").merge(c, on="uwwCode")
    m = m[(m[[f"trt_secondary_{y}" for y in YEARS]] == 1).all(axis=1) & (m.country_2018 == m.country_2022)]
    m = m.assign(country=m.country_2018)
    m["pattern"] = (m[f"{t}_2018"].astype("Int64").astype(str) + m[f"{t}_2020"].astype("Int64").astype(str)
                    + m[f"{t}_2022"].astype("Int64").astype(str))
    return m


if __name__ == "__main__":
    R = {y: pc.load_round(p) for y, p in pc.ROUNDS.items()}
    R[2022] = pc.load_2022()
    rows, plants = [], []
    for P in pc.POLL:
        m = histories(R, P)
        plants.append(m[["uwwCode", "country", "pattern"]].assign(P=P))
        for lab, measured in [("all plausible", False), ("measured", True)]:
            ok = np.ones(len(m), bool)
            for y in YEARS:
                ok &= m[f"{P}_y_{y}"].notna().to_numpy()
                if measured:
                    ok &= (m[f"{P}_src_{y}"] == "M").to_numpy()
            mm = m[ok]
            for g in ("011", "001", "111"):
                k = mm.pattern.isin([g, "000"])
                sub = mm[k]
                rows.append(dict(P=P, labels=lab, group=g, year=2018, effect_log10=0.0, se=0.0,
                                 switchers=int((sub.pattern == g).sum()), never_treated=int((sub.pattern == "000").sum())))
                for y in (2020, 2022):
                    r = pc.within_country(sub, sub[f"{P}_y_{y}"] - sub[f"{P}_y_2018"], sub.pattern == g)
                    if r is None:
                        continue
                    rows.append(dict(P=P, labels=lab, group=g, year=y, effect_log10=r["effect_log10"], se=r["se"],
                                     switchers=r["switchers"], never_treated=r["never_treated"]))
    e = pd.DataFrame(rows)
    e["change_pct"] = (10 ** e.effect_log10 - 1) * 100
    e["lo_pct"] = (10 ** (e.effect_log10 - 1.96 * e.se) - 1) * 100
    e["hi_pct"] = (10 ** (e.effect_log10 + 1.96 * e.se) - 1) * 100
    e.to_csv(RES / "20_panel_event.csv", index=False)
    loc = pd.read_parquet(pc.ROOT / "data" / "uwwtp_clean.parquet")[["uwwCode", "lat", "lon", "load_pe"]]
    pl = pd.concat(plants).merge(loc.drop_duplicates("uwwCode"), on="uwwCode", how="left")
    pl.to_parquet(RES / "20_panel_plants.parquet", index=False)
    pd.set_option("display.width", 200)
    print(e.round(3).to_string())
    print(pl.groupby(["P", "pattern"]).size().unstack(0).loc[["111", "000", "011", "001", "010", "110", "100", "101"]])
