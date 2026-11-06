"""Before-and-after check on earlier registry rounds (reference years 2018 and 2020).

The 2018 and 2020 treatment-plant releases are cleaned with the rules used for 2022 and linked
by plant code. Plants whose nutrient-removal flag switched on between two rounds ("switchers")
are compared with plants that had no removal in either round ("never treated"): the change in
log10 per-capita discharge, differenced against never-treated plants of the same country, is a
within-plant estimate of the upgrade effect that does not rest on the cross-sectional
no-unmeasured-confounding assumption (it assumes parallel trends instead).

Outputs: results/19_panel_counts.csv, results/19_panel_did.csv.
"""
import sqlite3
import numpy as np
import pandas as pd
from common import ROOT, RES

RAW = ROOT.parent / "raw" / "earlier_releases"
ROUNDS = {2018: RAW / "dc2019_public" / "UWWTD_TreatmentPlants.gpkg",
          2020: RAW / "dc2021_final_public" / "UWWTD_TreatmentPlants.gpkg"}
POLL, SRC = ["N", "P"], ["Measured", "Calculated", "Estimated"]
LO, HI = 0.005, 50                      # plausible per-capita discharge, kg p.e.^-1 yr^-1 (as in the main analysis)


def to_num(s):
    if s.dtype == object:
        s = s.astype(str).str.strip().str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        s = s.replace({"None": np.nan, "nan": np.nan, "": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")


def flag(s):
    """Treatment flags arrive as 0/1, booleans or text; return 0/1 with NaN for missing."""
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower().map({"1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0})
    return pd.to_numeric(s, errors="coerce")


def clean(df):
    out = pd.DataFrame({"uwwCode": df.uwwCode.astype(str).str.strip(), "country": df.rptMStateKey,
                        "load_pe": to_num(df.uwwLoadEnteringUWWTP), "capacity_pe": to_num(df.uwwCapacity),
                        "trt_secondary": flag(df.uwwSecondaryTreatment),
                        "f_NRemoval": flag(df.uwwNRemoval), "f_PRemoval": flag(df.uwwPRemoval)})
    for p in POLL:
        vals = np.full(len(df), np.nan); src = np.array([None] * len(df), dtype=object)
        for s in SRC:                                   # priority M > C > E, a value counts when > 0
            v = to_num(df[f"uww{p}Discharge{s}"]).to_numpy()
            take = np.isfinite(v) & (v > 0) & ~np.isfinite(vals)
            vals[take] = v[take]; src[take] = s[0]
        q = vals * 1000 / out.load_pe.to_numpy()
        ok = np.isfinite(q) & (q > LO) & (q < HI)
        out[f"{p}_y"] = np.where(ok, np.log10(np.where(ok, q, 1)), np.nan)
        out[f"{p}_src"] = np.where(ok, src, None)
    return out


def load_round(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    df = pd.read_sql("SELECT * FROM UWWTD_TreatmentPlants", con).drop(columns=["geom"])
    con.close()
    return clean(df)


def load_2022():
    df = pd.read_parquet(ROOT / "data" / "uwwtp_raw.parquet")
    return clean(df)


def did(a, b, P, measured_only=False):
    """Switchers (0 -> 1) against never treated (0 -> 0), differenced within country."""
    t = f"f_{P}Removal"
    m = a.merge(b, on="uwwCode", suffixes=("_0", "_1"))
    m = m[(m.trt_secondary_0 == 1) & (m.trt_secondary_1 == 1) & (m.country_0 == m.country_1)]
    m = m[(m[f"{t}_0"] == 0) & m[f"{t}_1"].isin([0, 1])]
    m = m[m[f"{P}_y_0"].notna() & m[f"{P}_y_1"].notna()]
    if measured_only:
        m = m[(m[f"{P}_src_0"] == "M") & (m[f"{P}_src_1"] == "M")]
    m = m.assign(dy=m[f"{P}_y_1"] - m[f"{P}_y_0"], sw=m[f"{t}_1"].astype(int), country=m.country_0)
    # countries contributing both switchers and never-treated plants
    g = m.groupby("country").sw.agg(["sum", "size"])
    keep = g[(g["sum"] > 0) & (g["sum"] < g["size"])].index
    m = m[m.country.isin(keep)]
    if m.sw.sum() == 0:
        return None
    # within-country difference: dy demeaned by the never-treated mean of the same country
    base = m[m.sw == 0].groupby("country").dy.mean()
    m = m.assign(dd=m.dy - m.country.map(base))
    s = m[m.sw == 1]
    th = s.dd.mean()
    # country-clustered SE of the switcher mean plus the variance of the country baselines
    w = s.groupby("country").size()
    v0 = (m[m.sw == 0].groupby("country").dy.var(ddof=1) / m[m.sw == 0].groupby("country").size()).fillna(0)
    cl = s.groupby("country").dd.sum() - w * th
    se = np.sqrt((cl ** 2).sum() / len(s) ** 2 + ((w / len(s)) ** 2 * v0.reindex(w.index).fillna(0)).sum())
    return dict(switchers=int(len(s)), never_treated=int((m.sw == 0).sum()), countries=int(len(keep)),
                switcher_countries=", ".join(f"{c} {n}" for c, n in w.sort_values(ascending=False).items()),
                effect_log10=th, se=se, reduction_pct=(1 - 10 ** th) * 100,
                red_lo=(1 - 10 ** (th + 1.96 * se)) * 100, red_hi=(1 - 10 ** (th - 1.96 * se)) * 100,
                raw_switcher_dy=s.dy.mean(), raw_never_dy=m[m.sw == 0].dy.mean())


def within_country(m, dy, sw):
    """Mean of dy at sw==1 minus the never-treated mean of the same country; country-clustered SE."""
    m = m.assign(dy=dy, sw=sw.astype(int))
    g = m.groupby("country").sw.agg(["sum", "size"])
    m = m[m.country.isin(g[(g["sum"] > 0) & (g["sum"] < g["size"])].index)]
    if m.sw.sum() == 0:
        return None
    ctl = m[m.sw == 0].groupby("country").dy
    s = m[m.sw == 1].assign(dd=lambda x: x.dy - x.country.map(ctl.mean()))
    th = s.dd.mean(); w = s.groupby("country").size()
    v0 = (ctl.var(ddof=1) / ctl.size()).reindex(w.index).fillna(0)
    se = np.sqrt(((s.groupby("country").dd.sum() - w * th) ** 2).sum() / len(s) ** 2 + ((w / len(s)) ** 2 * v0).sum())
    within_country.used = s.uwwCode.tolist() if "uwwCode" in s else []
    return dict(switchers=int(len(s)), never_treated=int((m.sw == 0).sum()), countries=int(m.country.nunique()),
                switcher_countries=", ".join(f"{c} {n}" for c, n in w.sort_values(ascending=False).items()),
                effect_log10=th, se=se, reduction_pct=(1 - 10 ** th) * 100,
                red_lo=(1 - 10 ** (th + 1.96 * se)) * 100, red_hi=(1 - 10 ** (th - 1.96 * se)) * 100)


def three_round(R, P, measured_only):
    """Plants in all three rounds with secondary treatment throughout; persistent switches only."""
    t = f"f_{P}Removal"
    a, b, c = (R[y][["uwwCode", "country", "trt_secondary", t, f"{P}_y", f"{P}_src"]].rename(
        columns=lambda k, y=y: k if k == "uwwCode" else f"{k}_{y}") for y in (2018, 2020, 2022))
    m = a.merge(b, on="uwwCode").merge(c, on="uwwCode")
    m = m[(m[["trt_secondary_2018", "trt_secondary_2020", "trt_secondary_2022"]] == 1).all(axis=1)
          & (m.country_2018 == m.country_2022)].assign(country=lambda x: x.country_2018)
    pat = m[f"{t}_2018"].astype("Int64").astype(str) + m[f"{t}_2020"].astype("Int64").astype(str) + \
        m[f"{t}_2022"].astype("Int64").astype(str)
    ok = lambda y: m[f"{P}_y_{y}"].notna() & ((m[f"{P}_src_{y}"] == "M") if measured_only else True)
    out = {"patterns": pat.value_counts().to_dict()}
    lab = "measured" if measured_only else "all plausible"
    rows = []
    # long difference 2018 -> 2022: switched by 2020 and kept (011) or switched by 2022 (001) vs 000
    for name, sw_pat in [("switched 2018-2020, kept to 2022", "011"), ("switched 2020-2022", "001"),
                         ("any persistent switch", ("011", "001"))]:
        k = pat.isin(sw_pat if isinstance(sw_pat, tuple) else (sw_pat,)) | (pat == "000")
        mm = m[k & ok(2018) & ok(2022)]
        r = within_country(mm, mm[f"{P}_y_2022"] - mm[f"{P}_y_2018"], pat[mm.index] != "000")
        if r:
            # cross-sectional prediction for the same switchers: plant-level effect from the 2022 model
            tau = PRED[P].reindex(within_country.used)
            r.update(pred_reduction_pct=float((1 - 10 ** tau.mean()) * 100), pred_n=int(tau.notna().sum()),
                     pred_in_support=float(SUP[P].reindex(tau.index).mean()))
            rows.append(dict(P=P, labels=lab, design=f"2018->2022, {name}", **r))
    # pre-trend: plants that switch only after 2020 (001) should not decline faster in 2018 -> 2020
    k = pat.isin(("001", "000"))
    mm = m[k & ok(2018) & ok(2020)]
    r = within_country(mm, mm[f"{P}_y_2020"] - mm[f"{P}_y_2018"], pat[mm.index] == "001")
    if r: rows.append(dict(P=P, labels=lab, design="pre-trend 2018->2020 of later switchers (001)", **r))
    return out, rows


_pe = {P: pd.read_parquet(RES / f"13_plant_effects_{P}.parquet").drop_duplicates("uwwCode").set_index("uwwCode")
       for P in POLL}
PRED = {P: _pe[P].tau for P in POLL}
SUP = {P: _pe[P].in_support.astype(float) for P in POLL}


if __name__ == "__main__":
    R = {y: load_round(p) for y, p in ROUNDS.items()}
    R[2022] = load_2022()
    rows, est = [], []
    for y, d in R.items():
        sec = d[d.trt_secondary == 1]
        rows.append(dict(item="plants", year=y, n=len(d), secondary=len(sec),
                         N_reported=int(sec.N_y.notna().sum()), P_reported=int(sec.P_y.notna().sum()),
                         N_removal_share=sec.f_NRemoval.mean(), P_removal_share=sec.f_PRemoval.mean(),
                         unique_codes=d.uwwCode.nunique()))
    for y0, y1 in [(2018, 2020), (2020, 2022), (2018, 2022)]:
        a, b = R[y0], R[y1]
        m = a.merge(b, on="uwwCode", suffixes=("_0", "_1"))
        for P in POLL:
            t = f"f_{P}Removal"
            sec = m[(m.trt_secondary_0 == 1) & (m.trt_secondary_1 == 1)]
            both = sec[f"{P}_y_0"].notna() & sec[f"{P}_y_1"].notna()
            rows.append(dict(item=f"{P} {y0}->{y1}", year=y1, n=len(m), secondary=len(sec),
                             on_switch=int(((sec[f"{t}_0"] == 0) & (sec[f"{t}_1"] == 1)).sum()),
                             on_switch_reported=int(((sec[f"{t}_0"] == 0) & (sec[f"{t}_1"] == 1) & both).sum()),
                             off_switch=int(((sec[f"{t}_0"] == 1) & (sec[f"{t}_1"] == 0)).sum()),
                             never_reported=int(((sec[f"{t}_0"] == 0) & (sec[f"{t}_1"] == 0) & both).sum())))
            for mo in (False, True):
                r = did(a, b, P, measured_only=mo)
                if r:
                    est.append(dict(P=P, pair=f"{y0}->{y1}", labels="measured both rounds" if mo else "all plausible", **r))
    # where do removal flags switch off? (recoding rather than de-commissioning)
    off = []
    for y0, y1 in [(2018, 2020), (2020, 2022)]:
        m = R[y0].merge(R[y1], on="uwwCode", suffixes=("_0", "_1"))
        for P in POLL:
            t = f"f_{P}Removal"
            g = m.assign(on=(m[f"{t}_0"] == 0) & (m[f"{t}_1"] == 1), offs=(m[f"{t}_0"] == 1) & (m[f"{t}_1"] == 0)) \
                .groupby("country_0")[["on", "offs"]].sum()
            for cc, r in g[(g.on + g.offs) > 0].iterrows():
                off.append(dict(P=P, pair=f"{y0}->{y1}", country=cc, on=int(r.on), off=int(r.offs)))
    pd.DataFrame(off).to_csv(RES / "19_panel_switch_by_country.csv", index=False)
    three, pats = [], {}
    for P in POLL:
        for mo in (False, True):
            o, r3 = three_round(R, P, mo)
            pats[P] = o["patterns"]; three += r3
    c = pd.DataFrame(rows); e = pd.DataFrame(est)
    c.to_csv(RES / "19_panel_counts.csv", index=False); e.to_csv(RES / "19_panel_did.csv", index=False)
    t3 = pd.DataFrame(three); t3.to_csv(RES / "19_panel_three_round.csv", index=False)
    pd.DataFrame([dict(P=P, pattern=k, plants=v) for P, d in pats.items() for k, v in d.items()]) \
        .to_csv(RES / "19_panel_patterns.csv", index=False)
    print("flag patterns 2018-2020-2022:", pats)
    print(pd.DataFrame(off).pivot_table(index=["P", "country"], columns="pair", values=["on", "off"], fill_value=0)
          .loc[lambda x: x.sum(axis=1) > 40].to_string())
    print(t3.drop(columns=["switcher_countries"]).round(3).to_string()); print()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(c.round(3).to_string()); print()
    print(e.drop(columns=["switcher_countries"]).round(3).to_string()); print()
    print(e[["P", "pair", "labels", "switcher_countries"]].to_string())
