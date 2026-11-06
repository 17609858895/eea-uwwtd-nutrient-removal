"""Clean the UWWTP table: parse numbers, merge M/C/E load columns with provenance."""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "uwwtp_raw.parquet")


def to_num(s):
    if s.dtype == object:
        s = s.astype(str).str.strip().str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        s = s.replace({"None": np.nan, "nan": np.nan, "": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")


POLL = ["BOD", "COD", "N", "P"]
SRC = ["Measured", "Calculated", "Estimated"]
for p in POLL:
    for io in ["Incoming", "Discharge"]:
        for s in SRC:
            c = f"uww{p}{io}{s}"
            df[c] = to_num(df[c])

out = pd.DataFrame({
    "uwwCode": df.uwwCode,
    "country": df.rptMStateKey,
    "nuts": df.uwwNUTS,
    "nuts2": df.uwwNUTS.str[:4],
    "lat": df.uwwLatitude,
    "lon": df.uwwLongitude,
    "capacity_pe": df.uwwCapacity,
    "load_pe": df.uwwLoadEnteringUWWTP,
})
for c in ["Primary", "Secondary", "Other"]:
    out[f"trt_{c.lower()}"] = df[f"uww{c}Treatment"]
for c in ["NRemoval", "PRemoval", "UV", "Chlorination", "Ozonation",
          "SandFiltration", "MicroFiltration", "Other"]:
    out[f"f_{c}"] = df[f"uww{c}"]
for p in ["BOD5", "COD", "TSS", "NTot", "PTot"]:
    out[f"perf_{p}"] = df[f"uww{p}Perf"]
out["bad_perf"] = to_num(df.uwwBadPerformance)
out["bad_design"] = to_num(df.uwwBadDesign)
out["accidents"] = to_num(df.uwwAccidents)
out["ww_treated_m3"] = df.uwwWasteWaterTreated
out["ww_method"] = df.uwwMethodWasteWaterTreated
out["reuse_pct"] = to_num(df.uwwWasteWaterReuse)
bl = pd.to_datetime(df.uwwBeginLife.str[:10], errors="coerce")
out["begin_year"] = bl.dt.year

# Merge sources: a value counts as reported when > 0; priority M > C > E.
for p in POLL:
    for io in ["Incoming", "Discharge"]:
        vals = np.full(len(df), np.nan)
        src = np.array([None] * len(df), dtype=object)
        nrep = np.zeros(len(df), dtype=int)
        for s in SRC:
            v = df[f"uww{p}{io}{s}"].to_numpy()
            ok = np.isfinite(v) & (v > 0)
            nrep += ok
            take = ok & ~np.isfinite(vals)
            vals[take] = v[take]
            src[take] = s[0]
        key = f"{p}_{'in' if io == 'Incoming' else 'out'}"
        out[f"{key}_t"] = vals
        out[f"{key}_src"] = src
        out[f"{key}_nsrc"] = nrep

out.to_parquet(ROOT / "data" / "uwwtp_clean.parquet", index=False)

# --- quick profile
print(out.shape)
for p in POLL:
    for k in ["in", "out"]:
        print(p, k, out[f"{p}_{k}_src"].value_counts(dropna=False).to_dict(),
              "multi-source rows:", int((out[f"{p}_{k}_nsrc"] > 1).sum()))
both = out.N_in_t.notna() & out.N_out_t.notna()
print("N in&out:", both.sum(), " P in&out:", (out.P_in_t.notna() & out.P_out_t.notna()).sum())
print("N out only (no in):", (out.N_out_t.notna() & out.N_in_t.isna()).sum())
print("No N out:", out.N_out_t.isna().sum())
print(out.groupby("country").apply(lambda g: pd.Series({
    "n": len(g), "N_out%": g.N_out_t.notna().mean() * 100,
    "N_outM%": (g.N_out_src == "M").mean() * 100,
    "NRem%": g.f_NRemoval.mean() * 100, "PRem%": g.f_PRemoval.mean() * 100,
    "medPE": g.load_pe.median()})).round(1).to_string())
