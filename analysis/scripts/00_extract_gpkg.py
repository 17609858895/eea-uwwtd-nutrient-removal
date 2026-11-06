"""Extract UWWTD_TreatmentPlants from the official ZIP (in memory) to parquet/csv."""
import os, sqlite3, zipfile, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
ZIP = pathlib.Path(os.environ.get("UWWTD_SOURCE_ZIP", ROOT / "raw" / "EEA_UWWTD_PUBLIC_FINAL_2022-2023_v10_r00.zip"))
OUT = ROOT / "analysis" / "data"
OUT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ZIP) as z:
    name = [n for n in z.namelist() if n.endswith(".gpkg")][0]
    blob = z.read(name)

con = sqlite3.connect(":memory:")
con.deserialize(blob)
df = pd.read_sql("SELECT * FROM UWWTD_TreatmentPlants", con)
df = df.drop(columns=["geom"])
df.to_parquet(OUT / "uwwtp_raw.parquet", index=False)
df.to_csv(OUT / "uwwtp_raw.csv", index=False, encoding="utf-8-sig")
print(df.shape)
print(df.dtypes.to_string())
