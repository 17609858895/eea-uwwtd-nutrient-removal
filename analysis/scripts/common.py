"""Shared feature construction for the UWWTD nutrient study."""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

FLAGS = ["f_NRemoval", "f_PRemoval", "f_UV", "f_Chlorination", "f_Ozonation",
         "f_SandFiltration", "f_MicroFiltration", "f_Other"]
TRT = ["trt_secondary", "trt_other"]


def load(pollutant="N"):
    d = pd.read_parquet(DATA / "uwwtp_clean.parquet")
    d = d[(d.load_pe > 0) & (d.capacity_pe > 0)].copy()
    d["log_load"] = np.log10(d.load_pe)
    d["log_cap"] = np.log10(d.capacity_pe)
    d["util"] = np.clip(d.load_pe / d.capacity_pe, 0, 3)
    d["age"] = 2022 - d.begin_year
    d["age_missing"] = d.age.isna().astype(int)
    for p in ["NTot", "PTot", "BOD5", "COD"]:
        # regulatory requirement status (pre-treatment): P/F means a limit applies
        d[f"req_{p}"] = d[f"perf_{p}"].isin(["P", "F"]).astype(float)
        d.loc[d[f"perf_{p}"].isna() | d[f"perf_{p}"].isin(["NA"]), f"req_{p}"] = np.nan
    d["ww_per_pe"] = np.log10(d.ww_treated_m3 / d.load_pe).where(d.ww_treated_m3 > 0)
    p = pollutant
    q_in = d[f"{p}_in_t"] * 1000 / d.load_pe          # kg / p.e. / yr
    q_out = d[f"{p}_out_t"] * 1000 / d.load_pe
    d["y_logq_out"] = np.log10(q_out)
    d["y_logq_in"] = np.log10(q_in)
    rem = 1 - d[f"{p}_out_t"] / d[f"{p}_in_t"]
    d["removal"] = rem
    d["y_logit_pass"] = np.log(np.clip(1 - rem, 1e-3, 1) / np.clip(rem, 1e-3, 1))  # logit(out/in)
    d["src_out"] = d[f"{p}_out_src"]
    d["src_in"] = d[f"{p}_in_src"]
    return d


BASE_FEATS = ["log_load", "log_cap", "util", "lat", "lon", "age", "age_missing",
              "ww_per_pe"] + TRT + FLAGS + ["req_NTot", "req_PTot"]


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
