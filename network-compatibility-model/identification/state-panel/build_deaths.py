"""Aggregate DGIS/INEGI death-certificate microdata to state x year-of-occurrence counts.

Input: data/external/state-panel/dgis_defunciones/DEFUN_<year>.zip (registration years
2015-2023; some archives contain a nested zip). Run with `python -I`.
Output: outputs/deaths_state_year.csv  (state of residence, year of occurrence,
all-cause deaths, COVID deaths = ICD-10 U071/U072 as underlying cause).
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "external" / "state-panel" / "dgis_defunciones"
OUT = Path(__file__).resolve().parent / "outputs"


def open_csv(zpath):
    z = zipfile.ZipFile(zpath)
    big = max(z.infolist(), key=lambda i: i.file_size)
    data = z.read(big)
    if big.filename.lower().endswith(".zip"):
        inner = zipfile.ZipFile(io.BytesIO(data))
        big2 = max(inner.infolist(), key=lambda i: i.file_size)
        data = inner.read(big2)
    return io.BytesIO(data)


frames = []
for reg_year in range(2015, 2024):
    f = open_csv(RAW / f"DEFUN_{reg_year}.zip")
    df = pd.read_csv(f, encoding="latin-1", dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    occ = "anio_ocur" if "anio_ocur" in df.columns else "anio_ocurr"
    d = pd.DataFrame({
        "ent": pd.to_numeric(df["ent_resid"].str.strip(), errors="coerce"),
        "year": pd.to_numeric(df[occ].str.strip(), errors="coerce"),
        "covid": df["causa_def"].str.strip().str.upper().str[:4].isin(["U071", "U072"]),
    })
    d["reg_year"] = reg_year
    g = d.groupby(["ent", "year", "reg_year"]).agg(deaths=("covid", "size"), covid=("covid", "sum")).reset_index()
    frames.append(g)
    print(reg_year, len(df), file=sys.stderr)

allg = pd.concat(frames)
allg = allg[(allg.ent >= 1) & (allg.ent <= 32) & (allg.year >= 2015) & (allg.year <= 2023)]
allg.to_csv(OUT / "deaths_state_year_by_regyear.csv", index=False)
agg = allg.groupby(["ent", "year"])[["deaths", "covid"]].sum().reset_index()
agg[["ent", "year"]] = agg[["ent", "year"]].astype(int)
agg.to_csv(OUT / "deaths_state_year.csv", index=False)
print(agg.groupby("year")[["deaths", "covid"]].sum())
