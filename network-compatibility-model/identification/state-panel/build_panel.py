"""Build the Mexican state panel: remittances (Banxico CE166), need shocks, controls.

Reads raw downloads in data/external/state-panel/ (untrusted data; run with `python -I`).
Requires outputs/deaths_state_year.csv from build_deaths.py.
Writes:
  outputs/remittances_state_quarterly.csv   state x quarter, USD millions (2013Q1-2026Q2)
  data/external/state-panel/derived/remittances_municipal_quarterly.csv municipality x quarter,
      USD millions (4.5 MB, kept out of git; rebuild with this script)
  outputs/state_panel.csv                   one row per state: outcomes, need shocks, controls
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "data" / "external" / "state-panel"
OUT = HERE / "outputs"
DERIVED = RAW / "derived"

# INEGI codes and short names
STATES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur", 4: "Campeche",
    5: "Coahuila", 6: "Colima", 7: "Chiapas", 8: "Chihuahua", 9: "Ciudad de Mexico",
    10: "Durango", 11: "Guanajuato", 12: "Guerrero", 13: "Hidalgo", 14: "Jalisco",
    15: "Mexico", 16: "Michoacan", 17: "Morelos", 18: "Nayarit", 19: "Nuevo Leon",
    20: "Oaxaca", 21: "Puebla", 22: "Queretaro", 23: "Quintana Roo", 24: "San Luis Potosi",
    25: "Sinaloa", 26: "Sonora", 27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala",
    30: "Veracruz", 31: "Yucatan", 32: "Zacatecas",
}
ABBR = {1: "AGS", 2: "BC", 3: "BCS", 4: "CAM", 5: "COA", 6: "COL", 7: "CHS", 8: "CHH", 9: "CDMX",
        10: "DGO", 11: "GTO", 12: "GRO", 13: "HGO", 14: "JAL", 15: "MEX", 16: "MICH", 17: "MOR",
        18: "NAY", 19: "NL", 20: "OAX", 21: "PUE", 22: "QRO", 23: "QROO", 24: "SLP", 25: "SIN",
        26: "SON", 27: "TAB", 28: "TAMS", 29: "TLAX", 30: "VER", 31: "YUC", 32: "ZAC"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z ]", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^\d+\s*", "", s)
    for long, short in [("coahuila de zaragoza", "coahuila"), ("michoacan de ocampo", "michoacan"),
                        ("veracruz de ignacio de la llave", "veracruz"), ("estado de mexico", "mexico"),
                        ("distrito federal", "ciudad de mexico")]:
        if s == long:
            s = short
    return s


NAME2CODE = {norm(v): k for k, v in STATES.items()}


def code(name):
    return NAME2CODE.get(norm(name))


# ---------------------------------------------------------------- remittances
def remittances():
    d = pd.read_excel(RAW / "banxico_CE166_remesas_entidad_municipio.xlsx", header=None)
    periods = d.iloc[9, 2:].tolist()
    qmap = {"Ene-Mar": 1, "Abr-Jun": 2, "Jul-Sep": 3, "Oct-Dic": 4}
    yq = [(int(p.split()[1]), qmap[p.split()[0]]) for p in periods]
    rows_s, rows_m = [], []
    state = None
    for _, r in d.iloc[12:].iterrows():
        lab = str(r[1])
        vals = pd.to_numeric(r.iloc[2:], errors="coerce").values
        if "●" in lab:
            state = code(lab.split("●")[1])
            for (y, q), v in zip(yq, vals):
                rows_s.append((state, y, q, v))
        elif "⚬" in lab and state is not None:
            muni = lab.split("⚬")[1].strip()
            for (y, q), v in zip(yq, vals):
                rows_m.append((state, muni, y, q, v))
    s = pd.DataFrame(rows_s, columns=["ent", "year", "quarter", "usd_m"])
    m = pd.DataFrame(rows_m, columns=["ent", "municipio", "year", "quarter", "usd_m"])
    assert s.ent.nunique() == 32 and s.ent.notna().all()
    s["state"] = s.ent.map(STATES)
    s.to_csv(OUT / "remittances_state_quarterly.csv", index=False)
    DERIVED.mkdir(exist_ok=True)
    m.to_csv(DERIVED / "remittances_municipal_quarterly.csv", index=False)
    return s


def trend_excess(s, first, last):
    """Log deviation of annual remittances from each state's own log-linear quarterly trend
    (with quarter dummies) fitted on [first, last]. Returns ent x year table and the slope."""
    out, slopes = {}, {}
    for ent, g in s.groupby("ent"):
        g = g.copy()
        g["t"] = g.year + (g.quarter - 1) / 4
        fit = g[(g.year >= first) & (g.year <= last)]
        X = np.column_stack([np.ones(len(fit)), fit.t - 2019] + [(fit.quarter == q).astype(float) for q in (2, 3, 4)])
        b = np.linalg.lstsq(X, np.log(fit.usd_m.values), rcond=None)[0]
        Xa = np.column_stack([np.ones(len(g)), g.t - 2019] + [(g.quarter == q).astype(float) for q in (2, 3, 4)])
        g["pred"] = np.exp(Xa @ b)
        a = g.groupby("year")[["usd_m", "pred"]].sum()
        out[ent] = np.log(a.usd_m / a.pred)
        slopes[ent] = b[1]
        g["dev"] = np.log(g.usd_m / g.pred)
        g["ent"] = ent
        out[("q", ent)] = g[["ent", "year", "quarter", "dev"]]
    ann = pd.DataFrame({k: v for k, v in out.items() if not isinstance(k, tuple)}).T
    q = pd.concat([v for k, v in out.items() if isinstance(k, tuple)])
    return ann, pd.Series(slopes), q


# ---------------------------------------------------------------- population
def population():
    d = pd.read_excel(RAW / "cpv2020_b_eum_01_poblacion.xlsx", sheet_name="04", header=None)
    rows = []
    for _, r in d.iterrows():
        m = re.match(r"^(\d\d) ", str(r[0]))
        if m:
            rows.append((int(m.group(1)), float(r[1]), float(r[4])))
    p = pd.DataFrame(rows, columns=["ent", "pop2020", "median_age"]).drop_duplicates("ent")
    assert len(p) == 32
    return p.set_index("ent")


# ---------------------------------------------------------------- deaths
def deaths(pop):
    d = pd.read_csv(OUT / "deaths_state_year.csv")
    w = d.pivot(index="ent", columns="year", values="deaths")
    c = d.pivot(index="ent", columns="year", values="covid")
    yrs = np.arange(2015, 2020)
    exp = {}
    for ent, r in w.iterrows():
        b = np.polyfit(yrs - 2019, r[yrs].values.astype(float), 1)
        exp[ent] = {y: np.polyval(b, y - 2019) for y in (2020, 2021, 2022)}
    exp = pd.DataFrame(exp).T
    o = pd.DataFrame(index=w.index)
    p = pop.pop2020
    for y in (2020, 2021):
        o[f"excess_pc_{y}"] = (w[y] - exp[y]) / p * 1000
        o[f"excess_pct_{y}"] = w[y] / exp[y] - 1
        o[f"covid_pc_{y}"] = c[y] / p * 1000
    o["excess_pc_2020_21"] = o.excess_pc_2020 + o.excess_pc_2021
    o["excess_pct_2020_21"] = (w[2020] + w[2021]) / (exp[2020] + exp[2021]) - 1
    o["covid_pc_2020_21"] = o.covid_pc_2020 + o.covid_pc_2021
    o["deaths_2019"] = w[2019]
    return o


# ---------------------------------------------------------------- labour income (CONEVAL ITLP)
def itlp():
    f = RAW / "coneval" / "itlp" / "Cuadro_de_indicadores_ITLP_4T2024.xlsx"
    # real labour income per capita (INPC-deflated), Cuadro 11
    d = pd.read_excel(f, sheet_name="Cuadro 11", header=None)
    years = d.iloc[2].ffill()
    qs = d.iloc[3]
    cols = {}
    rom = {"I": 1, "II": 2, "III": 3, "IV": 4}
    for c in d.columns[4:]:
        q = str(qs[c]).strip().rstrip("*").upper().replace("L", "I")
        if q in rom:
            cols[c] = (int(years[c]), rom[q])
    inc = {}
    for _, r in d.iterrows():
        if isinstance(r[3], str) and r[3].startswith("Deflactado con el INPC"):
            ent = code(r[2]) if isinstance(r[2], str) else None
            if ent:
                inc[ent] = {k: pd.to_numeric(r[c], errors="coerce") for c, k in cols.items()}
    inc = pd.DataFrame(inc).T
    # labour poverty %, Cuadro 10
    d = pd.read_excel(f, sheet_name="Cuadro 10", header=None)
    years = d.iloc[2].ffill()
    qs = d.iloc[3]
    cols = {}
    for c in d.columns[3:]:
        q = str(qs[c]).strip().rstrip("*").upper().replace("L", "I")
        if q in rom:
            cols[c] = (int(years[c]), rom[q])
    pov = {}
    for _, r in d.iterrows():
        ent = code(r[2]) if isinstance(r[2], str) else None
        if ent:
            pov[ent] = {k: pd.to_numeric(r[c], errors="coerce") for c, k in cols.items()}
    pov = pd.DataFrame(pov).T
    o = pd.DataFrame(index=inc.index)
    base_h2 = inc[[(2019, 3), (2019, 4)]].mean(axis=1)
    base = inc[[(2019, q) for q in range(1, 5)]].mean(axis=1)
    o["dlog_labinc_2020"] = np.log(inc[[(2020, 3), (2020, 4)]].mean(axis=1) / base_h2)  # 2020Q2 not collected
    o["dlog_labinc_2021"] = np.log(inc[[(2021, q) for q in range(1, 5)]].mean(axis=1) / base)
    pb = pov[[(2019, q) for q in range(1, 5)]].mean(axis=1)
    o["d_labpov_2020"] = pov[[(2020, 3), (2020, 4)]].mean(axis=1) - pov[[(2019, 3), (2019, 4)]].mean(axis=1)
    o["d_labpov_2021"] = pov[[(2021, q) for q in range(1, 5)]].mean(axis=1) - pb
    o["labpov_2019"] = pb
    # labour income loss expressed as a positive "need" shock
    o["labinc_loss_2020"] = -o.dlog_labinc_2020
    o["labinc_loss_2021"] = -o.dlog_labinc_2021
    return o


# ---------------------------------------------------------------- migration intensity (CONAPO 2020 components)
GRADO = {  # CONAPO 2020 degree of migration intensity, as listed in BBVA Research (1 Oct 2021)
    "Muy alto": ["Zacatecas", "Michoacan", "Nayarit", "Guanajuato"],
    "Alto": ["San Luis Potosi", "Aguascalientes", "Oaxaca", "Durango", "Guerrero", "Hidalgo"],
    "Medio": ["Chihuahua", "Colima", "Baja California", "Jalisco", "Queretaro", "Sonora", "Morelos", "Sinaloa"],
    "Bajo": ["Tamaulipas", "Puebla", "Coahuila", "Veracruz", "Tlaxcala", "Chiapas", "Nuevo Leon"],
    "Muy bajo": ["Mexico", "Baja California Sur", "Yucatan", "Campeche", "Ciudad de Mexico", "Quintana Roo", "Tabasco"],
}


def iim():
    r = pypdf.PdfReader(RAW / "bbva_remesas_iim_2021.pdf")
    t = r.pages[5].extract_text()
    rows = []
    for line in t.splitlines():
        m = re.match(r"^\s*(\d{1,2}) (.+?) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)\s*$", line)
        if m:
            rows.append((int(m.group(1)), *[float(m.group(i)) for i in range(3, 7)]))
    o = pd.DataFrame(rows, columns=["ent", "hh_emig_pct", "hh_circ_pct", "hh_return_pct", "hh_remit_pct"]).set_index("ent")
    assert len(o) == 32
    g = {code(s): lvl for lvl, ss in GRADO.items() for s in ss}
    o["iim_grado"] = pd.Series(g).map({"Muy bajo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5})
    assert o.iim_grado.notna().all()
    return o


# ---------------------------------------------------------------- transit (UPM irregular-migration events)
def upm():
    out = {}
    for y in range(2019, 2024):
        d = pd.read_excel(RAW / "upm" / f"upm_{y}_cuadro3.1.xls", header=None)
        hdr = d.iloc[:8].astype(str)
        tc = [c for c in d.columns if (hdr[c] == "Total").any()][0]
        vals = {}
        for _, r in d.iterrows():
            ent = code(r[1]) if isinstance(r[1], str) else None
            if ent:
                vals[ent] = pd.to_numeric(r[tc], errors="coerce")
        out[y] = pd.Series(vals)
    return pd.DataFrame(out).add_prefix("upm_events_")


TRADITIONAL = [1, 6, 10, 11, 14, 16, 18, 24, 32]  # Durand-Massey traditional region
SOUTH_TRANSIT_CORE = [7, 27, 20, 30]  # Chiapas, Tabasco, Oaxaca, Veracruz
SOUTH_TRANSIT_EXT = [7, 27, 20, 30, 12, 23, 4]  # + Guerrero, Quintana Roo, Campeche
NORTH_BORDER = [2, 26, 8, 5, 19, 28]


def main():
    s = remittances()
    pop = population()
    ann17, slope17, q17 = trend_excess(s, 2017, 2019)
    ann15, slope15, q15 = trend_excess(s, 2015, 2019)
    ann13, slope13, _ = trend_excess(s, 2013, 2019)
    q17.to_csv(OUT / "remittances_state_quarterly_dev_trend1719.csv", index=False)
    a = s.groupby(["ent", "year"]).usd_m.sum().unstack()
    panel = pd.DataFrame(index=pd.Index(range(1, 33), name="ent"))
    panel["state"] = panel.index.map(STATES)
    panel["abbr"] = panel.index.map(ABBR)
    panel = panel.join(pop)
    panel["remit_2019_usd_m"] = a[2019]
    panel["remit_pc_2019_usd"] = a[2019] * 1e6 / panel.pop2020
    for y in (2020, 2021, 2022, 2023, 2024, 2025):
        panel[f"dlog_{y}"] = np.log(a[y] / a[2019])
        panel[f"exc1719_{y}"] = ann17[y]
        panel[f"exc1519_{y}"] = ann15[y]
        panel[f"exc1319_{y}"] = ann13[y]
    panel["pretrend_1719"] = slope17
    panel["pretrend_1519"] = slope15
    panel["pretrend_1319"] = slope13
    panel["growth_1619"] = np.log(a[2019] / a[2016]) / 3
    panel = panel.join(deaths(pop)).join(itlp()).join(iim()).join(upm())
    for y in range(2019, 2024):
        panel[f"upm_pc_{y}"] = panel[f"upm_events_{y}"] / panel.pop2020 * 1000
    panel["traditional"] = panel.index.isin(TRADITIONAL)
    panel["south_transit_core"] = panel.index.isin(SOUTH_TRANSIT_CORE)
    panel["south_transit_ext"] = panel.index.isin(SOUTH_TRANSIT_EXT)
    panel["north_border"] = panel.index.isin(NORTH_BORDER)
    panel.to_csv(OUT / "state_panel.csv")
    print(panel.describe().T.to_string())


if __name__ == "__main__":
    main()
