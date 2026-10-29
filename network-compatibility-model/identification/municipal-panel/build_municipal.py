"""Build the Mexican municipal panel: Banxico CE166 remittances matched to INEGI codes,
death certificates by municipality of residence, census 2020 controls, CONAPO IIM 2020.

Inputs (untrusted downloads, gitignored; run with `python -I -X utf8`):
  --state-raw  data/external/state-panel      (derived/remittances_municipal_quarterly.csv,
                                               dgis_defunciones/DEFUN_<y>.zip)
  --mun-raw    data/external/municipal-panel  (iter/iter.zip, conapo/iim_base2020m.csv)
Writes:
  <mun-raw>/derived/deaths_mun_year.csv             residence municipality x year of occurrence
  <mun-raw>/derived/municipal_panel.csv             one row per analysis unit
  <mun-raw>/derived/municipal_quarterly_dev.csv     unit x quarter log deviation from own trend
  outputs/match_report.csv, outputs/match_summary.csv   name-matching diagnostics (catalogue names only)
"""
import argparse
import io
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"

ap = argparse.ArgumentParser()
ap.add_argument("--state-raw", default=str(ROOT / "data" / "external" / "state-panel"))
ap.add_argument("--mun-raw", default=str(ROOT / "data" / "external" / "municipal-panel"))
ap.add_argument("--skip-deaths", action="store_true", help="reuse derived/deaths_mun_year.csv")
args = ap.parse_args()
SRAW, MRAW = Path(args.state_raw), Path(args.mun_raw)
DER = MRAW / "derived"
DER.mkdir(exist_ok=True)

# Municipalities created from a parent after the Banxico series starts (2013). Banxico keeps
# reporting them inside the parent, and pre-split death certificates code them to the parent,
# so they are collapsed into the parent to form a consistent analysis unit. (ent, new) -> parent
SPLITS = {
    (2, 6): 1,      # San Quintin <- Ensenada (2020)
    (4, 12): 4,     # Seybaplaya <- Champoton (2021)
    (7, 120): 80,   # Capitan Luis Angel Vidal <- Siltepec
    (7, 121): 72,   # Rincon Chamula San Pedro <- Pueblo Nuevo Solistahuacan
    (7, 122): 107,  # El Parral <- Villa Corzo
    (7, 123): 108,  # Emiliano Zapata <- Villaflores
    (7, 124): 92,   # Mezcalapa <- Tecpatan
    (7, 125): 80,   # Honduras de la Sierra <- Siltepec
    (17, 34): 15,   # Coatetelco <- Miacatlan
    (17, 35): 17,   # Xoxocotla <- Puente de Ixtla
    (17, 36): 22,   # Hueyapan <- Tetela del Volcan
    (23, 10): 4,    # Bacalar <- Othon P. Blanco
    (23, 11): 5,    # Puerto Morelos <- Benito Juarez
}
# Banxico name (normalized) -> INEGI 2020 municipal code, where names differ
ALIASES = {  # values: INEGI 2020 name (matched after normalization) or, for duplicated names, the code
    (8, "batopilas"): "Batopilas de Manuel Gomez Morin",
    (11, "dolores hidalgo cuna de la independencia nal"): "Dolores Hidalgo Cuna de la Independencia Nacional",
    (11, "silao"): "Silao de la Victoria",
    (14, "tlaquepaque"): "San Pedro Tlaquepaque",
    (15, "acambay"): "Acambay de Ruiz Castaneda",
    (17, "jonacatepec"): "Jonacatepec de Leandro Valle",
    (17, "tlaltizapan"): "Tlaltizapan de Zapata",
    (17, "zacualpan"): "Zacualpan de Amilpas",
    (19, "carmen"): "El Carmen",
    (20, "heroica ciudad de juchitan de zaragoza"): "Juchitan de Zaragoza",
    (20, "san juan mixtepec distr 08"): 208,  # Juxtlahuaca district (pop 6,941)
    (20, "san juan mixtepec distr 26"): 209,  # Miahuatlan district (pop 607)
    (20, "san pedro mixtepec distr 22"): 318,  # Juquila district, Puerto Escondido (pop 49,780)
    (20, "san pedro mixtepec distr 26"): 319,  # Miahuatlan district (pop 972)
    (20, "santiago chazumba"): "Villa de Santiago Chazumba",
    (20, "tezoatlan de segura y luna"): "Heroica Villa Tezoatlan de Segura y Luna, Cuna de la Independencia de Oaxaca",
    (20, "villa de tututepec de melchor ocampo"): "Villa de Tututepec",
    (20, "zapotitlan del rio"): "San Mateo Yucutindoo",  # renamed
    (30, "medellin"): "Medellin de Bravo",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^dr ", "doctor ", s)
    s = re.sub(r"^gral ", "general ", s)
    return s


def unit(ent, mun):
    return SPLITS.get((ent, mun), mun)


# ---------------------------------------------------------------- census 2020 (ITER)
def iter_census():
    z = zipfile.ZipFile(MRAW / "iter" / "iter.zip")
    name = [n for n in z.namelist() if n.endswith("CSV20.csv") and "conjunto" in n][0]
    d = pd.read_csv(z.open(name), usecols=["ENTIDAD", "MUN", "NOM_MUN", "LOC", "POBTOT", "POB65_MAS"],
                    dtype=str, encoding="utf-8-sig")
    d = d[d.MUN != "000"].copy()
    d["ent"] = d.ENTIDAD.astype(int)
    d["mun"] = d.MUN.astype(int)
    d["pop"] = pd.to_numeric(d.POBTOT, errors="coerce")
    tot = d[d.LOC == "0000"].copy()
    tot["pop65"] = pd.to_numeric(tot.POB65_MAS, errors="coerce")
    loc = d[~d.LOC.isin(["0000", "9998", "9999"])]
    urb = loc[loc["pop"] >= 2500].groupby(["ent", "mun"])["pop"].sum().rename("pop_urban")
    tot = tot.set_index(["ent", "mun"]).join(urb).fillna({"pop_urban": 0}).reset_index()
    return tot[["ent", "mun", "NOM_MUN", "pop", "pop65", "pop_urban"]]


# ---------------------------------------------------------------- CONAPO IIM 2020
def iim():
    d = pd.read_csv(MRAW / "conapo" / "iim_base2020m.csv", encoding="latin-1")
    d = d[d.CVE_GEO > 1000].copy()
    d["ent"] = d.CVE_GEO // 1000
    d["mun"] = d.CVE_GEO % 1000
    d = d.rename(columns={"IIM-DP2": "iim_dp2", "VIV_REM": "viv_rem_pct", "VIV_EMIG": "viv_emig_pct",
                          "VIV_TOT": "viv_tot"})
    return d[["ent", "mun", "viv_tot", "viv_rem_pct", "viv_emig_pct", "iim_dp2"]]


# ---------------------------------------------------------------- deaths
def open_csv(zpath):
    z = zipfile.ZipFile(zpath)
    big = max(z.infolist(), key=lambda i: i.file_size)
    data = z.read(big)
    if big.filename.lower().endswith(".zip"):
        inner = zipfile.ZipFile(io.BytesIO(data))
        big2 = max(inner.infolist(), key=lambda i: i.file_size)
        data = inner.read(big2)
    return io.BytesIO(data)


def deaths():
    frames = []
    for reg in range(2015, 2024):
        f = open_csv(SRAW / "dgis_defunciones" / f"DEFUN_{reg}.zip")
        df = pd.read_csv(f, encoding="latin-1", dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        occ = "anio_ocur" if "anio_ocur" in df.columns else "anio_ocurr"
        d = pd.DataFrame({
            "ent": pd.to_numeric(df["ent_resid"].str.strip(), errors="coerce"),
            "mun": pd.to_numeric(df["mun_resid"].str.strip(), errors="coerce"),
            "year": pd.to_numeric(df[occ].str.strip(), errors="coerce"),
            "covid": df["causa_def"].str.strip().str.upper().str[:4].isin(["U071", "U072"]),
        })
        g = d.groupby(["ent", "mun", "year"]).agg(deaths=("covid", "size"), covid=("covid", "sum")).reset_index()
        frames.append(g)
        print("deaths", reg, len(df), file=sys.stderr)
    a = pd.concat(frames)
    a = a[(a.ent.between(1, 32)) & (a.year.between(2015, 2023))]
    a = a.groupby(["ent", "mun", "year"])[["deaths", "covid"]].sum().reset_index()
    a[["ent", "mun", "year"]] = a[["ent", "mun", "year"]].astype(int)
    a.to_csv(DER / "deaths_mun_year.csv", index=False)
    return a


# ---------------------------------------------------------------- trend deviations
def trend_dev(q, first, last):
    """Per unit: log-linear quarterly trend with quarter dummies fitted on [first, last];
    returns annual log deviations, quarterly deviations and the slope (log points per year)."""
    ann, qs, slopes = [], [], {}
    for u, g in q.groupby("uid"):
        g = g.sort_values(["year", "quarter"])
        t = g.year + (g.quarter - 1) / 4 - last
        X = np.column_stack([np.ones(len(g)), t] + [(g.quarter == k).astype(float) for k in (2, 3, 4)])
        f = ((g.year >= first) & (g.year <= last)).values
        b = np.linalg.lstsq(X[f], np.log(g.usd_m.values[f]), rcond=None)[0]
        pred = np.exp(X @ b)
        gg = pd.DataFrame({"uid": u, "year": g.year.values, "quarter": g.quarter.values,
                           "usd_m": g.usd_m.values, "pred": pred})
        with np.errstate(divide="ignore", invalid="ignore"):
            gg["dev"] = np.log(gg.usd_m / gg.pred)
        a = gg.groupby("year")[["usd_m", "pred"]].sum()
        ann.append(pd.Series(np.log(a.usd_m / a.pred), name=u))
        qs.append(gg)
        slopes[u] = b[1]
    return pd.DataFrame(ann), pd.concat(qs), pd.Series(slopes)


def main():
    cen = iter_census()
    M = pd.read_csv(SRAW / "derived" / "remittances_municipal_quarterly.csv", encoding="utf-8")
    M = M[M.year <= 2024]
    tot19 = M[M.year == 2019].usd_m.sum()
    unid = M.municipio.str.strip().str.lower() == "no identificado"
    unid19 = M[unid & (M.year == 2019)].usd_m.sum()
    M = M[~unid].copy()

    # --- name matching
    names = M.groupby(["ent", "municipio"]).usd_m.sum().reset_index()[["ent", "municipio"]]
    names["key"] = names.municipio.map(norm)
    cen["key"] = cen.NOM_MUN.map(norm)
    ckeys = cen.drop_duplicates(["ent", "key"], keep=False)  # ambiguous names (Mixtepec) only via ALIASES
    names = names.merge(ckeys[["ent", "key", "mun"]], on=["ent", "key"], how="left")
    names["method"] = np.where(names.mun.notna(), "exact_normalized", "")
    for i, r in names[names.mun.isna()].iterrows():
        if (r.ent, r.key) in ALIASES:
            t = ALIASES[(r.ent, r.key)]
            if isinstance(t, str):
                t = cen.loc[(cen.ent == r.ent) & (cen.key == norm(t)), "mun"].item()
            names.loc[i, "mun"] = t
            names.loc[i, "method"] = "manual_alias"
    names["method"] = names.method.replace("", "unmatched")
    names = names.merge(cen[["ent", "mun", "NOM_MUN"]], on=["ent", "mun"], how="left")
    assert not names.dropna(subset=["mun"]).duplicated(["ent", "mun"]).any(), "two Banxico names on one code"
    M = M.merge(names[["ent", "municipio", "mun"]], on=["ent", "municipio"], how="left")
    m19 = M[(M.year == 2019)]
    usd19 = m19.groupby(m19.mun.notna()).usd_m.sum()
    summ = pd.DataFrame([
        ("banxico_names_identified", len(names)),
        ("matched_exact_normalized", (names.method == "exact_normalized").sum()),
        ("matched_manual_alias", (names.method == "manual_alias").sum()),
        ("unmatched", (names.method == "unmatched").sum()),
        ("match_rate_names", names.mun.notna().mean()),
        ("usd_2019_total_m", tot19),
        ("usd_2019_no_identificado_m", unid19),
        ("share_2019_no_identificado", unid19 / tot19),
        ("share_2019_usd_matched_of_total", usd19.get(True, 0) / tot19),
        ("share_2019_usd_matched_of_identified", usd19.get(True, 0) / usd19.sum()),
        ("inegi_2020_municipalities", len(cen)),
        ("inegi_municipalities_collapsed_into_parent", len(SPLITS)),
    ], columns=["item", "value"])
    summ.to_csv(OUT / "match_summary.csv", index=False)
    names[names.method != "exact_normalized"].to_csv(OUT / "match_report.csv", index=False)
    print(summ.to_string(), file=sys.stderr)
    M = M.dropna(subset=["mun"])
    M["mun"] = M.mun.astype(int)

    # --- analysis units (collapse post-2013 splits into parent)
    for d in (M, cen):
        d["umun"] = [unit(e, m) for e, m in zip(d.ent, d.mun)]
        d["uid"] = d.ent * 1000 + d.umun
    q = M.groupby(["uid", "ent", "year", "quarter"]).usd_m.sum().reset_index()
    full = pd.MultiIndex.from_product([q.uid.unique(), range(2013, 2025), range(1, 5)], names=["uid", "year", "quarter"])
    q = q.set_index(["uid", "year", "quarter"]).usd_m.reindex(full).reset_index()
    q["ent"] = q.uid // 1000
    U = cen.groupby("uid").agg(ent=("ent", "first"), pop2020=("pop", "sum"), pop65=("pop65", "sum"),
                               pop_urban=("pop_urban", "sum"), name=("NOM_MUN", "first"),
                               n_inegi=("mun", "size"))
    # IIM: household-weighted average across collapsed municipalities
    I = iim()
    I["uid"] = I.ent * 1000 + [unit(e, m) for e, m in zip(I.ent, I.mun)]
    I["w"] = I.viv_tot
    agg = I.groupby("uid").apply(lambda g: pd.Series({
        "iim_dp2": np.average(g.iim_dp2, weights=g.w), "viv_rem_pct": np.average(g.viv_rem_pct, weights=g.w),
        "viv_emig_pct": np.average(g.viv_emig_pct, weights=g.w)}), include_groups=False)
    U = U.join(agg)

    # --- deaths
    D = pd.read_csv(DER / "deaths_mun_year.csv") if args.skip_deaths else deaths()
    tot_d = D.groupby("year").deaths.sum()
    D = D[D.mun.between(1, 600)].copy()
    D["uid"] = D.ent * 1000 + [unit(e, m) for e, m in zip(D.ent, D.mun)]
    known = D.groupby("year").deaths.sum() / tot_d
    print("share of deaths with known municipality of residence\n", known.round(4), file=sys.stderr)
    W = D.groupby(["uid", "year"]).deaths.sum().unstack().reindex(U.index).fillna(0)
    C = D.groupby(["uid", "year"]).covid.sum().unstack().reindex(U.index).fillna(0)
    p = U.pop2020 / 1000
    base1519 = W[list(range(2015, 2020))].mean(axis=1)
    base1719 = W[list(range(2017, 2020))].mean(axis=1)
    yrs = np.arange(2015, 2020) - 2019
    Xt = np.column_stack([np.ones(5), yrs])
    bt = np.linalg.lstsq(Xt, W[list(range(2015, 2020))].values.T, rcond=None)[0]  # 2 x units
    for y in (2020, 2021):
        U[f"excess_pc_{y}"] = (W[y] - base1519) / p
        U[f"excess1719_pc_{y}"] = (W[y] - base1719) / p
        U[f"excesstr_pc_{y}"] = (W[y] - (bt[0] + bt[1] * (y - 2019))) / p
        U[f"covid_pc_{y}"] = C[y] / p
    for v in ("excess", "excess1719", "excesstr", "covid"):
        U[f"{v}_pc_2020_21"] = U[f"{v}_pc_2020"] + U[f"{v}_pc_2021"]
    U["excess_pct_2020_21"] = (W[2020] + W[2021]) / (2 * base1519) - 1
    U["deaths_base1519"] = base1519
    U["deaths_pc_2019"] = W[2019] / p

    # --- remittance outcomes
    a = q.groupby(["uid", "year"]).usd_m.sum(min_count=4).unstack()
    U = U.join(a[[2016, 2019]].rename(columns={2016: "remit_2016_usd_m", 2019: "remit_2019_usd_m"}))
    pos1719 = q[q.year.between(2017, 2019)].groupby("uid").usd_m.apply(lambda s: (s > 0).all() and s.notna().all())
    pos1316 = q[q.year.between(2013, 2016)].groupby("uid").usd_m.apply(lambda s: (s > 0).all() and s.notna().all())
    U["pos_1719"] = pos1719.reindex(U.index).fillna(False).astype(bool)
    U["pos_1316"] = pos1316.reindex(U.index).fillna(False).astype(bool)
    q17 = q[q.uid.isin(U.index[U.pos_1719])]
    ann, qdev, slope = trend_dev(q17, 2017, 2019)
    for y in range(2019, 2025):
        U[f"exc1719_{y}"] = ann[y]
    U["pretrend_1719"] = slope
    q13 = q[q.uid.isin(U.index[U.pos_1316])]
    ann13, _, slope13 = trend_dev(q13, 2013, 2015)
    U["exc1315_2016"] = ann13[2016]
    U["pretrend_1315"] = slope13
    ann15, _, slope15 = trend_dev(q[q.uid.isin(U.index[U.pos_1719 & U.pos_1316])], 2015, 2019)
    U["exc1519_2021"] = ann15[2021]
    U["pretrend_1519"] = slope15
    for y in (2020, 2021, 2022, 2023):
        U[f"dlog_{y}"] = np.log(a[y] / a[2019]).reindex(U.index)
    U["growth_1619"] = (np.log(a[2019] / a[2016]) / 3).reindex(U.index)
    U["remit_pc_2019_usd"] = U.remit_2019_usd_m * 1e6 / U.pop2020
    U["urban_share"] = U.pop_urban / U.pop2020
    U["pop65_share"] = U.pop65 / U.pop2020
    U["has_remit"] = U.index.isin(q.uid.unique())
    U.to_csv(DER / "municipal_panel.csv")
    qdev[["uid", "year", "quarter", "usd_m", "dev"]].to_csv(DER / "municipal_quarterly_dev.csv", index=False)
    print(U.describe().T.round(3).to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
