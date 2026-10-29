"""ENIGH 2018-2024: remittance receipt and amounts by household need cell.

Design-based standard errors by Taylor linearization (with-replacement PSU
variance within strata; PSU = upm, strata = est_dis, weights = factor).
Outputs are aggregate tables only (no household rows); cells with unweighted
n < 30 are suppressed.

Run:  C:/miniforge3/python.exe -I enigh_heterogeneity.py   (from any directory)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
EXT = DATA / "external" / "enigh"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

YEARS = [2018, 2020, 2022, 2024]
MIN_N = 30
TRAD = {"01", "06", "10", "11", "14", "16", "18", "24", "32"}
STATE_NAMES = {f"{i+1:02d}": n for i, n in enumerate([
    "Aguascalientes",
    "Baja California",
    "Baja California Sur",
    "Campeche",
    "Coahuila",
    "Colima",
    "Chiapas",
    "Chihuahua",
    "Ciudad de Mexico",
    "Durango",
    "Guanajuato",
    "Guerrero",
    "Hidalgo",
    "Jalisco",
    "Mexico",
    "Michoacan",
    "Morelos",
    "Nayarit",
    "Nuevo Leon",
    "Oaxaca",
    "Puebla",
    "Queretaro",
    "Quintana Roo",
    "San Luis Potosi",
    "Sinaloa",
    "Sonora",
    "Tabasco",
    "Tamaulipas",
    "Tlaxcala",
    "Veracruz",
    "Yucatan",
    "Zacatecas",
])}


# ---------------------------------------------------------------- deflator
def inpc_august():
    """INEGI INPC (base 2Q Jul 2018 = 100), via IMF CPI on DBnomics."""
    j = json.loads((EXT / "dbnomics_imf_cpi_mx.json").read_text(encoding="utf-8"))
    s = j["series"]["docs"][0]
    cpi = dict(zip(s["period"], s["value"]))
    return {y: float(cpi[f"{y}-08"]) for y in YEARS}


INPC = inpc_august()
DEFL = {y: INPC[2024] / INPC[y] for y in YEARS}  # to August-2024 pesos


# ---------------------------------------------------------------- data
USE = ["ubica_geo", "tam_loc", "est_dis", "upm", "factor", "remesas", "ocupados",
       "p65mas", "sexo_jefe", "edad_jefe", "tot_integ", "ingtrab", "ing_cor"]


def load(year):
    d = pd.read_csv(DATA / f"concentradohogar_{year % 100}.csv", usecols=USE,
                    dtype={"ubica_geo": str, "est_dis": str, "upm": str},
                    encoding="utf-8-sig")
    d["w"] = d["factor"].astype(float)
    d["state"] = d["ubica_geo"].str.zfill(5).str[:2]
    d["rec"] = (d["remesas"] > 0).astype(float)
    d["rem"] = d["remesas"] * DEFL[year]          # quarterly, Aug-2024 MXN
    d["lab_pc"] = d["ingtrab"] / d["tot_integ"]
    # weighted 20th percentile of labour income per capita (ties at 0 included)
    o = d.sort_values("lab_pc")
    cw = o["w"].cumsum() / o["w"].sum()
    p20 = o["lab_pc"].values[np.searchsorted(cw.values, 0.20)]
    d["lab_q1"] = d["lab_pc"] <= p20
    d["psu"] = d["est_dis"] + "_" + d["upm"]
    return d


def cells(d):
    """Boolean masks: name -> (group, label, mask)."""
    c = {"All": ("all", "All households", np.ones(len(d), bool))}
    c["no_worker"] = ("worker", "No employed resident", (d.ocupados == 0).values)
    c["worker"] = ("worker", "Some employed resident", (d.ocupados > 0).values)
    c["old"] = ("old", "Resident aged 65+", (d.p65mas > 0).values)
    c["no_old"] = ("old", "No resident 65+", (d.p65mas == 0).values)
    c["rural"] = ("rural", "Rural (<2,500)", (d.tam_loc == 4).values)
    c["urban"] = ("rural", "Urban (2,500+)", (d.tam_loc != 4).values)
    c["trad"] = ("trad", "Traditional states", d.state.isin(TRAD).values)
    c["nontrad"] = ("trad", "Other states", (~d.state.isin(TRAD)).values)
    c["lab_q1"] = ("labq", "Labour inc. p.c. bottom 20%", d.lab_q1.values)
    c["lab_q25"] = ("labq", "Labour inc. p.c. top 80%", (~d.lab_q1).values)
    c["head_f"] = ("sex", "Female head", (d.sexo_jefe == 2).values)
    c["head_m"] = ("sex", "Male head", (d.sexo_jefe == 1).values)
    for lo, hi, nm in [(0, 34, "<35"), (35, 49, "35-49"), (50, 64, "50-64"), (65, 200, "65+")]:
        c[f"age_{nm}"] = ("head_age", f"Head aged {nm}",
                          ((d.edad_jefe >= lo) & (d.edad_jefe <= hi)).values)
    for lo, hi, nm in [(1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")]:
        c[f"size_{nm}"] = ("size", f"Household size {nm}",
                           ((d.tot_integ >= lo) & (d.tot_integ <= hi)).values)
    # composite high-need: no worker or resident 65+
    c["need_any"] = ("need", "No worker or 65+ resident",
                     ((d.ocupados == 0) | (d.p65mas > 0)).values)
    c["need_none"] = ("need", "Worker and no 65+", ((d.ocupados > 0) & (d.p65mas == 0)).values)
    return c


PAIRS = [("no_worker", "worker"), ("old", "no_old"), ("rural", "urban"),
         ("trad", "nontrad"), ("lab_q1", "lab_q25"), ("head_f", "head_m"),
         ("age_65+", "age_35-49"), ("size_1", "size_3-4"), ("need_any", "need_none")]


# ---------------------------------------------------------------- variance
class Design:
    def __init__(self, d):
        self.psu = d["psu"].values
        self.stratum = d["est_dis"].values
        self.key = pd.MultiIndex.from_arrays([self.stratum, self.psu])

    def psu_totals(self, z):
        return pd.Series(z).groupby([self.stratum, self.psu]).sum()

    @staticmethod
    def var(t):
        """t: PSU totals of linearized values, index (stratum, psu)."""
        g = t.groupby(level=0)
        nh = g.transform("size")
        dev2 = (t - g.transform("mean")) ** 2
        return float((dev2 * nh / (nh - 1)).sum())


def ratio(d, des, y, x, m):
    """R = sum w m y / sum w m x; returns (R, PSU totals of linearized z, n)."""
    w = d["w"].values * m
    Y, X = (w * y).sum(), (w * x).sum()
    R = Y / X
    z = w * (y - R * x) / X
    return R, des.psu_totals(z), int(m.sum() if np.isscalar(x) else (m & (x > 0)).sum())


def woodruff_median(d, des, y, m):
    """Weighted median among m with Woodruff SE."""
    w = d["w"].values[m]
    yy = y[m]
    o = np.argsort(yy)
    cw = np.cumsum(w[o]) / w.sum()
    def q(p):
        return yy[o][min(np.searchsorted(cw, p), len(yy) - 1)]
    med = q(0.5)
    ind = (y <= med).astype(float)
    _, t, _ = ratio(d, des, ind, np.ones(len(d)), m)
    se_p = np.sqrt(Design.var(t))
    lo, hi = q(max(0.5 - 1.96 * se_p, 0)), q(min(0.5 + 1.96 * se_p, 1))
    return med, (hi - lo) / (2 * 1.96)


# ---------------------------------------------------------------- estimation
def estimate_year(year):
    d = load(year)
    des = Design(d)
    ones = np.ones(len(d))
    rec, rem = d["rec"].values, d["rem"].values
    rows, Z = [], {}
    for name, (grp, lab, m) in cells(d).items():
        n = int(m.sum())
        nr = int((m & (rec > 0)).sum())
        p, tp, _ = ratio(d, des, rec, ones, m)                # receipt rate
        a, ta, _ = ratio(d, des, rem, rec, m)                 # mean amount | recipient
        h, th, _ = ratio(d, des, rem, ones, m)                # mean per household
        Z[name] = {"p": tp, "a": ta, "h": th,
                   "lp": tp / p, "la": ta / a, "lh": th / h,
                   "val": {"p": p, "a": a, "h": h}}
        med, med_se = woodruff_median(d, des, rem, m & (rec > 0)) if nr >= MIN_N else (np.nan, np.nan)
        wsum = d["w"].values[m].sum()
        rows.append(dict(year=year, cell=name, group=grp, label=lab, n=n, n_recip=nr,
                         households=wsum, share_of_households=wsum / d["w"].sum(),
                         receipt=p, receipt_se=np.sqrt(des.var(tp)),
                         amount_mean=a, amount_mean_se=np.sqrt(des.var(ta)),
                         amount_median=med, amount_median_se=med_se,
                         per_hh=h, per_hh_se=np.sqrt(des.var(th)),
                         total_quarterly_mxn_m=h * wsum / 1e6))
    st = []
    for s in sorted(d["state"].unique()):
        m = (d["state"] == s).values
        n, nr = int(m.sum()), int((m & (rec > 0)).sum())
        p, tp, _ = ratio(d, des, rec, ones, m)
        a, ta, _ = ratio(d, des, rem, rec, m)
        h, th, _ = ratio(d, des, rem, ones, m)
        wsum = d["w"].values[m].sum()
        st.append(dict(year=year, state=s, state_name=STATE_NAMES[s], traditional=s in TRAD,
                       n=n, n_recip=nr, households=wsum,
                       receipt=p, receipt_se=np.sqrt(des.var(tp)),
                       amount_mean=a, amount_mean_se=np.sqrt(des.var(ta)),
                       per_hh=h, per_hh_se=np.sqrt(des.var(th)),
                       total_quarterly_mxn_m=h * wsum / 1e6))
    return pd.DataFrame(rows), pd.DataFrame(st), Z, d


def suppress(df, ncol_rate="n", ncol_amt="n_recip", rate_cols=(), amt_cols=()):
    df = df.copy()
    df.loc[df[ncol_rate] < MIN_N, list(rate_cols)] = np.nan
    df.loc[df[ncol_amt] < MIN_N, list(amt_cols)] = np.nan
    return df


def main():
    levels, states, Zs, nrec = [], [], {}, {}
    for y in YEARS:
        lv, st, Z, d = estimate_year(y)
        levels.append(lv); states.append(st); Zs[y] = Z
        print(y, "done", f"receipt {lv.iloc[0].receipt:.4f}")
    lv = pd.concat(levels, ignore_index=True)
    st = pd.concat(states, ignore_index=True)
    ncount = lv.set_index(["year", "cell"])[["n", "n_recip"]]

    RATE = ["receipt", "receipt_se"]
    AMT = ["amount_mean", "amount_mean_se", "amount_median", "amount_median_se"]
    lv = suppress(lv, rate_cols=RATE + ["per_hh", "per_hh_se", "total_quarterly_mxn_m"], amt_cols=AMT)
    st = suppress(st, rate_cols=RATE + ["per_hh", "per_hh_se", "total_quarterly_mxn_m"],
                  amt_cols=["amount_mean", "amount_mean_se"])
    lv.to_csv(OUT / "levels_by_cell.csv", index=False, float_format="%.6g")
    lv[lv.cell == "All"].to_csv(OUT / "national.csv", index=False, float_format="%.6g")
    st.to_csv(OUT / "states.csv", index=False, float_format="%.6g")

    # changes and difference-in-changes
    periods = [(2018, 2020), (2020, 2022), (2022, 2024), (2018, 2022)]
    meas = {"p": "receipt (pp)", "a": "mean amount | recipient (MXN)",
            "h": "mean per household (MXN)", "lp": "log receipt",
            "la": "log mean amount | recipient", "lh": "log mean per household"}

    def value(y, c, k):
        v = Zs[y][c]["val"]
        return {"p": v["p"], "a": v["a"], "h": v["h"], "lp": np.log(v["p"]),
                "la": np.log(v["a"]), "lh": np.log(v["h"])}[k]

    def ok(y, c, k):
        nn = ncount.loc[(y, c)]
        return nn["n_recip"] >= MIN_N and nn["n"] >= MIN_N  # all measures need recipients

    ch = []
    for c in Zs[2018]:
        for t0, t1 in periods:
            for k in meas:
                if not (ok(t0, c, k) and ok(t1, c, k)):
                    continue
                est = value(t1, c, k) - value(t0, c, k)
                se = np.sqrt(Design.var(Zs[t1][c][k]) + Design.var(Zs[t0][c][k]))
                ch.append(dict(cell=c, from_year=t0, to_year=t1, measure=k,
                               measure_label=meas[k], change=est, se=se, z=est / se))
    ch = pd.DataFrame(ch)
    ch.to_csv(OUT / "changes_by_cell.csv", index=False, float_format="%.6g")

    dd = []
    for A, B in PAIRS:
        for t0, t1 in periods:
            for k in meas:
                if not all(ok(t, c, k) for t in (t0, t1) for c in (A, B)):
                    continue
                est = (value(t1, A, k) - value(t1, B, k)) - (value(t0, A, k) - value(t0, B, k))
                v1 = Design.var(Zs[t1][A][k].sub(Zs[t1][B][k], fill_value=0))
                v0 = Design.var(Zs[t0][A][k].sub(Zs[t0][B][k], fill_value=0))
                se = np.sqrt(v0 + v1)
                dd.append(dict(cell_a=A, cell_b=B, from_year=t0, to_year=t1, measure=k,
                               measure_label=meas[k], did=est, se=se, z=est / se))
    dd = pd.DataFrame(dd)
    dd.to_csv(OUT / "diff_in_changes.csv", index=False, float_format="%.6g")

    # margin decomposition: dlog(per hh) = dlog(receipt) + dlog(amount | recipient)
    dec = ch[ch.measure.isin(["lp", "la", "lh"])].pivot_table(
        index=["cell", "from_year", "to_year"], columns="measure", values=["change", "se"])
    dec.columns = [f"{m}_{s}" for s, m in dec.columns]
    dec = dec.rename(columns={"lp_change": "dlog_receipt", "la_change": "dlog_amount",
                              "lh_change": "dlog_per_hh", "lp_se": "dlog_receipt_se",
                              "la_se": "dlog_amount_se", "lh_se": "dlog_per_hh_se"})
    dec["extensive_share"] = dec["dlog_receipt"] / dec["dlog_per_hh"]
    dec.reset_index().to_csv(OUT / "margin_decomposition.csv", index=False, float_format="%.6g")

    pd.DataFrame({"year": YEARS, "inpc_august": [INPC[y] for y in YEARS],
                  "factor_to_aug2024": [DEFL[y] for y in YEARS]}).to_csv(
        OUT / "deflator.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
