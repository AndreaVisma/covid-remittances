"""ENIGH pre-period (2016) and monthly remittance profile (ingresos, 2016/2018/2020).

A. National series 2016-2024 from concentradohogar: receipt, amount per recipient
   (real and USD), mean per household, receiving households; margin decomposition by
   interval; Banxico comparison (counts, USD, implied transfers per receiving household)
   and the model's transfers-per-receiver rise (option 6a).
B. Monthly profile from the person-level ingresos table, clave P041 (income from abroad):
   household receipt and amount by calendar month of the six reference months.

Design-based SEs as in enigh_heterogeneity.py (Taylor linearization, PSU = upm within
est_dis, weight = factor). Outputs are weighted aggregates only; cells with unweighted
n < 30 are suppressed.

Run (paths to the extracted INEGI files are arguments; downloaded files are untrusted):
  C:/miniforge3/python.exe -I enigh_pre_and_monthly.py \
      --c16 data/external/enigh/unzipped/enigh2016_ns_concentradohogar_csv/concentradohogar.csv \
      --ing16 data/external/enigh/unzipped/enigh2016_ns_ingresos_csv/ingresos.csv \
      --ing18 data/external/enigh/unzipped/enigh2018_ns_ingresos_csv/ingresos.csv \
      --ing20 data/external/enigh/unzipped/enigh2020_ns_ingresos_csv/ingresos.csv \
      --fx data/external/fx/fred_DEXMXUS_full.csv
(relative paths are resolved against the repository root)
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
EXT = DATA / "external" / "enigh"
OUT = HERE / "outputs"
FIG = HERE / "figures"
BANXICO = ROOT / "oct-26/data-prep/outputs/mexico-rem-monthly-total.parquet"
HHMODEL = ROOT / "network-compatibility-model/recipient-level/outputs/household-model"

YEARS = [2016, 2018, 2020, 2022, 2024]
MIN_N = 30
WINDOW = range(2, 11)                       # Feb..Oct, the ENIGH reference window
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
INTERVALS = [(2016, 2018), (2018, 2020), (2018, 2022), (2020, 2022), (2022, 2024), (2016, 2022)]

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#6b6b6b", "#e6e6e3"
GREY = "#9a9a96"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})


def rpath(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- prices and FX
def load_inpc():
    j = json.loads((EXT / "dbnomics_imf_cpi_mx.json").read_text(encoding="utf-8"))
    s = j["series"]["docs"][0]
    return {p: float(v) for p, v in zip(s["period"], s["value"]) if v not in (None, "NA")}


def load_fx(path):
    fx = pd.read_csv(path, parse_dates=["observation_date"])
    fx["v"] = pd.to_numeric(fx["DEXMXUS"], errors="coerce")
    fx = fx.dropna(subset=["v"])
    return fx.groupby(fx.observation_date.dt.to_period("M"))["v"].mean()


# ---------------------------------------------------------------- design (as enigh_heterogeneity.py)
class Design:
    def __init__(self, d):
        self.psu = d["psu"].values
        self.stratum = d["est_dis"].values

    def psu_totals(self, z):
        return pd.Series(z).groupby([self.stratum, self.psu]).sum()

    @staticmethod
    def var(t):
        g = t.groupby(level=0)
        nh = g.transform("size")
        dev2 = (t - g.transform("mean")) ** 2
        return float((dev2 * nh / (nh - 1)).sum())


def ratio(d, des, y, x, m):
    w = d["w"].values * m
    Y, X = (w * y).sum(), (w * x).sum()
    R = Y / X
    return R, des.psu_totals(w * (y - R * x) / X)


def total(d, des, y, m):
    w = d["w"].values * m
    return (w * y).sum(), des.psu_totals(w * y)


def se(t):
    return np.sqrt(Design.var(t))


def combine(*terms):
    """Variance of sum c_i * t_i for PSU-total series t_i from the same survey."""
    out = None
    for c, t in terms:
        out = c * t if out is None else out.add(c * t, fill_value=0)
    return out


# ---------------------------------------------------------------- A. concentradohogar
USE = ["folioviv", "foliohog", "est_dis", "upm", "factor", "remesas"]


def load_ch(year, c16):
    path = c16 if year == 2016 else DATA / f"concentradohogar_{year % 100}.csv"
    d = pd.read_csv(path, usecols=USE, dtype={"folioviv": str, "foliohog": str, "est_dis": str,
                                              "upm": str}, encoding="utf-8-sig")
    d["w"] = d["factor"].astype(float)
    d["psu"] = d["est_dis"] + "_" + d["upm"]
    d["rec"] = (d["remesas"] > 0).astype(float)
    return d


def national(year, d, inpc, fx):
    des = Design(d)
    one = np.ones(len(d))
    allm = np.ones(len(d), bool)
    defl = inpc["2024-08"] / inpc[f"{year}-08"]
    fxw = fx[[pd.Period(f"{year}-{m:02d}", "M") for m in WINDOW]].mean()
    rem_real = d["remesas"].values * defl
    rem_usd = d["remesas"].values / fxw            # Aug-of-year pesos / mean Feb-Oct FX
    rec = d["rec"].values
    p, tp = ratio(d, des, rec, one, allm)
    a, ta = ratio(d, des, rem_real, rec, allm)
    au, tau = ratio(d, des, rem_usd, rec, allm)
    h, th = ratio(d, des, rem_real, one, allm)
    N, tN = total(d, des, rec, allm)
    U, tU = total(d, des, rem_usd, allm)
    row = dict(year=year, n=len(d), n_recip=int(rec.sum()), households=d["w"].sum(),
               receipt=p, receipt_se=se(tp), amount_mean=a, amount_mean_se=se(ta),
               amount_mean_usd=au, amount_mean_usd_se=se(tau), per_hh=h, per_hh_se=se(th),
               receiving_households=N, receiving_households_se=se(tN),
               total_usd_quarterly_m=U / 1e6, total_usd_quarterly_m_se=se(tU) / 1e6,
               inpc_august=inpc[f"{year}-08"], factor_to_aug2024=defl, fx_feb_oct=fxw)
    # log linearizations for changes
    Z = {"lp": tp / p, "la": ta / a, "lau": tau / au, "lh": th / h, "lN": tN / N, "lU": tU / U}
    return row, Z


def part_a(c16, inpc, fx):
    rows, Zs = [], {}
    for y in YEARS:
        r, Z = national(y, load_ch(y, c16), inpc, fx)
        rows.append(r); Zs[y] = Z
        print(y, f"receipt {r['receipt']:.4f} amount {r['amount_mean']:.0f} n_recip {r['n_recip']}")
    nat = pd.DataFrame(rows)
    nat.to_csv(OUT / "pre_national.csv", index=False, float_format="%.6g")
    N = nat.set_index("year")

    def lval(y, k):
        r = N.loc[y]
        return np.log({"lp": r.receipt, "la": r.amount_mean, "lau": r.amount_mean_usd,
                       "lh": r.per_hh, "lN": r.receiving_households,
                       "lU": r.total_usd_quarterly_m}[k])

    def dlog(t0, t1, k):
        return lval(t1, k) - lval(t0, k), np.sqrt(Design.var(Zs[t1][k]) + Design.var(Zs[t0][k]))

    # Banxico, summed over Feb-Oct of each year (the ENIGH reference window)
    bx = pd.read_parquet(BANXICO)
    bx["y"], bx["m"] = bx.date.dt.year, bx.date.dt.month
    bw = bx[bx.m.isin(list(WINDOW))].groupby("y")[["transactions", "total_usd"]].sum()
    bw["avg_usd"] = bw.total_usd / bw.transactions

    dec = []
    for t0, t1 in INTERVALS:
        r = dict(from_year=t0, to_year=t1)
        for k, nm in [("lh", "dlog_per_hh"), ("lp", "dlog_receipt"), ("la", "dlog_amount"),
                      ("lau", "dlog_amount_usd"), ("lN", "dlog_receiving_hh"),
                      ("lU", "dlog_enigh_total_usd")]:
            r[nm], r[nm + "_se"] = dlog(t0, t1, k)
        r["years"] = t1 - t0
        r["dlog_banxico_counts"] = np.log(bw.loc[t1, "transactions"] / bw.loc[t0, "transactions"])
        r["dlog_banxico_usd"] = np.log(bw.loc[t1, "total_usd"] / bw.loc[t0, "total_usd"])
        r["dlog_banxico_avg_usd"] = np.log(bw.loc[t1, "avg_usd"] / bw.loc[t0, "avg_usd"])
        # (1) Banxico counts per ENIGH receiving household
        r["dlog_tpr_counts"] = r["dlog_banxico_counts"] - r["dlog_receiving_hh"]
        r["dlog_tpr_counts_se"] = r["dlog_receiving_hh_se"]
        # (2) ENIGH USD per recipient / Banxico USD per transaction (assumes ENIGH receipts
        #     have Banxico's average transfer size, or that the ratio is stable)
        r["dlog_tpr_amounts"] = r["dlog_amount_usd"] - r["dlog_banxico_avg_usd"]
        r["dlog_tpr_amounts_se"] = r["dlog_amount_usd_se"]
        dec.append(r)
    dec = pd.DataFrame(dec)
    for c in ["dlog_per_hh", "dlog_receipt", "dlog_amount", "dlog_tpr_counts", "dlog_tpr_amounts"]:
        dec[c + "_per_year"] = dec[c] / dec["years"]
    dec["intensive_share"] = dec["dlog_amount"] / dec["dlog_per_hh"]
    dec.to_csv(OUT / "pre_margin_decomposition.csv", index=False, float_format="%.6g")

    # excess over the 2016->18 trend: d(t0->t1) - (t1-t0)/2 * d(16->18); 2018 appears in both.
    # In levels x: excess = x_t1 - x_2018 - k (x_2018 - x_2016), k = (t1 - 2018)/2.
    ex = []
    for t1 in [2020, 2022, 2024]:
        k = (t1 - 2018) / 2
        for meas, base_k, add in [("tpr_counts", "lN", "counts"), ("tpr_amounts", "lau", "avg"),
                                  ("receipt", "lp", None), ("amount", "la", None),
                                  ("per_hh", "lh", None)]:
            sign = -1 if meas == "tpr_counts" else 1
            xv = {y: sign * lval(y, base_k) for y in (2016, 2018, t1)}
            if add == "counts":
                for y in xv:
                    xv[y] += np.log(bw.loc[y, "transactions"])
            if add == "avg":
                for y in xv:
                    xv[y] -= np.log(bw.loc[y, "avg_usd"])
            est = xv[t1] - xv[2018] - k * (xv[2018] - xv[2016])
            var = (Design.var(Zs[t1][base_k]) + (1 + k) ** 2 * Design.var(Zs[2018][base_k])
                   + k ** 2 * Design.var(Zs[2016][base_k]))
            ex.append(dict(measure=meas, from_year=2018, to_year=t1,
                           change=xv[t1] - xv[2018], trend_from_2016_18=k * (xv[2018] - xv[2016]),
                           excess=est, excess_se=np.sqrt(var), excess_ratio=np.exp(est)))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / "pre_excess_over_trend.csv", index=False, float_format="%.6g")

    bwo = bw.loc[YEARS].reset_index().rename(columns={"y": "year"})
    bwo["enigh_total_usd_m_x2"] = (N.loc[YEARS, "total_usd_quarterly_m"].values * 2)
    bwo["enigh_coverage_of_banxico_feb_oct"] = (bwo.enigh_total_usd_m_x2 * 9 / 6) / (bwo.total_usd / 1e6)
    bwo["banxico_transactions_per_enigh_receiving_hh_feb_oct"] = (
        bwo.transactions / N.loc[YEARS, "receiving_households"].values)
    bwo.to_csv(OUT / "pre_banxico_feb_oct.csv", index=False, float_format="%.6g")
    return nat, dec, ex, bwo


def model_tpr():
    """Model transfers per receiving household, COVID run / no-COVID run (= index_model /
    index_receipt), option 6a, at the fitted beta per kappa (nearest grid value)."""
    q = pd.read_csv(HHMODEL / "quarterly_paths.csv")
    fit = pd.read_csv(HHMODEL / "need_fit.csv")
    q["tpr"] = q.index_model / q.index_receipt
    rows = []
    for _, f in fit.iterrows():
        kq = q[np.isclose(q.kappa, f.kappa)]
        bgrid = np.sort(kq.beta.unique())
        b = bgrid[np.argmin(abs(bgrid - f.beta))]
        s = kq[np.isclose(kq.beta, b)].set_index("quarter")
        def m(qs, col="tpr"):
            return float(s.loc[qs, col].mean())
        rows.append(dict(kappa=f.kappa, beta_fitted=f.beta, beta_grid=b,
                         tpr_2020q1_q3=m(["2020Q1", "2020Q2", "2020Q3"]),
                         tpr_2021=m(["2021Q1", "2021Q2", "2021Q3", "2021Q4"]),
                         tpr_2021q4=m(["2021Q4"]),
                         tpr_2022q1_q3=m(["2022Q1", "2022Q2", "2022Q3"]),
                         tpr_2022=m(["2022Q1", "2022Q2", "2022Q3", "2022Q4"]),
                         receipt_2020q1_q3=m(["2020Q1", "2020Q2", "2020Q3"], "index_receipt"),
                         receipt_2022q1_q3=m(["2022Q1", "2022Q2", "2022Q3"], "index_receipt"),
                         banxico_2022q1_q3=m(["2022Q1", "2022Q2", "2022Q3"], "index_banxico")))
    out = pd.DataFrame(rows)
    out["dlog_tpr_2022q1_q3"] = np.log(out.tpr_2022q1_q3)
    out["dlog_tpr_2020q1_q3"] = np.log(out.tpr_2020q1_q3)
    out.to_csv(OUT / "pre_model_tpr.csv", index=False, float_format="%.6g")
    return out


# ---------------------------------------------------------------- B. ingresos, monthly
def household_months(ing_path, year, inpc, fx):
    """Per household: interview month, and per calendar month the P041 amount (nominal,
    real Aug-2024 MXN, USD). Returns (window table, P041 long table), aggregated to
    household x month. Nothing is written to disk."""
    i = pd.read_csv(ing_path, dtype=str, encoding="utf-8-sig")
    i["mes_1"] = i["mes_1"].str.strip()
    dated = i[i.mes_1 != ""]
    iv = dated.groupby(["folioviv", "foliohog"])["mes_1"].agg(["min", "max"])
    n_incons = int((iv["min"] != iv["max"]).sum())
    iv = iv["min"].astype(int).rename("m1").reset_index()   # m1 = month before interview
    p = i[i.clave == "P041"].copy()
    long = []
    for k in range(1, 7):
        x = p[["folioviv", "foliohog", f"mes_{k}", f"ing_{k}"]].copy()
        x.columns = ["folioviv", "foliohog", "month", "amt"]
        x["month"] = pd.to_numeric(x.month.str.strip(), errors="coerce")
        x["amt"] = pd.to_numeric(x.amt.str.strip(), errors="coerce").fillna(0)
        x["lag"] = k
        long.append(x)
    long = pd.concat(long).dropna(subset=["month"])
    long["month"] = long.month.astype(int)
    long = long.groupby(["folioviv", "foliohog", "month", "lag"], as_index=False)["amt"].sum()
    per = [pd.Period(f"{year}-{m:02d}", "M") for m in long.month]
    long["real"] = long.amt * [inpc["2024-08"] / inpc[str(pp)] for pp in per]
    long["usd"] = long.amt / fx[per].values
    # check: ing_tri = sum_k ing_k * INPC(Aug)/INPC(month_k) / 2
    pk = p.copy()
    tri = pd.to_numeric(pk.ing_tri, errors="coerce")
    rebuilt = 0
    for k in range(1, 7):
        mm = pd.to_numeric(pk[f"mes_{k}"].str.strip(), errors="coerce")
        f = mm.map(lambda v: inpc[f"{year}-08"] / inpc[f"{year}-{int(v):02d}"] if v == v else np.nan)
        rebuilt = rebuilt + pd.to_numeric(pk[f"ing_{k}"].str.strip(), errors="coerce").fillna(0) * f
    chk = float(np.nanmax(np.abs(rebuilt / 2 - tri) / np.maximum(tri, 1)))
    return iv, long, n_incons, chk


def part_b(c_paths, ing_paths, inpc, fx):
    res, Zm, checks, lagrows, cnt = [], {}, [], [], []
    for year in [2016, 2018, 2020]:
        d = load_ch(year, c_paths[2016])
        iv, long, n_incons, chk = household_months(ing_paths[year], year, inpc, fx)
        d = d.merge(iv, on=["folioviv", "foliohog"], how="left")
        undated = d.m1.isna()
        des = Design(d)
        one = np.ones(len(d))
        key = d.folioviv + "_" + d.foliohog
        long["key"] = long.folioviv + "_" + long.foliohog
        # receipt in at least one of the six months (consistency with remesas > 0)
        anypos = set(long.loc[long.amt > 0, "key"])
        any6 = key.isin(anypos).values.astype(float)
        allm = np.ones(len(d), bool)
        r6, t6 = ratio(d, des, any6, one, allm)
        rch, tch = ratio(d, des, d.rec.values, one, allm)
        nmon = long[long.amt > 0].groupby("key").month.nunique()
        months_rec = key.map(nmon).fillna(0).values
        mr, tmr = ratio(d, des, months_rec, any6, allm)
        six, tsix = ratio(d, des, (months_rec == 6).astype(float), any6, allm)
        checks.append(dict(year=year, n=len(d), undated_households=int(undated.sum()),
                           undated_weighted_share=d.w[undated].sum() / d.w.sum(),
                           undated_with_remesas=int((undated & (d.rec > 0)).sum()),
                           households_inconsistent_interview_month=n_incons,
                           max_rel_error_ing_tri_rebuilt=chk,
                           receipt_concentradohogar=rch, receipt_concentradohogar_se=se(tch),
                           receipt_any_of_six_months=r6, receipt_any_of_six_months_se=se(t6),
                           mean_months_received_of_6=mr, mean_months_received_se=se(tmr),
                           share_recipients_all_6_months=six, share_all_6_se=se(tsix)))
        # interview-month distribution (weighted)
        for m1, g in d[~undated].groupby("m1"):
            cnt.append(dict(year=year, interview_month=int(m1) + 1, n=len(g),
                            weighted_share=g.w.sum() / d.w[~undated].sum()))
        # by calendar month
        Zm[year] = {}
        for cm in WINDOW:
            cover = (~undated & (d.m1 >= cm) & (d.m1 - 5 <= cm)).values
            lm = long[long.month == cm].groupby("key")[["amt", "real", "usd"]].sum()
            amt = key.map(lm.real).fillna(0).values
            usd = key.map(lm.usd).fillna(0).values
            rec = (amt > 0).astype(float)
            n, nr = int(cover.sum()), int((cover & (rec > 0)).sum())
            p, tp = ratio(d, des, rec, one, cover)
            a, ta = ratio(d, des, amt, rec, cover)
            au, tau = ratio(d, des, usd, rec, cover)
            h, th = ratio(d, des, amt, one, cover)
            Zm[year][cm] = {"lp": tp / p, "la": ta / a, "lh": th / h,
                            "val": {"lp": np.log(p), "la": np.log(a), "lh": np.log(h)},
                            "ok": n >= MIN_N and nr >= MIN_N}
            res.append(dict(year=year, month=cm, month_name=MONTHS[cm - 1], n=n, n_recip=nr,
                            receipt=p, receipt_se=se(tp), amount_mean_real=a,
                            amount_mean_real_se=se(ta), amount_mean_usd=au,
                            amount_mean_usd_se=se(tau), per_hh_real=h, per_hh_real_se=se(th)))
        # by lag (recall check): k = 1 is the month before the interview
        for k in range(1, 7):
            lk = long[long.lag == k].groupby("key").amt.sum()
            rec = (key.map(lk).fillna(0).values > 0).astype(float)
            m = (~undated).values
            p, tp = ratio(d, des, rec, one, m)
            lagrows.append(dict(year=year, lag=k, n=int(m.sum()), n_recip=int((m & (rec > 0)).sum()),
                                receipt=p, receipt_se=se(tp)))
        print(year, "monthly done; any-of-6", round(r6, 4), "vs remesas>0", round(rch, 4))
    mon = pd.DataFrame(res)
    bad = (mon.n < MIN_N) | (mon.n_recip < MIN_N)
    mon.loc[mon.n < MIN_N, ["receipt", "receipt_se", "per_hh_real", "per_hh_real_se"]] = np.nan
    mon.loc[bad, ["amount_mean_real", "amount_mean_real_se", "amount_mean_usd",
                  "amount_mean_usd_se"]] = np.nan
    mon.to_csv(OUT / "monthly_profile.csv", index=False, float_format="%.6g")
    pd.DataFrame(checks).to_csv(OUT / "monthly_checks.csv", index=False, float_format="%.6g")
    pd.DataFrame(lagrows).to_csv(OUT / "monthly_by_lag.csv", index=False, float_format="%.6g")
    pd.DataFrame(cnt).to_csv(OUT / "monthly_interview_months.csv", index=False, float_format="%.6g")

    # ratios across years, same calendar month (independent samples)
    rat = []
    for y1, y0 in [(2020, 2018), (2018, 2016)]:
        for cm in WINDOW:
            A, B = Zm[y1][cm], Zm[y0][cm]
            if not (A["ok"] and B["ok"]):
                continue
            r = dict(year=y1, base_year=y0, month=cm, month_name=MONTHS[cm - 1])
            for k, nm in [("lp", "receipt"), ("la", "amount"), ("lh", "per_hh")]:
                e = A["val"][k] - B["val"][k]
                s = np.sqrt(Design.var(A[k]) + Design.var(B[k]))
                r[f"dlog_{nm}"], r[f"dlog_{nm}_se"], r[f"ratio_{nm}"] = e, s, np.exp(e)
            rat.append(r)
    rat = pd.DataFrame(rat)
    rat.to_csv(OUT / "monthly_ratio.csv", index=False, float_format="%.6g")

    # month-to-month changes within a year (covariance kept), and the difference 2020 - 2018
    mm = []
    for a_, b_ in [(3, 2), (4, 3), (5, 4), (6, 5), (5, 3), (4, 2)]:
        r = dict(from_month=MONTHS[b_ - 1], to_month=MONTHS[a_ - 1])
        vv = {}
        for y in [2016, 2018, 2020]:
            A, B = Zm[y][a_], Zm[y][b_]
            for k, nm in [("lp", "receipt"), ("la", "amount"), ("lh", "per_hh")]:
                e = A["val"][k] - B["val"][k]
                v = Design.var(combine((1, A[k]), (-1, B[k])))
                r[f"{nm}_{y}"], r[f"{nm}_{y}_se"] = e, np.sqrt(v)
                vv[(y, nm)] = (e, v)
        for nm in ["receipt", "amount", "per_hh"]:
            for y1, y0 in [(2020, 2018), (2018, 2016)]:
                e = vv[(y1, nm)][0] - vv[(y0, nm)][0]
                r[f"{nm}_{y1}_minus_{y0}"] = e
                r[f"{nm}_{y1}_minus_{y0}_se"] = np.sqrt(vv[(y1, nm)][1] + vv[(y0, nm)][1])
        mm.append(r)
    mm = pd.DataFrame(mm)
    mm.to_csv(OUT / "monthly_month_to_month.csv", index=False, float_format="%.6g")

    # model receipt index (COVID / no-COVID) by month, household model
    e1 = pd.read_csv(HHMODEL / "e1_monthly.csv", parse_dates=["date"])
    e1 = e1[e1.date.dt.year == 2020]
    e1["model_receipt_index"] = e1.receipt_covid / e1.receipt_cf
    e1["month"] = e1.date.dt.month
    mod = e1.pivot_table(index="month", columns="kappa", values="model_receipt_index")
    mod.columns = [f"model_receipt_index_kappa_{c:g}" for c in mod.columns]
    mod.reset_index().to_csv(OUT / "monthly_model_receipt_2020.csv", index=False, float_format="%.6g")
    return mon, rat, mm, mod


# ---------------------------------------------------------------- figures
def figures(nat, dec, ex, mon, rat, mod):
    # F4: national 2016-2024 and margin decomposition
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    ax[0].errorbar(nat.year, 100 * nat.receipt, yerr=196 * nat.receipt_se, color=BLUE, lw=2,
                   marker="o", ms=6, capsize=3)
    ax[0].set_title("Households receiving (%)", loc="left", color=INK)
    ax[0].set_xticks(YEARS)
    ax[1].errorbar(nat.year, nat.amount_mean_usd, yerr=1.96 * nat.amount_mean_usd_se, color=BLUE,
                   lw=2, marker="o", ms=6, capsize=3)
    ax[1].set_title("Quarterly amount per recipient household (USD)", loc="left", color=INK)
    ax[1].set_xticks(YEARS)
    ax[1].set_ylim(0, None)
    sel = dec[dec.from_year.isin([2016, 2018, 2020]) & dec.to_year.isin([2018, 2020, 2022])]
    sel = sel[~((sel.from_year == 2016) & (sel.to_year == 2022))]
    labels = [f"{a}-{b}" for a, b in zip(sel.from_year, sel.to_year)]
    x = np.arange(len(sel))
    wdt = 0.27
    for o, (c, col, lab) in enumerate([("dlog_receipt", GREY, "Receipt (extensive)"),
                                       ("dlog_amount", BLUE, "Amount per recipient (intensive)"),
                                       ("dlog_tpr_counts", ORANGE, "Banxico counts per ENIGH receiving hh")]):
        ax[2].bar(x + (o - 1) * wdt, sel[c], wdt, color=col, label=lab,
                  yerr=1.96 * sel[c + "_se"], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    ax[2].axhline(0, color=INK, lw=0.8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels)
    ax[2].set_title("Log change by interval", loc="left", color=INK)
    ax[2].grid(axis="x", visible=False)
    ax[2].set_ylim(-0.17, 0.62)
    ax[2].legend(frameon=False, fontsize=7, loc="upper left")
    fig.text(0.01, -0.03, "ENIGH concentradohogar 2016-2024 (real amounts in Aug 2024 MXN in the tables; USD at the "
             "mean Feb-Oct rate); Banxico counts summed Feb-Oct. 95% design-based CIs.", color=MUTED, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "F4_pre_period.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # F5: monthly receipt rate and 2020/2018 ratio
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    cols = {2016: GREY, 2018: BLUE, 2020: ORANGE}
    for y in [2016, 2018, 2020]:
        s = mon[mon.year == y]
        ax[0].errorbar(s.month, 100 * s.receipt, yerr=196 * s.receipt_se, color=cols[y], lw=2,
                       marker="o", ms=4, capsize=2, label=str(y))
        ax[1].errorbar(s.month, s.amount_mean_usd, yerr=1.96 * s.amount_mean_usd_se, color=cols[y],
                       lw=2, marker="o", ms=4, capsize=2, label=str(y))
    ax[0].set_title("Households receiving in the month (%)", loc="left", color=INK)
    ax[1].set_title("Monthly amount per recipient (USD)", loc="left", color=INK)
    ax[0].legend(frameon=False, fontsize=7)
    r = rat[rat.year == 2020]
    ax[2].errorbar(r.month, r.ratio_receipt, yerr=[r.ratio_receipt - np.exp(r.dlog_receipt - 1.96 * r.dlog_receipt_se),
                                                   np.exp(r.dlog_receipt + 1.96 * r.dlog_receipt_se) - r.ratio_receipt],
                   color=ORANGE, lw=2, marker="o", ms=4, capsize=2, label="ENIGH receipt 2020 / 2018")
    r0 = rat[rat.year == 2018]
    ax[2].plot(r0.month, r0.ratio_receipt, color=GREY, lw=1.5, marker="o", ms=3, label="ENIGH receipt 2018 / 2016")
    kcol = [c for c in mod.columns if c.endswith("_1")]
    if kcol:
        mm_ = mod.loc[list(WINDOW), kcol[0]]
        ax[2].plot(mm_.index, mm_.values, color=INK, lw=1.5, ls="--", label="Model receipt, COVID / no-COVID (kappa 1)")
    ax[2].axhline(1, color=INK, lw=0.8)
    ax[2].set_title("Receipt ratio, same calendar month", loc="left", color=INK)
    ax[2].legend(frameon=False, fontsize=7, loc="lower left")
    for a in ax:
        a.set_xticks(list(WINDOW))
        a.set_xticklabels([MONTHS[m - 1] for m in WINDOW])
    fig.text(0.01, -0.03, "ENIGH ingresos, clave P041 (income from abroad), six reference months mapped to calendar "
             "months; Feb and Oct come from one interview cohort only. 95% design-based CIs.", color=MUTED, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "F5_monthly_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c16", required=True)
    ap.add_argument("--ing16", required=True)
    ap.add_argument("--ing18", required=True)
    ap.add_argument("--ing20", required=True)
    ap.add_argument("--fx", required=True)
    a = ap.parse_args()
    inpc = load_inpc()
    fx = load_fx(rpath(a.fx))
    nat, dec, ex, bwo = part_a(rpath(a.c16), inpc, fx)
    mt = model_tpr()
    mon, rat, mm, mod = part_b({2016: rpath(a.c16)},
                               {2016: rpath(a.ing16), 2018: rpath(a.ing18), 2020: rpath(a.ing20)}, inpc, fx)
    figures(nat, dec, ex, mon, rat, mod)
    pd.set_option("display.width", 250)
    print(nat.round(4).to_string())
    print(dec.round(4).to_string())
    print(ex.round(4).to_string())
    print(bwo.round(3).to_string())
    print(mt.round(4).to_string())
    print(mon.round(4).to_string())
    print(rat.round(4).to_string())
    print(mm.round(4).to_string())
    print(mod.round(4).to_string())


if __name__ == "__main__":
    main()
