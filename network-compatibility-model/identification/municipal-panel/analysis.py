"""Municipal test of the need hypothesis: within-state regressions, event study, terciles, placebo.

Reads <mun-raw>/derived/municipal_panel.csv and municipal_quarterly_dev.csv (from build_municipal.py).
Writes aggregate tables to outputs/ and three figures to figures/. Run with `python -I -X utf8`.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS  # statsmodels.api import is broken in this env

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT, FIG = HERE / "outputs", HERE / "figures"
ap = argparse.ArgumentParser()
ap.add_argument("--mun-raw", default=str(ROOT / "data" / "external" / "municipal-panel"))
ap.add_argument("--min-usd", type=float, default=1.0, help="2019 USD millions threshold")
args = ap.parse_args()
DER = Path(args.mun_raw) / "derived"
RNG = np.random.default_rng(20261029)
NPERM = 2000

SOUTH_TRANSIT_CORE = [7, 27, 20, 30]  # same lists as the state panel
SOUTH_TRANSIT_EXT = [7, 27, 20, 30, 12, 23, 4]

U = pd.read_csv(DER / "municipal_panel.csv", index_col=0)
U = U.replace([np.inf, -np.inf], np.nan)
S = U[U.pos_1719 & (U.remit_2019_usd_m >= args.min_usd) & U.iim_dp2.notna()].copy()
S["log_remit_pc_2019"] = np.log(S.remit_pc_2019_usd)
S["log_pop"] = np.log(S.pop2020)
YEARS = [2020, 2021, 2022, 2023, 2024]


def winsor(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


for y in YEARS:
    S[f"w_exc1719_{y}"] = winsor(S[f"exc1719_{y}"])
S["w_exc1315_2016"] = winsor(S.exc1315_2016)
S["w_exc1519_2021"] = winsor(S.exc1519_2021)
S["w_dlog_2021"] = winsor(S.dlog_2021)
NEEDRAW = ["excess_pc_2020_21", "covid_pc_2020_21", "excess1719_pc_2020_21", "excesstr_pc_2020_21",
           "excess_pct_2020_21", "excess_pc_2020", "excess_pc_2021"]
for v in NEEDRAW:
    S["z_" + v] = (S[v] - S[v].mean()) / S[v].std()
# within-state SD of the main need variable, for interpretation
WSD = (S.excess_pc_2020_21 - S.groupby("ent").excess_pc_2020_21.transform("mean")).std()
CONTROLS = ["pretrend_1719", "log_remit_pc_2019", "log_pop", "urban_share", "iim_dp2"]
SAMPLES = {"all": S.index == S.index, "excl_south_core": ~S.ent.isin(SOUTH_TRANSIT_CORE),
           "excl_south_ext": ~S.ent.isin(SOUTH_TRANSIT_EXT)}
NEEDS = {"excess_mort": "z_excess_pc_2020_21", "covid_deaths": "z_covid_pc_2020_21"}

C_LOW, C_MID, C_HIGH, C_ALT = "#2a78d6", "#9a9993", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})


def design(d, xs, fe=True):
    X = d[xs].astype(float)
    if fe:
        X = X.join(pd.get_dummies(d.ent, prefix="ent", drop_first=True, dtype=float))
    X.insert(0, "const", 1.0)
    return X


def fit(y, xs, d, fe=True, weights=None):
    """Returns (HC3 fit, state-cluster fit) on rows with non-missing data."""
    d = d.dropna(subset=[y] + xs)
    X = design(d, xs, fe)
    mod = OLS(d[y], X) if weights is None else WLS(d[y], X, weights=d[weights])
    return mod.fit(cov_type="HC3"), mod.fit(cov_type="cluster", cov_kwds={"groups": d.ent.values}), d


def perm_p(y, xs, d, var, fe=True):
    """Permute `var` within state; statistic |b| from the FWL regression (sharp null, within-state exchangeability)."""
    d = d.dropna(subset=[y] + xs)
    Z = design(d, [x for x in xs if x != var], fe).values
    Q, _ = np.linalg.qr(Z)
    res = lambda v: v - Q @ (Q.T @ v)
    ry = res(d[y].values.astype(float))
    x = d[var].values.astype(float)
    rx = res(x)
    b0 = abs(rx @ ry / (rx @ rx))
    groups = [np.where(d.ent.values == e)[0] for e in np.unique(d.ent.values)]
    cnt = 0
    for _ in range(NPERM):
        xp = x.copy()
        for g in groups:
            xp[g] = x[RNG.permutation(g)]
        r = res(xp)
        cnt += abs(r @ ry / (r @ r)) >= b0
    return (cnt + 1) / (NPERM + 1)


def row(h, c, var, **kw):
    return dict(**kw, coef=h.params[var], se_hc3=h.bse[var], p_hc3=h.pvalues[var], se_cl=c.bse[var],
                p_cl=c.pvalues[var], ci_lo_cl=c.params[var] - 1.96 * c.bse[var],
                ci_hi_cl=c.params[var] + 1.96 * c.bse[var], n=int(h.nobs), n_states=None, r2=h.rsquared)


# ------------------------------------------------------------- main regressions
rows = []
for y in YEARS:
    for oname, ov in [("raw", f"exc1719_{y}"), ("winsor1_99", f"w_exc1719_{y}")]:
        for sname, mask in SAMPLES.items():
            d = S[mask]
            for wname, w in [("unweighted", None), ("w_usd2019", "remit_2019_usd_m")]:
                for nname, nv in NEEDS.items():
                    for spec, xs, fe in [("bivariate_noFE", [nv], False), ("stateFE", [nv], True),
                                         ("stateFE_controls", [nv] + CONTROLS, True)]:
                        h, c, dd = fit(ov, xs, d, fe, w)
                        r = row(h, c, nv, year=y, outcome=oname, sample=sname, weight=wname, need=nname, spec=spec)
                        r["n_states"] = dd.ent.nunique()
                        rows.append(r)
reg = pd.DataFrame(rows)
reg["mde_80_cl"] = 2.8 * reg[["se_hc3", "se_cl"]].max(axis=1)
reg["p_perm_within_state"] = np.nan
for i, r in reg.iterrows():
    if (r.weight == "unweighted" and r.spec == "stateFE_controls" and r["sample"] == "all"
            and r.need == "excess_mort" and r.year in (2020, 2021, 2022, 2023)):
        ov = f"exc1719_{r.year}" if r.outcome == "raw" else f"w_exc1719_{r.year}"
        reg.loc[i, "p_perm_within_state"] = perm_p(ov, [NEEDS["excess_mort"]] + CONTROLS, S, NEEDS["excess_mort"])
reg.to_csv(OUT / "regressions_need.csv", index=False, float_format="%.5g")

# ------------------------------------------------------------- robustness (2021, excess mortality, FE + controls)
rob = []
nv = NEEDS["excess_mort"]
S["pop10k"] = S.pop2020 >= 10000
variants = [
    ("base", "exc1719_2021", [nv] + CONTROLS, S),
    ("base, winsorized", "w_exc1719_2021", [nv] + CONTROLS, S),
    ("excess vs 2017-19 mean", "exc1719_2021", ["z_excess1719_pc_2020_21"] + CONTROLS, S),
    ("excess vs 2015-19 linear trend", "exc1719_2021", ["z_excesstr_pc_2020_21"] + CONTROLS, S),
    ("excess in % of 2015-19 deaths", "exc1719_2021", ["z_excess_pct_2020_21"] + CONTROLS, S),
    ("excess 2020 only", "exc1719_2021", ["z_excess_pc_2020"] + CONTROLS, S),
    ("excess 2021 only", "exc1719_2021", ["z_excess_pc_2021"] + CONTROLS, S),
    ("+ share 65+", "exc1719_2021", [nv] + CONTROLS + ["pop65_share"], S),
    ("+ COVID deaths (joint)", "exc1719_2021", [nv, "z_covid_pc_2020_21"] + CONTROLS, S),
    ("outcome: 2015-19 trend window (winsor)", "w_exc1519_2021", [nv, "pretrend_1519"] + CONTROLS[1:], S),
    ("outcome: dlog 2019-21 + growth 2016-19 (winsor)", "w_dlog_2021", [nv, "growth_1619"] + CONTROLS[1:], S),
    ("pop >= 10,000", "exc1719_2021", [nv] + CONTROLS, S[S.pop10k]),
    ("2019 USD >= 5m", "exc1719_2021", [nv] + CONTROLS, S[S.remit_2019_usd_m >= 5]),
    ("excl. southern transit core", "exc1719_2021", [nv] + CONTROLS, S[SAMPLES["excl_south_core"]]),
    ("excl. southern transit ext.", "exc1719_2021", [nv] + CONTROLS, S[SAMPLES["excl_south_ext"]]),
    ("excl. southern ext., winsorized", "w_exc1719_2021", [nv] + CONTROLS, S[SAMPLES["excl_south_ext"]]),
]
for name, y, xs, d in variants:
    for wname, w in [("unweighted", None), ("w_usd2019", "remit_2019_usd_m")]:
        h, c, dd = fit(y, xs, d, True, w)
        r = row(h, c, xs[0], spec=name, weight=wname)
        r["n_states"] = dd.ent.nunique()
        rob.append(r)
# pre-period placebo: 2016 vs own 2013-15 trend
P = S[S.exc1315_2016.notna()]
for oname, y in [("placebo 2016 vs 2013-15 trend", "exc1315_2016"), ("placebo 2016, winsorized", "w_exc1315_2016")]:
    for wname, w in [("unweighted", None), ("w_usd2019", "remit_2019_usd_m")]:
        xs = [nv, "pretrend_1315"] + CONTROLS[1:]
        h, c, dd = fit(y, xs, P, True, w)
        r = row(h, c, nv, spec=oname, weight=wname)
        r["n_states"] = dd.ent.nunique()
        rob.append(r)
rob = pd.DataFrame(rob)
rob["mde_80_cl"] = 2.8 * rob[["se_hc3", "se_cl"]].max(axis=1)
rob["p_perm_within_state"] = np.nan
for i, r in rob.iterrows():
    if r.spec.startswith("placebo") and r.weight == "unweighted":
        y = "exc1315_2016" if "winsor" not in r.spec else "w_exc1315_2016"
        rob.loc[i, "p_perm_within_state"] = perm_p(y, [nv, "pretrend_1315"] + CONTROLS[1:], P, nv)
rob.to_csv(OUT / "robustness_2021.csv", index=False, float_format="%.5g")

# ------------------------------------------------------------- quarterly event study 2019Q1-2024Q4
Q = pd.read_csv(DER / "municipal_quarterly_dev.csv")
Q = Q[Q.uid.isin(S.index) & Q.year.between(2019, 2024)].replace([np.inf, -np.inf], np.nan)
ev = []
for (yy, qq), g in Q.groupby(["year", "quarter"]):
    d = S.join(g.set_index("uid").dev)
    d["w_dev"] = winsor(d.dev)
    for oname, ov in [("raw", "dev"), ("winsor1_99", "w_dev")]:
        for sname in ["all", "excl_south_ext"]:
            for nname in NEEDS:
                h, c, dd = fit(ov, [NEEDS[nname]] + CONTROLS, d[SAMPLES[sname]], True)
                ev.append(row(h, c, NEEDS[nname], year=yy, quarter=qq, outcome=oname, sample=sname, need=nname))
ev = pd.DataFrame(ev).drop(columns="n_states")
ev["t"] = ev.year + (ev.quarter - 1) / 4
ev.to_csv(OUT / "event_study_quarterly.csv", index=False, float_format="%.5g")

# ------------------------------------------------------------- within-state terciles of excess mortality
S["need_pct_ws"] = S.groupby("ent").excess_pc_2020_21.rank(pct=True)
S["tercile"] = pd.cut(S.need_pct_ws, [0, 1 / 3, 2 / 3, 1], labels=["low", "mid", "high"], include_lowest=True)
G = Q.merge(S[["tercile", "ent", "remit_2019_usd_m"]], left_on="uid", right_index=True)
G["w_dev"] = G.groupby(["year", "quarter"]).dev.transform(winsor)
terc = []
for (yy, qq, t), g in G.groupby(["year", "quarter", "tercile"], observed=True):
    terc.append(dict(year=yy, quarter=qq, tercile=t, n=len(g), mean_dev=g.dev.mean(), median_dev=g.dev.median(),
                     mean_dev_winsor=g.w_dev.mean(),
                     usdw_mean_dev=np.average(g.dev.fillna(0), weights=g.remit_2019_usd_m)))
terc = pd.DataFrame(terc)
terc["t"] = terc.year + (terc.quarter - 1) / 4
terc.to_csv(OUT / "terciles_within_state_quarterly.csv", index=False, float_format="%.5g")
# annual tercile means and high-minus-low within state
ta = []
for y in YEARS:
    d = S.assign(high=(S.tercile == "high").astype(float), mid=(S.tercile == "mid").astype(float))
    h, c, _ = fit(f"exc1719_{y}", ["high", "mid"], d, True)
    hw, cw, _ = fit(f"w_exc1719_{y}", ["high", "mid"], d, True)
    m = S.groupby("tercile", observed=True)[f"exc1719_{y}"].agg(["mean", "median"])
    ta.append(dict(year=y, low_mean=m.loc["low", "mean"], mid_mean=m.loc["mid", "mean"], high_mean=m.loc["high", "mean"],
                   low_median=m.loc["low", "median"], mid_median=m.loc["mid", "median"],
                   high_median=m.loc["high", "median"], high_minus_low_FE=h.params["high"], se_cl=c.bse["high"],
                   high_minus_low_FE_winsor=hw.params["high"], se_cl_winsor=cw.bse["high"]))
ta = pd.DataFrame(ta)
ta.to_csv(OUT / "terciles_within_state_annual.csv", index=False, float_format="%.5g")
tmean = S.groupby("tercile", observed=True).excess_pc_2020_21.mean()

# ------------------------------------------------------------- sample description
desc = S[["remit_2019_usd_m", "remit_pc_2019_usd", "pop2020", "urban_share", "iim_dp2", "pretrend_1719",
          "excess_pc_2020_21", "covid_pc_2020_21", "exc1719_2019", "exc1719_2020", "exc1719_2021", "exc1719_2022",
          "exc1719_2023", "exc1315_2016"]].describe(percentiles=[.01, .1, .5, .9, .99]).T
desc.loc["_sample", "count"] = len(S)
desc.loc["_states", "count"] = S.ent.nunique()
desc.loc["_share_2019_usd_in_sample", "count"] = S.remit_2019_usd_m.sum() / U.remit_2019_usd_m.sum()
desc.loc["_share_pop_in_sample", "count"] = S.pop2020.sum() / U.pop2020.sum()
desc.loc["_units_positive_1719", "count"] = U.pos_1719.sum()
desc.loc["_within_state_sd_excess_pc", "count"] = WSD
desc.loc["_corr_excess_covid", "count"] = S.excess_pc_2020_21.corr(S.covid_pc_2020_21)
desc.to_csv(OUT / "sample_description.csv", float_format="%.5g")

# ------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, oname, lab in [(axes[0], "raw", "raw"), (axes[1], "winsor1_99", "winsorized 1/99")]:
    for sname, col, off in [("all", C_LOW, -0.03), ("excl_south_ext", C_HIGH, 0.03)]:
        d = ev[(ev["sample"] == sname) & (ev.need == "excess_mort") & (ev.outcome == oname)]
        ax.errorbar(d.t + off, d.coef, yerr=1.96 * d.se_cl, fmt="o", ms=3.5, color=col, lw=1, capsize=0,
                    label="all states" if sname == "all" else "excl. southern transit")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(2020.125, color=MUTED, lw=0.6, ls=":")
    ax.set_title(f"Coef. on excess mortality (1 SD), FE + controls, {lab}", fontsize=9, loc="left")
    ax.set_xlabel("quarter (2019 is inside the trend window)")
axes[0].set_ylabel("coef. on log deviation from own 2017-19 trend")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "F1_event_study.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, col_v, lab in [(axes[0], "mean_dev_winsor", "mean (winsorized 1/99)"), (axes[1], "median_dev", "median")]:
    for t, col in [("low", C_LOW), ("mid", C_MID), ("high", C_HIGH)]:
        d = terc[terc.tercile == t]
        ax.plot(d.t, d[col_v], color=col, lw=2, marker="o", ms=3,
                label=f"{t} (excess {tmean[t]:.1f}/1,000)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(f"Within-state excess-mortality terciles: {lab}", fontsize=9, loc="left")
axes[0].set_ylabel("log deviation from own 2017-19 trend")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "F2_terciles_within_state.png")
plt.close(fig)

# binned scatter, 2021, residualized on state FE (+ controls)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, xs_c, lab in [(axes[0], [], "state FE only"), (axes[1], CONTROLS, "state FE + controls")]:
    d = S.dropna(subset=["w_exc1719_2021", "excess_pc_2020_21"] + CONTROLS)
    Z = design(d, xs_c, True).values
    Qz, _ = np.linalg.qr(Z)
    rx = d.excess_pc_2020_21.values - Qz @ (Qz.T @ d.excess_pc_2020_21.values)
    ry = d.w_exc1719_2021.values - Qz @ (Qz.T @ d.w_exc1719_2021.values)
    bins = pd.qcut(rx, 20, labels=False)
    bx = pd.Series(rx).groupby(bins).mean()
    by = pd.Series(ry).groupby(bins).mean()
    ax.scatter(rx, ry, s=4, color=C_MID, alpha=0.25, lw=0)
    ax.scatter(bx, by, s=30, color=C_LOW, edgecolor="white", zorder=3, label="20 bins")
    b = rx @ ry / (rx @ rx)
    xx = np.linspace(np.percentile(rx, 1), np.percentile(rx, 99), 10)
    ax.plot(xx, b * xx, color=INK, lw=1.2, label=f"slope {b:.4f} per death/1,000")
    ax.set_xlim(np.percentile(rx, 0.5), np.percentile(rx, 99.5))
    ax.set_ylim(-0.6, 0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("excess deaths 2020-21 per 1,000, within state (residual)")
    ax.set_title(f"2021 deviation (winsorized), {lab}", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False)
axes[0].set_ylabel("2021 log deviation from own trend (residual)")
fig.tight_layout()
fig.savefig(FIG / "F3_binscatter_2021_within_state.png")
plt.close(fig)

# ------------------------------------------------------------- console summary
pd.set_option("display.width", 220)
print("N", len(S), "states", S.ent.nunique(), "within-state SD excess", round(WSD, 3), "overall SD",
      round(S.excess_pc_2020_21.std(), 3))
cols = ["coef", "se_hc3", "se_cl", "p_cl", "p_perm_within_state", "mde_80_cl", "n"]
h = reg[(reg.spec == "stateFE_controls") & (reg.weight == "unweighted")]
print(h.set_index(["need", "outcome", "sample", "year"])[cols].round(4).to_string())
w = reg[(reg.spec != "stateFE_controls") & (reg.weight == "unweighted") & (reg["sample"] == "all") & (reg.need == "excess_mort")]
print(w.set_index(["spec", "outcome", "year"])[["coef", "se_cl", "p_cl"]].round(4).to_string())
w = reg[(reg.spec == "stateFE_controls") & (reg.weight == "w_usd2019") & (reg.need == "excess_mort")]
print(w.set_index(["outcome", "sample", "year"])[["coef", "se_hc3", "se_cl", "p_cl"]].round(4).to_string())
print(rob.set_index(["spec", "weight"])[["coef", "se_hc3", "se_cl", "p_cl", "p_perm_within_state", "n"]].round(4).to_string())
e = ev[(ev.need == "excess_mort") & (ev["sample"] == "all")]
print(e.pivot_table(index="t", columns="outcome", values=["coef", "se_cl"]).round(3).to_string())
e = ev[(ev.need == "covid_deaths") & (ev["sample"] == "all") & (ev.outcome == "winsor1_99")]
print("covid ES", e.set_index("t")[["coef", "se_cl"]].round(3).T.to_string())
print(ta.round(3).to_string())
print(tmean.round(2))
