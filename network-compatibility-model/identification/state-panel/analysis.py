"""State-level test of the need hypothesis: regressions, influence, timing, transit checks.

Reads outputs/state_panel.csv and outputs/remittances_state_quarterly_dev_trend1719.csv
(from build_panel.py). Writes tables to outputs/ and figures to figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS  # statsmodels.api import is broken in this env
from statsmodels.tools.tools import add_constant

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DERIVED = HERE.parents[2] / "data" / "external" / "state-panel" / "derived"
FIG = HERE / "figures"
RNG = np.random.default_rng(20261029)
NPERM = 5000

P = pd.read_csv(OUT / "state_panel.csv", index_col=0)
P["log_hh_remit"] = np.log(P.hh_remit_pct)
P["log_upm_pc_2021"] = np.log(P.upm_pc_2021)
P["log_upm_pc_2022"] = np.log(P.upm_pc_2022)
P["log_pop"] = np.log(P.pop2020)
for v in ["excess_pc_2020_21", "excess_pct_2020_21", "covid_pc_2020_21", "excess_pc_2020", "covid_pc_2020",
          "labinc_loss_2020", "labinc_loss_2021", "d_labpov_2020", "log_upm_pc_2021"]:
    P["z_" + v] = (P[v] - P[v].mean()) / P[v].std()

YEARS = [2020, 2021, 2022, 2023]
SAMPLES = {
    "all": P.index == P.index,
    "excl_south_core": ~P.south_transit_core.astype(bool),
    "excl_south_ext": ~P.south_transit_ext.astype(bool),
    "excl_south_ext_north": ~(P.south_transit_ext.astype(bool) | P.north_border.astype(bool)),
}
CONTROLS = ["pretrend_1719", "log_hh_remit"]

C_OTHER, C_SOUTH, C_NORTH = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150})


def fit(y, xs, d, weights=None):
    X = add_constant(d[xs].astype(float))
    if weights is None:
        m = OLS(d[y].astype(float), X).fit(cov_type="HC3")
    else:
        m = WLS(d[y].astype(float), X, weights=d[weights]).fit(cov_type="HC3")
    return m


def perm_p(y, xs, d, var, weights=None):
    """Randomization-inference p-value: permute `var` across states, compare |t| (HC3)."""
    t0 = abs(fit(y, xs, d, weights).tvalues[var])
    j = xs.index(var) + 1
    X0 = np.column_stack([np.ones(len(d))] + [d[x].astype(float).values for x in xs])
    yv = d[y].astype(float).values
    cnt = 0
    for _ in range(NPERM):
        X = X0.copy()
        X[:, j] = RNG.permutation(X0[:, j])
        XtXi = np.linalg.inv(X.T @ X)
        b = XtXi @ X.T @ yv
        e = yv - X @ b
        h = np.einsum("ij,jk,ik->i", X, XtXi, X)
        meat = (X * (e / (1 - h))[:, None] ** 2).T @ X
        V = XtXi @ meat @ XtXi
        cnt += abs(b[j] / np.sqrt(V[j, j])) >= t0
    return (cnt + 1) / (NPERM + 1)


def row(m, var, **kw):
    ci = m.conf_int().loc[var]
    return dict(**kw, coef=m.params[var], se=m.bse[var], p_hc3=m.pvalues[var], ci_lo=ci[0], ci_hi=ci[1],
                n=int(m.nobs), r2=m.rsquared)


# ------------------------------------------------------------- main regressions
NEEDS = {
    "excess_mort": "z_excess_pc_2020_21",
    "covid_deaths": "z_covid_pc_2020_21",
    "labinc_loss": "z_labinc_loss_2020",
}
rows = []
for y in YEARS:
    for sname, mask in SAMPLES.items():
        d = P[mask]
        for wname, w in [("unweighted", None), ("w_remit2019", "remit_2019_usd_m")]:
            for nname, nv in NEEDS.items():
                for spec, xs in [("bivariate", [nv]), ("controls", [nv] + CONTROLS)]:
                    m = fit(f"exc1719_{y}", xs, d, w)
                    rows.append(row(m, nv, outcome=f"exc1719_{y}", year=y, sample=sname, weight=wname,
                                    need=nname, spec=spec))
            # joint: mortality + labour income
            xs = [NEEDS["excess_mort"], NEEDS["labinc_loss"]] + CONTROLS
            m = fit(f"exc1719_{y}", xs, d, w)
            for nname in ["excess_mort", "labinc_loss"]:
                rows.append(row(m, NEEDS[nname], outcome=f"exc1719_{y}", year=y, sample=sname, weight=wname,
                                need=nname, spec="joint"))
reg = pd.DataFrame(rows)

# randomization-inference p for the headline cells (unweighted, controls)
reg["p_perm"] = np.nan
for i, r in reg.iterrows():
    if r.weight == "unweighted" and r.spec == "controls" and r["sample"] in ("all", "excl_south_ext"):
        d = P[SAMPLES[r["sample"]]]
        nv = NEEDS[r.need]
        reg.loc[i, "p_perm"] = perm_p(r.outcome, [nv] + CONTROLS, d, nv)
reg["mde_80"] = 2.8 * reg.se  # minimum detectable effect, 5% two-sided, 80% power
reg.to_csv(OUT / "regressions_need.csv", index=False)

# robustness: alternative outcome definitions for 2021, mortality need, controls, all / excl south ext
rob = []
for sname in ["all", "excl_south_ext"]:
    d = P[SAMPLES[sname]]
    nv = NEEDS["excess_mort"]
    for oname, y, xs in [
        ("exc1719 (base)", "exc1719_2021", [nv] + CONTROLS),
        ("exc1519", "exc1519_2021", [nv, "pretrend_1519", "log_hh_remit"]),
        ("exc1319", "exc1319_2021", [nv, "pretrend_1319", "log_hh_remit"]),
        ("raw dlog 2019-21 + growth_1619", "dlog_2021", [nv, "growth_1619", "log_hh_remit"]),
        ("exc1719, excess % (not per capita)", "exc1719_2021", ["z_excess_pct_2020_21"] + CONTROLS),
        ("exc1719, 2020 deaths only", "exc1719_2021", ["z_excess_pc_2020"] + CONTROLS),
        ("exc1719 + labour poverty change", "exc1719_2021", [nv, "z_d_labpov_2020"] + CONTROLS),
        ("exc1719 + log pop + median age", "exc1719_2021", [nv, "log_pop", "median_age"] + CONTROLS),
        ("exc1719 + transit (log UPM events pc 2021)", "exc1719_2021", [nv, "z_log_upm_pc_2021"] + CONTROLS),
    ]:
        m = fit(y, xs, d)
        rob.append(row(m, xs[0], spec=oname, sample=sname))
        if "transit" in oname:
            rob.append(row(m, "z_log_upm_pc_2021", spec=oname + " [transit coef]", sample=sname))
# registered COVID deaths: is the excl.-south result a need effect or an urban/certification proxy?
P["cdmx_mex"] = P.abbr.isin(["CDMX", "MEX"]).astype(float)
for sname in ["all", "excl_south_ext"]:
    d = P[SAMPLES[sname]]
    nv = NEEDS["covid_deaths"]
    for oname, xs in [
        ("covid deaths, base controls", [nv] + CONTROLS),
        ("covid deaths + log pop + median age", [nv, "log_pop", "median_age"] + CONTROLS),
        ("covid deaths + excess mortality", [nv, NEEDS["excess_mort"]] + CONTROLS),
        ("covid deaths + CDMX/MEX dummy", [nv, "cdmx_mex"] + CONTROLS),
        ("covid deaths + labour income loss", [nv, NEEDS["labinc_loss"]] + CONTROLS),
    ]:
        m = fit("exc1719_2021", xs, d)
        rob.append(row(m, nv, spec=oname, sample=sname))
rob = pd.DataFrame(rob)
loo_c = []
dS = P[SAMPLES["excl_south_ext"]]
for e in dS.index:
    m = fit("exc1719_2021", [NEEDS["covid_deaths"]] + CONTROLS, dS.drop(e))
    loo_c.append(dict(dropped=P.loc[e, "abbr"], coef=m.params[NEEDS["covid_deaths"]], se=m.bse[NEEDS["covid_deaths"]]))
loo_c = pd.DataFrame(loo_c).sort_values("coef")
loo_c.to_csv(OUT / "leave_one_out_2021_covid_excl_south.csv", index=False)
rob.to_csv(OUT / "robustness_2021.csv", index=False)

# transit variable on its own, by year
tr = []
for y in YEARS:
    m = fit(f"exc1719_{y}", ["z_log_upm_pc_2021"] + CONTROLS, P)
    tr.append(row(m, "z_log_upm_pc_2021", year=y, sample="all"))
    for g in ["south_transit_core", "south_transit_ext", "north_border"]:
        dd = P.assign(g=P[g].astype(float))
        m = fit(f"exc1719_{y}", ["g"] + CONTROLS, dd)
        tr.append(row(m, "g", year=y, sample=f"dummy:{g}"))
pd.DataFrame(tr).to_csv(OUT / "transit_checks.csv", index=False)

# ------------------------------------------------------------- leave-one-out (2021, mortality, controls)
loo = []
nv = NEEDS["excess_mort"]
for e in P.index:
    m = fit("exc1719_2021", [nv] + CONTROLS, P.drop(e))
    loo.append(dict(dropped=P.loc[e, "abbr"], coef=m.params[nv], se=m.bse[nv]))
loo = pd.DataFrame(loo)
base = fit("exc1719_2021", [nv] + CONTROLS, P)
loo["shift"] = loo.coef - base.params[nv]
loo.sort_values("shift").to_csv(OUT / "leave_one_out_2021.csv", index=False)
loo_l = []
nvl = NEEDS["labinc_loss"]
for e in P.index:
    m = fit("exc1719_2021", [nvl] + CONTROLS, P.drop(e))
    loo_l.append(dict(dropped=P.loc[e, "abbr"], coef=m.params[nvl], se=m.bse[nvl]))
loo_l = pd.DataFrame(loo_l)
base_l = fit("exc1719_2021", [nvl] + CONTROLS, P)

# ------------------------------------------------------------- timing: quarter-by-quarter coefficient
Q = pd.read_csv(OUT / "remittances_state_quarterly_dev_trend1719.csv")
Q = Q[(Q.year >= 2018) & (Q.year <= 2024)]
tq = []
for (yy, qq), g in Q.groupby(["year", "quarter"]):
    d = P.join(g.set_index("ent").dev)
    for sname in ["all", "excl_south_ext"]:
        dd = d[SAMPLES[sname]]
        for nname in ["excess_mort", "covid_deaths", "labinc_loss"]:
            m = fit("dev", [NEEDS[nname]] + CONTROLS, dd)
            tq.append(row(m, NEEDS[nname], year=yy, quarter=qq, sample=sname, need=nname))
tq = pd.DataFrame(tq)
tq["t"] = tq.year + (tq.quarter - 1) / 4
tq.to_csv(OUT / "timing_quarterly_coef.csv", index=False)

# tercile group paths (unweighted mean of state deviations)
P["need_tercile"] = pd.qcut(P.excess_pc_2020_21, 3, labels=["low", "mid", "high"])
G = Q.merge(P[["need_tercile", "south_transit_ext"]], left_on="ent", right_index=True)
gp = G.groupby(["need_tercile", "year", "quarter"], observed=True).dev.mean().reset_index()
gp["t"] = gp.year + (gp.quarter - 1) / 4
gp.to_csv(OUT / "timing_need_terciles.csv", index=False)

# ------------------------------------------------------------- Chiapas within-state: transit hubs
M = pd.read_csv(DERIVED / "remittances_municipal_quarterly.csv")
chs = M[(M.ent == 7)].groupby(["municipio", "year"]).usd_m.sum().unstack()
hubs = ["Tapachula", "Suchiate", "Tuxtla Chico", "Frontera Comalapa", "Palenque"]
c = pd.DataFrame({"usd_2019": chs[2019], "usd_2021": chs[2021], "usd_2023": chs[2023]})
c["hub"] = c.index.isin(hubs)
tot = c.sum(numeric_only=True)
chs_tab = c.groupby("hub")[["usd_2019", "usd_2021", "usd_2023"]].sum()
chs_tab.loc["share_hub"] = chs_tab.loc[True] / tot[["usd_2019", "usd_2021", "usd_2023"]]
chs_tab.to_csv(OUT / "chiapas_transit_hubs.csv")
c.sort_values("usd_2023", ascending=False).head(15).to_csv(OUT / "chiapas_top_municipalities.csv")

# ------------------------------------------------------------- ENIGH comparison (relative patterns only)
E = pd.read_csv(HERE.parent / "enigh" / "outputs" / "states.csv", dtype={"state": str})
E["ent"] = E.state.astype(int)
Ew = E.pivot(index="ent", columns="year", values=["receipt", "amount_mean", "total_quarterly_mxn_m"])
enigh = pd.DataFrame(index=Ew.index)
enigh["enigh_dlog_total_18_22"] = np.log(Ew["total_quarterly_mxn_m"][2022] / Ew["total_quarterly_mxn_m"][2018])
enigh["enigh_dlog_receipt_18_22"] = np.log(Ew["receipt"][2022] / Ew["receipt"][2018])
enigh["enigh_dlog_amount_18_22"] = np.log(Ew["amount_mean"][2022] / Ew["amount_mean"][2018])
enigh["enigh_d_receipt_18_20"] = Ew["receipt"][2020] - Ew["receipt"][2018]
EP = P.join(enigh)
# Banxico 2018->2022 (USD; ENIGH MXN real; relative pattern only)
Rq = pd.read_csv(OUT / "remittances_state_quarterly.csv")
ra = Rq.groupby(["ent", "year"]).usd_m.sum().unstack()
EP["banxico_dlog_18_22"] = np.log(ra[2022] / ra[2018])
ecomp = []
for y in ["enigh_dlog_total_18_22", "enigh_dlog_receipt_18_22", "enigh_dlog_amount_18_22"]:
    d = EP.dropna(subset=[y])
    m = fit(y, ["banxico_dlog_18_22"], d)
    ecomp.append(row(m, "banxico_dlog_18_22", outcome=y, regressor="banxico_dlog_18_22"))
    m = fit(y, [NEEDS["excess_mort"]] + CONTROLS, d)
    ecomp.append(row(m, NEEDS["excess_mort"], outcome=y, regressor="excess_mort (+controls)"))
    m = fit(y, [NEEDS["labinc_loss"]] + CONTROLS, d)
    ecomp.append(row(m, NEEDS["labinc_loss"], outcome=y, regressor="labinc_loss (+controls)"))
pd.DataFrame(ecomp).to_csv(OUT / "enigh_comparison.csv", index=False)
EP[["abbr"] + list(enigh.columns) + ["banxico_dlog_18_22"]].to_csv(OUT / "enigh_vs_banxico_states.csv")


# ------------------------------------------------------------- figures
def group_color(r):
    if r.south_transit_ext:
        return C_SOUTH
    if r.north_border:
        return C_NORTH
    return C_OTHER


def scatter(ax, x, y, xlabel, title, line=True):
    for _, r in P.iterrows():
        ax.scatter(r[x], r[y], s=34, color=group_color(r), edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(r.abbr, (r[x], r[y]), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color=MUTED)
    if line:
        for mask, ls, lab in [(SAMPLES["all"], "-", "all 32"), (SAMPLES["excl_south_ext"], "--", "excl. south")]:
            d = P[mask]
            b = np.polyfit(d[x], d[y], 1)
            xx = np.linspace(P[x].min(), P[x].max(), 10)
            ax.plot(xx, np.polyval(b, xx), ls, color=INK, lw=1.2, label=f"OLS {lab}: slope {b[0]:.3f}")
        ax.legend(fontsize=7, frameon=False, loc="best")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9, loc="left")


def legend_groups(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
         [(C_SOUTH, "southern transit (CHS TAB OAX VER GRO QROO CAM)"), (C_NORTH, "northern border"), (C_OTHER, "other")]]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5)


fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
for ax, y in zip(axes.flat, YEARS):
    scatter(ax, "excess_pc_2020_21", f"exc1719_{y}", "excess deaths 2020-21 per 1,000 (trend 2015-19)",
            f"{y}: log remittances vs own 2017-19 trend")
fig.tight_layout(rect=(0, 0.04, 1, 1))
legend_groups(fig)
fig.savefig(FIG / "F1_scatter_excess_mortality.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
scatter(axes[0], "labinc_loss_2020", "exc1719_2021", "real labour income loss, 2020H2 vs 2019H2 (log, ITLP)",
        "2021 excess vs labour income loss")
scatter(axes[1], "log_upm_pc_2021", "exc1719_2021", "log irregular-migrant events per 1,000, 2021 (UPM)",
        "2021 excess vs transit intensity")
fig.tight_layout(rect=(0, 0.07, 1, 1))
legend_groups(fig)
fig.savefig(FIG / "F2_scatter_income_transit.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, nname, lab in [(axes[0], "excess_mort", "excess mortality (1 SD)"), (axes[1], "labinc_loss", "labour income loss (1 SD)")]:
    for sname, col, off in [("all", C_OTHER, -0.03), ("excl_south_ext", C_SOUTH, 0.03)]:
        d = tq[(tq["sample"] == sname) & (tq.need == nname)]
        ax.errorbar(d.t + off, d.coef, yerr=1.96 * d.se, fmt="o", ms=3.5, color=col, lw=1, capsize=0,
                    label="all 32" if sname == "all" else "excl. southern transit")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(2020.125, color=MUTED, lw=0.6, ls=":")
    ax.set_title(f"Coefficient on {lab}, by quarter", fontsize=9, loc="left")
    ax.set_xlabel("quarter")
axes[0].set_ylabel("log deviation from own 2017-19 trend")
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "F3_timing_coefficients.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
for terc, col in [("low", C_OTHER), ("mid", "#9a9993"), ("high", C_SOUTH)]:
    d = gp[gp.need_tercile == terc]
    ax.plot(d.t, d.dev, color=col, lw=2, marker="o", ms=3, label=f"{terc} excess mortality")
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel("mean log deviation from own 2017-19 trend")
ax.set_title("Remittances by tercile of excess mortality 2020-21 (unweighted state means)", fontsize=9, loc="left")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "F4_need_terciles_quarterly.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ll = loo.sort_values("coef")
ax.errorbar(ll.coef, range(len(ll)), xerr=1.96 * ll.se, fmt="o", ms=3.5, color=C_OTHER, lw=1)
ax.set_yticks(range(len(ll)))
ax.set_yticklabels(ll.dropped, fontsize=6)
ax.axvline(base.params[nv], color=INK, lw=1, label=f"full sample {base.params[nv]:.3f}")
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlabel("2021 coefficient on excess mortality (1 SD), dropping the labelled state")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "F5_leave_one_out_2021.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
d = EP.dropna(subset=["enigh_dlog_total_18_22"])
for _, r in d.iterrows():
    ax.scatter(r.banxico_dlog_18_22, r.enigh_dlog_total_18_22, s=30, color=group_color(r), edgecolor="white", zorder=3)
    ax.annotate(r.abbr, (r.banxico_dlog_18_22, r.enigh_dlog_total_18_22), xytext=(3, 2), textcoords="offset points",
                fontsize=6.5, color=MUTED)
ax.set_xlabel("Banxico log change 2018-2022 (USD)")
ax.set_ylabel("ENIGH log change 2018-2022 (real MXN total)")
ax.set_title("State remittance growth: Banxico vs ENIGH", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig(FIG / "F6_enigh_vs_banxico.png")
plt.close(fig)

# ------------------------------------------------------------- console summary
pd.set_option("display.width", 200)
h = reg[(reg.weight == "unweighted") & (reg.spec == "controls")]
print(h.pivot_table(index=["need", "sample"], columns="year", values="coef").round(3))
print(h.pivot_table(index=["need", "sample"], columns="year", values="se").round(3))
print(h[h.year == 2021][["need", "sample", "coef", "se", "p_hc3", "p_perm", "mde_80"]].round(3).to_string())
w = reg[(reg.weight == "w_remit2019") & (reg.spec == "controls") & (reg.year == 2021)]
print(w[["need", "sample", "coef", "se", "p_hc3"]].round(3).to_string())
j = reg[(reg.weight == "unweighted") & (reg.spec == "joint") & (reg.year == 2021)]
print(j[["need", "sample", "coef", "se", "p_hc3"]].round(3).to_string())
print(rob.round(3).to_string())
print(pd.DataFrame(tr).round(3).to_string())
print("LOO range", loo.coef.min().round(3), loo.coef.max().round(3), "base", round(base.params[nv], 3))
print(loo.sort_values("shift").iloc[[0, 1, -2, -1]].round(3))
print("LOO labinc range", loo_l.coef.min().round(3), loo_l.coef.max().round(3), "base", round(base_l.params[nvl], 3))
print(tq[(tq["sample"] == "all")].pivot_table(index="t", columns="need", values="coef").round(3).T.to_string())
print(chs_tab)
print("LOO covid excl south", loo_c.iloc[[0, 1, -2, -1]].round(3).to_string())
print(tq[(tq["sample"] == "excl_south_ext")].pivot_table(index="t", columns="need", values="coef").round(3).T.to_string())
print(pd.DataFrame(ecomp).round(3).to_string())
