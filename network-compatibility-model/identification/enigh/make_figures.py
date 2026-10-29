"""Figures and a USD view from the aggregate ENIGH tables in outputs/.

Run: C:/miniforge3/python.exe -I make_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "outputs"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#6b6b6b", "#e6e6e3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
YEARS = [2018, 2020, 2022, 2024]

lv = pd.read_csv(OUT / "levels_by_cell.csv")
dd = pd.read_csv(OUT / "diff_in_changes.csv")
defl = pd.read_csv(OUT / "deflator.csv").set_index("year")["factor_to_aug2024"]

# ---- USD view of the national mean amount (nominal pesos / mean FX over Feb-Oct reference window)
fx = pd.read_csv(ROOT / "data" / "DEXMXUS.csv", parse_dates=["observation_date"])
fx["DEXMXUS"] = pd.to_numeric(fx["DEXMXUS"], errors="coerce")
nat = lv[lv.cell == "All"].set_index("year")
rows = []
for y in YEARS:
    f = fx[(fx.observation_date >= f"{y}-02-01") & (fx.observation_date < f"{y}-11-01")]["DEXMXUS"].mean()
    nominal = nat.loc[y, "amount_mean"] / defl[y]   # pesos of August of survey year
    rows.append(dict(year=y, fx_mxn_per_usd_feb_oct=f, amount_mean_nominal_mxn=nominal,
                     amount_mean_usd_quarterly=nominal / f,
                     amount_mean_usd_quarterly_se=nat.loc[y, "amount_mean_se"] / defl[y] / f))
pd.DataFrame(rows).to_csv(OUT / "national_usd.csv", index=False, float_format="%.6g")

# ---- Fig 1: national receipt and real amount
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].errorbar(YEARS, 100 * nat.receipt, yerr=196 * nat.receipt_se, color=BLUE, lw=2,
               marker="o", ms=6, capsize=3)
ax[0].set_title("Households receiving remittances (%)", loc="left", color=INK)
ax[0].set_ylim(3.5, 5.2)
ax[1].errorbar(YEARS, nat.amount_mean / 1e3, yerr=1.96 * nat.amount_mean_se / 1e3, color=BLUE,
               lw=2, marker="o", ms=6, capsize=3, label="Mean")
ax[1].plot(YEARS, nat.amount_median / 1e3, color=ORANGE, lw=2, marker="s", ms=6, label="Median")
ax[1].set_title("Quarterly amount per recipient household\n(thousand MXN, Aug 2024 prices)",
                loc="left", color=INK)
ax[1].legend(frameon=False)
for a in ax:
    a.set_xticks(YEARS)
fig.text(0.01, -0.02, "ENIGH concentradohogar; 95% design-based CIs. Source: INEGI.", color=MUTED, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "F1_national.png", dpi=200, bbox_inches="tight")

# ---- Fig 2: receipt and amount by need cell pair
pairs = [("no_worker", "worker"), ("old", "no_old"), ("rural", "urban"),
         ("trad", "nontrad"), ("lab_q1", "lab_q25")]
fig, ax = plt.subplots(2, len(pairs), figsize=(12, 5), sharex=True)
for j, (a_, b_) in enumerate(pairs):
    for c, col in [(a_, ORANGE), (b_, BLUE)]:
        s = lv[lv.cell == c].set_index("year")
        ax[0, j].errorbar(YEARS, 100 * s.receipt, yerr=196 * s.receipt_se, color=col, lw=2,
                          marker="o", ms=5, capsize=2, label=s.label.iloc[0])
        ax[1, j].errorbar(YEARS, s.amount_mean / 1e3, yerr=1.96 * s.amount_mean_se / 1e3,
                          color=col, lw=2, marker="o", ms=5, capsize=2)
    ax[0, j].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0, j].set_ylim(0, ax[0, j].get_ylim()[1] * 1.35)
    ax[1, j].set_ylim(0, None)
    ax[1, j].set_xticks(YEARS)
ax[0, 0].set_ylabel("Receipt rate (%)")
ax[1, 0].set_ylabel("Mean amount | recipient\n(thousand MXN, Aug 2024)")
fig.tight_layout()
fig.savefig(FIG / "F2_cells.png", dpi=200, bbox_inches="tight")

# ---- Fig 3: difference-in-changes (log receipt, log amount), high-need minus comparison
lab = {"no_worker": "No worker vs some", "old": "65+ resident vs none", "rural": "Rural vs urban",
       "trad": "Traditional vs other states", "lab_q1": "Bottom 20% labour inc. vs rest",
       "need_any": "No worker or 65+ vs rest", "head_f": "Female vs male head"}
pers = [(2018, 2020), (2020, 2022), (2022, 2024)]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
cols = {2018: MUTED, 2020: ORANGE, 2022: BLUE}
for k, (m, title) in enumerate([("lp", "Extensive: change in log receipt rate"),
                                ("la", "Intensive: change in log amount | recipient")]):
    for i, a_ in enumerate(lab):
        for o, (t0, t1) in enumerate(pers):
            r = dd[(dd.cell_a == a_) & (dd.measure == m) & (dd.from_year == t0) & (dd.to_year == t1)]
            if r.empty:
                continue
            y = i + (o - 1) * 0.25
            ax[k].errorbar(r.did, y, xerr=1.96 * r.se, fmt="o", color=cols[t0], ms=5, capsize=2,
                           label=f"{t0}-{t1}" if i == 0 else None)
    ax[k].axvline(0, color=INK, lw=0.8)
    ax[k].set_title(title, loc="left", color=INK)
    ax[k].set_yticks(range(len(lab)))
    ax[k].set_yticklabels(list(lab.values()))
    ax[k].invert_yaxis()
    ax[k].grid(axis="y", visible=False)
fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center",
           ncol=3, bbox_to_anchor=(0.6, -0.06))
fig.text(0.01, -0.10, "Difference in changes (high-need cell minus comparison), log points, 95% CIs.",
         color=MUTED, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "F3_diff_in_changes.png", dpi=200, bbox_inches="tight")
print(pd.read_csv(OUT / "national_usd.csv").round(1).to_string())
