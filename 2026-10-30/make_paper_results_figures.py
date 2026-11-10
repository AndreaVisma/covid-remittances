"""Results figures for the paper draft (3 Nov 2026). No estimation: reads existing outputs.

Run: python 2026-10-30/make_paper_results_figures.py  -> figures/paper/R1_national_fit.png, R2_need_series.png,
R3_household_losses.png
R1 national index vs Banco de Mexico (job loss only; stepping-in kappa 0 / 0.5 / 1 with fitted need response) and the
   January/March 2021 stimulus spikes by kappa (outputs/shocks).
R2 need measured as cumulative excess deaths, monthly excess deaths, and a placebo ramp, kappa = 1, beta fitted
   (outputs/shocks-quick-base, shocks-need-E0, shocks-need-R1).
R3 household layer: months with receipt among receiving households (ENIGH 2018 vs model without and with the habit)
   and how concentrated losses are (outputs/household-layer).
Colours: kappa as an ordinal blue ramp (project palette K0/K2/K4); need series and model variants use the first three
categorical slots (blue, orange, aqua; validated all-pairs); data in ink.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import willing_able as wa
import sample_receipt_figure as srf

mf = srf.mf
FIG = wa.HERE / 'figures' / 'paper'
O = wa.OUT
QS = [f'{y}Q{q}' for y in (2019, 2020, 2021, 2022) for q in range(1, 5)]
KAPPA_COL = {0.0: mf.K0, 0.5: mf.K2, 1.0: mf.K4}


def qaxis(ax):
    x = np.arange(len(QS))
    ax.axvspan(QS.index('2022Q1') - 0.5, len(QS) - 0.5, color=mf.HOLDOUT_WASH, zorder=0)
    ax.text(QS.index('2022Q1') - 0.3, 0.875, '2022: not fitted', fontsize=7.5, color=mf.MUTED, va='bottom')
    ax.set_xticks(x[::2], [q.replace('Q', ' Q') for q in QS[::2]], fontsize=7.5)
    ax.set_xlim(-0.5, len(QS) - 0.5)
    ax.axhline(1, color=mf.AXIS, lw=0.8, zorder=1)
    ax.set_ylabel('Transactions relative to no-COVID path')
    ax.grid(axis='x', visible=False)
    return x


def r1():
    q = pd.read_csv(O / 'shocks/quarterly_paths.csv')
    f = pd.read_csv(O / 'shocks/beta_fit.csv')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1.7, 1]))
    x = qaxis(ax)
    bx = q[(q.kappa == 1.0) & (q.fitted == True)].set_index('quarter').index_banxico.reindex(QS)
    jl = q[(q.kappa == 0.0) & (q.beta == 0.0) & (q.fitted != True)].set_index('quarter').index_transfers.reindex(QS)
    ax.plot(x, jl, color=mf.MUTED, lw=1.6, ls='--', zorder=3)
    ax.text(x[5] + 0.2, jl.iloc[5] - 0.004, 'job loss only', fontsize=8, color=mf.INK2, va='top')
    for k, c in KAPPA_COL.items():
        y = q[(q.kappa == k) & (q.fitted == True)].set_index('quarter').index_transfers.reindex(QS)
        ax.plot(x, y, color=c, lw=2, zorder=4, label=f'model, stepping-in κ = {k:g}')
    ax.plot(x, bx, color=mf.INK, lw=1.4, marker='D', markersize=4.5, zorder=5, label='Banco de México')
    ax.set_ylim(0.86, 1.16)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    ax.set_title('a  National transactions', fontsize=10, loc='left')
    ks = f.kappa.to_numpy()
    for i, (m, lab) in enumerate((('spike_2021_01', 'January 2021'), ('spike_2021_03', 'March 2021'))):
        ax2.plot(ks + (i - 0.5) * 0.04, 100 * f[m], color=[mf.BLUE, mf.ORANGE][i], lw=1.6, marker='o', markersize=7,
                 markeredgecolor=mf.SURFACE, label=f'model, {lab}')
    import json
    s = json.loads((O / 'shocks/run_settings.json').read_text())['spike_banxico']
    for i, (m, v) in enumerate(s.items()):
        ax2.axhline(100 * v, color=[mf.BLUE, mf.ORANGE][i], lw=1, ls=':')
    ax2.text(-0.03, 100 * s['2021-01'] + 0.15, 'data, January', fontsize=7.5, color=mf.INK2, va='bottom', ha='left')
    ax2.text(-0.03, 100 * s['2021-03'] - 0.15, 'data, March', fontsize=7.5, color=mf.INK2, va='top', ha='left')
    ax2.set_xlabel('Stepping-in κ')
    ax2.set_ylabel('Stimulus spike (log points)')
    ax2.set_xticks([0, 0.5, 0.8, 1])
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    ax2.set_title('b  Stimulus spikes', fontsize=10, loc='left')
    ax2.grid(axis='x', visible=False)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(FIG / 'R1_national_fit.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


def r2():
    series = [('shocks-quick-base', 'cumulative excess deaths', mf.BLUE), ('shocks-need-E0', 'monthly excess deaths', mf.ORANGE),
              ('shocks-need-R1', 'placebo: steady ramp', mf.AQUA)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = qaxis(ax)
    for d, lab, c in series:
        q = pd.read_csv(O / d / 'quarterly_paths.csv')
        y = q[(q.kappa == 1.0) & (q.fitted == True)].set_index('quarter')
        ax.plot(x, y.index_transfers.reindex(QS), color=c, lw=2, label=f'need = {lab}')
        bx = y.index_banxico.reindex(QS)
    ax.plot(x, bx, color=mf.INK, lw=1.4, marker='D', markersize=4.5, zorder=5, label='Banco de México')
    ax.set_ylim(0.86, 1.16)
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.savefig(FIG / 'R2_need_series.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


def r3():
    d = pd.read_csv(O / 'household-layer/months_distribution.csv')
    w = pd.read_csv(O / 'household-layer/who_gains_loses.csv')
    enigh = [0.203, 0.117, 0.104, 0.047, 0.034, 0.496]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(1, 7)
    rows = [(enigh, 'ENIGH 2018', mf.INK), ]
    for rho, lab, c in ((0.0, 'model, months independent', mf.ORANGE), (d.rho.max(), 'model, with habit', mf.BLUE)):
        r = d[(np.isclose(d.rho, rho)) & (d.year == 2018)].iloc[0]
        rows.append(([r[f'share_{k}'] for k in range(1, 7)], lab, c))
    for i, (v, lab, c) in enumerate(rows):
        ax.bar(x + (i - 1) * 0.27, np.array(v) * 100, width=0.25, color=c, label=lab, edgecolor=mf.SURFACE, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xlabel('Months with money from abroad, out of six')
    ax.set_ylabel('Share of receiving households (%)')
    ax.legend(loc='upper center', frameon=False, fontsize=8)
    ax.set_title('a  How regularly households receive (2018)', fontsize=10, loc='left')
    ax.grid(axis='x', visible=False)
    nets = ['Core only', 'Core + kin', 'Kin only', 'all']
    labels = ['Close family\nonly', 'Close family\nand kin', 'Kin only', 'All']
    xx = np.arange(len(nets))
    for i, (rho, lab, c) in enumerate(((0.0, 'months independent', mf.ORANGE), (w.rho.max(), 'with habit', mf.BLUE))):
        v = w[np.isclose(w.rho, rho)].set_index('network').reindex(nets).share_lost_months_in_3plus * 100
        ax2.bar(xx + (i - 0.5) * 0.36, v, width=0.34, color=c, label=f'model, {lab}', edgecolor=mf.SURFACE, linewidth=0.6)
        for xi, vi in zip(xx + (i - 0.5) * 0.36, v):
            ax2.text(xi, vi + 1, f'{vi:.0f}%', ha='center', va='bottom', fontsize=7.5, color=mf.INK2)
    ax2.set_xticks(xx, labels, fontsize=8)
    ax2.set_ylabel('Share of lost months borne by\nhouseholds losing 3+ months (%)')
    ax2.set_ylim(0, 70)
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    ax2.set_title('b  How concentrated the losses are (Apr 2020 – Dec 2021)', fontsize=10, loc='left')
    ax2.grid(axis='x', visible=False)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(FIG / 'R3_household_losses.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    mf.style()
    FIG.mkdir(parents=True, exist_ok=True)
    r1(), r2(), r3()
    print('saved R1-R3 in', FIG)


if __name__ == '__main__':
    main()
