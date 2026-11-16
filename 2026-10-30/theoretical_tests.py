"""Theoretical tests: how the calibrated system responds to unemployment and need (13 Nov 2026).

Run: python 2026-10-30/theoretical_tests.py      (exact computation, no simulation; a few minutes)
Writes figures/theoretical-tests/T1..T5 (PNG), captions.md and theoretical_tests.csv (aggregates only).

Baseline: the 2019 calibration (calibration_simple.csv, simplified forms), the twelve months of 2019, no stimulus,
no crisis. Scenarios change one or two things and keep everything else at 2019:
    unemployment  +u percentage points for every occupation: a_j,t = max(e_o(j),t - u, 0)
    need          a shift Delta on the log-odds of every migrant's desire (as beta N*_t would do)
    kappa         0, 0.5, 1 (stepping-in treated as a dimension, user 13 Nov 2026)
With kappa = 1, effective senders are E = W + min(L, R) = min(D, A): the smaller of how many want to send (D) and
how many can (A). A household-month is 'untouched' if nobody who wants to send is out of work (L = 0),
'absorbed' if the reserve covers every lost sender (0 < L <= R), and 'short' if it does not (L > R).
T1 national response to unemployment by kappa (transfers and households receiving)
T2 the household buffer: untouched / absorbed / short as unemployment rises
T3 unemployment x need: how much of the transfers survive unemployment at each level of need
T4 who bears the loss: change in receipt by network type and size
T5 the two margins: households receiving vs transfers per receiving household, unemployment and need
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import willing_able as wa
import sample_receipt_figure as srf

mf = srf.mf
FIG = wa.HERE / 'figures' / 'theoretical-tests'
U_GRID = np.round(np.arange(0, 30.01, 2.5), 2)                        # unemployment increase, pp
D_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]                             # desire shift, log-odds
KAPPAS = [0.0, 0.5, 1.0]
K_COLORS = {0.0: mf.ORANGE, 0.5: mf.AQUA, 1.0: mf.BLUE}
K_LABEL = {0.0: 'κ = 0 (nobody steps in)', 0.5: 'κ = 0.5', 1.0: 'κ = 1 (full stepping-in)'}
CAPTIONS = {}


def save(fig, name, title, caption):
    import time
    fig.suptitle(title, x=0.01, ha='left', fontsize=12.5, fontweight='semibold', color=mf.INK)
    for attempt in range(10):
        try:
            fig.savefig(FIG / f'{name}.png', dpi=mf.DPI, bbox_inches='tight')
            break
        except OSError:
            if attempt == 9:
                raise
            time.sleep(1.0)
    plt.close(fig)
    CAPTIONS[name] = (title, caption)
    print('saved', name, flush=True)


# %% Computation
def groups(pop):
    """Network type and size of every household."""
    nc, nk = pop['n_core'], pop['n_kin']
    net = np.select([(nc > 0) & (nk == 0), nc > 0], ['Core only', 'Core + kin'], 'Kin only')
    n = nc + nk
    size = np.select([n == 1, n == 2, n <= 4], ['1', '2', '3–4'], '5+')
    return net, size


def regimes(tables):
    """(3,) shares of household-months untouched (L = 0), absorbed (0 < L <= R), short (L > R)."""
    T, H = tables.T, tables.pop['H']
    untouched = 0.0
    for nc, nk, hh, Pc, Pk in tables.blocks:
        lc, lk = wa._triples(nc)[1], wa._triples(nk)[1]
        untouched += (Pc[:, :, lc == 0].sum(-1) * Pk[:, :, lk == 0].sum(-1)).sum()
    short = tables.shortage(1.0)[2].sum()                              # P(L > R) summed over households, months
    untouched, short = untouched / (T * H), short / (T * H)
    return np.array([untouched, 1 - untouched - short, short])


def run_all(pop, p, e):
    rows, hh = [], {}
    T = e.shape[0]
    for delta_d in D_GRID:
        d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta=1.0, need_stock=np.full(T, delta_d), delta=p['delta'])
        for u in U_GRID:
            a = np.clip(e - u / 100, 0, 1)
            tables = wa.StateTables(pop, a, d)
            reg = regimes(tables)
            lost, unfilled, _ = tables.shortage(1.0).sum(1)
            for kappa in KAPPAS:
                P, Tr = wa.receipt_and_transfers(pop, tables.pairs(kappa), p['b'])
                rows.append(dict(u=u, delta_d=delta_d, kappa=kappa, receiving=P.mean(0).sum(),
                                 transfers=Tr.mean(0).sum(), able=a.mean(), wants=d.mean(),
                                 untouched=reg[0], absorbed=reg[1], short=reg[2],
                                 lost_senders=lost / T, lost_unreplaceable=unfilled / T))
                if delta_d == 0.0:
                    hh[(u, kappa)] = P.mean(0)
            print(f'delta {delta_d:+.1f} u {u:4.1f} done', flush=True)
    r = pd.DataFrame(rows)
    base = r.query('u == 0 and delta_d == 0').set_index('kappa')
    r['transfers_index'] = r.transfers / r.kappa.map(base.transfers)
    r['receiving_index'] = r.receiving / r.kappa.map(base.receiving)
    r['per_receiving'] = r.transfers / r.receiving
    return r, hh


# %% Figures
def t1_response(r):
    z = r.query('delta_d == 0')
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, col, title in [(axes[0], 'transfers_index', 'Transfers'), (axes[1], 'receiving_index', 'Households receiving')]:
        ax.plot(U_GRID, 1 - U_GRID / 100 / z.able.iloc[0], color=mf.MUTED, lw=1.2, ls=':', zorder=1)
        for k in KAPPAS:
            s = z.loc[z.kappa.eq(k)]
            ax.plot(s.u, s[col], color=K_COLORS[k], lw=2, marker='o', markersize=4.5, markeredgecolor=mf.SURFACE)
            ax.text(30.6, s[col].iloc[-1], f'{s[col].iloc[-1]:.0%}', va='center', fontsize=8, color=mf.INK2)
        ax.set_title(title, fontsize=9.5, loc='left')
        ax.set_xlabel('Increase in unemployment (percentage points)')
        ax.set_xlim(-0.5, 33)
        mf.pct_axis(ax)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('Relative to 2019')
    axes[0].text(14, 1 - 14 / 100 / z.able.iloc[0] - 0.035, 'proportional to jobs', fontsize=7.5, color=mf.MUTED, rotation=-17)
    axes[0].legend(handles=[Line2D([], [], color=K_COLORS[k], lw=2, marker='o', markersize=5, label=K_LABEL[k])
                            for k in KAPPAS], loc='lower left', frameon=False, fontsize=8)
    v10 = z.loc[z.u.eq(10)].set_index('kappa').transfers_index
    save(fig, 'T1_response_to_unemployment', 'How the system responds to unemployment',
         'Expected monthly transfers (left) and households receiving (right) under 2019 conditions when the '
         'unemployment rate of every occupation rises by the same number of percentage points, relative to 2019, '
         'for three levels of stepping-in. Dotted: a fall proportional to the number of migrants in work. '
         f'At +10 points transfers fall by {1 - v10[0.0]:.0%} without stepping-in and by {1 - v10[1.0]:.0%} with '
         'full stepping-in. Exact computation over all MMP migrant households, 2019 calibration, no need response.')


def t2_buffer(r):
    z = r.query('delta_d == 0 and kappa == 1')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    cols = [mf.BLUE, mf.ORANGE]
    names = ['Absorbed: the reserve covers every lost sender', 'Short: more lost senders than reserve']
    ax1.stackplot(z.u, z.absorbed, z.short, colors=cols, edgecolor=mf.SURFACE, linewidth=0.8)
    last = z.iloc[-1]
    ax1.text(29.5, last.absorbed / 2, f'Absorbed {last.absorbed:.1%}', ha='right', va='center', fontsize=8,
             color='white', fontweight='semibold')
    ax1.text(29.5, last.absorbed + last.short / 2, f'Short {last.short:.1%}', ha='right', va='center', fontsize=8,
             color='white', fontweight='semibold')
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, None)
    mf.pct_axis(ax1)
    ax1.set_ylabel('Share of all household-months')
    ax1.set_xlabel('Increase in unemployment (percentage points)')
    ax1.set_title('Households hit by a lost sender', fontsize=9.5, loc='left')
    ax1.grid(axis='x', visible=False)
    ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=n) for c, n in zip(cols, names)],
               loc='upper left', frameon=False, fontsize=7.5)
    rep = 1 - z.lost_unreplaceable / z.lost_senders
    ax2.plot(z.u, z.lost_senders, color=mf.INK, lw=2, marker='o', markersize=4.5, markeredgecolor=mf.SURFACE,
             label='Lost senders')
    ax2.plot(z.u, z.lost_senders - z.lost_unreplaceable, color=mf.BLUE, lw=2, marker='o', markersize=4.5,
             markeredgecolor=mf.SURFACE, label='of whom someone in the reserve can replace')
    ax2.fill_between(z.u, z.lost_senders - z.lost_unreplaceable, z.lost_senders, color=mf.ORANGE, alpha=0.25, lw=0)
    ax2.text(z.u.iloc[-1] + 0.6, z.lost_senders.iloc[-1], f'{rep.iloc[-1]:.0%}\nreplaceable', va='center', fontsize=8,
             color=mf.INK2)
    ax2.text(10.4, z.lost_senders.iloc[4] - z.lost_unreplaceable.iloc[4] - 8, f'{rep.iloc[4]:.0%} replaceable',
             va='top', ha='left', fontsize=8, color=mf.INK2)
    ax2.set_xlim(-0.5, 35)
    ax2.set_ylim(0, None)
    ax2.set_xlabel('Increase in unemployment (percentage points)')
    ax2.set_ylabel('Migrants in a month')
    ax2.set_title('Lost senders and the reserve', fontsize=9.5, loc='left')
    ax2.legend(loc='upper left', frameon=False, fontsize=8, bbox_to_anchor=(0, 0.92))
    ax2.grid(axis='x', visible=False)
    z10 = z.loc[z.u.eq(10)].iloc[0]
    save(fig, 'T2_household_buffer', 'The household buffer: who can step in',
         'Left: household-months in which a migrant who wants to send is out of work, under 2019 conditions with '
         'unemployment raised in every occupation (all other household-months are untouched). Absorbed: the household\'s '
         'reserve (migrants who can send but were not planning to) is at least as large as its lost senders. '
         'Short: it is not. With full stepping-in, effective senders equal min(D, A), the smaller of how many '
         'want to send and how many can; only short households lose transfers. '
         f'In 2019, {z.iloc[0].short:.1%} of household-months are short; at +10 points {z10.short:.1%}, while '
         f'{z10.absorbed:.1%} are absorbed. Right: expected lost senders per month and how many of them a reserve '
         'migrant of the same household could replace (orange gap: no one available).')


def t3_interaction(r):
    k1 = r.query('kappa == 1').copy()
    k1['survive'] = k1.transfers / k1.delta_d.map(k1.query('u == 0').set_index('delta_d').transfers)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    M = k1.pivot(index='delta_d', columns='u', values='survive')
    im = ax1.imshow(M.values, aspect='auto', origin='lower', cmap='Blues_r', vmin=M.values.min(), vmax=1,
                    extent=[U_GRID[0] - 1.25, U_GRID[-1] + 1.25, -0.5, len(D_GRID) - 0.5])
    for i, dd in enumerate(M.index):
        for u in (10, 20, 30):
            v = M.loc[dd, u]
            ax1.text(u, i, f'{v:.0%}', ha='center', va='center', fontsize=7.5,
                     color='white' if v < (M.values.min() + 1) / 2 else mf.INK)
    wants = k1.query('u == 0').set_index('delta_d').wants
    ax1.set_yticks(range(len(D_GRID)), [f'{dd:+.1f}  ({wants[dd]:.0%})' for dd in D_GRID])
    ax1.set_ylabel('Desire shift, log-odds (share wanting to send)')
    ax1.set_xlabel('Increase in unemployment (percentage points)')
    ax1.set_title('Share of transfers that survive unemployment, κ = 1', fontsize=9.5, loc='left')
    ax1.grid(visible=False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.05, pad=0.02)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for k in KAPPAS:
        z = r.query('kappa == @k').copy()
        b0 = z.query('u == 0').set_index('delta_d').transfers
        s = z.query('u == 10').set_index('delta_d').transfers / b0
        ax2.plot(D_GRID, 1 - s.reindex(D_GRID), color=K_COLORS[k], lw=2, marker='o', markersize=5,
                 markeredgecolor=mf.SURFACE, label=K_LABEL[k])
    ax2.set_xlabel('Desire shift (log-odds)')
    ax2.set_ylabel('Transfers lost at +10 points')
    mf.pct_axis(ax2)
    ax2.set_ylim(0, None)
    ax2.set_title('Higher need, larger losses from job loss', fontsize=9.5, loc='left')
    ax2.legend(loc='center right', frameon=False, fontsize=8)
    ax2.grid(axis='x', visible=False)
    fig.subplots_adjust(wspace=0.38)
    save(fig, 'T3_unemployment_x_need', 'Unemployment and need interact',
         'Left: transfers under unemployment as a share of transfers at the same level of need without the '
         'unemployment increase, with full stepping-in. Rows shift every migrant\'s desire by the same amount on the '
         'log-odds scale (as the crisis term β·N* does); in brackets the resulting share of migrants who want to '
         'send. Right: share of transfers lost when unemployment rises by 10 points, by level of need and '
         'stepping-in. Higher need moves households from wanting fewer transfers than they can send towards '
         'wanting more, so the reserve that absorbs job losses shrinks.')


def t4_who(pop, hh, r):
    net, size = groups(pop)
    df = pd.DataFrame(dict(net=net, size=size))
    sizes = ['1', '2', '3–4', '5+']
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, t in zip(axes, ['Core only', 'Core + kin', 'Kin only']):
        x = np.arange(len(sizes))
        for i, k in enumerate([0.0, 1.0]):
            ch = []
            for s in sizes:
                m = (df.net.eq(t) & df['size'].eq(s)).to_numpy()
                ch.append(hh[(10.0, k)][m].sum() / hh[(0.0, k)][m].sum() - 1 if m.any() else np.nan)
            ax.bar(x + (i - 0.5) * 0.36, ch, width=0.34, color=K_COLORS[k], label=K_LABEL[k])
        ns = [int((df.net.eq(t) & df['size'].eq(s)).sum()) for s in sizes]
        ax.axhline(0, color=mf.INK2, lw=0.8)
        ax.set_xticks(x, [f'{s}\n(n = {n:,})' if n else f'{s}\n(none)' for s, n in zip(sizes, ns)], fontsize=7.5)
        ax.set_xlabel('Linked migrants')
        ax.set_title(t, fontsize=9.5, loc='left')
        ax.grid(axis='x', visible=False)
        mf.pct_axis(ax)
    axes[0].set_ylabel('Change in households receiving')
    axes[2].legend(loc='lower right', frameon=False, fontsize=8)
    save(fig, 'T4_who_bears_the_loss', 'Who bears the loss',
         'Change in expected receipt when unemployment rises by 10 points in every occupation, by network type and '
         'number of linked migrants, without and with full stepping-in (2019 conditions, no need response). '
         'Households with one migrant have nobody to step in, so stepping-in cannot protect them; larger networks '
         'are protected almost fully.')


def t5_margins(r):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for k, ls in [(0.0, '--'), (1.0, '-')]:
        z = r.query('delta_d == 0 and kappa == @k')
        b = z.iloc[0]
        ax1.plot(z.u, np.log(z.receiving / b.receiving), color=mf.BLUE, lw=2, ls=ls)
        ax1.plot(z.u, np.log(z.per_receiving / b.per_receiving), color=mf.ORANGE, lw=2, ls=ls)
    z = r.query('u == 0 and kappa == 1')
    b = z.set_index('delta_d').loc[0.0]
    ax2.plot(z.delta_d, np.log(z.receiving / b.receiving), color=mf.BLUE, lw=2, marker='o', markersize=5,
             markeredgecolor=mf.SURFACE)
    ax2.plot(z.delta_d, np.log(z.per_receiving / b.per_receiving), color=mf.ORANGE, lw=2, marker='o', markersize=5,
             markeredgecolor=mf.SURFACE)
    for ax in (ax1, ax2):
        ax.axhline(0, color=mf.INK2, lw=0.8)
        ax.grid(axis='x', visible=False)
    ax1.set_xlabel('Increase in unemployment (percentage points)')
    ax1.set_ylabel('Log change relative to 2019')
    ax1.set_title('Unemployment', fontsize=9.5, loc='left')
    ax2.set_xlabel('Desire shift (log-odds), no unemployment change')
    ax2.set_title('Need', fontsize=9.5, loc='left')
    ax1.legend(handles=[Line2D([], [], color=mf.BLUE, lw=2, label='Households receiving'),
                        Line2D([], [], color=mf.ORANGE, lw=2, label='Transfers per receiving household'),
                        Line2D([], [], color=mf.INK2, lw=1.5, ls='--', label='κ = 0'),
                        Line2D([], [], color=mf.INK2, lw=1.5, label='κ = 1')], loc='lower left', frameon=False, fontsize=8)
    save(fig, 'T5_two_margins', 'Two margins: who receives and how much',
         'Log change in the number of households receiving (blue) and in transfers per receiving household '
         '(orange); the two add up to the log change in transfers. Left: unemployment raised in every occupation, '
         'without (dashed) and with (solid) full stepping-in. Right: every migrant\'s desire shifted on the '
         'log-odds scale, κ = 1, 2019 employment.')


def main():
    mf.style()
    FIG.mkdir(parents=True, exist_ok=True)
    pop = wa.load_population()
    p = wa.params_for(wa.load_calibration(), 0.0)
    e = wa.ability(pop, wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns']))
    r, hh = run_all(pop, p, e)
    r.to_csv(FIG / 'theoretical_tests.csv', index=False)
    t1_response(r)
    t2_buffer(r)
    t3_interaction(r)
    t4_who(pop, hh, r)
    t5_margins(r)
    lines = ['# Theoretical tests (13 Nov 2026)', '', 'Made by `2026-10-30/theoretical_tests.py` (2019 calibration, '
             'simplified forms; exact computation). Data: `theoretical_tests.csv`.', '']
    for name, (title, cap) in CAPTIONS.items():
        lines += [f'## {name}: {title}', '', cap, '']
    (FIG / 'captions.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
