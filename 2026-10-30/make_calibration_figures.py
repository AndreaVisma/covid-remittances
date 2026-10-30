"""Figures that give a flavour of the 2019 calibration of the willing-and-able model (30 Oct 2026).

Run: python 2026-10-30/make_calibration_figures.py      (reads outputs/calibration.csv; no estimation)
Writes figures/calibration/C1..C6 (PNG) and figures/calibration/captions.md. Aggregates only.

C1 Households by network type, including MMP households with no linked migrant (not used in the calibration):
   how many, observed share receiving (95% Wilson CI) and the model's average.
C2 Migrants across the able x willing dimensions: the four states in an average month of 2019 and in April
   2020 (CPS employment only; desire unchanged), core and kin migrants.
C3 Desire and the number of core migrants: each core migrant's monthly desire, the chance that at least one is
   an able sender, and observed vs fitted receipt by number of core migrants (with g, and the Stage 1 fit g = 0).
C4 Where households sit on the receipt curve: illustrative compositions (not MMP households), job loss and
   stepping-in (kappa = 1).
C5 All households on the receipt curve by network type, including households with no linked migrant, with
   observed receipt by score band.
C6 Fit by household need and network type: observed vs fitted receipt.
Colours: network types use the first three categorical slots (blue, orange, aqua; validated all-pairs); the
no-linked-migrant group is a hollow ring with a direct label (a fourth hue fails the all-pairs checks). States
in C2: blue, aqua, violet, orange in grid order (adjacent pairs validated; tiles are directly labelled).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.special import expit, logit

import willing_able as wa
import sample_receipt_figure as srf                 # household_receipt (exact, one household) and the style module

mf = srf.mf
sys.path.insert(0, str(wa.NCM / 'data-prep'))
import build_population as bp                        # noqa: E402

FIG = wa.HERE / 'figures' / 'calibration'
TYPES = ['Core only', 'Core + kin', 'Kin only']
BUBBLE = 0.30                                         # marker area (pt^2) per household
TYPE_COLORS = {'Core only': mf.BLUE, 'Core + kin': mf.ORANGE, 'Kin only': mf.AQUA}
NONE = 'No linked migrant'
STATES = [('Able sender', 'able, wants to send', mf.BLUE), ('Reserve', 'able, does not want to', mf.AQUA),
          ('Inactive', 'not able, does not want to', '#6250d6'), ('Lost sender', 'wants to send, not able', mf.ORANGE)]
CAPTIONS = {}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def save(fig, name, title, caption):
    fig.suptitle(title, x=0.01, ha='left', fontsize=12.5, fontweight='semibold', color=mf.INK)
    fig.savefig(FIG / f'{name}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)
    CAPTIONS[name] = (title, caption)
    print('saved', name)


# %% Data
def load():
    cal = pd.read_csv(wa.OUT / 'calibration.csv')
    p = wa.params_for(cal, 0.0)
    pop = wa.load_population()
    ok, y = wa.target(pop)
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    a = wa.ability(pop, inputs)
    d = wa.desire(pop, a.shape[0], p['a_c'], p['a_k'], p['eta'], g=p['g'])
    P, _ = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, d).pairs(0.0), p['b'])
    h = pop['h'].assign(P=P.mean(0), observed=y, ok=ok, n_core=pop['n_core'], n_kin=pop['n_kin'])
    h['type'] = np.select([h.n_core.gt(0) & h.n_kin.eq(0), h.n_core.gt(0)], ['Core only', 'Core + kin'], 'Kin only')
    raw = np.column_stack([h.resident_worker.eq(0), h.resident_older.gt(0)]).astype(float)
    known = (h.resident.gt(0) & h[['resident_worker', 'resident_older']].notna().all(axis=1)).to_numpy()
    h['no_worker'] = np.where(known, raw[:, 0], np.nan)
    h['older'] = np.where(known, raw[:, 1], np.nan)
    need_mean = raw[known].mean(0)

    # MMP households of the same surveys with no linked migrant (dropped from the model population)
    hh, pp = bp.load()
    hh = bp.composition(hh, pp)
    lk, flags, _ = bp.links(hh, pp)
    hh = hh.merge(flags, on='mmp_household_id', validate='one_to_one')
    n_links = hh.mmp_household_id.map(lk.groupby('mmp_household_id').size()).fillna(0)
    none = hh.loc[n_links.eq(0) & hh.recorded_fields_complete.astype(bool) & hh.remus.isin([1, 2])]
    none = pd.DataFrame(dict(type=NONE, observed=none.remus.eq(1).astype(float).to_numpy(), P=expit(p['b'][0]),
                             ok=True, n_core=0, n_kin=0))
    return dict(p=p, pop=pop, a=a, d=d, h=h, none=none, need_mean=need_mean)


# %% C1 households by type
def c1_types(D):
    h, none = D['h'], D['none']
    rows = []
    for t in [NONE] + TYPES[::-1]:
        g = none if t == NONE else h.loc[h.ok & h.type.eq(t)]
        n, k = len(g), g.observed.sum()
        lo, hi = wilson(k, n)
        rows.append(dict(type=t, n=n, observed=k / n, lo=lo, hi=hi, model=g.P.mean()))
    r = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
    yy = np.arange(len(r))
    cols = [TYPE_COLORS.get(t, mf.SURFACE) for t in r.type]
    ax1.barh(yy, r.n, color=cols, edgecolor=[mf.INK2 if t == NONE else c for t, c in zip(r.type, cols)],
             linewidth=[1.2 if t == NONE else 0 for t in r.type], height=0.62)
    for y_, n in zip(yy, r.n):
        ax1.text(n + 60, y_, f'{n:,}', va='center', fontsize=8.5, color=mf.INK2)
    ax1.set_yticks(yy, r.type)
    ax1.set_xlabel('Households (MMP, interviewed 2010–19)')
    ax1.set_xlim(0, r.n.max() * 1.18)
    ax1.grid(axis='y', visible=False)
    for y_, row, c in zip(yy, r.itertuples(), cols):
        ax2.plot([row.lo, row.hi], [y_, y_], color=mf.INK, lw=1.2, zorder=2)
        ax2.scatter(row.observed, y_, marker='D', s=36, color=mf.INK, zorder=4)
        ax2.scatter(row.model, y_ - 0.22, s=60, facecolor=mf.SURFACE if row.type == NONE else c,
                    edgecolor=mf.INK2 if row.type == NONE else mf.SURFACE, linewidth=1.3, zorder=4)
        ax2.text(max(row.hi, row.model) + 0.03, y_, f'{row.observed:.0%} observed · {row.model:.0%} model',
                 va='center', fontsize=8, color=mf.INK2)
    mf.pct_axis(ax2, 'x')
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('Share receiving remittances from the US')
    ax2.grid(axis='y', visible=False)
    ax2.legend(handles=[Line2D([], [], marker='D', ls='-', color=mf.INK, lw=1.2, markersize=6, label='MMP (95% CI)'),
                        Line2D([], [], marker='o', ls='', color=mf.MUTED, markeredgecolor=mf.SURFACE, markersize=8,
                               label='Model, 2019 average')], loc='upper left', frameon=False)
    save(fig, 'C1_households_by_type', 'Who is in the data, and who receives',
         'MMP origin households interviewed 2010–19, by the US migrants linked to them: core (head, spouse or '
         'child in the US) and kin (close kin and one aggregated distant-kin link). The calibration uses the '
         f'{int(h.ok.sum()):,} connected households with complete records. The {len(none):,} households with no '
         'linked migrant are not used in the calibration; the model gives them the baseline receipt '
         f'expit(b0) = {expit(D["p"]["b"][0]):.1%} a month, against {none.observed.mean():.1%} observed, an '
         'out-of-sample check. Model: average monthly receipt probability in 2019 (κ = 0 calibration).')


# %% C2 able x willing
def state_shares(a, d, mask):
    a, d = a[:, mask], d[:, mask]
    return {'Able sender': (a * d).mean(), 'Reserve': (a * (1 - d)).mean(),
            'Inactive': ((1 - a) * (1 - d)).mean(), 'Lost sender': ((1 - a) * d).mean()}


CELLS = {'Able sender': (0, 1), 'Lost sender': (1, 1), 'Reserve': (0, 0), 'Inactive': (1, 0)}   # (column, row)


def state_grid(ax, s, n):
    """2 x 2 grid: columns able / not able, rows wants / does not want to send. Square area = share."""
    colors = {k: c for k, _, c in STATES}
    for k, (cx, cy) in CELLS.items():
        ax.add_patch(Rectangle((cx + 0.03, cy + 0.03), 0.94, 0.94, facecolor=mf.HOLDOUT_WASH, edgecolor='none'))
        side = 0.66 * np.sqrt(s[k])
        x0, y0 = cx + 0.5 - side / 2, cy + 0.62 - side / 2
        ax.add_patch(Rectangle((x0, y0), side, side, facecolor=colors[k], edgecolor='none'))
        ax.text(cx + 0.5, cy + 0.13, f'{k}' + chr(10) + f'{s[k]:.0%} · {s[k] * n:,.0f}', ha='center', va='center', fontsize=8,
                color=mf.INK, fontweight='semibold' if k in ('Able sender', 'Lost sender') else 'normal')
    able = s['Able sender'] + s['Reserve']
    willing = s['Able sender'] + s['Lost sender']
    ax.text(0.5, 2.04, f'Able ({able:.0%})', ha='center', va='bottom', fontsize=8.5, color=mf.INK2)
    ax.text(1.5, 2.04, f'Not able ({1 - able:.0%})', ha='center', va='bottom', fontsize=8.5, color=mf.INK2)
    ax.text(-0.04, 1.5, f'Wants to send\n({willing:.0%})', ha='right', va='center', fontsize=8.5, color=mf.INK2)
    ax.text(-0.04, 0.5, f'Does not\n({1 - willing:.0%})', ha='right', va='center', fontsize=8.5, color=mf.INK2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.axis('off')


def c2_able_willing(D):
    pop, d = D['pop'], D['d']
    apr = wa.abm.make_inputs(pd.DatetimeIndex(['2020-04-01']), pop['rate_columns'])
    a_apr = wa.ability(pop, apr)
    d_apr = d[:1]                                                            # desire unchanged (2019 desire, beta = 0)
    groups = [('Core migrants', pop['core']), ('Kin migrants', ~pop['core'])]
    periods = [('average month of 2019', D['a'], d), ('April 2020, employment only', a_apr, d_apr)]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.4))
    table = []
    for i, (gname, mask) in enumerate(groups):
        n = int(mask.sum())
        for j, (pname, a, dd) in enumerate(periods):
            s = state_shares(a, dd, mask)
            state_grid(axes[i, j], s, n)
            axes[i, j].set_title(f'{gname} ({n:,}), {pname}', fontsize=9.5, loc='left', pad=22)
            table += [dict(group=gname, period=pname, state=k, share=v, migrants=v * n) for k, v in s.items()]
    fig.subplots_adjust(hspace=0.32, wspace=0.42, left=0.12)
    pd.DataFrame(table).to_csv(FIG / 'C2_able_willing.csv', index=False)
    save(fig, 'C2_able_willing', 'Migrants across the able and willing dimensions',
         'Each migrant is able to send in a month if employed (CPS employment rate of their occupation group) and '
         'wants to send with the calibrated desire probability; the two are independent. Square area: share of '
         'migrants in that state in a month; labels give the share and the number of migrants. An able sender sends; '
         'a lost sender wanted to send but is out of work, the gap that stepping-in can fill; the reserve (able, not '
         'planning to send) is who can step in. April 2020 uses that month\'s CPS employment with 2019 desire (no '
         'crisis response, no stimulus) to show where lost senders come from. Kin migrants rarely want to send '
         '(3% a month), so nearly all of them are in the reserve.')


# %% C3 desire and network size
def c3_desire(D):
    h, p, pop = D['h'], D['p'], D['pop']
    s1 = pd.read_csv(wa.OUT / 'stage1_by_core.csv')
    ncs = np.arange(1, 5)
    desire = expit(p['a_c'] + p['g'] * np.log(ncs))
    able = D['a'][:, pop['core']].mean()
    at_least_one = 1 - (1 - able * desire) ** ncs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.7))
    x = np.arange(4)
    ax1.bar(x, desire, width=0.55, color=mf.BLUE, label='Each core migrant wants to send')
    ax1.plot(x, at_least_one, color=mf.INK, lw=2, marker='o', markersize=7, markerfacecolor=mf.SURFACE,
             label='At least one is an able sender')
    for xi, dv, al in zip(x, desire, at_least_one):
        ax1.text(xi, 0.025, f'{dv:.0%}', ha='center', va='bottom', fontsize=8, color='white', fontweight='semibold')
        ax1.text(xi, al + 0.04, f'{al:.0%}', ha='center', va='bottom', fontsize=8, color=mf.INK2)
    ax1.set_xticks(x, ['1', '2', '3', '4'])
    ax1.set_xlabel('Core migrants in the household')
    mf.pct_axis(ax1)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel('Probability in a month')
    ax1.set_title('Desire falls with each extra core migrant…', fontsize=9.5)
    ax1.legend(loc='upper left', frameon=False, fontsize=8)
    ax1.grid(axis='x', visible=False)

    g = h.loc[h.ok].assign(nc=lambda z: z.n_core.clip(upper=4))
    obs = g.groupby('nc').agg(n=('observed', 'size'), k=('observed', 'sum'), model=('P', 'mean')).reset_index()
    obs['lo'], obs['hi'] = wilson(obs.k, obs.n)
    xs = obs.nc.to_numpy()
    ax2.vlines(xs, obs.lo, obs.hi, color=mf.INK, lw=1.2)
    ax2.scatter(xs, obs.k / obs.n, marker='D', s=36, color=mf.INK, zorder=4, label='MMP (95% CI)')
    ax2.plot(xs + 0.12, obs.model, color=mf.BLUE, lw=2, marker='o', markersize=7, markeredgecolor=mf.SURFACE,
             zorder=5, label='Calibrated model')
    ax2.plot(s1.n_core + 0.12, s1.fitted_g0, color=mf.ORANGE, lw=2, ls='--', marker='o', markersize=7,
             markeredgecolor=mf.SURFACE, zorder=3, label='Same desire for all (g = 0)')
    ax2.text(4.2, s1.fitted_g0.iloc[-1], f'{s1.fitted_g0.iloc[-1]:.0%}', va='center', fontsize=8, color=mf.INK2)
    ax2.text(4.2, obs.model.iloc[-1], f'{obs.model.iloc[-1]:.0%}', va='center', fontsize=8, color=mf.INK2)
    ax2.set_xticks(xs, ['0', '1', '2', '3', '4+'])
    ax2.set_xlabel('Core migrants in the household')
    mf.pct_axis(ax2)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(-0.4, 4.6)
    ax2.set_title('…which keeps receipt in line with the data', fontsize=9.5)
    ax2.legend(loc='lower right', frameon=False, fontsize=8)
    ax2.grid(axis='x', visible=False)
    save(fig, 'C3_desire_network_size', 'Desire and the number of core migrants',
         f'Left: monthly probability that a core migrant wants to send, expit(a_c + g·log N), with a_c = '
         f'{p["a_c"]:.2f} and g = {p["g"]:.2f} (household of average need), and the chance that at least one of '
         f'them is an able sender (employment {able:.0%}). Right: share of households receiving, MMP 2019 (95% '
         'Wilson CI), the calibrated model, and the Stage 1 fit with the same desire for every core migrant '
         '(g = 0). With the same desire for all, receipt rises far too steeply with the number of core migrants; '
         'a desire that falls with each extra core migrant (one reading: a division of labour among them) fits '
         'every group.')


# %% C4 receipt curve examples
def c4_examples(D):
    p, need_mean = D['p'], D['need_mean']
    examples = [
        ('Two cousins in the US;\nworkers at home', 'Kin only', (0, 2, 0, 0), (-2.2, 0.27), None),
        ('Two cousins; no worker,\na grandparent at home', 'Kin only', (0, 2, 1, 1), (-1.2, 0.08), None),
        ('One son in the US; workers at home', 'Core only', (1, 0, 0, 0), (0.55, 0.30), ('core', 'he loses his job')),
        ('Three children and two cousins;\nno worker, a grandparent', 'Core + kin', (3, 2, 1, 1), (-1.75, 0.92),
         ('core', 'one child loses their job')),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.linspace(-5.2, 3.5, 300)
    ax.plot(x, expit(x), color=mf.GRAY_LIGHT, lw=2.5, zorder=1)
    mf.pct_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Probability of receiving in a month')
    for label, net, (c, k, nw, ol), pos, loss in examples:
        P0 = srf.household_receipt(p, c, k, nw, ol, need_mean)
        s0 = logit(P0)
        text = f'{label}: {P0:.0%}'
        if loss is not None:
            who, what = loss
            P1 = srf.household_receipt(p, c, k, nw, ol, need_mean, lost=who, kappa=0.0)
            P2 = srf.household_receipt(p, c, k, nw, ol, need_mean, lost=who, kappa=1.0)
            ax.annotate('', (logit(P1), P1), (s0, P0), arrowprops=dict(arrowstyle='-|>', color=mf.RED, lw=1.6,
                        shrinkA=6, shrinkB=5, connectionstyle='arc3,rad=0.25'))
            ax.scatter(logit(P1), P1, s=40, facecolor=mf.SURFACE, edgecolor=mf.RED, linewidth=1.5, zorder=5)
            if abs(P2 - P1) > 0.005:
                ax.annotate('', (logit(P2), P2), (logit(P1), P1), arrowprops=dict(arrowstyle='-|>', color=mf.K4,
                            lw=1.4, ls='--', shrinkA=5, shrinkB=5, connectionstyle='arc3,rad=-0.9'))
                text += f'\n{what}: {P1:.0%}; someone steps in: {P2:.0%}'
            else:
                text += f'\n{what}: {P1:.0%}; nobody can step in'
        mf.ring_dot(ax, s0, P0, TYPE_COLORS[net], size=70, zorder=6)
        ax.annotate(text, (s0, P0), xytext=pos, textcoords='data', fontsize=8, color=mf.INK2, va='center',
                    arrowprops=dict(arrowstyle='-', color=mf.MUTED, lw=0.7, shrinkA=2, shrinkB=6))
    ax.set_xlabel('Household score (log-odds of expected receipt)')
    ax.set_xlim(x[0], x[-1])
    handles = [Line2D([], [], marker='o', ls='', color=c, markeredgecolor=mf.SURFACE, markersize=8, label=n)
               for n, c in TYPE_COLORS.items()]
    handles += [Line2D([], [], color=mf.RED, lw=1.6, marker='>', markersize=5, label='One migrant loses their job'),
                Line2D([], [], color=mf.K4, lw=1.4, ls='--', marker='>', markersize=5, label='Someone steps in (κ = 1)')]
    ax.legend(handles=handles, loc='upper left', frameon=False)
    b0, b_any, b_c, b_k0 = p['b']
    ax.text(0.0, -0.17,
            f"receipt given able senders = expit({b0:.2f} + {b_any:.2f}·[any able core sender] + {b_c:.2f}·log(1 + able "
            f"core senders) + {b_k0:.2f}·[no core migrant]·log(1 + able kin senders))\n"
            f"desire: core expit({p['a_c']:.2f} {p['g']:+.2f}·log(core migrants) + need), kin expit({p['a_k']:.2f} + "
            f"need);  need = {p['eta'][0]:.2f}·no worker + {p['eta'][1]:.2f}·older (centred)",
            transform=ax.transAxes, fontsize=7, color=mf.MUTED, va='top')
    save(fig, 'C4_receipt_curve_examples', 'Where households sit on the receipt curve',
         'Illustrative compositions (not MMP households), all migrants employed, 2019 calibration (κ = 0). A '
         'household\'s score is the log-odds of its expected monthly receipt, averaged over which migrants are '
         'willing. Red: one migrant loses their job and nobody steps in. Dashed blue: someone in the household\'s '
         'reserve (able but not planning to send) steps in, κ = 1. An only migrant cannot be replaced; in a large '
         'network, the loss is fully restored.')


# %% C5 all households on the curve
def c5_households(D):
    h, none, p = D['h'], D['none'], D['p']
    g = h.loc[h.ok, ['type', 'P', 'observed']]
    allh = pd.concat([g, none[['type', 'P', 'observed']]], ignore_index=True)
    allh['score'] = logit(allh.P)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.15]))
    x = np.linspace(-5.6, 2.6, 300)
    ax.plot(x, expit(x), color=mf.GRAY_LIGHT, lw=2.5, zorder=1)
    for t in ['Kin only', 'Core + kin', 'Core only']:
        z = allh.loc[allh.type.eq(t)]
        b = z.groupby(z.score.round(1)).size()
        ax.scatter(b.index, expit(b.index), s=4 + b.values * BUBBLE, color=TYPE_COLORS[t], alpha=0.85,
                   edgecolor=mf.SURFACE, linewidth=0.6, zorder=3)
    s_none = logit(expit(p['b'][0]))
    ax.scatter([s_none], [expit(s_none)], s=4 + len(none) * BUBBLE, facecolor='none', edgecolor=mf.INK2, linewidth=1.3,
               zorder=3, clip_on=False)
    ax.annotate(f'{NONE}\n{len(none):,} households, all at the baseline', (s_none, expit(s_none)), xytext=(-5.3, 0.30),
                fontsize=8, color=mf.INK2, arrowprops=dict(arrowstyle='-', color=mf.MUTED, lw=0.7, shrinkB=22))
    # observed receipt by score band (bands with >= 30 households); the no-linked group is its own band
    edges = np.r_[-np.inf, -3.0, -2.5, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, np.inf]
    conn = allh.loc[allh.type.ne(NONE)].copy()
    conn['band'] = pd.cut(conn.score, edges)
    bands = conn.groupby('band', observed=True).agg(n=('observed', 'size'), k=('observed', 'sum'),
                                                    s=('score', 'mean')).reset_index()
    bands = pd.concat([bands.loc[bands.n >= 30],
                       pd.DataFrame([dict(n=len(none), k=none.observed.sum(), s=s_none)])], ignore_index=True)
    lo, hi = wilson(bands.k, bands.n)
    ax.vlines(bands.s, lo, hi, color=mf.INK, lw=1.2, zorder=6)
    ax.scatter(bands.s, bands.k / bands.n, marker='D', s=36, color=mf.INK, edgecolor=mf.SURFACE, zorder=7)
    mf.pct_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Probability of receiving in a month')
    counts = allh.type.value_counts()
    handles = [Line2D([], [], marker='o', ls='', color=TYPE_COLORS[t], markeredgecolor=mf.SURFACE, markersize=9,
                      label=f'{t} ({counts[t]:,})') for t in TYPES]
    handles += [Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor=mf.INK2, markersize=9,
                       label=f'{NONE} ({len(none):,})'),
                Line2D([], [], marker='D', ls='-', color=mf.INK, lw=1.2, markersize=6, label='MMP receipt by score band (95% CI)')]
    ax.legend(handles=handles, loc='upper left', frameon=False, title='Bubble area = households', title_fontsize=8)
    # histogram stacked by type
    bins = np.arange(-5.6, 2.61, 0.2)
    bottom = np.zeros(len(bins) - 1)
    for t, col in [(NONE, None)] + [(t, TYPE_COLORS[t]) for t in TYPES]:
        v, _ = np.histogram(allh.loc[allh.type.eq(t), 'score'], bins)
        axh.bar(bins[:-1], v, width=0.18, align='edge', bottom=bottom, color=col if col else mf.SURFACE,
                edgecolor=mf.INK2 if col is None else mf.SURFACE, linewidth=1.0 if col is None else 0.6)
        bottom += v
    axh.text(s_none + 0.32, len(none), f'{len(none):,}', va='top', fontsize=8, color=mf.INK2)
    axh.set_ylabel('Households')
    axh.set_xlabel('Household score in 2019 (log-odds of expected receipt)')
    axh.set_xlim(x[0], x[-1])
    axh.grid(axis='x', visible=False)
    fig.subplots_adjust(hspace=0.08)
    save(fig, 'C5_households_on_curve', 'All households on the receipt curve',
         'Every MMP household interviewed 2010–19 with complete records, at its expected 2019 receipt '
         '(calibrated model, κ = 0; score = log-odds), bubbles sized by the number of households with that '
         'score. Households with no linked migrant all sit at the baseline expit(b0) and are not used in the '
         'calibration. Black diamonds: observed MMP receipt by score band (bands with at least 30 households). '
         'Bottom: number of households by score. The model separates households with a core migrant (upper half '
         'of the curve) from kin-only and unlinked households (bottom).')


# %% C6 fit by need group
def c6_need(D):
    h = D['h'].loc[D['h'].ok & D['h'].no_worker.notna()].copy()
    h['net'] = np.where(h.type.eq('Kin only'), 'Kin only', 'With a core migrant')
    h['need'] = np.select([h.no_worker.eq(0) & h.older.eq(0), h.no_worker.eq(1) & h.older.eq(0),
                           h.no_worker.eq(0) & h.older.eq(1)],
                          ['Worker at home, no older resident', 'No worker, no older resident',
                           'Worker at home, older resident'], 'No worker, older resident')
    order = ['Worker at home, no older resident', 'No worker, no older resident', 'Worker at home, older resident',
             'No worker, older resident']
    r = h.groupby(['net', 'need']).agg(n=('observed', 'size'), k=('observed', 'sum'), model=('P', 'mean')).reset_index()
    r['lo'], r['hi'] = wilson(r.k, r.n)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, (net, col) in zip(axes, [('With a core migrant', mf.BLUE), ('Kin only', mf.AQUA)]):
        z = r.loc[r.net.eq(net)].set_index('need').reindex(order)
        yy = np.arange(len(order))[::-1]
        ax.hlines(yy, z.lo, z.hi, color=mf.INK, lw=1.2)
        ax.scatter(z.k / z.n, yy, marker='D', s=36, color=mf.INK, zorder=4)
        ax.scatter(z.model, yy - 0.2, s=60, color=col, edgecolor=mf.SURFACE, zorder=5)
        for y_, row in zip(yy, z.itertuples()):
            ax.text(1.0, y_, f'n = {row.n:,}', va='center', ha='right', fontsize=7.5, color=mf.MUTED)
        ax.set_title(net, fontsize=9.5, loc='left')
        mf.pct_axis(ax, 'x')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('Share receiving')
    axes[0].set_yticks(np.arange(len(order))[::-1], order)
    axes[1].legend(handles=[Line2D([], [], marker='D', ls='-', color=mf.INK, lw=1.2, markersize=6, label='MMP (95% CI)'),
                            Line2D([], [], marker='o', ls='', color=mf.MUTED, markeredgecolor=mf.SURFACE, markersize=8,
                                   label='Model')], loc='center right', frameon=False)
    fig.subplots_adjust(top=0.80)
    save(fig, 'C6_fit_by_need', 'How well the calibration fits, by household need',
         'Share of households receiving, MMP 2019 (95% Wilson CI) and the calibrated model, by whether anyone at '
         'home works and whether a resident is 65 or older, for households with a core migrant and kin-only '
         'households (households with a complete roster). Need raises desire through two parameters (η_w, η_o) '
         'shared by all households. The clear miss: kin-only households with an older resident receive more '
         'often than the model gives them.')


def write_captions():
    lines = ['# Calibration figures (2019, willing-and-able model)', '',
             'Made by `2026-10-30/make_calibration_figures.py` from `outputs/calibration.csv` (κ = 0 calibration).', '']
    for name, (title, cap) in CAPTIONS.items():
        lines += [f'## {name}: {title}', '', cap, '']
    (FIG / 'captions.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    mf.style()
    FIG.mkdir(parents=True, exist_ok=True)
    D = load()
    for f in (c1_types, c2_able_willing, c3_desire, c4_examples, c5_households, c6_need):
        f(D)
    write_captions()


if __name__ == '__main__':
    main()
