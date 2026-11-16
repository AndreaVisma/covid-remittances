"""How migrants moved between the four states of the willing-and-able model during COVID (11 Nov 2026).

Run: python 2026-10-30/make_transition_figures.py [--seeds 200] [--plot-only]
    without --plot-only: simulates and writes the aggregate tables to outputs/transitions/, then plots;
    with --plot-only: plots from the tables already there.
Writes figures/paper/R4_state_shares.png, R4_cohorts.png, R4_transitions.png (PNG, as R1-R3) and outputs/transitions/*.csv. Aggregates only (no household records).

Base settings: kappa and beta from outputs/shocks/beta_fit.csv (best fit: kappa = 1, beta = 0.153), base stimulus
(Jan and Mar 2021, pi = 1), need = cumulative excess deaths, household habit rho from
outputs/household-layer/run_settings.json (0.928), latent AR(1) employment per migrant.

States of migrant j in month t: able sender (able, wants to send), lost sender (wants to send, not able), reserve
(able, does not want to), inactive (neither). A reserve migrant who sends in place of a lost sender of the same
household "steps in" (a replacement); in the figures the reserve is split into "steps in" and the rest.

1. State shares by month (exact): shares of the four states from the marginals a_j,t d_j,t etc. (no simulation;
   the habit keeps each migrant's monthly mean desire at d_j,t, so the exact shares hold with rho = 0.928). The
   replacements: total expected number exact (StateTables.shortage: drawn - unfilled); the split by core / kin
   migrants who step in comes from the simulation (it depends on who steps in, which the model leaves open).
2. Transitions (simulation, run_households.simulate with the same random-number design, plus a seventh stream that
   picks who steps in): paired COVID and no-COVID runs, seeds 9000 + s as in household_layer.py.

Modelling choice (identity of the replacement; it does not change any count, receipt or transfer): the model fixes
how many reserve migrants of a household step in each month (min(X_h, R_h), X_h ~ Bin(lost senders, kappa)), not
which ones. Who steps in: last month's replacements first if they are still in the reserve (a replacement keeps
the role while it is needed and they remain able and not otherwise wanting to send), then other reserve migrants of
the household at random (core and kin alike).

Colours: five states in stack order steps in / able sender / lost sender / reserve / inactive = magenta, blue,
orange, aqua, violet (C2 colours for the four states; adjacent pairs validated with the dataviz validator: worst
CVD dE 9.2, normal-vision 27.5; magenta and aqua under 3:1 contrast, so every band is directly labelled and the
tables are written). Transition matrices: one-hue blue sequential ramp.
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm

import willing_able as wa
import run_household_model as hm
import sample_receipt_figure as srf

mf = srf.mf
FIG = wa.HERE / 'figures' / 'paper'
OUT = wa.OUT / 'transitions'
DATES = hm.DATES                                            # 2019-01 .. 2022-12
STATES = ['Steps in', 'Able sender', 'Lost sender', 'Reserve', 'Inactive']   # codes 0..4 (stack order)
STEP, ABLE, LOST, RES, INACT = range(5)
COLORS = dict(zip(STATES, ['#e87ba4', mf.BLUE, mf.ORANGE, mf.AQUA, '#6250d6']))
GROUPS = ['core', 'kin']
ORIGIN = '2020-02'
DESTS = ['2020-04', '2020-05', '2020-12', '2021-01', '2021-12']
SEQ = ['#f0efec', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def mi(month):
    return int(np.flatnonzero(DATES.strftime('%Y-%m') == month)[0])


# %% Simulation
def simulate_states(pop, inputs, p, kappa, beta, need, pi, stim, seed, rho):
    """run_households.simulate with the same random numbers (streams 0-5), returning (T, M) state codes and the
    (T, H) receipt (to check identity with simulate). Stream 6 picks who steps in."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
    phi = wa.abm.employment_phi()
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['delta'])
    nc, nk = pop['n_core'], pop['n_kin']
    rec, state = np.zeros((T, H), bool), np.zeros((T, M), np.int8)
    z = streams[0].standard_normal(M)
    hsum = lambda v: np.bincount(hid, weights=v, minlength=H).astype(int)
    prev_step = np.zeros(M, bool)
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        employed = norm.cdf(z) < inputs.rates[t, pop['job_index']]
        checked = streams[2].random(M) < pi * stim[t]
        able = (employed | checked) & pop['eligible']
        u_d = streams[3].random(M)
        if rho and t > 0:
            p01 = d[t] - rho * d[t - 1]
            assert p01.min() >= -1e-12 and p01.max() + rho <= 1 + 1e-12, 'desire moves too fast for this rho'
            wants = u_d < np.where(wants, p01 + rho, p01)
        else:
            wants = u_d < d[t]
        lost = wants & ~able
        u = streams[4].random(M)
        cand = lost * ((u < kappa).astype(int) if kappa <= 1 else 1 + (u < kappa - 1))
        Wc, Wk = hsum(wants & able & core), hsum(wants & able & ~core)
        Xc, Xk, R = hsum(cand * core), hsum(cand * ~core), hsum(able & ~wants)
        xc = np.minimum(Xc, R)
        xk = np.minimum(Xk, R - xc)
        Ec, Ek = wa.relabel(Wc + xc, Wk + xk, nc, nk)
        s = wa.receipt_prob(p['b'], Ec, Ek)
        rec[t] = streams[5].random(H) < s
        # who steps in: xc + xk reserve migrants of the household; last month's replacements first, then at random
        reserve = np.flatnonzero(able & ~wants)
        key = streams[6].random(M)
        order = reserve[np.lexsort((key[reserve], ~prev_step[reserve], hid[reserve]))]
        h_sorted = hid[order]
        first = np.searchsorted(h_sorted, h_sorted, side='left')
        rank = np.arange(len(order)) - first
        steps = np.zeros(M, bool)
        steps[order[rank < (xc + xk)[h_sorted]]] = True
        st = np.full(M, INACT, np.int8)
        st[wants & able] = ABLE
        st[lost] = LOST
        st[able & ~wants] = RES
        st[steps] = STEP
        state[t] = st
        prev_step = steps
    return state, rec


def spells(is_rep):
    """(T, M) length of the run of consecutive replacement months starting at each month (0 if not replacing)."""
    run = np.zeros(is_rep.shape, int)
    for t in range(is_rep.shape[0] - 1, -1, -1):
        run[t] = is_rep[t] * (1 + (run[t + 1] if t + 1 < is_rep.shape[0] else 0))
    return run


class Acc:
    """Aggregates (sums over seeds) of one run (COVID or no-COVID)."""

    def __init__(self, T):
        self.n = 0
        self.shares = np.zeros((T, 2, 5))                 # migrants by month, group, state
        self.trans = np.zeros((len(DESTS), 2, 5, 5))      # ORIGIN state -> DEST state, by group
        self.cohort_able = np.zeros((T, 2, 5))            # states over time of ORIGIN able senders
        self.cohort_rep = np.zeros((T, 2, 5))             # states over time of April 2020 replacements
        self.came_from = np.zeros((T, 2, 5))              # state at t-1 of migrants stepping in at t
        self.spell_len = np.zeros((T, 2, T + 1))          # new replacement spells starting at t, by length
        self.spell_exit = np.zeros((T, 2, 6))             # state after the spell ends (5 = still replacing, Dec 2022)

    def add(self, state, core):
        T = state.shape[0]
        self.n += 1
        g = np.where(core, 0, 1)
        o, a = mi(ORIGIN), mi('2020-04')
        is_rep = state == STEP
        run = spells(is_rep)
        for gi in range(2):
            S = state[:, g == gi]
            for k in range(5):
                self.shares[:, gi, k] += (S == k).sum(1)
            for i, dm in enumerate(DESTS):
                np.add.at(self.trans[i, gi], (S[o], S[mi(dm)]), 1)
            for mask, arr in ((S[o] == ABLE, self.cohort_able), (S[a] == STEP, self.cohort_rep)):
                for k in range(5):
                    arr[:, gi, k] += (S[:, mask] == k).sum(1)
            Rg, runs = is_rep[:, g == gi], run[:, g == gi]
            for t in range(1, T):
                for k in range(5):
                    self.came_from[t, gi, k] += (Rg[t] & (S[t - 1] == k)).sum()
                new = Rg[t] & ~Rg[t - 1]
                L = runs[t, new]
                self.spell_len[t, gi] += np.bincount(L, minlength=T + 1)
                end = t + L
                ex = np.where(end < T, S[np.minimum(end, T - 1), np.flatnonzero(new)], 5)
                self.spell_exit[t, gi] += np.bincount(ex, minlength=6)


def run_simulation(seeds):
    pop = wa.load_population()
    cal = wa.load_calibration()
    fits = pd.read_csv(wa.SHOCKS / 'beta_fit.csv')
    settings = json.loads((wa.SHOCKS / 'run_settings.json').read_text())
    rho = json.loads((wa.OUT / 'household-layer' / 'run_settings.json').read_text())['rho']
    best = fits.loc[fits.rmse_fit.idxmin()]
    kappa, beta = float(best.kappa), float(best.beta)
    p = wa.params_for(cal, kappa)
    pi, rounds = settings['pi'], settings['stimulus_rounds']
    covid = wa.abm.make_inputs(DATES, pop['rate_columns'])
    cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in DATES]), pop['rate_columns'])
    need, stim, zero = hm.need_stock(DATES), wa.stimulus_indicator(DATES, rounds), np.zeros(len(DATES))
    T, core = len(DATES), pop['core']
    print(f'kappa {kappa}, beta {beta:.4f}, rho {rho}, pi {pi}, rounds {rounds}, migrants {pop["M"]} '
          f'({core.sum()} core), households {pop["H"]}', flush=True)

    # exact state shares and expected replacements
    exact = []
    for run, inp, b, nd, st in (('covid', covid, beta, need, stim), ('no_covid', cf, 0.0, None, zero)):
        a = wa.ability(pop, inp, pi, st)
        d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], b, nd, p['delta'])
        drawn, unfilled, _ = wa.StateTables(pop, a, d).shortage(kappa)
        for gi, (gname, m) in enumerate((('core', core), ('kin', ~core))):
            sh = {'Able sender': (a * d)[:, m].mean(1), 'Lost sender': ((1 - a) * d)[:, m].mean(1),
                  'Reserve': (a * (1 - d))[:, m].mean(1), 'Inactive': ((1 - a) * (1 - d))[:, m].mean(1)}
            for t, dt in enumerate(DATES):
                exact.append(dict(run=run, month=dt.strftime('%Y-%m'), group=gname, migrants=int(m.sum()),
                                  **{k: v[t] for k, v in sh.items()},
                                  replacements_all_migrants=(drawn[t] - unfilled[t]) if gi == 0 else np.nan,
                                  replacements_unfilled=unfilled[t] if gi == 0 else np.nan))
    exact = pd.DataFrame(exact)

    import run_households as rh
    acc = {'covid': Acc(T), 'no_covid': Acc(T)}
    for s in range(seeds):
        st_c, rec_c = simulate_states(pop, covid, p, kappa, beta, need, pi, stim, 9000 + s, rho)
        st_f, _ = simulate_states(pop, cf, p, kappa, 0.0, None, 0.0, zero, 9000 + s, rho)
        if s == 0:                                         # same random numbers as the household simulator
            rec_ref, _ = rh.simulate(pop, covid, p, kappa, beta, need, pi, stim, 9000, rho)
            assert (rec_ref == rec_c).all(), 'receipt differs from run_households.simulate'
        acc['covid'].add(st_c, core)
        acc['no_covid'].add(st_f, core)
        if (s + 1) % 20 == 0:
            print(f'seed {s + 1}/{seeds}', flush=True)
    write_tables(acc, exact, pop, dict(kappa=kappa, beta=beta, rho=rho, pi=pi, stimulus_rounds=rounds, seeds=seeds))


def write_tables(acc, exact, pop, settings):
    OUT.mkdir(parents=True, exist_ok=True)
    months = DATES.strftime('%Y-%m')
    nmig = {'core': int(pop['core'].sum()), 'kin': int((~pop['core']).sum())}
    sim, trans, coh, came, spl = [], [], [], [], []
    for run, A in acc.items():
        n = A.n
        for gi, g in enumerate(GROUPS):
            for t, m in enumerate(months):
                sim.append(dict(run=run, month=m, group=g, **{k: A.shares[t, gi, i] / n / nmig[g] for i, k in enumerate(STATES)}))
                for name, arr in (('able_senders_' + ORIGIN, A.cohort_able), ('replacements_2020-04', A.cohort_rep)):
                    tot = arr[t, gi].sum()
                    coh.append(dict(run=run, cohort=name, group=g, month=m, cohort_size_share_of_group=tot / n / nmig[g],
                                    **{k: arr[t, gi, i] / tot if tot else np.nan for i, k in enumerate(STATES)}))
                tot = A.came_from[t, gi].sum()
                came.append(dict(run=run, month=m, group=g, replacements_share_of_group=tot / n / nmig[g],
                                 **{f'from_{k}': A.came_from[t, gi, i] / tot if tot else np.nan for i, k in enumerate(STATES)}))
                L = A.spell_len[t, gi]
                if L.sum():
                    spl.append(dict(run=run, start_month=m, group=g, new_spells_share_of_group=L.sum() / n / nmig[g],
                                    **{f'length_{k}': L[k] / L.sum() for k in range(1, len(L))},
                                    **{f'exit_{k}': A.spell_exit[t, gi, i] / L.sum()
                                       for i, k in enumerate(STATES + ['still replacing Dec 2022'])}))
            for i, dm in enumerate(DESTS):
                M_ = A.trans[i, gi]
                for o in range(5):
                    for d in range(5):
                        trans.append(dict(run=run, group=g, origin_month=ORIGIN, dest_month=dm, origin=STATES[o],
                                          dest=STATES[d], share_of_group=M_[o, d] / n / nmig[g],
                                          share_of_origin=M_[o, d] / M_[o].sum() if M_[o].sum() else np.nan))
    exact.to_csv(OUT / 'state_shares_exact.csv', index=False)
    pd.DataFrame(sim).to_csv(OUT / 'state_shares_simulated.csv', index=False)
    pd.DataFrame(trans).to_csv(OUT / 'transitions.csv', index=False)
    pd.DataFrame(coh).to_csv(OUT / 'cohorts.csv', index=False)
    pd.DataFrame(came).to_csv(OUT / 'replacements_came_from.csv', index=False)
    pd.DataFrame(spl).to_csv(OUT / 'replacement_spells.csv', index=False)
    (OUT / 'run_settings.json').write_text(json.dumps(dict(
        **settings, origin_month=ORIGIN, dest_months=DESTS, seed_base=9000,
        migrants=nmig, households=int(pop['H']),
        who_steps_in='last month\'s replacements first if still in the reserve, then other reserve migrants of the '
                     'household at random (core and kin alike); does not change counts, receipt or transfers',
        states=STATES), indent=2))
    print('tables in', OUT)


# %% Figures
def month_axis(ax, dates, every=6):
    x = np.arange(len(dates))
    ticks = sorted({0} | {i for i, d in enumerate(dates) if d.month in (1, 7)})
    labels = [d.strftime('%b %Y') if (d.month == 1 or i == 0) else d.strftime('%b') for i, d in zip(ticks, dates[ticks])]
    ax.set_xticks(ticks, labels, fontsize=7.5)
    ax.set_xlim(-0.5, len(dates) - 0.5)
    ax.grid(axis='x', visible=False)
    return x


def mark_events(ax, dates, y, labels=True):
    """Dotted verticals at April 2020 and the stimulus months, labelled above the plot."""
    m = list(dates.strftime('%Y-%m'))
    for mo, lab in (('2020-04', 'Apr 2020'), ('2021-01', 'stimulus'), ('2021-03', '')):
        if mo in m:
            i = m.index(mo)
            ax.axvline(i, color=mf.INK2, lw=0.8, ls=':', zorder=6)
            if labels and lab:
                ax.text(i + (0.4 if mo == '2020-04' else 0.4), y, lab + (' (Jan, Mar 2021)' if lab == 'stimulus' else ''),
                        fontsize=7, color=mf.INK2, va='bottom', ha='left', transform=ax.get_xaxis_transform())


def stack(ax, x, shares, label_at=None, min_label=0.035):
    """100% stacked bands in STATES order with a surface hairline between bands; direct labels on wide bands."""
    lo = np.zeros(len(x))
    for k in STATES:
        hi = lo + shares[k]
        ax.fill_between(x, lo, hi, color=COLORS[k], lw=0.8, edgecolor=mf.SURFACE, step=None, zorder=2)
        if label_at is not None:
            i = label_at
            if shares[k][i] >= min_label:
                ax.text(x[i] + 0.4, (lo[i] + hi[i]) / 2, f'{k.lower()} {shares[k][i]:.0%}', ha='left', va='center',
                        fontsize=7.5, color='white', fontweight='semibold', zorder=7)
        lo = hi
    ax.set_ylim(0, 1)
    mf.pct_axis(ax)


def load_tables():
    T = {n: pd.read_csv(OUT / f'{n}.csv') for n in ('state_shares_exact', 'state_shares_simulated', 'transitions',
                                                   'cohorts', 'replacements_came_from', 'replacement_spells')}
    T['settings'] = json.loads((OUT / 'run_settings.json').read_text())
    return T


def shares_for(T, run, group):
    """Exact four-state shares with the reserve split into steps in (simulated) and the rest."""
    e = T['state_shares_exact'].query('run == @run and group == @group').set_index('month')
    s = T['state_shares_simulated'].query('run == @run and group == @group').set_index('month')
    out = {k: e[k].to_numpy() for k in ('Able sender', 'Lost sender', 'Reserve', 'Inactive')}
    out['Steps in'] = np.minimum(s['Steps in'].reindex(e.index).to_numpy(), out['Reserve'])
    out['Reserve'] = out['Reserve'] - out['Steps in']
    return out


def legend_row(fig, y=1.0, extra=()):
    h = [Patch(facecolor=COLORS[k], edgecolor='none', label='steps in (replacement)' if k == 'Steps in' else k.lower())
         for k in STATES] + list(extra)
    fig.legend(handles=h, loc='lower left', bbox_to_anchor=(0.01, y), ncol=len(h), frameon=False, fontsize=8.5,
               handlelength=1.2, columnspacing=1.4)


def save(fig, name):
    fig.savefig(FIG / f'{name}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def pct1(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))


def lost_and_replaced(T, run):
    """All migrants: share who are lost senders, share stepping in (exact), share stepping in who are kin (sim)."""
    nm = T['settings']['migrants']
    N = sum(nm.values())
    e = T['state_shares_exact'].query('run == @run').pivot(index='month', columns='group')
    s = T['state_shares_simulated'].query('run == @run').pivot(index='month', columns='group').reindex(e.index)
    lost = (e['Lost sender']['core'] * nm['core'] + e['Lost sender']['kin'] * nm['kin']) / N
    rep = e['replacements_all_migrants']['core'] / N
    rep_kin = s['Steps in']['kin'] * nm['kin'] / N
    return lost.to_numpy(), rep.to_numpy(), rep_kin.to_numpy()


def r4_state_shares(T):
    """a, b: shares of core / kin migrants in each state by month (exact; steps in simulated). c: lost senders and
    replacements, all migrants (COVID run; lost senders also for the no-COVID run)."""
    fig = plt.figure(figsize=(10, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.42, wspace=0.16)
    lab = mi('2019-02')
    for j, (g, title) in enumerate((('core', 'a  Core migrants'), ('kin', 'b  Kin migrants'))):
        ax = fig.add_subplot(gs[0, j])
        n = T['settings']['migrants'][g]
        x = month_axis(ax, DATES)
        stack(ax, x, shares_for(T, 'covid', g), label_at=lab, min_label=0.06)
        mark_events(ax, DATES, 1.01)
        ax.set_title(f'{title} ({n:,})', fontsize=10, loc='left', pad=16)
        ax.set_ylabel('Share of migrants' if j == 0 else '')
        ax.grid(axis='y', visible=False)
    legend_row(fig, y=0.955)
    ax = fig.add_subplot(gs[1, :])
    x = month_axis(ax, DATES)
    lost, rep, rep_kin = lost_and_replaced(T, 'covid')
    lost_cf = lost_and_replaced(T, 'no_covid')[0]
    ax.fill_between(x, rep, lost, where=lost >= rep, color=COLORS['Lost sender'], alpha=0.16, lw=0, zorder=1,
                    label='lost senders not replaced')
    ax.plot(x, lost, color=COLORS['Lost sender'], lw=2, zorder=4, label='lost senders')
    ax.plot(x, rep, color=COLORS['Steps in'], lw=2, zorder=4, label='replacements (reserve migrants who step in)')
    ax.plot(x, lost_cf, color=mf.MUTED, lw=1.3, ls=':', zorder=3, label='lost senders, no-COVID run')
    ax.plot(x, rep_kin, color=COLORS['Steps in'], lw=1.4, ls='--', zorder=4, label='replacements who are kin migrants')
    a = mi('2020-04')
    ax.annotate(f'{lost[a]:.1%} of migrants are lost senders,\n{rep[a] / lost[a]:.0%} of them replaced',
                (x[a], lost[a]), xytext=(10, 0), textcoords='offset points', fontsize=7.5, color=mf.INK2, va='center')
    mark_events(ax, DATES, 1.01)
    ax.set_ylim(0, lost.max() * 1.12)
    pct1(ax)
    ax.set_ylabel('Share of all migrants')
    ax.set_title('c  Lost senders and replacements, all migrants', fontsize=10, loc='left', pad=16)
    ax.legend(loc='upper right', frameon=False, fontsize=8, ncol=1)
    save(fig, 'R4_state_shares')


def combined_cohort(T, cohort, run):
    """Core and kin cohort shares combined (weighted by cohort size); returns shares by month and cohort size."""
    c = T['cohorts'].query('cohort == @cohort and run == @run')
    w = c.group.map(T['settings']['migrants']) * c.cohort_size_share_of_group
    out = c[STATES].mul(w, axis=0).groupby(c.month).sum()
    n = w.groupby(c.month).sum()
    return out.div(n, axis=0), float(n.iloc[0])


def spell_survival(T, run, start='2020-04'):
    """Share of replacement spells starting in `start` (core and kin) still running after k months, k = 1, 2, ..."""
    r = T['replacement_spells'].query('run == @run and start_month == @start')
    w = r.new_spells_share_of_group * r.group.map(T['settings']['migrants'])
    L = (r[[f'length_{k}' for k in range(1, len(DATES) + 1)]].mul(w, axis=0).sum() / w.sum()).to_numpy()
    return 1 - np.concatenate([[0], np.cumsum(L)[:-1]]), L


def r4_cohorts(T):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1.3, 0.8]))
    # a: Feb 2020 able senders, Feb 2020 - Dec 2022
    ax = axes[0]
    sh, n = combined_cohort(T, 'able_senders_' + ORIGIN, 'covid')
    ref, _ = combined_cohort(T, 'able_senders_' + ORIGIN, 'no_covid')
    keep = DATES >= pd.Timestamp(ORIGIN + '-01')
    d = DATES[keep]
    x = month_axis(ax, d)
    stack(ax, x, {k: sh[k].to_numpy()[keep] for k in STATES}, label_at=list(d.strftime('%Y-%m')).index('2021-10'),
          min_label=0.08)
    yref = ref['Able sender'].to_numpy()[keep]
    ax.plot(x, yref, color=mf.INK, lw=1.3, ls='--', zorder=8)
    i = list(d.strftime('%Y-%m')).index('2020-09')
    ax.annotate('able senders,\nno-COVID run', (x[i], yref[i]), xytext=(4, 16), textcoords='offset points',
                fontsize=7.5, color=mf.INK, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='-', color=mf.INK, lw=0.7))
    mark_events(ax, d, 1.01)
    ax.set_ylabel('Share of the group')
    ax.set_title('a  Where the Feb 2020 able senders went', fontsize=10, loc='left', pad=16)
    ax.grid(axis='y', visible=False)
    # b: April 2020 replacements, before and after
    ax = axes[1]
    sh, n = combined_cohort(T, 'replacements_2020-04', 'covid')
    keep = DATES >= pd.Timestamp('2019-07-01')
    d = DATES[keep]
    x = month_axis(ax, d)
    stack(ax, x, {k: sh[k].to_numpy()[keep] for k in STATES}, label_at=list(d.strftime('%Y-%m')).index('2019-07'),
          min_label=0.08)
    mark_events(ax, d, 1.01)
    ax.set_title('b  Apr 2020 replacements, before and after', fontsize=10, loc='left', pad=16)
    ax.grid(axis='y', visible=False)
    # c: spell length
    ax = axes[2]
    K = 12
    for run, col, ls, lab in (('covid', COLORS['Steps in'], '-', 'COVID run'),
                              ('no_covid', mf.MUTED, '--', 'no-COVID run')):
        surv, _ = spell_survival(T, run)
        ax.plot(np.arange(1, K + 1), surv[:K], color=col, lw=2, ls=ls, marker='o', markersize=5,
                markeredgecolor=mf.SURFACE, label=lab)
    ax.set_xlabel('Months since stepping in (Apr 2020)')
    ax.set_ylabel('Share still replacing')
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_ylim(0, 1.05)
    mf.pct_axis(ax)
    ax.axvline(10, color=mf.INK2, lw=0.8, ls=':')
    ax.text(10.25, 0.97, 'Jan 2021\nstimulus', fontsize=7, color=mf.INK2, va='top')
    ax.legend(loc='upper right', bbox_to_anchor=(0.97, 0.8), frameon=False, fontsize=8)
    ax.set_title('c  How long replacements stay', fontsize=10, loc='left', pad=16)
    ax.grid(axis='x', visible=False)
    legend_row(fig, y=0.99, extra=[Line2D([], [], color=mf.INK, lw=1.3, ls='--', label='no-COVID run (a)')])
    fig.subplots_adjust(wspace=0.25)
    save(fig, 'R4_cohorts')


def r4_transitions(T):
    """Transition matrices: state in Feb 2020 (rows) -> state in a later month (columns), share of the row."""
    from matplotlib.colors import LinearSegmentedColormap
    panels = [('no_covid', '2020-04', 'Apr 2020, no-COVID run'), ('covid', '2020-04', 'Apr 2020'),
              ('covid', '2020-12', 'Dec 2020'), ('covid', '2021-12', 'Dec 2021')]
    order = ['Able sender', 'Steps in', 'Lost sender', 'Reserve', 'Inactive']
    short = {k: ('steps in' if k == 'Steps in' else k.lower()) for k in order}
    cmap = LinearSegmentedColormap.from_list('seq', SEQ)
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.6))
    t = T['transitions']
    for i, g in enumerate(GROUPS):
        for j, (run, dm, title) in enumerate(panels):
            ax = axes[i, j]
            m = t.query('run == @run and group == @g and dest_month == @dm')
            P = m.pivot(index='origin', columns='dest', values='share_of_origin').reindex(index=order, columns=order)
            orig = m.groupby('origin').share_of_group.sum().reindex(order)
            edges = np.arange(6) - 0.5
            ax.pcolormesh(edges, edges, P.to_numpy(), cmap=cmap, vmin=0, vmax=1, edgecolors=mf.SURFACE, linewidth=2)
            for r in range(5):
                for c in range(5):
                    v = P.iat[r, c]
                    ax.text(c, r, f'{v:.0%}' if v >= 0.005 else '·', ha='center', va='center', fontsize=8,
                            color='white' if v > 0.45 else mf.INK2)
            ax.set_xticks(range(5), [short[k] for k in order], rotation=40, ha='right', fontsize=7.5)
            ax.set_yticks(range(5), [f'{short[k]} ({orig[k]:.1%})' for k in order] if j == 0 else [], fontsize=7.5)
            ax.grid(False)
            ax.tick_params(which='both', length=0)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(4.5, -0.5)
            ax.set_title(title, fontsize=9, loc='left')
            if j == 0:
                ax.set_ylabel(f'{"Core" if g == "core" else "Kin"} migrants: state in Feb 2020\n(share of group)',
                              fontsize=8.5)
            if i == 1:
                ax.set_xlabel('State in the later month', fontsize=8.5)
    fig.subplots_adjust(wspace=0.1, hspace=0.5)
    save(fig, 'R4_transitions')


def key_numbers(T):
    e, s, nm = T['state_shares_exact'], T['state_shares_simulated'], T['settings']['migrants']
    rows = []
    for run in ('covid', 'no_covid'):
        lost, rep, rep_kin = lost_and_replaced(T, run)
        for m in ('2020-02', '2020-04', '2020-05', '2020-12', '2021-12'):
            i = mi(m)
            q = e.query('run == @run and month == @m').set_index('group')
            rows.append(dict(run=run, month=m, lost_senders_share_all=lost[i], replacements_share_all=rep[i],
                             share_of_lost_senders_replaced=rep[i] / lost[i], kin_share_of_replacements=rep_kin[i] / rep[i],
                             core_able_sender=q.at['core', 'Able sender'], core_lost_sender=q.at['core', 'Lost sender'],
                             core_inactive=q.at['core', 'Inactive'], kin_inactive=q.at['kin', 'Inactive']))
    tr = T['transitions'].query('origin == "Able sender"')
    for run in ('covid', 'no_covid'):
        for dm in DESTS:
            x = tr.query('run == @run and dest_month == @dm')
            w = x.groupby('dest').share_of_group.apply(lambda v: v.to_numpy() @ np.array([nm[g] for g in x.loc[v.index, 'group']]))
            rows.append(dict(run=run, month=f'Feb 2020 able senders in {dm}', **(w / w.sum()).to_dict()))
    for run in ('covid', 'no_covid'):
        surv, L = spell_survival(T, run)
        rows.append(dict(run=run, month='replacement spells starting Apr 2020', mean_length=float(L @ np.arange(1, len(L) + 1)),
                         still_after_3=surv[3], still_after_6=surv[6]))
        c, _ = combined_cohort(T, 'replacements_2020-04', run)
        for m in ('2020-02', '2020-03', '2020-12', '2021-12'):
            rows.append(dict(run=run, month=f'Apr 2020 replacements in {m}', **c.loc[m].to_dict()))
    kn = pd.DataFrame(rows)
    kn.to_csv(OUT / 'key_numbers.csv', index=False)
    pd.set_option('display.width', 250)
    print(kn.round(3).to_string(index=False))


def plot_all():
    mf.style()
    FIG.mkdir(parents=True, exist_ok=True)
    T = load_tables()
    r4_state_shares(T)
    r4_cohorts(T)
    r4_transitions(T)
    key_numbers(T)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    if not args.plot_only:
        run_simulation(args.seeds)
    plot_all()
