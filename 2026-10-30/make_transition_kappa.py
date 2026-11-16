"""R4_state_shares panel a (core migrants, four states plus those who step in) for kappa = 0, 0.5, 1 (11 Nov 2026).

Run: python 2026-10-30/make_transition_kappa.py [--seeds 40]
Each kappa uses its own fitted beta (outputs/shocks/beta_fit.csv). The four state shares are exact (marginals);
the 'steps in' band (core migrants stepping in, a subset of the reserve) is simulated with
make_transition_figures.simulate_states (rho from the household layer), COVID run only.
Outputs: figures/paper/R4_state_shares_by_kappa.png, outputs/transitions/state_shares_by_kappa.csv (aggregates).
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import willing_able as wa
import run_household_model as hm
import make_transition_figures as tf

KAPPAS = [0.0, 0.5, 1.0]


def compute(seeds):
    pop = wa.load_population()
    cal = wa.load_calibration()
    fits = pd.read_csv(wa.SHOCKS / 'beta_fit.csv')
    settings = json.loads((wa.SHOCKS / 'run_settings.json').read_text())
    rho = json.loads((wa.OUT / 'household-layer' / 'run_settings.json').read_text())['rho']
    pi, rounds = settings['pi'], settings['stimulus_rounds']
    dates = tf.DATES
    covid = wa.abm.make_inputs(dates, pop['rate_columns'])
    need, stim = hm.need_stock(dates), wa.stimulus_indicator(dates, rounds)
    core, T = pop['core'], len(dates)
    rows = []
    for kappa in KAPPAS:
        beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
        p = wa.params_for(cal, kappa)
        a = wa.ability(pop, covid, pi, stim)
        d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['delta'])
        steps = np.zeros(T)
        if kappa > 0:
            for s in range(seeds):
                st, _ = tf.simulate_states(pop, covid, p, kappa, beta, need, pi, stim, 9000 + s, rho)
                steps += (st[:, core] == tf.STEP).mean(1)
            steps /= seeds
        m = core
        for t, dt in enumerate(dates):
            rows.append(dict(kappa=kappa, beta=beta, month=dt.strftime('%Y-%m'),
                             able=(a * d)[t, m].mean(), lost=((1 - a) * d)[t, m].mean(),
                             reserve=(a * (1 - d))[t, m].mean(), inactive=((1 - a) * (1 - d))[t, m].mean(),
                             steps_in=steps[t]))
        print(f'kappa {kappa}: beta {beta:.3f} done', flush=True)
    df = pd.DataFrame(rows)
    tf.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(tf.OUT / 'state_shares_by_kappa.csv', index=False)
    return df, int(core.sum())


def plot(df, n_core):
    tf.mf.style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    lab = tf.mi('2019-02')
    for ax, kappa in zip(axes, KAPPAS):
        g = df[np.isclose(df.kappa, kappa)]
        steps = np.minimum(g.steps_in.to_numpy(), g.reserve.to_numpy())
        shares = {'Steps in': steps, 'Able sender': g.able.to_numpy(), 'Lost sender': g.lost.to_numpy(),
                  'Reserve': g.reserve.to_numpy() - steps, 'Inactive': g.inactive.to_numpy()}
        x = tf.month_axis(ax, tf.DATES)
        tf.stack(ax, x, shares, label_at=lab if kappa == 0 else None, min_label=0.06)
        tf.mark_events(ax, tf.DATES, 1.01)
        a = tf.mi('2020-04')
        ax.set_title(f'κ = {kappa:g}  (β = {g.beta.iloc[0]:.2f}; Apr 2020: {steps[a]:.1%} step in)',
                     fontsize=9.5, loc='left', pad=16)
        ax.grid(axis='y', visible=False)
    axes[0].set_ylabel(f'Share of core migrants ({n_core:,})')
    tf.legend_row(fig, y=0.985)
    tf.save(fig, 'R4_state_shares_by_kappa')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    if args.plot_only:
        df = pd.read_csv(tf.OUT / 'state_shares_by_kappa.csv')
        n = int(wa.load_population()['core'].sum())
    else:
        df, n = compute(args.seeds)
    plot(df, n)
