"""What stepping-in (kappa) does: senders, receiving households and transfers against the no-COVID run (11 Nov 2026).

Run: python 2026-10-30/make_kappa_effect_figure.py      (exact, no simulation; about a minute)
Lines: kappa = 0, 0.5, 1, each with its fitted beta (outputs/shocks/beta_fit.csv), plus job loss only (kappa = 0,
beta = 0). Each series is COVID run / no-COVID run (same kappa), monthly:
  a  effective senders: able senders sum(a d) + replacements (drawn - unfilled, StateTables.shortage)
  b  households receiving: sum_h P_h,t
  c  transfers: sum_h T_h,t, with Banxico transactions / 2017-19 trend
Outputs: figures/paper/R5_kappa_effect.png, outputs/transitions/kappa_effect.csv (aggregates).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import willing_able as wa
import run_household_model as hm
import run_shocks as rs
import make_transition_figures as tf

mf = tf.mf
RUNS = [('job loss only', 0.0, 0.0), ('κ = 0', 0.0, None), ('κ = 0.5', 0.5, None), ('κ = 1', 1.0, None)]
COL = {'job loss only': mf.MUTED, 'κ = 0': mf.ORANGE, 'κ = 0.5': '#e87ba4', 'κ = 1': mf.BLUE}
LS = {'job loss only': ':', 'κ = 0': '-', 'κ = 0.5': '-', 'κ = 1': '-'}


def senders(pop, a, d, kappa):
    drawn, unfilled, _ = wa.StateTables(pop, a, d).shortage(kappa)
    return (a * d).sum(1) + (np.asarray(drawn) - np.asarray(unfilled))


def compute():
    import json
    pop, cal = wa.load_population(), wa.load_calibration()
    st = json.loads((wa.SHOCKS / 'run_settings.json').read_text())
    sh = rs.Shocks(pop, cal, st['stimulus_rounds'], st['pi'])
    fits = pd.read_csv(wa.SHOCKS / 'beta_fit.csv')
    T = len(hm.DATES)
    rows = []
    for name, kappa, beta in RUNS:
        if beta is None:
            beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
        p = wa.params_for(cal, kappa)
        rc, tc, rf, tfr = sh.run(kappa, beta)
        d_c = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, sh.need, p['delta'])
        d_f = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], delta=p['delta'])
        ec, ef = senders(pop, sh.a_covid, d_c, kappa), senders(pop, sh.a_cf, d_f, kappa)
        for t, dt in enumerate(hm.DATES):
            rows.append(dict(run=name, kappa=kappa, beta=beta, month=dt.strftime('%Y-%m'),
                             senders=ec[t] / ef[t], receipt=rc.iloc[t] / rf.iloc[t], transfers=tc.iloc[t] / tfr.iloc[t]))
        print(f'{name}: beta {beta:.3f}', flush=True)
    df = pd.DataFrame(rows)
    bx, trend, _ = hm.e1.banxico()
    df = df.merge(pd.DataFrame({'month': hm.DATES.strftime('%Y-%m'),
                                'banxico': (bx / trend).reindex(hm.DATES).to_numpy()}), on='month')
    tf.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(tf.OUT / 'kappa_effect.csv', index=False)
    return df


def plot(df):
    mf.style()
    start = '2020-01'
    months = df.month.unique()
    keep = months[months >= start]
    dates = pd.DatetimeIndex(pd.to_datetime(keep))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    panels = [('senders', 'a  Effective senders'), ('receipt', 'b  Households receiving'),
              ('transfers', 'c  Transfers')]
    for ax, (col, title) in zip(axes, panels):
        x = tf.month_axis(ax, dates)
        ax.axhline(1, color=mf.INK2, lw=0.8, zorder=1)
        if col == 'transfers':
            g = df[df.run == 'κ = 1'].set_index('month').loc[keep]
            ax.plot(x, g.banxico, color='black', lw=1.6, marker='o', ms=2.5, zorder=5)
        for name, _, _ in RUNS:
            g = df[df.run == name].set_index('month').loc[keep]
            ax.plot(x, g[col], color=COL[name], lw=1.4 if name == 'job loss only' else 2, ls=LS[name], zorder=4)
        tf.mark_events(ax, dates, 1.01)
        ax.set_title(title, fontsize=10, loc='left', pad=16)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
    axes[0].set_ylabel('COVID run / no-COVID run')
    lo = df[df.month >= start][['senders', 'receipt', 'transfers', 'banxico']].min().min()
    hi = df[df.month >= start][['senders', 'receipt', 'transfers', 'banxico']].max().max()
    axes[0].set_ylim(lo - 0.02, hi + 0.02)
    handles = [Line2D([], [], color=COL[n], ls=LS[n], lw=2, label=n + (', fitted β' if n.startswith('κ') else
                                                                         ' (κ = 0, β = 0)')) for n, _, _ in RUNS]
    handles.append(Line2D([], [], color='black', marker='o', ms=3, lw=1.6, label='Banxico transactions / trend (c)'))
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.01, 0.99), ncol=len(handles), frameon=False,
               fontsize=8.5)
    tf.save(fig, 'R5_kappa_effect')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    df = pd.read_csv(tf.OUT / 'kappa_effect.csv') if args.plot_only else compute()
    plot(df)
