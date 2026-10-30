"""Figure 1 of the paper draft: Banxico transfer counts against their pre-COVID trend (29 Oct 2026).

Run: python paper/make_paper_figures.py   (seconds)
Reads Banxico counts through model/run_e1_job_loss.banxico (public aggregate) and the MXN/USD rate in
data/DEXMXUS.csv (FRED, public; local copy). Writes paper/figures/F1_puzzle.png. Uses the palette and
helpers of network-compatibility-model/model/make_figures.py.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
NCM = HERE.parent / 'network-compatibility-model'
sys.path[:0] = [str(NCM / 'model')]
import run_e1_job_loss as e1
import make_figures as mf

OUT = HERE / 'figures'
FX = HERE.parent / 'data/DEXMXUS.csv'


def main():
    OUT.mkdir(exist_ok=True)
    mf.style()
    bx, trend, info = e1.banxico()
    idx = bx / trend
    q = bx.groupby(bx.index.to_period('Q')).sum() / trend.groupby(trend.index.to_period('Q')).sum()
    fx = pd.read_csv(FX, parse_dates=['observation_date']).set_index('observation_date').DEXMXUS
    fx = pd.to_numeric(fx, errors='coerce').resample('MS').mean().reindex(idx.index)

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.1]))
    ax.axhline(1, color=mf.AXIS, lw=1, zorder=1)
    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), color=mf.HOLDOUT_WASH, lw=0, zorder=0)
    ax.plot(idx.index, idx.values, color=mf.INK, lw=1.6, marker='o', markersize=3, zorder=3,
            label='Monthly transfers / 2017–19 trend')
    for i, (per, v) in enumerate(q.items()):
        ax.hlines(v, per.start_time, per.end_time, color=mf.BLUE, lw=2.2, zorder=2,
                  label='Quarterly' if i == 0 else None)
    notes = [('2020-03-01', 'March 2020:\nrush at the weak peso', -10, 6),
             ('2020-04-01', 'April 2020:\npayback', 10, -24),
             ('2021-01-01', 'Stimulus checks\n(Jan and Mar 2021)', 4, 34), ('2021-03-01', '', 0, 0)]
    for d, text, dx, dy in notes:
        t = pd.Timestamp(d)
        mf.ring_dot(ax, t, idx[t], mf.ORANGE, size=46, zorder=5)
        if text:
            ax.annotate(text, (t, idx[t]), xytext=(dx, dy), textcoords='offset points', fontsize=8.5,
                        color=mf.INK2, ha='left' if dx >= 0 else 'right')
    ax.text(pd.Timestamp('2022-07-01'), 0.875, '2022: held out of the fit', ha='center', fontsize=8.5, color=mf.INK2)
    ax.set_ylabel('Relative to pre-COVID trend')
    ax.set_ylim(0.86, 1.2)
    ax.legend(loc='upper left')
    ax2.plot(fx.index, fx.values, color=mf.MUTED, lw=1.6)
    ax2.set_ylabel('MXN per USD')
    fig.tight_layout()
    fig.savefig(OUT / 'F1_puzzle.png', dpi=mf.DPI, bbox_inches='tight')
    print('saved F1_puzzle', f"trend growth {info['trend_growth_per_year']:.3%}")
    print(q.round(3).to_string())


if __name__ == '__main__':
    main()
