"""Collect the robustness runs 6a-6h into one table (29 Oct 2026).

Run: python network-compatibility-model/recipient-level/robustness/collect.py   (seconds; after the eight
     scripts in this folder, see run_all.py)
Output: outputs/robustness/summary.csv, one row per test x variant x kappa in {0.8, 1}, with the base model's
     standard columns (beta refitted continuously for each kappa; quarterly log RMSE 2020Q2-2021Q4 vs Banxico /
     2017-19 trend, 2022 held out; Jan / Mar 2021 spikes in log points). Test-specific results stay in each
     test's own folder and README. Decision on the stimulus (user, 29 Oct 2026): keep the base, pi = 1 in
     Jan and Mar 2021 fitted to the spikes, and report that the spikes look like transfers shifted between
     months (stimulus_gamma: the quarterly fit is best without any stimulus).
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / 'outputs/robustness'
COLS = ['beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
        'spike_2021_01', 'spike_2021_03', 'spike_rmse']
KAPPAS = [0.8, 1.0]


def pick(d, test, variant_col, label=None, rename=None):
    d = d.rename(columns=rename or {})
    d = d[np.isclose(d.kappa.to_numpy(float)[:, None], KAPPAS).any(1)]
    out = d.assign(test=test, variant=d[variant_col].astype(str) if label is None else label(d))
    return out[['test', 'variant', 'kappa'] + [c for c in COLS if c in out]].drop_duplicates(['test', 'variant', 'kappa'])


def main():
    parts = [
        pick(pd.read_csv(OUT / 'savings_margin/summary.csv'), '6a savings margin', 'variant'),
        pick(pd.read_csv(OUT / 'pi_april_2020/summary.csv'), '6b pi in April 2020', 'variant'),
        pick(pd.read_csv(OUT / 'stimulus_gamma/summary.csv'), '6c stimulus as gamma', 'variant'),
        pick(pd.read_csv(OUT / 'pooled_mar_apr/summary.csv'), '6d Mar-Apr 2020 pooled', 'variant'),
        pick(pd.read_csv(OUT / 'need_series_decay/summary.csv'), '6f need series and decay', 'row',
             rename={'slope': 'beta'}),
        pick(pd.read_csv(OUT / 'lower_bound_mapping/fits.csv'), '6g count mapping', 'w',
             label=lambda d: 'w = ' + d.w.map('{:g}'.format) + np.where(d.w.eq(0), ' (base)', '')),
        pick(pd.read_csv(OUT / 'trend_window/summary.csv'), '6h trend window', 'variant'),
    ]
    oos = pd.read_csv(OUT / 'out_of_sample_2019/summary.csv')
    oos = oos[np.isclose(oos.kappa.to_numpy(float)[:, None], KAPPAS).any(1)]
    parts.append(pd.DataFrame(dict(test='6e out of sample 2019', variant=oos.direction, kappa=oos.kappa,
                                   rmse_fit=oos.rmse_log_vs_1_quarterly)))
    s = pd.concat(parts, ignore_index=True)
    s.to_csv(OUT / 'summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(s.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
