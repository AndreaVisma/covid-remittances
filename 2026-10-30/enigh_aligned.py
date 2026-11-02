"""Model margins measured as ENIGH measures them (receipt-vs-transfers test 2; 2 Nov 2026).

Run: python 2026-10-30/enigh_aligned.py            (after run_shocks.py; reads outputs/shocks/beta_fit.csv)
     --kappas 1 0.8   (beta: the fitted value for each);  --seeds 200
ENIGH counts a household as receiving if it received at least once in the six months before an Aug-Nov interview.
The shock-run index instead compares the monthly receipt probability. With persistent receipt within households,
"at least once in six months" can rise less than monthly receipt. Paired simulations (run_households.simulate,
common random numbers; COVID run vs no-COVID run with 2019 employment, no stimulus, no crisis) for the 2020 and
2022 survey windows give, interview-weighted (Aug-Nov 10/31/32/27%):
    receiving      share receiving at least once in the six months, log COVID / no-COVID
    tpr            transfers in the six months per receiving household, log COVID / no-COVID
    monthly        monthly receipt probability over the same six months, log COVID / no-COVID
compared with ENIGH's excess over the 2016-18 trend (identification/enigh/outputs/pre_excess_over_trend.csv):
receipt (receiving households) and tpr_counts (Banxico transactions per ENIGH receiving household). Both are counts,
so prices do not enter them.
Output (aggregates): outputs/enigh_aligned.csv
"""
import argparse
import numpy as np
import pandas as pd

import willing_able as wa
import run_household_model as hm
import run_households as rh

ENIGH = wa.NCM / 'identification/enigh/outputs/pre_excess_over_trend.csv'


def window_measures(rec, trans, dates, year):
    """Interview-weighted share receiving at least once, transfers per receiving household, and mean monthly
    receipt, over the six months before each Aug-Nov interview of `year`. rec, trans: (T, H)."""
    idx = {(d.year, d.month): i for i, d in enumerate(pd.DatetimeIndex(dates))}
    out = np.zeros(3)
    for m, w in rh.INTERVIEW.items():
        rows = [idx[(year, m - k)] for k in range(1, 7)]
        any_rec = rec[rows].any(0)
        out += w * np.array([any_rec.mean(), trans[rows].sum(0).sum() / max(any_rec.sum(), 1), rec[rows].mean()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='*', default=[1.0])
    ap.add_argument('--seeds', type=int, default=200)
    args = ap.parse_args()
    fits = pd.read_csv(wa.OUT / 'shocks' / 'beta_fit.csv')
    settings = pd.read_json(wa.OUT / 'shocks' / 'run_settings.json', typ='series')
    pi, rounds = float(settings['pi']), list(settings['stimulus_rounds'])
    pop = wa.load_population()
    cal = wa.load_calibration()
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need, stim, zero = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, rounds), np.zeros(len(hm.DATES))
    enigh = pd.read_csv(ENIGH).query("from_year == 2018 and to_year in (2020, 2022) and measure in ('receipt', 'tpr_counts')")
    rows = []
    for kappa in args.kappas:
        p = wa.params_for(cal, kappa)
        beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
        acc = {y: [] for y in (2020, 2022)}
        for seed in range(args.seeds):
            rc, tc = rh.simulate(pop, covid, p, kappa, beta, need, pi, stim, 9000 + seed)
            rf, tf = rh.simulate(pop, cf, p, kappa, 0.0, None, 0.0, zero, 9000 + seed)
            for y in acc:
                acc[y].append((window_measures(rc, tc, hm.DATES, y), window_measures(rf, tf, hm.DATES, y)))
        for y, v in acc.items():
            c = np.array([a for a, _ in v]).mean(0)
            f = np.array([b for _, b in v]).mean(0)
            lr = np.log(c / f)
            e = enigh.set_index(['measure', 'to_year'])
            rows.append(dict(kappa=kappa, beta=beta, year=y, seeds=args.seeds,
                             model_receiving=lr[0], model_tpr=lr[1], model_monthly_receipt=lr[2],
                             receiving_share_cf=f[0], monthly_receipt_cf=f[2],
                             enigh_receipt=e.loc[('receipt', y), 'excess'], enigh_receipt_se=e.loc[('receipt', y), 'excess_se'],
                             enigh_tpr_counts=e.loc[('tpr_counts', y), 'excess'],
                             enigh_tpr_counts_se=e.loc[('tpr_counts', y), 'excess_se']))
        print(f'kappa {kappa}: done', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(wa.OUT / 'enigh_aligned.csv', index=False)
    print(out.round(3).T.to_string())


if __name__ == '__main__':
    main()
