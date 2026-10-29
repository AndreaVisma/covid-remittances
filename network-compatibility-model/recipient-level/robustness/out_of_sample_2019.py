"""Robustness 6e: out-of-sample normal-times check of the job-loss channel (29 Oct 2026).

Run (from the repo root): python network-compatibility-model/recipient-level/robustness/out_of_sample_2019.py

Forward: the 7 receipt coefficients are fitted per kappa (KAPPAS) by ML to the MMP receipt targets
(same targets as the base model) with 2018 CPS employment instead of 2019. The household model is then
run with 2019 CPS rates, and the index is run 2019 / run with the same calendar month of 2018
(beta = 0, N* = 0, no stimulus in 2018-19). Reverse: fitted with 2019 CPS (the base calibration),
run 2018 / same month of 2019.
Banxico comparisons for 2019: (a) transactions / 2017-19 trend (in-sample for the trend: 2019 is in the
fit window, so ~1 by construction); (b) transactions / 2015-18 trend extrapolated to 2019 (out of
sample; run_e1_job_loss.banxico with TREND_WINDOW and DATES patched in this process only);
(c) raw 2019 / same month of 2018. Reverse: Banxico 2018 / 2019 raw, and 2018 against a 2015-17 trend.
Counts at beta = 0: base mapping = sum_h P_h,t; lower bound = sum_h P_h,t r_h,t (expected able senders).
Outputs (aggregates only) in recipient-level/outputs/robustness/out_of_sample_2019/.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RL = HERE.parent
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent / 'model'))
import run_household_model as hm
import abm
import run_e1_job_loss as e1

OUT = RL / 'outputs/robustness/out_of_sample_2019'
KAPPAS = [0.0, 0.5, 0.8, 1.0]
Y18 = pd.date_range('2018-01-01', '2018-12-01', freq='MS')
Y19 = pd.date_range('2019-01-01', '2019-12-01', freq='MS')


def banxico(window, dates):
    """e1.banxico with another trend window and output dates (patched in this process only)."""
    saved = e1.TREND_WINDOW, e1.DATES
    e1.TREND_WINDOW, e1.DATES = window, pd.DatetimeIndex(dates)
    try:
        return e1.banxico()
    finally:
        e1.TREND_WINDOW, e1.DATES = saved


def q(s):
    return s.groupby(s.index.to_period('Q')).sum().rename(index=str)


def rmse(x, ref=None):
    x = np.log(np.asarray(x, float)) - (0 if ref is None else np.log(np.asarray(ref, float)))
    return float(np.sqrt(np.mean(x ** 2)))


def run(pop, y, ok, fit_dates, run_dates, base_dates, label):
    cols = pop['rate_columns']
    fit_cells = hm.Cells(pop, abm.make_inputs(fit_dates, cols))
    run_in, base_in = abm.make_inputs(run_dates, cols), abm.make_inputs(base_dates, cols)
    c_run, c_base = hm.Cells(pop, run_in), hm.Cells(pop, base_in)
    r_run = hm.Senders(pop, run_in, np.zeros(12)).r(0.0)
    r_base = hm.Senders(pop, base_in, np.zeros(12)).r(0.0)
    rows, fits, start = [], [], None
    for kappa in KAPPAS:
        x, _, nll, _ = hm.fit_2019(fit_cells, y, ok, kappa, start)
        start = x
        fits.append(dict(direction=label, kappa=kappa, loglik=-nll,
                         mean_receipt_fit_year=float(fit_cells.annual(x, kappa)[ok].mean()),
                         **dict(zip(['intercept'] + hm.TERMS, x))))
        Pr, Pb = c_run.receipt(x, kappa), c_base.receipt(x, kappa)
        s = lambda z, d: pd.Series(z.sum(1), index=d)
        base_r, base_b = s(Pr, run_dates), s(Pb, run_dates)
        low_r, low_b = s(Pr * r_run, run_dates), s(Pb * r_base, run_dates)
        rows.append(pd.DataFrame({'direction': label, 'kappa': kappa, 'date': run_dates,
                                  'index_model': (base_r / base_b).values,
                                  'index_model_lower_bound': (low_r / low_b).values,
                                  'receipt_run': base_r.values / pop['H'], 'receipt_base': base_b.values / pop['H']}))
        rows[-1].attrs['q'] = (q(base_r) / q(base_b), q(low_r) / q(low_b))
    return rows, fits


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hm.hl.load_population()
    h = pop['h']
    ok = hm.e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)

    # Banxico
    bx19_a, tr19_a, info_a = banxico(('2017-01-01', '2019-12-01'), Y19)
    bx19_b, tr19_b, info_b = banxico(('2015-01-01', '2018-12-01'), Y19)
    bx18, _, _ = banxico(('2017-01-01', '2019-12-01'), Y18)
    bx18_r, tr18_r, info_r = banxico(('2015-01-01', '2017-12-01'), Y18)
    bxm = pd.DataFrame({'banxico_vs_trend_2017_19': (bx19_a / tr19_a).values,
                        'banxico_vs_trend_2015_18': (bx19_b / tr19_b).values,
                        'banxico_raw_2019_over_2018': bx19_a.values / bx18.values}, index=Y19)
    bxq = pd.DataFrame({'banxico_vs_trend_2017_19': q(bx19_a) / q(tr19_a),
                        'banxico_vs_trend_2015_18': q(bx19_b) / q(tr19_b),
                        'banxico_raw_2019_over_2018': q(bx19_a).values / q(bx18).values})
    bxq_rev = pd.DataFrame({'banxico_vs_trend_2015_17': q(bx18_r) / q(tr18_r),
                            'banxico_raw_2018_over_2019': q(bx18).values / q(bx19_a).values})
    bxm_rev = pd.DataFrame({'banxico_vs_trend_2015_17': (bx18_r / tr18_r).values,
                            'banxico_raw_2018_over_2019': bx18.values / bx19_a.values}, index=Y18)

    fwd, fits_f = run(pop, y, ok, Y18, Y19, Y18, 'forward: fit 2018, run 2019 / 2018')
    rev, fits_r = run(pop, y, ok, Y19, Y18, Y19, 'reverse: fit 2019, run 2018 / 2019')

    monthly, quarterly, summary = [], [], []
    for frames, bq, bm, qs in [(fwd, bxq, bxm, [f'2019Q{i}' for i in range(1, 5)]),
                               (rev, bxq_rev, bxm_rev, [f'2018Q{i}' for i in range(1, 5)])]:
        for d in frames:
            qi, ql = d.attrs['q']
            m = d.set_index('date').join(bm)
            monthly.append(m.reset_index())
            qq = pd.DataFrame({'direction': d.direction.iloc[0], 'kappa': d.kappa.iloc[0], 'quarter': qi.index,
                               'index_model': qi.values, 'index_model_lower_bound': ql.values}).join(
                bq.reset_index(drop=True))
            quarterly.append(qq)
            row = dict(direction=d.direction.iloc[0], kappa=float(d.kappa.iloc[0]),
                       **{f'index_{k}': float(qi[k]) for k in qs},
                       mean_index_model=float(qi.mean()), max_abs_dev_monthly=float(np.abs(d.index_model - 1).max()),
                       rmse_log_vs_1_quarterly=rmse(qi), rmse_log_vs_1_monthly=rmse(d.index_model),
                       rmse_log_vs_1_quarterly_lower_bound=rmse(ql))
            for c in bq.columns:
                row[f'rmse_log_vs_{c}_quarterly'] = rmse(qi.values, bq[c].values)
                row[f'rmse_1_vs_{c}_quarterly'] = rmse(bq[c].values)   # the 'no change' benchmark
            summary.append(row)
    monthly = pd.concat(monthly, ignore_index=True)
    quarterly = pd.concat(quarterly, ignore_index=True)
    summary = pd.DataFrame(summary)

    # CPS employment changes 2018 -> 2019 by occupation (rate_model, annual mean)
    rates = abm.cps_rates()
    cps = pd.DataFrame({'rate_2018': rates.loc[Y18].mean(), 'rate_2019': rates.loc[Y19].mean()})
    cps['change_pp'] = 100 * (cps.rate_2019 - cps.rate_2018)
    cps['ratio_2019_over_2018'] = cps.rate_2019 / cps.rate_2018
    mdiff = rates.loc[Y19].to_numpy() / rates.loc[Y18].to_numpy()
    cps['monthly_ratio_min'], cps['monthly_ratio_max'] = mdiff.min(0), mdiff.max(0)
    cps['used_by_model'] = cps.index.isin(pop['rate_columns'])
    occ = pd.Series(pop['job_index']).map(dict(enumerate(pop['rate_columns']))).value_counts(normalize=True)
    cps['migrant_share'] = occ.reindex(cps.index).fillna(0.0)

    summary.to_csv(OUT / 'summary.csv', index=False)
    quarterly.to_csv(OUT / 'quarterly_paths.csv', index=False)
    monthly.to_csv(OUT / 'monthly_paths.csv', index=False)
    cps.rename_axis('job').reset_index().to_csv(OUT / 'cps_changes_2018_2019.csv', index=False)
    pd.DataFrame(fits_f + fits_r).to_csv(OUT / 'fits.csv', index=False)
    pd.set_option('display.width', 250)
    print(summary.round(4).T.to_string())
    print(quarterly.round(4).to_string(index=False))
    print(cps.round(4).to_string())
    print(pd.DataFrame(fits_f + fits_r).round(3).to_string(index=False))

    settings = dict(task='6e out_of_sample_2019', kappas=KAPPAS, households=int(ok.sum()), receiving=int(y[ok].sum()),
                    forward='coefficients by ML with 2018 CPS employment (MMP 2010-19 receipt targets); index = '
                            'run with 2019 CPS / run with same calendar month of 2018; beta = 0, N* = 0, no stimulus',
                    reverse='coefficients with 2019 CPS (base); index = run 2018 / same month 2019',
                    count_mapping='base: sum_h P_h,t (beta = 0); lower bound: sum_h P_h,t r_h,t',
                    banxico=dict(a=dict(note='2019 in trend window: ~1 by '
                                        'construction (calendar dummies + log-linear trend)', **info_a),
                                 b=dict(note='extrapolated to 2019', **info_b),
                                 c='raw 2019 / same month 2018',
                                 reverse=dict(note='extrapolated to 2018',
                                              **info_r)),
                    cps_source=str(abm.CPS_RATES.relative_to(abm.ROOT)), cps_column=abm.RATE_COLUMN,
                    cps_coverage='2017-01..2023-12 (2018 available, no rebuild needed)',
                    rate_columns=list(pop['rate_columns']))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=str))
    figure(quarterly, monthly)


def figure(quarterly, monthly):
    spec = importlib.util.spec_from_file_location('rl_make_figures', RL / 'make_figures.py')
    rmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rmf)
    mf = rmf.mf
    import matplotlib.pyplot as plt
    mf.style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    colors = {0.0: mf.BLUE, 0.5: mf.AQUA, 0.8: mf.ORANGE, 1.0: mf.INK2}
    for ax, key, bcols, title in [
            (axes[0], 'forward', [('banxico_vs_trend_2015_18', 'Banxico / 2015-18 trend'),
                                  ('banxico_raw_2019_over_2018', 'Banxico raw 2019 / 2018'),
                                  ('banxico_vs_trend_2017_19', 'Banxico / 2017-19 trend (in sample)')],
             'Fit 2018 CPS, index 2019 / 2018'),
            (axes[1], 'reverse', [('banxico_vs_trend_2015_17', 'Banxico / 2015-17 trend'),
                                  ('banxico_raw_2018_over_2019', 'Banxico raw 2018 / 2019')],
             'Fit 2019 CPS, index 2018 / 2019')]:
        m = monthly[monthly.direction.str.startswith(key)]
        for kappa, d in m.groupby('kappa'):
            ax.plot(d.date, d.index_model, color=colors[kappa], lw=1.8, label=f'Model, kappa {kappa:g}')
        d0 = m[m.kappa == 0]
        for (c, lab), ls in zip(bcols, ['-', '--', ':']):
            ax.plot(d0.date, d0[c], color=mf.MUTED, lw=1.2, ls=ls, label=lab)
        ax.axhspan(0.99, 1.01, color=mf.GRID, alpha=0.6, lw=0)
        ax.axhline(1, color=mf.AXIS, lw=0.8)
        ax.set_xlim(d0.date.min(), d0.date.max())
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=7, loc='lower left')
    axes[0].set_ylabel('Index (monthly)')
    fig.suptitle('Normal-times placebo: model receipt index vs Banxico, 2018-19', x=0.01, ha='left',
                 fontsize=12, fontweight='semibold', color=mf.INK)
    fig.savefig(OUT / 'out_of_sample_2019.png', dpi=mf.DPI, bbox_inches='tight')


if __name__ == '__main__':
    main()
