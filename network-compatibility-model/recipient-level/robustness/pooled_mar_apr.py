"""Robustness 6d (29 Oct 2026): the March 2020 rush at the weak peso pulled April transfers forward
(Banxico vs trend: Mar 1.087, Apr 0.912, Mar+Apr 1.000). The base criterion's 2020Q2 contains the
depressed April. Refit beta per kappa under alternative criteria:
  base        quarters 2020Q2..2021Q4 (reproduces outputs/household-model/need_fit.csv)
  pooled      [Mar-Apr 2020], [May-Jun 2020], 2020Q3..2021Q4 (8 periods); index = sum over the
              period of the model (Banxico) / sum of the no-COVID run (trend)
  q2_apr_avg  2020Q2 with April := (Mar + Apr) / 2 in both numerator and denominator (Banxico and
              trend, model COVID and no-COVID runs), then 2020Q3..2021Q4
  drop_q2     2020Q3..2021Q4 only
2022 held out (quarters) throughout. Uses run_household_model unchanged (imported).
Run from the repo root: python network-compatibility-model/recipient-level/robustness/pooled_mar_apr.py
Outputs (aggregates only): outputs/robustness/pooled_mar_apr/.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent / 'model'))
import run_household_model as hm
import run_e1_job_loss as e1

TASK = 'pooled_mar_apr'
OUT = RL / 'outputs/robustness' / TASK
REPORT_KAPPAS = [0.0, 0.5, 0.8, 1.0]
DATES = hm.DATES
Q = lambda a, b: [str(p) for p in pd.period_range(a, b, freq='Q')]


def months(a, b):
    return list(pd.date_range(a, b, freq='MS'))


def quarter_periods(qs):
    return {q: list(pd.date_range(pd.Period(q).start_time, periods=3, freq='MS')) for q in qs}


PERIODS = {
    'base': quarter_periods(Q('2020Q2', '2021Q4')),
    'pooled': {'2020-03..04': months('2020-03-01', '2020-04-01'), '2020-05..06': months('2020-05-01', '2020-06-01'),
               **quarter_periods(Q('2020Q3', '2021Q4'))},
    'q2_apr_avg': quarter_periods(Q('2020Q2', '2021Q4')),
    'drop_q2': quarter_periods(Q('2020Q3', '2021Q4')),
}
HOLD = quarter_periods(hm.HOLD_Q)
REPORT_PERIODS = {**PERIODS['pooled'], **HOLD}


def adjust(s, variant):
    """Monthly level series; q2_apr_avg replaces April 2020 by the Mar-Apr 2020 average."""
    if variant != 'q2_apr_avg':
        return s
    s = s.copy()
    s[pd.Timestamp('2020-04-01')] = 0.5 * (s[pd.Timestamp('2020-03-01')] + s[pd.Timestamp('2020-04-01')])
    return s


def period_index(num, den, periods, variant='base'):
    num, den = adjust(num, variant), adjust(den, variant)
    return pd.Series({k: num[m].sum() / den[m].sum() for k, m in periods.items()})


def rmse(model, data):
    return float(np.sqrt(np.mean(np.log(model / data[model.index]) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hm.hl.load_population()
    h = pop['h']
    ok = hm.e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    base = hm.Cells(pop, hm.abm.make_inputs(hm.abm.baseline_2019_dates(), pop['rate_columns']))
    x_by_kappa, start = {}, None
    for kappa in hm.KAPPAS:
        x, *_ = hm.fit_2019(base, y, ok, kappa, start)
        start = x_by_kappa[kappa] = x
    del base
    print('E0 done', flush=True)

    target_q, _ = hm.banxico_target()
    bx_m, trend_m, _ = e1.banxico()
    bx_m, trend_m = bx_m.reindex(DATES), trend_m.reindex(DATES)
    spike_bx = hm.spikes(bx_m / trend_m)
    covid = hm.abm.make_inputs(DATES, pop['rate_columns'])
    cf = hm.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in DATES]), pop['rate_columns'])
    cc, cfc = hm.Cells(pop, covid), hm.Cells(pop, cf)
    snd_c, snd_f = hm.Senders(pop, covid, hm.need_stock(DATES)), hm.Senders(pop, cf, np.zeros(len(DATES)))
    r_c0, r_f = snd_c.r(0.0), snd_f.r(0.0)
    a_core = hm.m1_alphas()[0]
    targets = {v: period_index(bx_m, trend_m, p, v) for v, p in PERIODS.items()}
    target_report = period_index(bx_m, trend_m, REPORT_PERIODS)
    target_hold = period_index(bx_m, trend_m, HOLD)

    # check: monthly-sum period target equals hm.banxico_target on quarters
    chk = max(abs(targets['base'] - target_q[list(PERIODS['base'])]).max(), abs(target_hold - target_q[hm.HOLD_Q]).max())
    assert chk < 1e-12, chk

    rows, paths, verify = [], [], []
    for kappa in hm.KAPPAS:
        x = x_by_kappa[kappa]
        Pf, Pc = cfc.receipt(x, kappa), cc.receipt(x, kappa)
        cache = {}

        def series(beta):
            if beta not in cache:
                r_c = snd_c.r(beta)
                S = lambda z: pd.Series(z.sum(1), index=DATES)
                cache[beta] = (S(Pc * r_c / r_c0), S(Pf), S(Pc * r_c), S(Pf * r_f))
            return cache[beta]

        # exact base replication path (as hm.main: quarterly sums, hm.log_rmse)
        base_q = lambda b: hm.quarterly(series(b)[0]) / hm.quarterly(series(b)[1])
        res0 = minimize_scalar(lambda b: hm.log_rmse(base_q(b), target_q, hm.FIT_Q), bounds=(0.0, hm.BETA_MAX),
                               method='bounded', options=dict(xatol=1e-4))
        q0 = base_q(float(res0.x))
        verify.append(dict(kappa=kappa, beta=float(res0.x), rmse_log_quarterly=float(res0.fun),
                           rmse_log_quarterly_holdout_2022=hm.log_rmse(q0, target_q, hm.HOLD_Q),
                           index_2020q2=float(q0['2020Q2']), index_2021q4=float(q0['2021Q4']),
                           mean_index_2022=float(q0[hm.HOLD_Q].mean()),
                           theta_core_2019=float(expit(a_core)), theta_core_2021_12=float(expit(a_core + res0.x))))

        for v, per in PERIODS.items():
            obj = lambda b: rmse(period_index(*series(b)[:2], per, v), targets[v])
            res = minimize_scalar(obj, bounds=(0.0, hm.BETA_MAX), method='bounded', options=dict(xatol=1e-4))
            b = float(res.x)
            c, f, tc, tf = series(b)
            obj_lb = lambda bb: rmse(period_index(*series(bb)[2:], per, v), targets[v])
            res_lb = minimize_scalar(obj_lb, bounds=(0.0, hm.BETA_MAX), method='bounded', options=dict(xatol=1e-4))
            rep = period_index(c, f, REPORT_PERIODS)
            hold = period_index(c, f, HOLD)
            sp = hm.spikes(c / f)
            q2 = period_index(c, f, PERIODS['q2_apr_avg'], v)['2020Q2'] if v != 'drop_q2' else np.nan
            rows.append(dict(
                variant=v, kappa=kappa, beta=b, rmse_fit=float(res.fun), n_periods=len(per),
                rmse_holdout_2022=rmse(hold, target_hold),
                index_2020q2=float(hm.quarterly(c)['2020Q2'] / hm.quarterly(f)['2020Q2']),
                index_2020q2_criterion=float(q2),
                index_mar_apr_2020=float(rep['2020-03..04']), index_may_jun_2020=float(rep['2020-05..06']),
                index_2021q4=float(rep['2021Q4']), mean_index_2022=float(hold.mean()),
                spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                spike_rmse=float(np.sqrt(np.mean([(sp[m] - spike_bx[m]) ** 2 for m in hm.SPIKE_MONTHS]))),
                rmse_lower_bound_at_beta=rmse(period_index(tc, tf, per, v), targets[v]),
                beta_lower_bound=float(res_lb.x), rmse_lower_bound_refit=float(res_lb.fun)))
            paths.append(pd.DataFrame({'variant': v, 'kappa': kappa, 'beta': b, 'period': list(rep.index),
                                       'in_criterion': [p in per for p in rep.index],
                                       'index_model': rep.values, 'index_banxico': target_report.values}))
        print(f'kappa {kappa}: done', flush=True)

    # verification against the base need_fit.csv
    ref = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')
    ver = pd.DataFrame(verify)
    max_diff = float((ver[ref.columns] - ref).abs().to_numpy().max())
    print('max abs diff vs base need_fit.csv:', max_diff)

    grid = pd.DataFrame(rows)
    base_cols = ['beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
                 'spike_2021_01', 'spike_2021_03', 'spike_rmse', 'rmse_lower_bound_at_beta']
    basev = grid[grid.variant == 'base'].set_index('kappa')[base_cols].add_prefix('base_')
    grid = grid.join(basev, on='kappa')
    best = {v: dict(kappa_min_rmse_fit=float(d.loc[d.rmse_fit.idxmin(), 'kappa']),
                    kappa_min_spike_rmse=float(d.loc[d.spike_rmse.idxmin(), 'kappa']),
                    kappa_min_rmse_lower_bound=float(d.loc[d.rmse_lower_bound_refit.idxmin(), 'kappa']))
            for v, d in grid.groupby('variant', sort=False)}
    grid['best_kappa_rmse_fit'] = grid.variant.map(lambda v: best[v]['kappa_min_rmse_fit'])
    grid['best_kappa_spike_rmse'] = grid.variant.map(lambda v: best[v]['kappa_min_spike_rmse'])
    grid['is_best_kappa'] = grid.kappa == grid.best_kappa_rmse_fit
    grid.to_csv(OUT / 'all_kappas.csv', index=False)
    summary = grid[grid.kappa.isin(REPORT_KAPPAS) | grid.is_best_kappa]
    summary.to_csv(OUT / 'summary.csv', index=False)
    paths = pd.concat(paths, ignore_index=True)
    paths.to_csv(OUT / 'quarterly_paths.csv', index=False)
    pd.set_option('display.width', 250)
    print(summary.drop(columns=[c for c in summary if c.startswith('base_')]).round(4).to_string(index=False))
    print(json.dumps(best, indent=1))
    print('Banxico periods:', target_report.round(4).to_dict())

    settings = dict(
        task=TASK, experiment='6d: April 2020 timing shift; refit beta per kappa under alternative criteria',
        periods={v: list(p) for v, p in PERIODS.items()}, holdout=hm.HOLD_Q,
        q2_apr_avg='April 2020 level := (Mar + Apr 2020)/2 in Banxico, trend, model COVID and no-COVID series',
        index='period sum of the model count series / no-COVID sum; Banxico period sum / trend period sum',
        count_mapping='base (hm): sum P r(beta)/r(0); lower bound: sum P r(beta) (rmse at the fitted beta and refit)',
        beta_bounds=[0.0, hm.BETA_MAX], kappas=hm.KAPPAS, spike_banxico=spike_bx,
        banxico_period_index=target_report.to_dict(),
        verification=dict(reference='outputs/household-model/need_fit.csv', max_abs_diff=max_diff),
        best=best)
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=float))
    figure(grid, paths, target_report)


def figure(grid, paths, target):
    spec = importlib.util.spec_from_file_location('rl_make_figures', RL / 'make_figures.py')
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    mf, plt = rf.mf, rf.plt
    colors = {'base': mf.INK2, 'pooled': mf.BLUE, 'q2_apr_avg': mf.ORANGE, 'drop_q2': mf.AQUA}
    labels = {'base': 'Base (2020Q2..2021Q4)', 'pooled': 'Mar–Apr pooled', 'q2_apr_avg': 'Q2 with Apr := Mar–Apr avg',
              'drop_q2': 'Drop 2020Q2'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for v, d in grid.groupby('variant', sort=False):
        ax.plot(d.kappa, d.rmse_fit, color=colors[v], label=labels[v], marker='o', ms=3)
    ax.set_xlabel('κ (replacement share)')
    ax.set_ylabel('Log RMSE at fitted β (own criterion)')
    ax.legend(loc='upper right')
    ax = axes[1]
    per = list(target.index)
    xs = np.arange(len(per))
    for k, ls in [(0.8, '--'), (1.0, '-')]:
        d = paths[(paths.variant == 'pooled') & (paths.kappa == k)].set_index('period').loc[per]
        ax.plot(xs, d.index_model, color=mf.BLUE, ls=ls, label=f'Model, pooled fit, κ = {k:g}')
    ax.plot(xs, target.values, color=mf.INK, label='Banxico vs 2017–19 trend')
    ax.axhline(1, color=mf.AXIS, lw=1)
    ax.set_xticks(xs, per, rotation=60, fontsize=7.5)
    ax.set_ylabel('Relative to no-COVID path')
    ax.legend(loc='upper left')
    fig.suptitle('Pooling March–April 2020 in the criterion', x=0.01, ha='left', fontsize=12.5,
                 fontweight='semibold', color=mf.INK)
    fig.tight_layout()
    fig.savefig(OUT / f'{TASK}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
