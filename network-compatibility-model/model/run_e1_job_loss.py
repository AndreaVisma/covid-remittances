"""E1a: US job loss on the real networks only (model-specification.md §6). Library.

Shared by the migrant-level comparison (run_e1_coordination.py: run_experiment) and the household
model (banxico, the target). The stand-alone E1 run under the lost-sending rule was removed in the 29 Oct 2026 cleanup (git history up to a83b60c).

beta = 0 (no need response), rho = 0, pi = 0 (no benefits; E1b adds them once
the benefit schedule, task 3, exists). For each kappa on the E0 grid, with the
E0 alphas for that kappa:
  * exact expected monthly transfers 2019-2022 under observed CPS employment
    and under the no-COVID counterfactual (2019 rates continued);
  * count index, monthly and quarterly: model = COVID run / no-COVID run (equal to the
    same period of 2019), Banxico = transactions / their 2017-19 trend (log-linear with
    month effects, +6.3% a year); the raw Banxico index (same period of 2019) is kept as
    index_banxico_raw. Log-index RMSE over 2020-04..2021-12 (2020Q2..2021Q4), reported
    only, since tau_C is not fixed yet; 2022 is held out;
  * household outcomes by network type, size and link type from paired
    stochastic runs (common random numbers): interruptions, drop in USD
    received, replacement by kin, and recovery month of group receipt.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
from run_e0_baseline import household_cells

DATES = pd.date_range('2019-01-01', '2022-12-01', freq='MS')
FIT = ('2020-04-01', '2021-12-01')           # March 2020 excluded (salience had not moved)
FIT_Q = ('2020Q2', '2021Q4')
HOLDOUT = ('2022-01-01', '2022-12-01')
TREND_WINDOW = ('2017-01-01', '2019-12-01')
SEEDS = range(101, 151)
AMOUNT_SEED = 7
RECOVERY_GAP = 0.01                          # group receipt within 1% of counterfactual
GROUPS = ['network_type', 'size_bin', 'links']


def counterfactual_dates():
    """Each month mapped to the same calendar month of 2019."""
    return pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in DATES])


def banxico():
    """Banxico transactions and their pre-COVID path: log-linear trend plus calendar-month effects
    fitted on 2017-2019, projected forward (user decision 20 Oct 2026: compare against the trend)."""
    bx = pd.read_parquet(abm.BANXICO)
    bx['date'] = pd.to_datetime(bx.date)
    t = bx.set_index('date').transactions
    design = lambda idx: np.column_stack([(idx.year - 2019) * 12 + idx.month]
                                         + [(idx.month == m).astype(float) for m in range(1, 13)])
    pre = t.loc[TREND_WINDOW[0]:TREND_WINDOW[1]]
    coef, *_ = np.linalg.lstsq(design(pre.index), np.log(pre.to_numpy()), rcond=None)
    trend = pd.Series(np.exp(design(DATES) @ coef), index=DATES)
    resid = np.log(pre.to_numpy()) - design(pre.index) @ coef
    info = dict(trend_window=list(TREND_WINDOW), trend_growth_per_year=float(np.expm1(12 * coef[0])),
                residual_sd_monthly=float(resid.std(ddof=13)))
    return t.loc[DATES], trend, info


def same_period_2019(series, freq):
    """Index relative to the same month (freq 'M') or quarter ('Q') of 2019."""
    if freq == 'Q':
        series = series.groupby(series.index.to_period('Q')).sum()
        base = series[series.index.year == 2019]
        return series / base.set_axis(base.index.quarter).loc[series.index.quarter].to_numpy()
    base = series[series.index.year == 2019]
    return series / base.set_axis(base.index.month).loc[series.index.month].to_numpy()


def log_rmse(model, data, window):
    w = slice(*window)
    return float(np.sqrt(np.mean(np.log(model[w] / data[w]) ** 2)))


def aggregate(pop, covid, cf, params, amounts, bx, trend):
    f = abm.expected_flows(pop, covid, params, amounts)
    g = abm.expected_flows(pop, cf, params, amounts).set_axis(f.index)
    monthly = f.add_suffix('_covid').join(g.add_suffix('_cf'))
    monthly['banxico_transactions'] = bx
    monthly['banxico_trend'] = trend
    for k, s in [('model', f.transfers), ('cf', g.transfers), ('banxico_raw', bx)]:
        monthly[f'index_{k}'] = same_period_2019(s, 'M')
    monthly['index_banxico'] = bx / trend                      # target: Banxico relative to its 2017-19 trend
    monthly['excess_over_cf'] = f.transfers / g.transfers
    monthly['mean_transfer_usd'] = f.usd / f.transfers
    q = lambda s: s.groupby(s.index.to_period('Q')).sum()
    quarterly = pd.DataFrame({'transfers_covid': q(f.transfers), 'transfers_cf': q(g.transfers),
                              'kin_transfers_covid': q(f.kin_transfers), 'kin_transfers_cf': q(g.kin_transfers),
                              'banxico_transactions': q(bx), 'banxico_trend': q(trend)})
    for k, s in [('model', f.transfers), ('cf', g.transfers), ('banxico_raw', bx)]:
        quarterly[f'index_{k}'] = same_period_2019(s, 'Q')
    quarterly['index_banxico'] = quarterly.banxico_transactions / quarterly.banxico_trend
    quarterly['excess_over_cf'] = quarterly.transfers_covid / quarterly.transfers_cf
    fit = dict(rmse_log_monthly=log_rmse(monthly.index_model, monthly.index_banxico, FIT),
               rmse_log_quarterly=log_rmse(quarterly.index_model, quarterly.index_banxico, FIT_Q),
               rmse_log_monthly_holdout_2022=log_rmse(monthly.index_model, monthly.index_banxico, HOLDOUT),
               rmse_log_quarterly_raw=log_rmse(quarterly.index_model, quarterly.index_banxico_raw, FIT_Q),
               min_index_model=float(monthly.index_model.min()),
               min_index_month=str(monthly.index_model.idxmin().date()),
               mean_index_model_2020=float(monthly.index_model['2020'].mean()),
               mean_index_model_2021=float(monthly.index_model['2021'].mean()),
               mean_index_banxico_2020=float(monthly.index_banxico['2020'].mean()),
               mean_index_banxico_2021=float(monthly.index_banxico['2021'].mean()),
               kin_share_of_lost_core_2020_04=float(
                   (f.kin_transfers - g.kin_transfers)['2020-04'].sum()
                   / (g.core_transfers - f.core_transfers)['2020-04'].sum()))
    return monthly.reset_index(), quarterly.reset_index().rename(columns={'index': 'quarter'}), fit


def recovery_months(pop, covid, cf, params, cells):
    """Group-level expected receipt (exact): ratio to counterfactual by month and
    the first month after April 2020 from which it stays within RECOVERY_GAP."""
    p = pd.DataFrame(abm.expected_receipt(pop, covid, params), index=DATES)
    p_cf = pd.DataFrame(abm.expected_receipt(pop, cf, params), index=DATES)
    rows, paths = [], []
    for g in GROUPS:
        for level, idx in cells.groupby(g, observed=True).groups.items():
            ratio = p[idx].mean(axis=1) / p_cf[idx].mean(axis=1)
            crisis = ratio['2020-04':'2021-12']
            below = crisis[crisis < 1 - RECOVERY_GAP]
            recovered = None if below.empty else (crisis.index[crisis.index > below.index[-1]].min())
            rows.append(dict(group=g, level=level, households=len(idx),
                             trough_ratio=float(crisis.min()), trough_month=str(crisis.idxmin().date()),
                             recovery_month=None if recovered is pd.NaT or recovered is None else str(recovered.date()),
                             months_below=int(len(below))))
            paths.append(ratio.rename('receipt_ratio').reset_index().rename(columns={'index': 'date'})
                         .assign(group=g, level=level))
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True)


def household_outcomes(pop, covid, cf, params, cells):
    """Paired stochastic runs over 2020-04..2021-12; group totals summed over seeds."""
    crisis = (DATES >= FIT[0]) & (DATES <= FIT[1])
    has_core = pop['has_core']
    acc = {}
    checks = []
    for seed in SEEDS:
        a = abm.simulate(pop, covid, params, seed, amount_seed=AMOUNT_SEED, keep_households=True)
        b = abm.simulate(pop, cf, params, seed, amount_seed=AMOUNT_SEED, keep_households=True)
        x, y = a['household'][crisis], b['household'][crisis]
        checks.append(a['summary'].transfers.to_numpy())
        rec, rec_cf = x[:, :, 0] > 0, y[:, :, 0] > 0
        kin_only_month = lambda z: (z[:, :, 0] > z[:, :, 2]) & (z[:, :, 2] == 0)
        per_h = dict(cf_receipt_months=rec_cf.sum(0), interrupted_months=(rec_cf & ~rec).sum(0),
                     usd=x[:, :, 1].sum(0), usd_cf=y[:, :, 1].sum(0),
                     kin_only_months=np.where(has_core, kin_only_month(x).sum(0), 0),
                     kin_only_months_cf=np.where(has_core, kin_only_month(y).sum(0), 0),
                     core_household_months=np.where(has_core, crisis.sum(), 0),
                     ever_interrupted=(rec_cf & ~rec).any(0).astype(float))
        for k, v in per_h.items():
            acc[k] = acc.get(k, 0) + v
    hh = pd.DataFrame(acc).join(cells[GROUPS])
    rows = []
    for g in GROUPS:
        t = hh.groupby(g, observed=True).sum(numeric_only=True)
        n = hh.groupby(g, observed=True).size()
        rows.append(pd.DataFrame({
            'group': g, 'level': t.index.astype(str), 'households': n.to_numpy(),
            'interruption_rate': t.interrupted_months / t.cf_receipt_months,
            'share_ever_interrupted': t.ever_interrupted / (n * len(SEEDS)),
            'usd_drop': 1 - t.usd / t.usd_cf,
            'kin_only_month_share': t.kin_only_months / t.core_household_months.replace(0, np.nan),
            'kin_only_month_share_cf': t.kin_only_months_cf / t.core_household_months.replace(0, np.nan)}))
    return pd.concat(rows, ignore_index=True), np.mean(checks, axis=0)


def run_experiment(fitted, out, rule, settings_extra=None):
    """E1a for each fitted row (dicts with kappa, alpha_core, alpha_kin) under a stepping-in rule."""
    OUT = out                                   # local: the output folder of this run
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    cells = household_cells(pop['h'])
    covid = abm.make_inputs(DATES, pop['rate_columns'])
    cf = abm.make_inputs(counterfactual_dates(), pop['rate_columns'])
    bx, bx_trend, bx_info = banxico()
    monthly, quarterly, fits, recovery, paths, outcomes, checks = [], [], [], [], [], [], []
    for row in fitted:
        params = abm.Params(row['alpha_core'], row['alpha_kin'], kappa=row['kappa'], rule=rule)
        amounts = abm.routine_amounts(pop, params, AMOUNT_SEED)
        m, q, fit = aggregate(pop, covid, cf, params, amounts, bx, bx_trend)
        r, p = recovery_months(pop, covid, cf, params, cells)
        o, sim_transfers = household_outcomes(pop, covid, cf, params, cells)
        kappa = dict(kappa=params.kappa)
        monthly.append(m.assign(**kappa))
        quarterly.append(q.assign(**kappa))
        fits.append(dict(**kappa, alpha_core=params.alpha_core, alpha_kin=params.alpha_kin, **fit))
        recovery.append(r.assign(**kappa))
        paths.append(p.assign(**kappa))
        outcomes.append(o.assign(**kappa))
        checks.append(dict(**kappa, max_abs_gap_transfers=float(np.max(np.abs(sim_transfers - m.transfers_covid))),
                           mean_analytic=float(m.transfers_covid.mean()), mean_simulated=float(sim_transfers.mean())))
        print(f"kappa {params.kappa}: done")
    tables = dict(aggregate_monthly=monthly, aggregate_quarterly=quarterly, fit_summary=fits,
                  receipt_recovery=recovery, receipt_ratio_paths=paths, household_outcomes=outcomes,
                  stochastic_check=checks)
    tables = {k: pd.concat(v, ignore_index=True) if isinstance(v[0], pd.DataFrame) else pd.DataFrame(v)
              for k, v in tables.items()}
    for k, v in tables.items():
        v.to_csv(OUT / f'{k}.csv', index=False)

    settings = dict(experiment='E1a: job loss only', beta=0.0, rho=0.0, pi=0.0, stepping_in_rule=rule,
                    kappas=[r['kappa'] for r in fitted], **(settings_extra or {}),
                    dates=[str(DATES[0].date()), str(DATES[-1].date())],
                    counterfactual='2019 CPS rates of the same calendar month, common random numbers',
                    index='model: month (quarter) / same month (quarter) 2019 = COVID run / no-COVID run; '
                          'Banxico: transactions / 2017-19 log-linear trend with month effects (index_banxico); '
                          'index_banxico_raw: same period of 2019',
                    banxico_trend=bx_info,
                    fit_window=list(FIT), fit_window_quarterly=list(FIT_Q), holdout=list(HOLDOUT),
                    tau_C='not fixed; RMSE reported only', seeds=[SEEDS.start, SEEDS.stop - 1],
                    amount_seed=AMOUNT_SEED, recovery_gap=RECOVERY_GAP, phi=abm.employment_phi(),
                    cps_rate_column=abm.RATE_COLUMN,
                    cps_population=json.loads(abm.CPS_SETTINGS.read_text())['population'])
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))

    pd.set_option('display.width', 220)
    print(tables['fit_summary'].round(3).to_string(index=False))
    print(tables['stochastic_check'].round(2).to_string(index=False))
    q = tables['aggregate_quarterly']
    print(q.pivot(index='quarter', columns='kappa', values='index_model').join(
        q.loc[q.kappa.eq(0)].set_index('quarter')[['index_banxico', 'index_cf']]).round(3).to_string())
    print(tables['household_outcomes'].loc[lambda d: d.kappa.isin([0, 4])].round(3).to_string(index=False))
    print(tables['receipt_recovery'].loc[lambda d: d.kappa.isin([0, 4])].to_string(index=False))
    return tables

