"""One core + three kin per household; paired allocation with real CPS inputs.

Run from any directory: python path/to/remittance-toy-core-kin-cps.py
Reuses the empirical ABM equations without running its calibration or exports.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'oct-26/outputs/remittance-toy-core-kin-cps'
MODEL = Path(__file__).with_name('remittance-minimal-cps-abm.py')
WORKBOOK = ROOT / 'oct-26/data-prep/outputs/cps_earnings_employment.xlsx'
PARAMETERS = ROOT / 'oct-26/outputs/remittance-minimal-cps-abm/run_settings.json'
BANXICO = ROOT / 'oct-26/data-prep/outputs/mexico-rem-monthly-total.parquet'
HOUSEHOLDS = 1000
JOB = 'Production/construction/transport'
POPULATION = 'Mexican origin'
SEEDS = [2026, 2027, 2028]
WARMUP = 36
EARNINGS_SEED = 1731
# Identical origin household features: worker present, no children/older people.
FEATURES = [0., 0., 0.]


def load_model():
    spec = importlib.util.spec_from_file_location('empirical_remittance_abm', MODEL)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def prepare(model, settings):
    with pd.ExcelFile(WORKBOOK) as book:
        baseline = pd.read_excel(book, sheet_name='baseline_2019')
        earnings = pd.read_excel(book, sheet_name='monthly_earnings')
        employment = pd.read_excel(book, sheet_name='monthly_employment')
    def select(frame):
        return frame.loc[frame.population.eq(POPULATION) & frame.job_group.eq(JOB)].copy()
    baseline = select(baseline).loc[lambda f: f.year.eq(2019)].set_index('job_group')
    assert len(baseline) == 1
    monthly = select(earnings).merge(select(employment),
        on=['population', 'year', 'month', 'job_group'], validate='one_to_one',
        suffixes=('', '_employment'))
    monthly['date'] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    monthly = monthly.set_index(['date', 'job_group']).sort_index()
    history_dates = pd.date_range('2020-01-01', '2022-12-01', freq='MS')
    selected = monthly.loc[(history_dates, [JOB]), :].reset_index()
    assert len(selected) == 36
    assert np.isfinite(selected.monthly_mean_usd).all()
    assert selected.monthly_mean_usd.gt(0).all()
    assert selected.unemployment_rate.between(0, 1).all()
    assert baseline.earnings_reliable.all() and baseline.unemployment_reliable.all()
    M = 4 * HOUSEHOLDS
    # Same occupation, with the empirical model's persistent within-job dispersion.
    sigma = np.sqrt(2 * np.log(float(baseline.monthly_mean_usd.iloc[0]
                                    / baseline.monthly_median_usd.iloc[0])))
    multiplier = np.random.default_rng(EARNINGS_SEED).lognormal(-sigma**2 / 2, sigma, M)
    multiplier /= multiplier.mean()
    migrants = pd.DataFrame(dict(synthetic_migrant_id=np.arange(M),
        synthetic_household_id=np.repeat(np.arange(HOUSEHOLDS), 4),
        relationship=np.tile(['Core', 'Kin', 'Kin', 'Kin'], HOUSEHOLDS),
        job_group=JOB, earnings_multiplier=multiplier))
    d = dict(H=HOUSEHOLDS, M=M, hid=migrants.synthetic_household_id.to_numpy(),
        core=migrants.relationship.eq('Core').to_numpy(), adult=np.ones(M, bool),
        eligible=np.ones(M, bool), jindex=np.zeros(M, int), jobs=[JOB],
        m=migrants, cps_base=baseline, cps_monthly=monthly,
        x=np.tile(FEATURES, (HOUSEHOLDS, 1)), sizes=np.full(HOUSEHOLDS, 4),
        offsets=np.arange(HOUSEHOLDS)*4, masks=np.ones((1, HOUSEHOLDS))/HOUSEHOLDS)
    params = np.array([settings['results']['coordinated']['parameters'][k]
                       for k in model.PARAMETER_NAMES])
    # Match the saved empirical run, even if its source defaults later change.
    model.ANNUAL_LIVING_ALLOWANCE_USD = settings['settings']['annual_living_allowance_usd']
    model.MONTHLY_LIVING_ALLOWANCE_USD = model.ANNUAL_LIVING_ALLOWANCE_USD / 12
    model.REMITTABLE_SURPLUS_SHARE = settings['settings']['remittable_surplus_share']
    return d, params, history_dates, selected, baseline


def run():
    model = load_model()
    settings = json.loads(PARAMETERS.read_text(encoding='utf-8'))
    d, params, history_dates, inputs, baseline = prepare(model, settings)
    dates = pd.date_range('2019-01-01', '2022-12-01', freq='MS')
    T = WARMUP + len(dates)
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, annual, effects = [], [], []
    for seed in SEEDS:
        bank = model.make_bank(d, T, seed, history_dates)
        results = {}
        for mode in ['uncoordinated', 'coordinated']:
            r = model.simulate(params, d, bank, mode, WARMUP, collect=True)
            h = np.stack(r['household'])[WARMUP:]
            transfers = np.stack(r['transfers'])[WARMUP:]
            budgets = bank['capacity'][WARMUP:]
            np.testing.assert_allclose(transfers.reshape(len(dates), HOUSEHOLDS, 4).sum(2), h[:, :, 0], atol=1e-7)
            assert (transfers <= budgets + model.EPS).all()
            assert not ((transfers > model.EPS) & ~bank['employed'][WARMUP:]).any()
            if mode == 'coordinated':
                assert (h[:, :, 0] <= h[:, :, 2] + 1e-5).all()
            s = r['summary'].iloc[WARMUP:].copy()
            s['date'], s['mode'], s['seed'] = dates, mode, seed
            s['mean_household_usd'] = h[:, :, 0].mean(1)
            s['excess_usd'] = np.maximum(h[:, :, 0]-h[:, :, 2], 0).sum(1)
            s['unmet_usd'] = np.maximum(h[:, :, 2]-h[:, :, 0], 0).sum(1)
            summaries.append(s)
            results[mode] = (h, transfers, s)
            for year, g in s.groupby(s.date.dt.year):
                annual.append(dict(mode=mode, seed=seed, year=int(year),
                    total_usd=float(g.total_usd.sum()), receipt_rate=float(g.receipt_rate.mean()),
                    senders_per_month=float(g.senders.mean()), kin_senders_per_month=float(g.kin_senders.mean()),
                    mean_transfer_usd=float(g.total_usd.sum()/g.senders.sum()),
                    unmet_share=float(g.unmet_share.mean()),
                    excess_support_rate=float(g.excess_support_rate.mean())))
            if seed == SEEDS[0]:
                pd.DataFrame(dict(date=np.repeat(dates, HOUSEHOLDS),
                    household_id=np.tile(np.arange(HOUSEHOLDS), len(dates)),
                    received_usd=h[:, :, 0].ravel(), senders=h[:, :, 1].ravel().astype(int),
                    need_usd=h[:, :, 2].ravel())).to_parquet(OUT/f'{mode}_household_months.parquet', index=False)
                pd.DataFrame(dict(date=np.repeat(dates, d['M']),
                    migrant_id=np.tile(np.arange(d['M']), len(dates)),
                    household_id=np.tile(d['hid'], len(dates)),
                    relationship=np.tile(d['m'].relationship, len(dates)),
                    employed=bank['employed'][WARMUP:].ravel(), gross_earnings_usd=bank['gross_earnings'][WARMUP:].ravel(),
                    budget_usd=budgets.ravel(), transfer_usd=transfers.ravel())).to_parquet(OUT/f'{mode}_migrant_months.parquet', index=False)
        a, b = results['uncoordinated'], results['coordinated']
        # Within-month capping cannot change whether any willing positive-budget
        # contributor exists. Subsequent receipt can differ via sending persistence.
        for t, date in enumerate(dates):
            effects.append(dict(seed=seed, date=date,
                total_difference_usd=float(b[0][t, :, 0].sum()-a[0][t, :, 0].sum()),
                receipt_difference_pp=float(100*((b[0][t, :, 1]>0).mean()-(a[0][t, :, 1]>0).mean())),
                sender_difference=int((b[1][t]>model.EPS).sum()-(a[1][t]>model.EPS).sum())))
    monthly = pd.concat(summaries, ignore_index=True)
    yearly = pd.DataFrame(annual)
    yearly['growth_vs_2019_pct'] = yearly.groupby(['mode', 'seed']).total_usd.transform(lambda x: 100*(x/x.iloc[0]-1))
    monthly.to_csv(OUT/'monthly_runs.csv', index=False)
    yearly.to_csv(OUT/'annual_runs.csv', index=False)
    pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv', index=False)
    inputs.to_csv(OUT/'cps_history_inputs.csv', index=False)
    baseline.to_csv(OUT/'cps_2019_baseline.csv')
    d['m'].to_csv(OUT/'migrants.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for mode, color in [('uncoordinated', '#2465a0'), ('coordinated', '#b75d24')]:
        g = monthly.loc[monthly['mode'].eq(mode)].groupby('date')
        for ax, col, factor, title in zip(axes.flat,
            ['total_usd', 'receipt_rate', 'senders', 'unmet_share', 'excess_support_rate', 'mean_transfer_usd'],
            [1, 100, 1, 100, 100, 1],
            ['Total remittances (USD/month)', 'Households receiving (%)', 'Positive contributors/month',
             'Unmet perceived need (%)', 'Households above need (%)', 'Mean positive contribution (USD)']):
            mean, lo, hi = g[col].mean()*factor, g[col].min()*factor, g[col].max()*factor
            ax.plot(mean.index, mean, label=mode, color=color, ls='-' if mode=='uncoordinated' else '--')
            ax.fill_between(mean.index, lo, hi, color=color, alpha=.12)
            ax.axvline(pd.Timestamp('2020-03-01'), color='grey', lw=.8, ls=':')
            ax.set_title(title, fontsize=10)
    axes[0, 0].legend()
    fig.suptitle('One core + three kin: same theta parameters, need, budgets and random draws')
    fig.text(.02, .015, '2019: pooled baseline capacity. 2020–2022: observed CPS inputs. Shading: three seeds, not confidence intervals.', fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(OUT/'comparison.png', dpi=160); plt.close(fig)
    aggregate_figures(monthly)
    config = dict(households=HOUSEHOLDS, migrants=d['M'], core_per_household=1, kin_per_household=3,
        occupation=JOB, cps_population=POPULATION, warmup_months=WARMUP, dates=['2019-01', '2022-12'],
        historical_cps_dates=['2020-01', '2022-12'], seeds=SEEDS, earnings_seed=EARNINGS_SEED,
        household_features=dict(zip(['no_worker', 'child_share', 'old_share'], FEATURES)),
        parameters=dict(zip(model.PARAMETER_NAMES, params.tolist())),
        need_usd=float(np.exp(params[6]+np.array(FEATURES)@params[2:5])),
        annual_allowance_usd=model.ANNUAL_LIVING_ALLOWANCE_USD,
        surplus_share=model.REMITTABLE_SURPLUS_SHARE,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), MODEL, PARAMETERS, WORKBOOK, BANXICO]},
        low_quality_history_months=int((~(inputs.earnings_reliable & inputs.unemployment_reliable)).sum()),
        checks='Transfer/household reconciliation, budget ceilings, employment, coordinated coverage passed')
    (OUT/'run_settings.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    report(config, monthly, yearly)
    print(yearly.groupby(['mode', 'year'])[['total_usd', 'receipt_rate', 'senders_per_month', 'growth_vs_2019_pct']].mean().round(4).to_string())
    print(f'Checks passed. Outputs: {OUT}')


def aggregate_figures(monthly):
    """Compare shapes without expanding the toy sample to national dollar levels."""
    observed = pd.read_parquet(BANXICO)[['date', 'total_usd']].copy()
    observed['date'] = pd.to_datetime(observed.date).dt.to_period('M').dt.to_timestamp()
    observed = observed.loc[observed.date.between('2019-01-01', '2022-12-01')].sort_values('date')
    assert observed.date.is_unique and len(observed) == 48
    assert observed.total_usd.gt(0).all()
    observed['series'] = 'Observed Banxico'
    observed['seed'] = -1
    model = monthly[['date', 'mode', 'seed', 'total_usd']].rename(columns={'mode': 'series'}).copy()
    data = pd.concat([model, observed], ignore_index=True)
    data['year'] = data.date.dt.year
    baseline = data.loc[data.year.eq(2019)].groupby(['series', 'seed']).total_usd.mean()
    assert baseline.gt(0).all()
    data['baseline_2019_monthly_usd'] = [baseline.loc[(s, seed)] for s, seed in zip(data.series, data.seed)]
    data['index_2019_mean_100'] = 100*data.total_usd/data.baseline_2019_monthly_usd
    data.to_csv(OUT/'aggregate_monthly_comparison.csv', index=False)
    annual = data.groupby(['series', 'seed', 'year'], as_index=False).agg(total_usd=('total_usd', 'sum'), months=('date', 'size'))
    assert annual.months.eq(12).all()
    annual['growth_vs_2019_pct'] = annual.groupby(['series', 'seed']).total_usd.transform(lambda x: 100*(x/x.iloc[0]-1))
    annual.to_csv(OUT/'aggregate_annual_comparison.csv', index=False)
    styles = [('Observed Banxico', 'black', '-'), ('uncoordinated', '#2465a0', '-'), ('coordinated', '#b75d24', '--')]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, color, ls in styles:
        g = data.loc[data.series.eq(name)].groupby('date').index_2019_mean_100
        ax.plot(g.mean().index, g.mean(), label=name, color=color, ls=ls, lw=2)
        if name != 'Observed Banxico':
            ax.fill_between(g.mean().index, g.min(), g.max(), color=color, alpha=.12)
    ax.axhline(100, color='grey', ls=':', lw=1)
    ax.axvline(pd.Timestamp('2020-03-01'), color='grey', ls=':', lw=1)
    ax.set(title='Model and observed remittances relative to 2019', ylabel='Index: own 2019 monthly average = 100', xlabel='Month')
    ax.legend(); fig.autofmt_xdate()
    fig.text(.02, .015, 'Nominal USD; no seasonal adjustment. Each model seed uses its own baseline. Bands: three-seed range, not confidence intervals.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1)); fig.savefig(OUT/'aggregate_monthly_index.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, color, ls in styles:
        g = annual.loc[annual.series.eq(name)].groupby('year').growth_vs_2019_pct
        ax.plot(g.mean().index, g.mean(), marker='o', label=name, color=color, ls=ls, lw=2)
        if name != 'Observed Banxico':
            ax.fill_between(g.mean().index, g.min(), g.max(), color=color, alpha=.12)
        ax.annotate(f'{g.mean().iloc[-1]:.1f}%', (2022, g.mean().iloc[-1]), xytext=(6, 7 if name!='coordinated' else -15), textcoords='offset points', color=color)
    ax.axhline(0, color='grey', ls=':', lw=1)
    ax.set(title='Annual remittance growth relative to 2019', ylabel='Change from own 2019 annual total (%)', xlabel='Year', xticks=[2019, 2020, 2021, 2022], xlim=(2018.9, 2022.35))
    ax.legend()
    fig.text(.02, .015, 'Observed: total inbound remittances to Mexico, nominal USD. Toy: one occupation, fixed need; no national scaling or pandemic fit.', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, 1)); fig.savefig(OUT/'aggregate_annual_growth.png', dpi=160); plt.close(fig)


def report(config, monthly, yearly):
    lines = ['# Toy ABM: independent versus coordinated contributions', '',
        'Run with `python oct-26/simulations/abms/remittance-toy-core-kin-cps.py`.', '',
        '## Specification', '',
        f"{HOUSEHOLDS:,} identical origin households; one core and three kin per household, all adult, all in {JOB}. "
        'Origin households have a worker, no children and no older residents. '
        'Migrant earnings vary through fixed lognormal multipliers inferred from pooled 2019 CPS mean/median earnings, as in the empirical model.', '',
        'Uses the saved coordinated fit for BOTH rules, without recalibration. '
        'The six theta coefficients and monetary scale are copied from the empirical run; applying them to this toy population does not preserve its empirical calibration.', '',
        '`theta = core_or_kin_intercept + beta_no_worker*no_worker + beta_child*child_share + beta_old*old_share + rho*sent_last_month`', '',
        '`P(willing) = logistic(theta)`; `need = exp(log_need_scale + demographic_index)`.', '',
        f"Need is fixed at ${config['need_usd']:,.2f} per household-month. Capacity is "
        f"{100*config['surplus_share']:.0f}% of gross monthly earnings above ${config['annual_allowance_usd']/12:,.2f}. "
        'The living allowance is a scenario assumption, not a verified historical MIT estimate.', '',
        '**Independent (uncoordinated):** each willing migrant sends min(budget, full household need). '
        '**Coordinated:** previous positive senders contribute first, random order within incumbent/nonincumbent groups; '
        'each sends min(budget, remaining need). Core migrants do not automatically go first.', '',
        'Willingness does not respond directly to another contributor losing work. Coordination operates through allocation; '
        'actual positive sending updates persistence, so later willingness may differ across rules despite shared uniforms.', '',
        '## CPS inputs and pairing', '',
        'Prepared Mexican-origin CPS results only; no raw CPS microdata are read. '
        'A 36-month baseline warm-up is discarded. Retained 2019 uses constant pooled 2019 earnings/unemployment. '
        'January 2020–December 2022 uses observed occupation-month means and unemployment rates. '
        'Employment draws are independent across migrants and months; same occupation does not impose common shocks. '
        'Earnings ranks remain fixed. Three seeds share the same banks between allocation rules. '
        f"Input months flagged low quality: {config['low_quality_history_months']} (retained and exported).", '',
        '## Results', '',
        'Annual USD totals are summed; receipt is the mean monthly rate; mean contribution is dollars / positive contributions. '
        'The table averages each statistic across three seeds.', '',
        '| Rule | Year | Annual USD | Receipt % | Contributors/month | Growth vs own 2019 % |',
        '|---|---:|---:|---:|---:|---:|']
    for (mode, year), g in yearly.groupby(['mode', 'year']):
        lines.append(f"| {mode} | {year} | {g.total_usd.mean():,.0f} | {100*g.receipt_rate.mean():.2f} | {g.senders_per_month.mean():.2f} | {g.growth_vs_2019_pct.mean():.2f} |")
    a = monthly.loc[monthly['mode'].eq('uncoordinated') & monthly.date.dt.year.ge(2020)]
    b = monthly.loc[monthly['mode'].eq('coordinated') & monthly.date.dt.year.ge(2020)]
    gap = 100*(b.total_usd.sum()/a.total_usd.sum()-1)
    lines += ['', f'Across 2020–2022, coordination changes total transfers by {gap:.4f}% relative to independent allocation. '
        f'The average receipt difference is {100*(b.receipt_rate.mean()-a.receipt_rate.mean()):.4f} percentage points.', '',
        f"Independent allocation exceeds household need in {100*a.excess_support_rate.mean():.4f}% of household-months; "
        f"coordination does so in {100*b.excess_support_rate.mean():.4f}%. "
        f"Mean unmet perceived need is {100*a.unmet_share.mean():.2f}% versus {100*b.unmet_share.mean():.2f}%.", '',
        '### Interpretation', '',
        'Coordination only changes current amounts when willing migrants jointly offer enough to cover need. '
        'If offered budgets remain below need, both rules make the same transfers; without a state divergence their future paths coincide too. '
        'Low kin willingness and tight budgets can therefore make the two rules almost indistinguishable. '
        'This is a conditional result to inspect, not a reason to tune parameters until a large coordination effect appears.', '',
        'For the same prior states and current positive capacities, both rules have the same any-receipt probability: '
        '`1 - product(1 - p_j)` over positive-capacity contributors. Coordination caps overcoverage; '
        'it does not directly create willingness. Differences in later receipt can arise through previous-positive-sender persistence.', '',
        'CPS-driven changes combine conditional wages and employment. Rising observed mean earnings can include workforce-selection effects; '
        'these are not identified individual wage gains. No increased Mexican need is supplied or inferred. '
        'Growth relative to this constant 2019 baseline is not a causal effect of COVID.', '',
        'This toy does not test network size, common-risk diversification, national fit, or inequality in Mexican resources. '
        'Its sample is illustrative, with no national population weights. One positive link-month contribution is not a unique observed Banxico transaction.', '',
        '## Files and verification', '',
        '- `aggregate_monthly_index.png`: model and observed Banxico totals, each indexed to its own 2019 monthly mean = 100.',
        '- `aggregate_annual_growth.png`: annual total growth relative to 2019, model versus observed.',
        '- `aggregate_monthly_comparison.csv`, `aggregate_annual_comparison.csv`: plotted values and baselines for all seeds.',
        'These comparisons use nominal USD, not inflation-adjusted amounts. Observed means actual recorded aggregate flows. '
        'Banxico covers total inbound remittances to Mexico; the toy represents US-connected households in one occupation. '
        'Indexing compares trajectories and does not establish national representativeness or causal COVID effects.', '',
        '- `comparison.png`: monthly seed means and ranges, not confidence intervals.',
        '- `monthly_runs.csv`, `annual_runs.csv`, `paired_effects.csv`: all seeds and paired differences.',
        '- `cps_history_inputs.csv`, `cps_2019_baseline.csv`: prepared inputs actually used.',
        '- `migrants.csv`: fixed relationship and earnings profiles.',
        '- `*_household_months.parquet`, `*_migrant_months.parquet`: retained 2019–2022, first seed, both rules.',
        '- `run_settings.json`: parameters, assumptions, input/source hashes and checks.', '',
        'Every run checks budget/employment limits, migrant-to-household accounting, and coordinated coverage. '
        'The original empirical ABM and its outputs are not written by this script. Rerunning replaces only this toy output directory\'s named files.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    run()
