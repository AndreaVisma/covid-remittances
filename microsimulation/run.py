"""CLI for generation, experiments, optional pre-COVID fitting and reporting."""
from pathlib import Path
from itertools import product
import argparse
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import generate_inputs, load_inputs, load_target
from model import simulate

ROOT = Path(__file__).resolve().parent


def variants(config):
    # A full factorial with a common reference target identifies interactions.
    for need, persistence, network in product([False, True], repeat=3):
        c = copy.deepcopy(config)
        c['target_mode'] = 'reference'
        c['mechanisms'] = dict(need=need, persistence=persistence, network=network)
        name = 'reference_' + ('_'.join(k for k, v in c['mechanisms'].items() if v) or 'only')
        yield name, c
    c = copy.deepcopy(config)
    c['target_mode'] = 'current'
    c['mechanisms'] = dict(need=False, persistence=False, network=False)
    yield 'baseline_current', c


def fit_pre_covid(inputs, config, target):
    """Optional one-parameter grid fit; never evaluates holdout outcomes."""
    c = copy.deepcopy(config)
    c['end'] = str((pd.Timestamp(config['holdout_start']) - pd.offsets.MonthBegin(1)).date())
    c['target_mode'] = 'current'
    c['mechanisms'] = dict(need=False, persistence=False, network=False)
    observed = target[(target.date >= c['start']) & (target.date <= c['end'])]
    expected = pd.date_range(c['start'], c['end'], freq='MS')
    if len(observed) != len(expected) or observed.empty:
        raise ValueError('Calibration requires complete pre-COVID observed coverage')
    scores = []
    for share in np.linspace(0.01, 0.5, 30):
        c['remittance_share'] = float(share)
        paths = [simulate(inputs, c, seed)[0].set_index('date').total_usd for seed in config['seeds']]
        mean = pd.concat(paths, axis=1).mean(axis=1)
        errors = mean.reindex(observed.date).to_numpy() - observed.observed_usd.to_numpy()
        scores.append({'remittance_share': share, 'pre_covid_rmse_usd': np.sqrt(np.mean(errors ** 2))})
    scores = pd.DataFrame(scores)
    best = scores.loc[scores.pre_covid_rmse_usd.idxmin(), 'remittance_share']
    return float(best), scores


def plot_results(runs, target, config, output):
    names = ['baseline_current', 'reference_only', 'reference_need',
             'reference_need_persistence', 'reference_need_persistence_network', 'configured']
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    observed = target[target.date.between(config['start'], config['end'])].set_index('date').observed_usd
    base_observed = observed[observed.index < config['holdout_start']].mean()
    axes[0].plot(observed.index, observed / 1e9, color='black', linewidth=2, label='Observed workbook (assumed USD)')
    if base_observed > 0:
        axes[1].plot(observed.index, observed / base_observed * 100, color='black', linewidth=2, label='Observed')
    for name in names:
        part = runs[runs.scenario == name]
        if part.empty:
            continue
        grouped = part.groupby('date').total_usd
        mean, low, high = grouped.mean(), grouped.min(), grouped.max()
        line, = axes[0].plot(mean.index, mean / 1e9, label=name.replace('_', ' '))
        axes[0].fill_between(mean.index, low / 1e9, high / 1e9, color=line.get_color(), alpha=0.12)
        base = mean[mean.index < config['holdout_start']].mean()
        if base > 0:
            axes[1].plot(mean.index, mean / base * 100, color=line.get_color())
    axes[0].set_ylabel('Monthly transfers (USD billions, assumed)')
    axes[1].set_ylabel('Index: own pre-COVID mean = 100')
    axes[0].legend(fontsize=8, loc='upper left')
    for ax in axes:
        ax.axvline(pd.Timestamp(config['holdout_start']), color='grey', linestyle='--')
        ax.grid(alpha=0.2)
    fig.suptitle('Synthetic-input experiments against real remittance outcomes\nShading: seed range, not a confidence interval; no empirical identification', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def report(runs, target, config, output, audit):
    mean = runs.groupby(['scenario', 'date'], as_index=False).mean(numeric_only=True)
    compared = mean.merge(target, on='date', how='left', validate='many_to_one')
    compared.to_csv(output / 'comparison.csv', index=False)
    scores = []
    for name, frame in compared.groupby('scenario'):
        for period, mask in [('pre_covid', frame.date < config['holdout_start']),
                             ('covid_holdout', frame.date >= config['holdout_start'])]:
            part = frame[mask & frame.observed_usd.notna()]
            if part.empty:
                continue
            error = part.total_usd - part.observed_usd
            scores.append({'scenario': name, 'period': period, 'months': len(part),
                           'rmse_usd': np.sqrt(np.mean(error ** 2)),
                           'mean_bias_usd': error.mean(),
                           'observed_usd': part.observed_usd.sum(), 'simulated_usd': part.total_usd.sum()})
    pd.DataFrame(scores).to_csv(output / 'metrics.csv', index=False)
    yearly = runs.assign(year=runs.date.dt.year).groupby(['scenario', 'seed', 'year'], as_index=False)[
        ['total_usd', 'network_usd', 'unmet_usd']].sum()
    yearly.to_csv(output / 'annual_by_seed.csv', index=False)
    # Paired effects isolate network support at a common contribution target.
    paired = runs.pivot(index=['date', 'seed'], columns='scenario', values='total_usd')
    effects = pd.DataFrame(index=paired.index)
    for name in paired.columns:
        if name.startswith('reference_'):
            effects[name + '_minus_reference_only'] = paired[name] - paired['reference_only']
    effects.to_csv(output / 'paired_effects.csv')
    plot_results(runs, target, config, output)
    summary = [
        '# Prototype results', '',
        'Synthetic inputs and assumed behavioural parameters; these runs are not an empirical explanation.', '',
        f"Target: `{audit['path']}`, {audit['start']} through {audit['end']} ({audit['observations']} monthly observations).",
        f"Units: {audit['currency_assumption']}. Values multiplied by {config['target_multiplier']:,}.",
        audit['coverage_warning'], '',
        f"Population represented by equal agent weights: {runs.attrs.get('population', 'see manifest'):,}.",
        'No outcome-based population scaling is applied. The indexed chart separately divides each series by its own pre-COVID mean.', '',
        'Default parameters are not calibrated. Optional calibration fits only the baseline remittance share before the holdout cutoff.',
        'The same parameters are retained across ablations; changes in baseline levels are part of the experiment.', '',
        'See comparison.png, metrics.csv, annual_by_seed.csv and paired_effects.csv. Sensitivity results are in sensitivity.csv.',
        'The seed range captures simulation randomness only. Network structure, population composition and shocks remain assumed.',
        'Aggregate fit cannot separately identify activation, persistence, transfer norms or support.', ''
    ]
    (output / 'REPORT.md').write_text('\n'.join(summary), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--target', type=Path)
    parser.add_argument('--generate', action='store_true', help='Create synthetic CSVs; refuses overwriting existing inputs')
    parser.add_argument('--calibrate', action='store_true', help='Exploratory pre-COVID baseline share fit only')
    parser.add_argument('--single', action='store_true', help='Run configured switches rather than full factorial')
    parser.add_argument('--skip-sensitivity', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    base = args.config.resolve().parent
    folder = (base / config['input_dir']).resolve()
    output = (base / config['output_dir']).resolve()
    if args.generate:
        generate_inputs(config, folder)
    inputs = load_inputs(folder, config)
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Provide distinct simulation seeds')
    dates = pd.date_range(config['start'], config['end'], freq='MS')
    if dates.empty or not (pd.Timestamp(config['start']) < pd.Timestamp(config['holdout_start']) <= pd.Timestamp(config['end'])):
        raise ValueError('Require a valid monthly horizon with pre-COVID and holdout periods')
    target_path = args.target.resolve() if args.target else (base / config['target_workbook']).resolve()
    target, audit = load_target(target_path, config)
    output.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        config['remittance_share'], fit = fit_pre_covid(inputs, config, target)
        fit.to_csv(output / 'calibration.csv', index=False)
        print(f"Exploratory pre-COVID fitted share: {config['remittance_share']:.4f}")
    runs = []
    scenarios = [('configured', config)] if args.single else list(variants(config))
    for name, variant in scenarios:
        for seed in config['seeds']:
            monthly, _ = simulate(inputs, variant, seed)
            runs.append(monthly.assign(scenario=name, seed=seed))
        print(f'Completed {name}', flush=True)
    runs = pd.concat(runs, ignore_index=True)
    runs.attrs['population'] = float(inputs[0].weight.sum())
    runs.to_csv(output / 'monthly_runs.csv', index=False)
    _, detail = simulate(inputs, config, config['seeds'][0], detail=True)
    detail.to_csv(output / 'configured_agent_trace.csv.gz', index=False, compression='gzip')
    sensitivity = []
    if not args.skip_sensitivity:
        for parameter, values in config['sensitivity'].items():
            for value in values:
                c = copy.deepcopy(config)
                c[parameter] = value
                for seed in config['seeds']:
                    monthly, _ = simulate(inputs, c, seed)
                    part = monthly[monthly.date >= config['holdout_start']]
                    sensitivity.append(dict(parameter=parameter, value=value, seed=seed,
                                            holdout_total_usd=part.total_usd.sum(),
                                            holdout_network_usd=part.network_usd.sum()))
    pd.DataFrame(sensitivity, columns=['parameter', 'value', 'seed', 'holdout_total_usd', 'holdout_network_usd']).to_csv(output / 'sensitivity.csv', index=False)
    target.to_csv(output / 'observed.csv', index=False)
    files = [folder / n for n in ['agents.csv', 'employment.csv', 'needs.csv']] + [target_path]
    manifest = dict(config=config, calibrated=args.calibrate,
                    represented_population=float(inputs[0].weight.sum()),
                    scenarios={name: c for name, c in scenarios},
                    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'target_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    report(runs, target, config, output, audit)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
