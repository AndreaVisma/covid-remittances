"""Rebuild every output of the paper, in order (29 Oct 2026).

Run from anywhere: python network-compatibility-model/run_all.py   (~75 min)
  --data-prep        first rebuild the inputs in data-prep/ (needs the local MMP, IPUMS CPS and Banxico
                     files in data/, and internet for FRED and the mortality series)
  --identification   also rerun the ENIGH, state and municipal analyses (need the downloads in
                     data/external/, see each folder's README)
  --skip-robustness  skip need_margins.py (option 6/7/8 experiments) and the robustness runs 6a-6h
                     (recipient-level/robustness/, ~30 min together)
  --from NAME        start at the step called NAME (see STEPS), e.g. --from household-model

Steps run as separate processes from the repository root and stop at the first failure.
Base model: recipient-level/run_household_model.py (household receipt logit, kappa = replacement
share, stimulus pi, need as more senders per receiving household, option 6a). Comparison: the
migrant-level model under the coordination rule (model/run_e1_coordination.py, run_e2_map.py).
Outputs are aggregates only; nothing under data/ is written to tracked folders.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import time

NCM = Path(__file__).resolve().parent
ROOT = NCM.parent
ENIGH_RAW = 'data/external/enigh/unzipped'

DATA_PREP = [
    ('population', ['data-prep/build_population.py']),
    ('cps', ['data-prep/cps_employment_rates.py']),
    ('banxico', ['data-prep/remittance_amounts.py']),
    ('need-series', ['data-prep/salience_inputs.py']),
    ('benefits', ['data-prep/benefit_schedule.py']),
    ('us-cpi', ['data-prep/us_cpi.py']),
]
MODEL = [
    ('logit-comparison', ['recipient-level/compare_household_logit.py']),     # also the M1 alphas
    ('household-model', ['recipient-level/run_household_model.py']),
    ('need-margins', ['recipient-level/need_margins.py']),                    # robustness; checks the base
    ('rob-savings', ['recipient-level/robustness/savings_margin.py']),
    ('rob-pi-april', ['recipient-level/robustness/pi_april_2020.py']),
    ('rob-gamma', ['recipient-level/robustness/stimulus_gamma.py']),
    ('rob-pooled', ['recipient-level/robustness/pooled_mar_apr.py']),
    ('rob-oos-2019', ['recipient-level/robustness/out_of_sample_2019.py']),
    ('rob-need-series', ['recipient-level/robustness/need_series_decay.py']),
    ('rob-mapping', ['recipient-level/robustness/lower_bound_mapping.py']),
    ('rob-trend', ['recipient-level/robustness/trend_window.py']),
    ('rob-collect', ['recipient-level/robustness/collect.py']),
    ('migrant-e1', ['model/run_e1_coordination.py']),
    ('migrant-e2', ['model/run_e2_map.py']),
    ('figures-data', ['model/make_figures.py']),
    ('figures-logit', ['recipient-level/make_figures.py']),
    ('figures-model', ['recipient-level/make_figures.py', '--model']),
]
IDENTIFICATION = [   # untrusted downloads: isolated mode (-I), as in the READMEs
    ('enigh-cells', ['-I', 'identification/enigh/enigh_heterogeneity.py']),
    ('enigh-figures', ['-I', 'identification/enigh/make_figures.py']),
    ('enigh-pre-monthly', ['-I', 'identification/enigh/enigh_pre_and_monthly.py',
                           '--c16', f'{ENIGH_RAW}/enigh2016_ns_concentradohogar_csv/concentradohogar.csv',
                           '--ing16', f'{ENIGH_RAW}/enigh2016_ns_ingresos_csv/ingresos.csv',
                           '--ing18', f'{ENIGH_RAW}/enigh2018_ns_ingresos_csv/ingresos.csv',
                           '--ing20', f'{ENIGH_RAW}/enigh2020_ns_ingresos_csv/ingresos.csv',
                           '--fx', 'data/external/fx/fred_DEXMXUS_full.csv']),
    ('state-deaths', ['-I', 'identification/state-panel/build_deaths.py']),
    ('state-panel', ['-I', 'identification/state-panel/build_panel.py']),
    ('state-analysis', ['-I', 'identification/state-panel/analysis.py']),
    ('municipal-panel', ['-I', '-X', 'utf8', 'identification/municipal-panel/build_municipal.py']),
    ('municipal-analysis', ['-I', '-X', 'utf8', 'identification/municipal-panel/analysis.py']),
]


def steps(args):
    out = (DATA_PREP if args.data_prep else []) + MODEL + (IDENTIFICATION if args.identification else [])
    if args.skip_robustness:
        out = [s for s in out if s[0] != 'need-margins' and not s[0].startswith('rob-')]
    if args.start:
        names = [n for n, _ in out]
        if args.start not in names:
            sys.exit(f'unknown step {args.start}; steps: {", ".join(names)}')
        out = out[names.index(args.start):]
    return out


def resolve(argv):
    """Script paths are relative to network-compatibility-model/ and passed as absolute paths."""
    return [str(NCM / a) if a.endswith('.py') else a for a in argv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-prep', action='store_true')
    ap.add_argument('--identification', action='store_true')
    ap.add_argument('--skip-robustness', action='store_true')
    ap.add_argument('--from', dest='start')
    todo = steps(ap.parse_args())
    t0 = time.time()
    for i, (name, argv) in enumerate(todo, 1):
        print(f'\n=== [{i}/{len(todo)}] {name}: {" ".join(argv)}', flush=True)
        t = time.time()
        res = subprocess.run([sys.executable, *resolve(argv)], cwd=ROOT)
        if res.returncode:
            sys.exit(f'step {name} failed (exit {res.returncode}); rerun with --from {name}')
        print(f'=== {name}: {time.time() - t:.0f} s', flush=True)
    print(f'\nall {len(todo)} steps done in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
