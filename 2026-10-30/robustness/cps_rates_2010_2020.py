"""CPS employment rates by occupation group, 2010-2020, with the definitions of the existing pipeline
(robustness check 'own interview year'; 10 Nov 2026).

Run: python 2026-10-30/robustness/cps_rates_2010_2020.py      (reads data/cps_00001.dat, about 5 minutes)

The model's rates (network-compatibility-model/data-prep/outputs/cps_employment_rates_monthly.csv) cover only
2017-2023, because cps_employment_rates.py restricts the extract to START_YM = 201701. cps_00001.dat itself is
the IPUMS basic monthly CPS from January 2010. This script reuses that module unchanged (same population:
Mexican origin by HISPAN, ages 18-64, ASEC records excluded, WTFINL > 0; same OCC2010 occupation groups;
rate = E / (E + U) weighted by WTFINL; same reliability rule and centred 3-month smoothing; model column
rate_model), only with the period widened to 2010-01..2020-12 (the end is past 2019 so that the December 2019
smoothing window is the same as in the base file). Checks that 2017-2020 agree with the base file.
Output (aggregates): 2026-10-30/outputs/robustness-own-year/cps_employment_rates_monthly_2010_2020.csv,
cps_rates_by_year.csv (annual mean of rate_model by group), cps_rates_check.json.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'network-compatibility-model/data-prep'))
import cps_employment_rates as cps          # noqa: E402

OUT = HERE.parent / 'outputs/robustness-own-year'
BASE = ROOT / 'network-compatibility-model/data-prep/outputs/cps_employment_rates_monthly.csv'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cps.START_YM, cps.END_YM = 201001, 202012
    p, population = cps.load(cps.read_layout())
    print(f'{population}: {len(p):,} person-months', flush=True)
    rates = cps.employment_rates(p)
    rates.insert(0, 'population', population)
    rates.to_csv(OUT / 'cps_employment_rates_monthly_2010_2020.csv', index=False)

    base = pd.read_csv(BASE)
    m = base.merge(rates, on=['year', 'month', 'job'], suffixes=('_base', '_new'))
    diff = (m.rate_model_base - m.rate_model_new).abs()
    check = dict(population=population, months=int(rates[['year', 'month']].drop_duplicates().shape[0]),
                 overlap_cells=len(m), max_abs_diff_rate_model_2017_2020=float(diff.max()),
                 max_abs_diff_2019=float(diff[m.year.eq(2019)].max()),
                 cells_differing=int((diff > 1e-12).sum()),
                 differing=[f'{a} {b} {c}' for a, b, c in m.loc[diff > 1e-12, ['year', 'month', 'job']].itertuples(index=False)],
                 unreliable_raw_cells_2010_2019=int((~rates.reliable_raw & rates.year.le(2019)).sum()),
                 unreliable_smooth_cells_2010_2019=int((~rates.reliable_smooth & rates.year.le(2019)).sum()))
    (OUT / 'cps_rates_check.json').write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))
    by_year = rates.loc[rates.year.le(2019)].pivot_table(index='year', columns='job', values='rate_model')
    by_year.to_csv(OUT / 'cps_rates_by_year.csv')
    print(by_year.round(3).to_string())


if __name__ == '__main__':
    main()
