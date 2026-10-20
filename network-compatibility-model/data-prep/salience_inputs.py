"""Task 2: monthly Mexican employment and deaths, and the salience index S_t (spec §4.3).

Run: python network-compatibility-model/data-prep/salience_inputs.py

Writes to data-prep/outputs/ (aggregate public series, safe to commit):
  salience_inputs_monthly.csv   components, 2019-01 to 2022-12
  salience_index_monthly.csv    S_t (base) and the reported-COVID-deaths variant
  salience_inputs_settings.json sources and definition

S_t = reported COVID-19 deaths in Mexico in month t (OWID), in units of 10,000 deaths (user
decisions, 20 Oct 2026: keep the index simple; use COVID deaths, not excess deaths). S_t = 0 in
2019. Variant: excess all-cause deaths (negative values set to 0).
The employment series is kept in salience_inputs_monthly.csv for possible later use; it is not
part of S_t.

Sources (raw files in data/, git-ignored; re-download with the URLs below):
  * Employed population, 15+, not seasonally adjusted, thousands: OECD Infra-annual labour
    statistics (monthly ENOE/ENOEN; employed = labour force - unemployed), file
    data/oecd_mex_labour_monthly.csv from
    https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,1.0/MEX........M?startPeriod=2018-01&endPeriod=2023-12&format=csvfilewithlabels
    The OECD series has no April-June 2020 (ENOE suspended); those months come from INEGI's
    telephone survey ETOE (press releases): April 43.4 M (45.3% of the 15+ population,
    ETOE May release), May 43.6 M (95.8% of a 45.5 M labour force, ETOE May release),
    June 48.3 M (comunicado 387/20, 5 Aug 2020). ETOE is calibrated to ENOE Q1 2020 but is
    not strictly comparable (about 10% of the ENOE sample, phone interviews).
  * Excess deaths: World Mortality Dataset (Karlinsky & Kobak 2021), weekly all-cause deaths,
    data/world_mortality.csv from
    https://raw.githubusercontent.com/akarlinsky/world_mortality/main/world_mortality.csv
    Baseline = 2015-2019 mean deaths for the same ISO week, scaled by the 2015-19 linear trend in
    annual deaths (without the trend, 2019 shows ~3,000 spurious excess deaths a month).
  * Reported COVID deaths: OWID, data/covid_cases.csv (already in the repo's data folder).
Weekly series are converted to calendar months by spreading each week evenly over its days.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
OUT = Path(__file__).resolve().parent / 'outputs'
MONTHS = pd.date_range('2019-01-01', '2022-12-01', freq='MS')
ETOE_EMPLOYED_THOUSANDS = {'2020-04-01': 43400.0, '2020-05-01': 43600.0, '2020-06-01': 48300.0}
BASELINE_YEARS = range(2015, 2020)
DEATHS_UNIT = 10_000
VARIANTS = {'S': 'covid_deaths', 'S_excess_deaths': 'excess_deaths_trend'}   # name: deaths column


def employed():
    d = pd.read_csv(DATA / 'oecd_mex_labour_monthly.csv')
    d = d[(d.SEX == '_T') & (d.AGE == 'Y_GE15') & (d.ADJUSTMENT == 'N') & d.MEASURE.isin(['LF', 'UNE_M'])]
    assert set(d.UNIT_MULT) == {3}, 'expected thousands of persons'
    p = d.pivot_table(index='TIME_PERIOD', columns='MEASURE', values='OBS_VALUE')
    p.index = pd.to_datetime(p.index)
    e = (p.LF - p.UNE_M).rename('employed_thousands')
    source = pd.Series('OECD (ENOE/ENOEN)', index=e.index)
    for month, value in ETOE_EMPLOYED_THOUSANDS.items():
        assert np.isnan(e.get(pd.Timestamp(month), np.nan)), f'OECD now has {month}; drop the ETOE patch'
        e[pd.Timestamp(month)] = value
        source[pd.Timestamp(month)] = 'INEGI ETOE press release'
    e, source = e.sort_index(), source.sort_index()
    assert e.reindex(MONTHS).notna().all()
    return e, source


def weekly_to_monthly(weekly):
    """weekly: Series indexed by ISO-week Monday. Spread each week over its 7 days."""
    days = weekly.reindex(pd.date_range(weekly.index.min(), weekly.index.max() + pd.Timedelta(days=6)))
    days = (days.ffill(limit=6) / 7).dropna()
    return days.groupby(days.index.to_period('M')).sum().to_timestamp()


def excess_deaths():
    w = pd.read_csv(DATA / 'world_mortality.csv')
    w = w[(w.iso3c == 'MEX') & (w.time_unit == 'weekly')].copy()
    w['monday'] = [pd.Timestamp.fromisocalendar(int(y), int(t), 1) for y, t in zip(w.year, w.time)]
    base = w[w.year.isin(BASELINE_YEARS)]
    mean_by_week = base.groupby('time').deaths.mean()
    mean_by_week[53] = mean_by_week.get(53, mean_by_week[52])
    w['expected_mean'] = w.time.map(mean_by_week)
    # Sensitivity: linear trend in annual deaths over the baseline, applied to the week profile.
    annual = base.groupby('year').deaths.sum()
    slope, intercept = np.polyfit(annual.index, annual.values, 1)
    growth = (intercept + slope * w.year) / annual.mean()
    w['expected_trend'] = w.expected_mean * growth
    s = w.set_index('monday')
    monthly = pd.DataFrame({c: weekly_to_monthly(s[c]) for c in ['deaths', 'expected_mean', 'expected_trend']})
    monthly['excess_deaths'] = monthly.deaths - monthly.expected_mean
    monthly['excess_deaths_trend'] = monthly.deaths - monthly.expected_trend
    return monthly.reindex(MONTHS)


def covid_deaths():
    c = pd.read_csv(DATA / 'covid_cases.csv', usecols=['country', 'date', 'new_deaths'])
    c = c[c.country == 'Mexico'].copy()
    c['date'] = pd.to_datetime(c.date)
    s = c.set_index('date').new_deaths.fillna(0)
    # OWID reports weekly totals on one day per week in recent vintages: monthly sums are robust either way.
    return s.groupby(s.index.to_period('M')).sum().to_timestamp().reindex(MONTHS).fillna(0).rename('covid_deaths')


def salience(inputs):
    out = pd.DataFrame(index=inputs.index)
    for name, col in VARIANTS.items():
        out[name] = (inputs[col] / DEATHS_UNIT).clip(lower=0)
    out.loc['2019', list(VARIANTS)] = 0.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e, source = employed()
    out = pd.DataFrame({'employed_thousands': e.reindex(MONTHS), 'employment_source': source.reindex(MONTHS)})
    base = out.loc['2019', 'employed_thousands']
    out['employed_same_month_2019'] = base.set_axis(base.index.month).loc[out.index.month].to_numpy()
    out['employment_loss'] = 1 - out.employed_thousands / out.employed_same_month_2019
    out = out.join(excess_deaths()).join(covid_deaths())
    out.index.name = 'date'
    out.to_csv(OUT / 'salience_inputs_monthly.csv', float_format='%.6g')
    index = salience(out)
    index.index.name = 'date'
    index.to_csv(OUT / 'salience_index_monthly.csv', float_format='%.6g')
    settings = dict(months=[str(MONTHS[0].date()), str(MONTHS[-1].date())],
                    employment='employed 15+ = OECD LF - UNE_M (monthly ENOE, NSA), ETOE patch Apr-Jun 2020',
                    etoe_employed_thousands=ETOE_EMPLOYED_THOUSANDS,
                    employment_loss='1 - employed / employed in the same month of 2019',
                    excess_baseline='2015-2019 mean deaths by ISO week, scaled by a linear trend in annual deaths',
                    covid_deaths='OWID new_deaths, monthly sum',
                    salience=f'S = reported COVID deaths / {DEATHS_UNIT}; S = 0 in 2019; variant: max(excess deaths, 0)',
                    variants=VARIANTS, base_variant='S', employment_role='kept as input only, not in S')
    (OUT / 'salience_inputs_settings.json').write_text(json.dumps(settings, indent=2))
    pd.set_option('display.width', 200)
    cols = ['employed_thousands', 'employment_loss', 'excess_deaths_trend', 'covid_deaths']
    print(out[cols].join(index[list(VARIANTS)]).round(3).to_string())
    print(index.groupby(index.index.year).agg(['mean', 'max']).round(2).to_string())


if __name__ == '__main__':
    main()
