"""Task 4: routine remittance amounts m_j (model-specification.md §4.4).

Run: python network-compatibility-model/data-prep/remittance_amounts.py
Dependencies: numpy, pandas. Source data are read only.

m_j ~ LogNormal(mu, sigma^2), drawn once per migrant, amount per sending month.
  * Level: Banxico 2019 mean USD per transaction (one transaction per sender-month).
  * Dispersion sigma: MMP self-reported average monthly remittances on the last
    US trip (MIG = household heads; MIGOTHER = other migrants, nearly all
    children/spouses), trips 2000-2019, deflated to 2019 USD, measured within
    5-year trip periods so that growth over time does not inflate it.
  * Kin (extended family) use the same distribution: MMP has too few kin amounts.
ENIGH 2018 is summarized as a validation target only. It averages receipt
over months (including months without receipt) and under-reports relative to
Banxico, so it does not set m_j.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

# %% Settings
ROOT = Path('C:/git-projects/covid-remittances')
MMP = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
BANXICO = ROOT / 'oct-26/data-prep/outputs/mexico-rem-monthly-total.parquet'
ENIGH_2018 = ROOT / 'data/concentradohogar_18.csv'
FX = ROOT / 'data/DEXMXUS.csv'
OUT = Path(__file__).resolve().parent / 'outputs'
TRIP_YEARS = (2000, 2019)
PERIODS = [2000, 2005, 2010, 2015, 2020]      # 5-year trip-year bins, right-open
MISSING_FROM = 9990                           # 9999 etc. = unknown
TRIM_QUANTILE = 0.99                          # Drop the top 1% of positive amounts
KEYS = ['country', 'commun', 'hhnum']
# US CPI-U annual averages (BLS series CUUR0000SA0, 1982-84=100).
CPI = {2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3,
       2006: 201.6, 2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056,
       2011: 224.939, 2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017,
       2016: 240.007, 2017: 245.120, 2018: 251.107, 2019: 255.657}
# Same MMP occupation groups as oct-26/simulations/synthetic-migrants-house-networks.py.
JOB_GROUPS = {
    'Agriculture': list(range(410, 420)),
    'Production/construction/transport': list(range(510, 560)),
    'Sales/personal services': list(range(710, 730)) + list(range(810, 840)),
    'Professional/administrative': list(range(110, 150)) + list(range(210, 220)) + list(range(610, 630)),
}
ROLES = {1: 'Head', 2: 'Spouse', 3: 'Child'}


# %% MMP amounts
def read_mmp(name, columns):
    return pd.read_stata(MMP / f'{name}.dta', convert_categoricals=False, columns=columns)


def load_mmp():
    cols = KEYS + ['surveyyr', 'usyrl', 'usoccl', 'usdocl', 'remit']
    heads = read_mmp('mig174', cols)
    heads['relhead'] = 1
    others = read_mmp('migother174', cols + ['persnum'])
    others = others.drop_duplicates(KEYS + ['persnum'])
    pers = read_mmp('pers174', KEYS + ['persnum', 'relhead'])
    others = others.merge(pers, on=KEYS + ['persnum'], how='left', validate='one_to_one')
    m = pd.concat([heads.assign(source='MIG'), others.assign(source='MIGOTHER')], ignore_index=True)
    m['role'] = m.relhead.map(ROLES).fillna('Other relative')
    job_map = {c: g for g, codes in JOB_GROUPS.items() for c in codes}
    m['job_group'] = m.usoccl.map(job_map).fillna('Other/unknown')
    # The DTA files carry no value labels. MMP codebook: usdocl 8 = undocumented.
    m['documentation'] = np.select([m.usdocl.eq(8), m.usdocl.lt(MISSING_FROM)],
                                   ['Undocumented', 'Documented'], 'Unknown')
    return m


def amount_sample(m):
    in_window = m.usyrl.between(*TRIP_YEARS)
    s = m.loc[in_window].copy()
    s['status'] = np.select([s.remit.ge(MISSING_FROM), s.remit.eq(0)], ['unknown', 'zero'], 'positive')
    pos = s.loc[s.status.eq('positive')].copy()
    pos['usd_2019'] = pos.remit * CPI[2019] / pos.usyrl.astype(int).map(CPI)
    cut = pos.usd_2019.quantile(TRIM_QUANTILE)
    pos = pos.loc[pos.usd_2019.le(cut)].copy()
    pos['log_usd'] = np.log(pos.usd_2019)
    pos['period'] = pd.cut(pos.usyrl, PERIODS, right=False, labels=[f'{a}-{b - 1}' for a, b in zip(PERIODS, PERIODS[1:])])
    return s, pos, cut


def dispersion(pos):
    resid = pos.log_usd - pos.groupby('period', observed=True).log_usd.transform('mean')
    q75, q25 = np.percentile(resid, [75, 25])
    return dict(sigma_within_period=float(resid.std(ddof=1)),
                sigma_within_period_robust=float((q75 - q25) / 1.349),
                sigma_pooled=float(pos.log_usd.std(ddof=1)), n=len(pos))


def describe(pos, by):
    g = pos.groupby(by, observed=True)
    out = g.usd_2019.agg(n='count', median='median', mean='mean')
    out['log_sd'] = g.log_usd.std()
    out['median_ratio_to_all'] = out['median'] / pos.usd_2019.median()
    return out.reset_index().rename(columns={by: 'group'}).assign(dimension=by)


# %% Banxico level and ENIGH validation
def banxico_2019_mean():
    b = pd.read_parquet(BANXICO)
    y = b.loc[b.date.dt.year.eq(2019)]
    return float(y.total_usd.sum() / y.transactions.sum()), b


def enigh_2018(banxico):
    e = pd.read_csv(ENIGH_2018, usecols=['folioviv', 'foliohog', 'factor', 'remesas'])
    fx = pd.read_csv(FX, parse_dates=['observation_date'], na_values='.')
    fx_2018 = float(fx.loc[fx.observation_date.dt.year.eq(2018), 'DEXMXUS'].mean())
    w = e.factor
    rec = e.loc[e.remesas.gt(0)]
    monthly_usd = rec.remesas / 3 / fx_2018      # ENIGH values are quarterly MXN.
    b18 = banxico.loc[banxico.date.dt.year.eq(2018), 'total_usd'].sum()
    enigh_total = float((e.remesas * w).sum() * 4 / fx_2018)
    order = np.argsort(monthly_usd.to_numpy())
    cum = np.cumsum(rec.factor.to_numpy()[order])
    wmedian = float(monthly_usd.to_numpy()[order][np.searchsorted(cum, cum[-1] / 2)])
    return dict(fx_2018_mxn_per_usd=fx_2018,
                recipient_share=float(np.average(e.remesas.gt(0), weights=w)),
                recipient_n=len(rec),
                recipient_monthly_usd_median=wmedian,
                recipient_monthly_usd_mean=float(np.average(monthly_usd, weights=rec.factor)),
                recipient_log_sd=float(np.log(monthly_usd).std()),
                enigh_annual_total_usd=enigh_total, banxico_2018_total_usd=float(b18),
                enigh_to_banxico_ratio=enigh_total / float(b18),
                note='ENIGH: quarterly MXN, converted at the 2018 mean FX. '
                     'Averages over months incl. months without receipt; validation only.')


# %% Main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_mmp()
    s, pos, cut = amount_sample(m)
    disp = dispersion(pos)
    level, banxico = banxico_2019_mean()
    sigma = disp['sigma_within_period']
    mu = float(np.log(level) - sigma ** 2 / 2)

    table = pd.concat([describe(pos.assign(all='All'), 'all'), describe(pos, 'period'),
                       describe(pos, 'role'), describe(pos, 'job_group'),
                       describe(pos, 'documentation')], ignore_index=True)
    table = table[['dimension', 'group', 'n', 'median', 'mean', 'log_sd', 'median_ratio_to_all']]
    table.to_csv(OUT / 'remittance_amounts_mmp_diagnostics.csv', index=False)

    status = s.groupby(['source', 'status']).size().unstack(fill_value=0)
    status.to_csv(OUT / 'remittance_amounts_mmp_sample.csv')
    enigh = enigh_2018(banxico)

    result = dict(
        distribution='lognormal', mean_usd_2019=level, mu=mu, sigma=sigma,
        median_usd_2019=float(np.exp(mu)),
        level_source='Banxico 2019 total USD / transactions (one transaction per sender-month)',
        sigma_source='MMP MIG+MIGOTHER, trips 2000-2019, 2019 USD, within 5-year trip periods',
        dispersion=disp, trim_top_usd_2019=float(cut), trip_years=TRIP_YEARS,
        mmp_zero_share=float(s.status.eq('zero').sum() / s.status.ne('unknown').sum()),
        mmp_median_usd_2019=float(pos.usd_2019.median()), mmp_mean_usd_2019=float(pos.usd_2019.mean()),
        kin_rule='Kin use the same distribution as core migrants (MMP has ~13 positive kin amounts).',
        sensitivity=dict(kin_amount_factor=0.5),
        enigh_2018_validation=enigh)
    (OUT / 'remittance_amounts.json').write_text(json.dumps(result, indent=2, default=str))

    print(status.to_string())
    print(table.round(2).to_string(index=False))
    print(json.dumps({k: result[k] for k in ['mean_usd_2019', 'mu', 'sigma', 'median_usd_2019',
                                             'mmp_zero_share', 'mmp_median_usd_2019', 'mmp_mean_usd_2019',
                                             'dispersion']}, indent=1))
    print(json.dumps(enigh, indent=1))


if __name__ == '__main__':
    main()
