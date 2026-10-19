"""Task 1: occupation-month employment rates and employment persistence from CPS.

Run: python network-compatibility-model/data-prep/cps_employment_rates.py
Dependencies: numpy, pandas, scipy. Source data are read only.

Produces the employment inputs of model-specification.md §4.1:
  * e_o,t  occupation-month employment rate, E / (E + U), with the occupation
           of the current or last job (CPS records no occupation for most people
           outside the labour force);
  * an exit-adjusted variant that counts labour-force exits since 2019 as job
    loss, spread proportionally over occupations;
  * phi    persistence of the latent employment process, matched to 2019
           month-to-month CPS transitions.

Population: Mexican-born (BPL == 20000) if the extract contains BPL, otherwise
the Mexican-origin fallback (HISPAN Mexican codes). cps_00001 has no BPL, so the
current outputs are Mexican origin and are flagged as such in every file.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

# %% Settings
ROOT = Path('C:/git-projects/covid-remittances')
DATA = ROOT / 'data/cps_00001.dat'
CODEBOOK = ROOT / 'oct-26/data-prep/cps_00001_codebook.txt'
OUT = Path(__file__).resolve().parent / 'outputs'
START_YM, END_YM = 201701, 202312
BASELINE_YEAR = 2019
AGE_RANGE = (18, 64)          # Working-age migrants; avoids retirement exits.
CHUNK_ROWS = 250_000
SMOOTH_MONTHS = 3             # Centred window, pooling weighted counts.
MIN_LF_N, MIN_UNEMPLOYED_N = 100, 5
MEXICO_BPL = 20000
MEXICAN_HISPAN = [100, 102, 103, 104, 108, 109]
EMPLOYED, UNEMPLOYED, NILF = [10, 12], [20, 21, 22], [30, 31, 32, 33, 34, 35, 36]
# Same occupation groups as oct-26/data-prep/cps-earnings-employment.py, which
# the network population (synthetic-migrants-house-networks.py) uses.
OCCUPATION_RANGES = {
    'Agriculture': [(6000, 6130)],
    'Production/construction/transport': [(6200, 8965), (9000, 9750)],
    'Sales/personal services': [(3600, 4650), (4700, 4965)],
    'Professional/administrative': [(10, 3540), (5000, 5940)],
}
JOBS = list(OCCUPATION_RANGES)
ALL = 'All occupations'


# %% Read the fixed-width extract
def read_layout():
    text = CODEBOOK.read_text(encoding='utf-8-sig')
    blocks = re.split(r'Variable: "([A-Z0-9_]+)"', text)
    layout = {}
    for name, block in zip(blocks[1::2], blocks[2::2]):
        start = int(re.search(r'Start Position:\s*(\d+)', block)[1]) - 1
        width = int(re.search(r'Width:\s*(\d+)', block)[1])
        decimals = int(re.search(r'Implied Decimal Places:\s*(\d+)', block)[1])
        layout[name] = (start, width, decimals)
    return layout


def occupation_group(occ):
    out = np.full(len(occ), '', dtype=object)
    for job, ranges in OCCUPATION_RANGES.items():
        mask = np.zeros(len(occ), dtype=bool)
        for lo, hi in ranges:
            mask |= (occ >= lo) & (occ <= hi)
        out[mask] = job
    return out


def load(layout):
    has_bpl = 'BPL' in layout
    names = ['YEAR', 'MONTH', 'MISH', 'AGE', 'EMPSTAT', 'OCC2010', 'WTFINL', 'CPSIDV']
    width = max(s + w for s, w, _ in layout.values())
    with DATA.open('rb') as f:
        stride = len(f.readline())
    assert stride - 1 == width or stride - 2 == width
    parts, seen = [], 0
    with DATA.open('rb') as f:
        while raw := f.read(stride * CHUNK_ROWS):
            n = len(raw) // stride

            def col(name, mask=None):
                s, w, d = layout[name]
                a = np.ndarray((n,), dtype=f'S{w}', buffer=raw, offset=s, strides=(stride,))
                if mask is not None:
                    a = a[mask]
                a = np.char.strip(a)
                a = np.where(a == b'', b'0', a).astype(np.int64)
                return a / 10 ** d if d else a

            ym = col('YEAR') * 100 + col('MONTH')
            age, asec = col('AGE'), col('ASECFLAG')
            target = col('BPL') == MEXICO_BPL if has_bpl else np.isin(col('HISPAN'), MEXICAN_HISPAN)
            keep = ((ym >= START_YM) & (ym <= END_YM) & (asec != 1) & target
                    & (age >= AGE_RANGE[0]) & (age <= AGE_RANGE[1]))
            parts.append(pd.DataFrame({k: col(k, keep) for k in names}))
            seen += n
            if len(parts) % 20 == 0:
                print(f'Read {seen:,} rows', flush=True)
    p = pd.concat(parts, ignore_index=True)
    p['ym'] = (p.YEAR * 100 + p.MONTH).astype(int)
    p['month_index'] = (p.YEAR * 12 + p.MONTH - 1).astype(int)
    p['E'] = p.EMPSTAT.isin(EMPLOYED)
    p['U'] = p.EMPSTAT.isin(UNEMPLOYED)
    p['N'] = p.EMPSTAT.isin(NILF)
    p['job'] = occupation_group(p.OCC2010.to_numpy())
    p['CPSIDV'] = p.CPSIDV.astype(np.int64)
    population = 'Mexican-born' if has_bpl else 'Mexican origin (fallback: extract has no BPL)'
    return p.loc[p.WTFINL.gt(0)], population


# %% Employment rates
def employment_rates(p):
    lf = p.loc[(p.E | p.U) & p.job.ne('')]
    g = (lf.assign(E_w=lf.WTFINL * lf.E, U_w=lf.WTFINL * lf.U)
           .groupby(['job', 'ym']).agg(E_n=('E', 'sum'), U_n=('U', 'sum'),
                                       E_w=('E_w', 'sum'), U_w=('U_w', 'sum')))
    total = g.groupby(level='ym').sum()
    total['job'] = ALL
    g = pd.concat([g.reset_index(), total.reset_index()], ignore_index=True)
    g = g.sort_values(['job', 'ym']).reset_index(drop=True)
    g['rate_raw'] = g.E_w / (g.E_w + g.U_w)
    roll = lambda s: s.rolling(SMOOTH_MONTHS, center=True, min_periods=1).sum()
    sm = g.groupby('job')[['E_w', 'U_w', 'E_n', 'U_n']].transform(roll)
    g['rate_smooth'] = sm.E_w / (sm.E_w + sm.U_w)
    g['reliable_raw'] = (g.E_n + g.U_n >= MIN_LF_N) & (g.U_n >= MIN_UNEMPLOYED_N)
    g['reliable_smooth'] = (sm.E_n + sm.U_n >= MIN_LF_N) & (sm.U_n >= MIN_UNEMPLOYED_N)
    # Model input: raw where the cell is large enough, smoothed otherwise.
    # Smoothing everywhere would blur the April 2020 step into March.
    g['rate_model'] = g.rate_raw.where(g.reliable_raw, g.rate_smooth)

    # Participation of the whole population (occupation unknown for most NILF).
    pop = p.assign(LF_w=p.WTFINL * (p.E | p.U), EP_w=p.WTFINL * p.E).groupby('ym')
    part = pd.DataFrame({'participation': pop.LF_w.sum() / pop.WTFINL.sum(),
                         'employment_population': pop.EP_w.sum() / pop.WTFINL.sum()})
    base = part.loc[part.index // 100 == BASELINE_YEAR, 'participation'].mean()
    part['participation_ratio_2019'] = part.participation / base
    g = g.merge(part.reset_index(), on='ym', how='left')
    # Exit-adjusted: labour-force exits relative to 2019 treated as job loss,
    # spread proportionally over occupations. Capped at the unadjusted rate
    # when participation is above its 2019 level.
    g['rate_exit_adjusted'] = g.rate_model * g.participation_ratio_2019.clip(upper=1)
    g.insert(0, 'year', g.ym // 100)
    g.insert(1, 'month', g.ym % 100)
    return g.drop(columns='ym')


# %% Persistence of the latent employment process
def phi_for(e, p_ee):
    """phi such that a Gaussian AR(1) latent with threshold Phi^-1(e) has
    P(employed in both months) = p_ee."""
    q = norm.ppf(e)
    f = lambda r: multivariate_normal.cdf([q, q], cov=[[1, r], [r, 1]]) - p_ee
    lo, hi = f(-0.99), f(0.999)
    if lo > 0 or hi < 0:
        return np.nan
    return brentq(f, -0.99, 0.999, xtol=1e-5)


def implied_ue(e, phi):
    """Implied P(employed in t | unemployed in t-1) under the latent model."""
    q = norm.ppf(e)
    p_ee = multivariate_normal.cdf([q, q], cov=[[1, phi], [phi, 1]])
    return (e - p_ee) / (1 - e)


def persistence(p):
    """Consecutive-month pairs within a CPS rotation spell (MISH 1-4 or 5-8),
    both months in the labour force, first month in BASELINE_YEAR. Weighted by
    the first month's WTFINL; occupation from the first month."""
    cols = ['CPSIDV', 'month_index', 'MISH', 'E', 'U', 'job', 'WTFINL', 'YEAR']
    a = p.loc[p.CPSIDV.gt(0) & p.MISH.isin([1, 2, 3, 5, 6, 7]), cols]
    b = p.loc[p.CPSIDV.gt(0), ['CPSIDV', 'month_index', 'MISH', 'E', 'U']].copy()
    b['month_index'] -= 1
    b['MISH'] -= 1
    m = a.merge(b, on=['CPSIDV', 'month_index', 'MISH'], suffixes=('', '_next'), validate='one_to_one')
    m = m.loc[m.YEAR.eq(BASELINE_YEAR) & (m.E | m.U) & (m.E_next | m.U_next)]
    rows = []
    for job in JOBS + [ALL]:
        g = m if job == ALL else m.loc[m.job.eq(job)]
        w = g.WTFINL
        e = np.average(g.E, weights=w)
        p_ee = np.average(g.E & g.E_next, weights=w)
        eu = np.average(~g.loc[g.E, 'E_next'], weights=g.loc[g.E, 'WTFINL'])
        ue = np.average(g.loc[g.U, 'E_next'], weights=g.loc[g.U, 'WTFINL'])
        phi = phi_for(e, p_ee)
        rows.append(dict(job=job, pairs_n=len(g), pairs_from_unemployed_n=int(g.U.sum()),
                         employment_rate=e, observed_EU=eu, observed_UE=ue, phi=phi,
                         implied_UE=implied_ue(e, phi) if np.isfinite(phi) else np.nan))
    return pd.DataFrame(rows), len(m)


# %% Main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    layout = read_layout()
    p, population = load(layout)
    print(f'{population}: {len(p):,} person-months', flush=True)

    rates = employment_rates(p)
    rates.insert(0, 'population', population)
    rates.to_csv(OUT / 'cps_employment_rates_monthly.csv', index=False)

    pers, n_pairs = persistence(p)
    pers.insert(0, 'population', population)
    pers.to_csv(OUT / 'cps_employment_persistence_2019.csv', index=False)

    pooled_phi = float(pers.loc[pers.job.eq(ALL), 'phi'].iloc[0])
    settings = dict(source=str(DATA.relative_to(ROOT)), population=population,
                    period=[START_YM, END_YM], age_range=AGE_RANGE, baseline_year=BASELINE_YEAR,
                    smooth_months=SMOOTH_MONTHS, reliability=dict(min_lf_n=MIN_LF_N, min_unemployed_n=MIN_UNEMPLOYED_N),
                    occupation_ranges_occ2010=OCCUPATION_RANGES, linked_pairs_2019=n_pairs,
                    recommended_phi=pooled_phi,
                    notes=['Model input column: rate_model (raw if reliable_raw, else centred smooth).',
                           'Rates use occupation of current or last job; NILF have no occupation.',
                           'rate_exit_adjusted counts labour-force exits since 2019 as job loss.',
                           'phi from raw CPS flows; classification error in month-to-month status '
                           'biases flows up and phi down.'])
    (OUT / 'cps_employment_run_settings.json').write_text(json.dumps(settings, indent=2))

    show = rates.loc[rates.year.between(2019, 2022) & rates.month.isin([1, 4, 7, 10])]
    print(show.pivot_table(index=['year', 'month'], columns='job', values='rate_model').round(3).to_string())
    print(pers.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
