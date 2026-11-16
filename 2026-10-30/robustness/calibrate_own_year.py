"""Robustness: 2019 calibration with each household evaluated at the employment of its own interview year
(10 Nov 2026).

Run: python 2026-10-30/robustness/cps_rates_2010_2020.py     (first: CPS rates 2010-2020, same definitions)
     python 2026-10-30/robustness/calibrate_own_year.py      (about 10 minutes)

The base (calibrate_mmp.py) evaluates every MMP household (interviews 2010-2019) under 2019 employment:
a_j,t = CPS rate of the migrant's occupation group in month t of 2019, P_h = mean over the 12 months. Here
a_j,t is the CPS rate of the same group in month t of the household's survey year (surveyyr; MMP house174 has
no interview month, so P_h is the mean over the 12 months of that year). Everything else is as in the base:
Stage 1 (a_c, a_k, g) and Stage 2 (eta_w, eta_o, b0, b_any, b_c >= 0, b_k0) at kappa = 0, reusing
calibrate_mmp.stage1 / stage2 unchanged (they take the (T, M) ability matrix as given; here T = the 12 months
of each household's own year).
Outputs (aggregates only) in 2026-10-30/outputs/robustness-own-year/: stage1.csv, stage1_by_core.csv,
calibration.csv (same columns as outputs/calibration.csv, usable by run_shocks.py --calibration),
fit_by_survey_year.csv (households, observed receipt, mean ability, fitted receipt), calibration_settings.json.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import willing_able as wa                    # noqa: E402
import calibrate_mmp as cm                   # noqa: E402

OUT = wa.OUT / 'robustness-own-year'
RATES = OUT / 'cps_employment_rates_monthly_2010_2020.csv'


def rate_table():
    r = pd.read_csv(RATES)
    r['date'] = pd.to_datetime(dict(year=r.year, month=r.month, day=1))
    return r.pivot(index='date', columns='job', values=wa.abm.RATE_COLUMN)


def own_year_ability(pop, years):
    """(12, M): a_j,t = CPS rate of j's occupation group in month t of year years[hid[j]] (pi = 0, no stimulus)."""
    table = rate_table()
    a = np.zeros((12, pop['M']))
    yr_m = years[pop['hid']]
    for y in np.unique(yr_m):
        dates = pd.date_range(f'{y}-01-01', f'{y}-12-01', freq='MS')
        R = table.loc[dates, pop['rate_columns']].to_numpy(float)          # (12, n_jobs)
        assert np.isfinite(R).all() and ((R > 0) & (R < 1)).all(), y
        cols = yr_m == y
        a[:, cols] = R[:, pop['job_index'][cols]]
    return a * pop['eligible'][None, :]


def stage2_receipt(pop, a, p, kappa=0.0):
    """(H,) P_h = mean over months of E[s_h] at parameters p (dict of a_c, a_k, g, eta, b)."""
    d = wa.desire(pop, a.shape[0], p['a_c'], p['a_k'], p['eta'], g=p['g'])
    P, _ = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, d).pairs(kappa), p['b'])
    return np.clip(P.mean(0), 1e-12, 1 - 1e-12)


def loglik(P, ok, y):
    return float(np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = wa.load_population()
    ok, y = wa.target(pop)
    sy = pop['h'].surveyyr.to_numpy()
    assert np.isfinite(sy[ok]).all()
    years = np.where(np.isfinite(sy), sy, 2019).astype(int)               # non-target households: 2019 (unused)
    assert years.min() >= 2010 and years.max() <= 2019

    # Check: the derived table gives exactly the base 2019 ability.
    a_base = wa.ability(pop, wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns']))
    a_2019 = own_year_ability(pop, np.full(pop['H'], 2019))
    assert np.allclose(a_base, a_2019, atol=1e-12, rtol=0), np.abs(a_base - a_2019).max()
    a = own_year_ability(pop, years)

    (a_c, a_k, g), ll1 = cm.stage1(pop, a, ok, y)
    x0, ll0 = cm.stage1(pop, a, ok, y, fix_g=0.0)
    s1 = pd.DataFrame([dict(a_c=a_c, a_k=a_k, g=g, loglik=ll1, d_core_single=expit(a_c), d_kin=expit(a_k),
                            g0_a_c=x0[0], g0_a_k=x0[1], g0_loglik=ll0)])
    s1.to_csv(OUT / 'stage1.csv', index=False)
    print(s1.round(4).to_string(index=False), flush=True)
    fit = pd.DataFrame(dict(n_core=np.minimum(pop['n_core'], 4), observed=y,
                            fitted=cm.stage1_receipt(pop, a, (a_c, a_k, g))))[ok]
    by = fit.groupby('n_core').agg(households=('observed', 'size'), observed=('observed', 'mean'),
                                   fitted=('fitted', 'mean')).reset_index()
    by.to_csv(OUT / 'stage1_by_core.csv', index=False)
    print(by.round(3).to_string(index=False), flush=True)

    r = cm.stage2(pop, a, ok, y, a_c, a_k, g, 0.0)
    k = len(wa.B_NAMES) + len(wa.ETA_NAMES)
    cal = pd.DataFrame([dict(kappa=0.0, a_c=a_c, a_k=a_k, g=g, **dict(zip(wa.ETA_NAMES, r['eta'])),
                             **dict(zip(wa.B_NAMES, r['b'])), loglik=r['loglik'], aic=2 * k - 2 * r['loglik'])])
    cal.to_csv(OUT / 'calibration.csv', index=False)
    print(cal.round(3).to_string(index=False), flush=True)

    # Fit by survey year: base parameters under 2019 employment (the base fit), base parameters under own-year
    # employment (timing alone), own-year parameters under own-year employment (this check).
    p_base = wa.params_for(wa.load_calibration(), 0.0)
    p_own = wa.params_for(cal, 0.0)
    P_base = stage2_receipt(pop, a_base, p_base)
    P_base_own_a = stage2_receipt(pop, a, p_base)
    P_own = stage2_receipt(pop, a, p_own)
    n_m = np.bincount(pop['hid'], minlength=pop['H'])
    df = pd.DataFrame(dict(surveyyr=years, observed=y,
                           ability_own=np.bincount(pop['hid'], weights=a.mean(0), minlength=pop['H']) / n_m,
                           ability_2019=np.bincount(pop['hid'], weights=a_base.mean(0), minlength=pop['H']) / n_m,
                           fitted_base=P_base, fitted_base_params_own_year_employment=P_base_own_a,
                           fitted_own_year=P_own))[ok]
    by_year = df.groupby('surveyyr').agg(households=('observed', 'size'), **{c: (c, 'mean') for c in df.columns[1:]})
    tot = df.drop(columns='surveyyr').mean().to_frame().T.assign(households=len(df))
    tot.index = ['all']
    by_year = pd.concat([by_year, tot])
    by_year.to_csv(OUT / 'fit_by_survey_year.csv', index_label='surveyyr')
    print(by_year.round(3).to_string(), flush=True)
    lls = dict(loglik_base=loglik(P_base, ok, y), loglik_base_params_own_year_employment=loglik(P_base_own_a, ok, y),
               loglik_own_year=loglik(P_own, ok, y))
    print(lls)

    settings = dict(
        check='robustness: own interview year employment', households=int(ok.sum()), receiving=int(y[ok].sum()),
        employment='CPS rate_model of the migrant occupation group, 12 months of the household survey year '
                   '(MMP surveyyr; no interview month in house174), derived by robustness/cps_rates_2010_2020.py '
                   'with the definitions of cps_employment_rates.py; rho = 0, no stimulus',
        stage1='as calibrate_mmp.stage1', stage2='as calibrate_mmp.stage2, kappa = 0', **lls,
        households_by_survey_year={int(k): int(v) for k, v in pd.Series(years[ok]).value_counts().sort_index().items()})
    (OUT / 'calibration_settings.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
