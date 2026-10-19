"""Network-compatibility ABM (model-specification.md §2-§4).

Library module: population loading, employment process, monthly decisions.
Experiments live in separate run_*.py scripts in this folder.

Monthly order of events (spec §3):
  1. inputs for month t: occupation employment rates e_o,t, salience S_t, benefit flags
  2. employment E_j,t from a latent Gaussian AR(1) with within-network factor (rho)
  3. ability A_j,t = eligible_j AND (E_j,t OR B_j,t)
  4. core migrants send with logit theta = alpha_core + beta*S_t
  5. U_h,t = household has eligible core migrants (adult, working occupation)
     and none of them is able to send this month
  6. kin send with logit theta = alpha_kin + beta*S_t + kappa*U_h,t
     (stepping in: triggered by core inability, i.e. job loss, not by a
     routine month in which an able core migrant chooses not to send)
  7. every sender transfers its fixed routine amount m_j
"""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

ROOT = Path('C:/git-projects/covid-remittances')
HERE = Path(__file__).resolve().parents[1]
POPULATION = HERE / 'data-prep/population'          # built by data-prep/build_population.py
CPS_RATES = HERE / 'data-prep/outputs/cps_employment_rates_monthly.csv'
CPS_SETTINGS = HERE / 'data-prep/outputs/cps_employment_run_settings.json'
AMOUNTS = HERE / 'data-prep/outputs/remittance_amounts.json'
BANXICO = ROOT / 'oct-26/data-prep/outputs/mexico-rem-monthly-total.parquet'

CPS_JOBS = ['Agriculture', 'Production/construction/transport',
            'Sales/personal services', 'Professional/administrative']
ALL_OCC = 'All occupations'
# MMP job groups without a CPS occupation use the all-occupation rate.
# 'Not working' migrants are never able to send (no earnings, no benefits).
JOB_PROXY = {'Unknown': ALL_OCC, 'Working student/pensioner': ALL_OCC, 'Not working': None}
RATE_COLUMN = 'rate_model'


@dataclass
class Params:
    alpha_core: float
    alpha_kin: float
    beta: float = 0.0          # need responsiveness (map axis 1)
    kappa: float = 0.0         # kin stepping-in when core cannot send (map axis 2)
    rho: float = 0.0           # within-network job-loss correlation (network map)
    pi: float = 0.0            # share of migrants eligible for US benefits (scenario)
    kin_amount_factor: float = 1.0


@dataclass
class Inputs:
    """Monthly exogenous inputs, one entry per simulated month."""
    dates: pd.DatetimeIndex
    rates: np.ndarray                      # (T, n_rate_columns) employment rates
    salience: np.ndarray                   # (T,) S_t, mean 0 in 2019
    ui_months: np.ndarray                  # (T,) bool, federal UI supplement paid
    stimulus_months: np.ndarray            # (T,) bool, stimulus payment month
    rate_columns: list = field(default_factory=list)


# %% Population
def network_type(h):
    return np.select([(h.core_count > 0) & (h.kin_count > 0), h.core_count > 0],
                     ['Core + kin', 'Core only'], 'Kin only')


def load_population():
    h = pd.read_parquet(POPULATION / 'households.parquet').sort_values('synthetic_household_id').reset_index(drop=True)
    m = pd.read_parquet(POPULATION / 'migrants.parquet').sort_values('synthetic_migrant_id').reset_index(drop=True)
    meta = json.loads((POPULATION / 'run_settings.json').read_text())
    assert meta['connection_groups']['Core'] == [1, 2, 3]
    assert not meta['use_mmp_weights']
    hid = m.synthetic_household_id.map(pd.Series(h.index, index=h.synthetic_household_id))
    assert hid.notna().all()
    hid = hid.to_numpy(int)
    assert np.array_equal(np.bincount(hid, minlength=len(h)), h.network_size)
    assert m.connection_group.isin(['Core', 'Kin']).all()

    rate_columns = CPS_JOBS + [ALL_OCC]
    cps_job = m.job_group.map(lambda j: JOB_PROXY.get(j, j))
    assert set(cps_job.dropna()).issubset(rate_columns), set(cps_job.dropna()) - set(rate_columns)
    job_index = cps_job.map({j: i for i, j in enumerate(rate_columns)}).fillna(-1).to_numpy(int)
    adult = m.age_years.ge(18).fillna(False).to_numpy()
    eligible = adult & (job_index >= 0)
    h['network_type'] = network_type(h)
    core = m.connection_group.eq('Core').to_numpy()
    return dict(h=h, m=m, H=len(h), M=len(m), hid=hid, core=core,
                has_core=np.bincount(hid, weights=core, minlength=len(h)) > 0,
                # Trigger only counts core migrants who could ever send (adult, not 'Not working').
                has_eligible_core=np.bincount(hid, weights=core & eligible, minlength=len(h)) > 0,
                job_index=np.where(job_index >= 0, job_index, 0), eligible=eligible,
                rate_columns=rate_columns)


def routine_amounts(pop, params, seed):
    a = json.loads(AMOUNTS.read_text())
    rng = np.random.default_rng(seed)
    m = rng.lognormal(a['mu'], a['sigma'], pop['M'])
    return np.where(pop['core'], m, m * params.kin_amount_factor)


# %% Inputs
def cps_rates():
    r = pd.read_csv(CPS_RATES)
    r['date'] = pd.to_datetime(dict(year=r.year, month=r.month, day=1))
    return r.pivot(index='date', columns='job', values=RATE_COLUMN)


def make_inputs(dates, rate_columns, salience=None, ui_months=None, stimulus_months=None):
    dates = pd.DatetimeIndex(dates)
    table = cps_rates()
    missing = dates.difference(table.index)
    assert missing.empty, f'No CPS rates for {list(missing)}'
    rates = table.loc[dates, rate_columns].to_numpy(float)
    assert np.isfinite(rates).all() and ((rates > 0) & (rates < 1)).all()
    T = len(dates)
    zeros = np.zeros(T, bool)
    return Inputs(dates=dates, rates=rates,
                  salience=np.zeros(T) if salience is None else np.asarray(salience, float),
                  ui_months=zeros if ui_months is None else np.asarray(ui_months, bool),
                  stimulus_months=zeros if stimulus_months is None else np.asarray(stimulus_months, bool),
                  rate_columns=list(rate_columns))


def baseline_2019_dates(repeats=1):
    """2019 months, optionally repeated (seasonal 2019 conditions continued)."""
    one = pd.date_range('2019-01-01', '2019-12-01', freq='MS')
    return one.append([one] * (repeats - 1)) if repeats > 1 else one


def employment_phi():
    return float(json.loads(CPS_SETTINGS.read_text())['recommended_phi'])


# %% Analytic receipt probability (rho = 0)
def expected_receipt(pop, inputs, params):
    """P(household receives anything in month t), exact for rho = 0, no benefits.

    With U = all core migrants unable (only for households with core):
      P(nothing) = P(U) * Q_kin(kappa) + [P(no core sends) - P(U)] * Q_kin(0)
    because U implies no core transfer, and kin ability is independent of core
    ability when rho = 0. Q_kin(x) = prod over kin of (1 - e_j theta_kin(x)).
    Returns an (T, H) array.
    """
    hid, H, core = pop['hid'], pop['H'], pop['core']
    e = inputs.rates[:, pop['job_index']] * pop['eligible'][None, :]
    s = inputs.salience[:, None]
    theta_core = expit(params.alpha_core + params.beta * s)
    theta_kin0 = expit(params.alpha_kin + params.beta * s)
    theta_kin1 = expit(params.alpha_kin + params.beta * s + params.kappa)
    clip = lambda x: np.log1p(-np.clip(x, 0, 1 - 1e-12))
    out = np.zeros((len(inputs.dates), H))
    for t in range(len(inputs.dates)):
        hsum = lambda w: np.bincount(hid, weights=w, minlength=H)
        p_no_core_send = np.exp(hsum(np.where(core, clip(e[t] * theta_core[t]), 0)))
        p_unable = np.where(pop['has_eligible_core'], np.exp(hsum(np.where(core, clip(e[t]), 0))), 0)
        q0 = np.exp(hsum(np.where(core, 0, clip(e[t] * theta_kin0[t]))))
        q1 = np.exp(hsum(np.where(core, 0, clip(e[t] * theta_kin1[t]))))
        out[t] = 1 - (p_unable * q1 + (p_no_core_send - p_unable) * q0)
    return out


# %% Simulation
def simulate(pop, inputs, params, seed, amount_seed=None, keep_households=False):
    """Run one stochastic path. Common random numbers: same seed gives the same
    employment, eligibility and decision draws across parameter values."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    phi = employment_phi()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    amounts = routine_amounts(pop, params, seed if amount_seed is None else amount_seed)
    benefit_eligible = streams[0].random(M) < params.pi
    z = streams[1].standard_normal(M)                       # stationary start
    rows, household = [], []
    for t in range(T):
        if t > 0:
            factor = streams[2].standard_normal(H)[hid]
            eps = streams[3].standard_normal(M)
            shock = np.sqrt(params.rho) * factor + np.sqrt(1 - params.rho) * eps
            z = phi * z + np.sqrt(1 - phi ** 2) * shock
        employed = (norm.cdf(z) < inputs.rates[t, pop['job_index']]) & pop['eligible']
        benefit = benefit_eligible & pop['eligible'] & (
            inputs.stimulus_months[t] | (inputs.ui_months[t] & ~employed))
        able = employed | benefit
        u = streams[4].random(M)
        s_t = inputs.salience[t]
        send_core = able & core & (u < expit(params.alpha_core + params.beta * s_t))
        core_able = np.bincount(hid, weights=able & core, minlength=H) > 0
        trigger = pop['has_eligible_core'] & ~core_able
        theta_kin = expit(params.alpha_kin + params.beta * s_t + params.kappa * trigger[hid])
        send_kin = able & ~core & (u < theta_kin)
        sent = send_core | send_kin
        usd = np.bincount(hid, weights=amounts * sent, minlength=H)
        n = np.bincount(hid, weights=sent, minlength=H)
        rows.append(dict(date=inputs.dates[t], transfers=int(sent.sum()), core_transfers=int(send_core.sum()),
                         kin_transfers=int(send_kin.sum()), usd=float(usd.sum()),
                         receipt_rate=float((n > 0).mean()), multi_transfer_rate=float((n >= 2).mean()),
                         triggered_households=int(trigger.sum()),
                         kin_transfers_triggered=int((send_kin & trigger[hid]).sum()),
                         employed_share=float(employed[pop['eligible']].mean()),
                         able_share=float(able[pop['eligible']].mean()),
                         benefit_only_share=float((benefit & ~employed)[pop['eligible']].mean())))
        if keep_households:
            household.append(np.column_stack([n, usd]))
    out = dict(summary=pd.DataFrame(rows))
    if keep_households:
        out['household'] = np.stack(household)            # (T, H, [transfers, usd])
    return out
