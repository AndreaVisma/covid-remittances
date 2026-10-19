"""Minimal remittance ABM, calibrated to MMP receipt and Banxico transfer size.

Run: python oct-26/simulations/abms/remittance-minimal-cps-abm.py
Reads ONLY prepared CPS Excel results, never CPS microdata. Dependencies:
numpy, pandas, scipy, matplotlib, openpyxl, pyarrow. See companion notes.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% 1. Paths and editable settings. Existing ABM/population files are untouched.
ROOT = Path('C:/git-projects/covid-remittances')
POPULATION = ROOT / 'oct-26/outputs/synthetic-migrants-house-networks'
CPS_WORKBOOK = ROOT / 'oct-26/data-prep/outputs/cps_earnings_employment.xlsx'
BANXICO_FILE = ROOT / 'oct-26/data-prep/outputs/mexico-rem-monthly-total.parquet'
OUT = ROOT / 'oct-26/outputs/remittance-minimal-cps-abm'
CPS_POPULATION = 'Mexican origin'
BASELINE_YEAR = 2019
MODES = ['uncoordinated', 'coordinated']
SEEDS = [2026, 2027, 2028]
MONTHS, BURN_IN = 120, 36
FIT_MONTHS, FIT_BURN_IN, FIT_SEED = 48, 18, 901
MAX_FIT_EVALUATIONS = 30
PROFILE_PERSISTENCE = [0., 1.5, 3.]
PROFILE_MAX_EVALUATIONS = 15
MIN_SOURCE_N, MIN_SYNTHETIC_N = 30, 10
STABILITY_WINDOW = 24
RECEIPT_TOLERANCE_PP, AMOUNT_TOLERANCE_REL = 8., .05
HISTORY_START, HISTORY_END = '2020-01-01', '2022-12-01'
SAVE_PANELS = True             # First seed, both models, baseline and history.
EPS = 1e-7

# ALL seven baseline coefficients are fitted. These are numerical starts and
# search bounds, not fixed behavioural effects. Negative need associations are
# allowed; receipt data estimate associations, not causal effects of dependence.
PARAMETER_NAMES = ['core_intercept', 'kin_intercept', 'no_worker', 'child_share',
                   'old_share', 'sent_last_month', 'log_need_scale_usd']
INITIAL = np.array([.3, -5., .3, .3, .3, .8, np.log(326.)])
LOWER = np.array([-8., -12., -3., -3., -3., 0., np.log(10.)])
UPPER = np.array([5., 1., 4., 4., 4., 4., np.log(3000.)])
FD_STEP = np.array([.10, .10, .12, .12, .12, .15, .04])
AMOUNT_OBJECTIVE_WEIGHT = .5  # Moment weighting, not a behavioural coefficient.

# Explicit CPS proxies for synthetic categories lacking a known occupation.
# Overall CPS earnings/u are a proxy, not an observed wage for these agents.
# Set a mapping to None to exclude that group from labour earnings instead.
JOB_PROXY = {'Unknown': 'All occupations', 'Working student/pensioner': 'All occupations',
             'Not working': None}
# User-supplied MIT table, first column: assumed one adult, no children.
# Fixed nominal allowance includes benchmark taxes because CPS pay is gross.
# Geography/price year were not supplied; this is a scenario, not a 2019 estimate.
ANNUAL_LIVING_ALLOWANCE_USD = 20000.
MONTHLY_LIVING_ALLOWANCE_USD = ANNUAL_LIVING_ALLOWANCE_USD / 12
REMITTABLE_SURPLUS_SHARE = .20  # Share of surplus, NOT 20% of gross earnings.
EARNINGS_HETEROGENEITY = True
EARNINGS_SEED = 1731  # Fixed across calibration/validation/history.
# No savings, benefits or nonlabour income; the allowance is fixed over time.


def remittable_budget(gross_earnings):
    assert MONTHLY_LIVING_ALLOWANCE_USD >= 0
    assert 0 < REMITTABLE_SURPLUS_SHARE <= 1
    return REMITTABLE_SURPLUS_SHARE * np.maximum(
        np.asarray(gross_earnings) - MONTHLY_LIVING_ALLOWANCE_USD, 0.)


# Future extension: CSV with date,covid_need_index, zero in the baseline.
# A nonzero index requires an explicitly supplied coefficient; it cannot be
# identified from 2019 observations. Default history is labour-capacity-only.
COVID_NEED_FILE = None
COVID_NEED_COEFFICIENT = None


# %% 2. Load prepared inputs, preserve household links, and define MMP moments.
def features(frame):
    resident = frame.resident.to_numpy(float)
    raw = frame[['resident', 'resident_child', 'resident_older', 'resident_worker']].to_numpy(float)
    complete = np.isfinite(raw).all(axis=1) & (resident > 0)
    # Flagged fallbacks only simulate incomplete households; never fit them.
    den = np.where(np.isfinite(resident) & (resident > 0), resident, 1.)
    x = np.column_stack([frame.resident_worker.eq(0).astype(float),
                          np.nan_to_num(raw[:, 1]) / den, np.nan_to_num(raw[:, 2]) / den])
    complete &= (x[:, 1:] <= 1).all(axis=1)
    return np.clip(x, 0, 1), complete


def network_type(frame):
    return np.select([(frame.core_count > 0) & (frame.kin_count > 0), frame.core_count > 0],
                     ['Core + kin', 'Core only'], default='Kin only')


def prepare():
    h = pd.read_parquet(POPULATION / 'households.parquet').sort_values('synthetic_household_id').reset_index(drop=True)
    m = pd.read_parquet(POPULATION / 'migrants.parquet').sort_values('synthetic_migrant_id').reset_index(drop=True)
    s = pd.read_csv(POPULATION / 'household_network_targets.csv')
    meta = json.loads((POPULATION / 'run_settings.json').read_text())
    assert not meta['use_mmp_weights'], 'Harmonize target/population weighting before using a weighted synthetic sample.'
    assert meta['connection_groups']['Core'] == [1, 2, 3]
    assert h.synthetic_household_id.is_unique and m.synthetic_migrant_id.is_unique and s.mmp_household_id.is_unique
    hid = m.synthetic_household_id.map(pd.Series(h.index, index=h.synthetic_household_id))
    assert hid.notna().all()
    hid = hid.to_numpy(int)
    size = np.bincount(hid, minlength=len(h))
    assert np.array_equal(size, h.network_size) and (size > 0).all()
    assert m.connection_group.isin(['Core', 'Kin']).all()
    x, hc = features(h)
    sx, sc = features(s)
    h['network_type'], s['network_type'] = network_type(h), network_type(s)
    y = s.remus.map({1: 1., 2: 0.}).to_numpy()
    hc &= h.recorded_fields_complete.to_numpy(bool)
    sc &= s.recorded_fields_complete.to_numpy(bool) & s.network_size.gt(0).to_numpy() & np.isfinite(y)
    target_rows, masks = [], []
    # Three overlapping panels provide information on each need coefficient.
    # Each panel receives equal objective weight; these are not independent data.
    for panel, col in [('Workers', 0), ('Children', 1), ('Older residents', 2)]:
        for net in ['Core only', 'Kin only', 'Core + kin']:
            for present in [False, True]:
                sm = sc & s.network_type.eq(net).to_numpy() & ((sx[:, col] > 0) == present)
                hm = hc & h.network_type.eq(net).to_numpy() & ((x[:, col] > 0) == present)
                label = ('No worker' if present else 'Worker(s)') if col == 0 else ('Present' if present else 'Absent')
                used = sm.sum() >= MIN_SOURCE_N and hm.sum() >= MIN_SYNTHETIC_N
                target_rows.append(dict(panel=panel, network=net, category=label, source_n=int(sm.sum()),
                    synthetic_n=int(hm.sum()), target=float(y[sm].mean()) if sm.any() else np.nan, fit_used=used))
                masks.append(hm.astype(float) / max(hm.sum(), 1))
    targets = pd.DataFrame(target_rows)
    assert targets.fit_used.sum() >= 7
    weights = np.zeros(len(targets))
    for _, idx in targets.loc[targets.fit_used].groupby('panel').groups.items():
        weights[idx] = np.sqrt(targets.loc[idx, 'source_n'] / targets.loc[idx, 'source_n'].sum() / 3)

    # Only aggregate workbook sheets are loaded; no raw CPS file is referenced.
    with pd.ExcelFile(CPS_WORKBOOK) as book:
        cb = pd.read_excel(book, sheet_name='baseline_2019')
        ce = pd.read_excel(book, sheet_name='monthly_earnings')
        cu = pd.read_excel(book, sheet_name='monthly_employment')
    cb = cb.loc[cb.population.eq(CPS_POPULATION) & cb.year.eq(BASELINE_YEAR)].copy()
    ce, cu = [f.loc[f.population.eq(CPS_POPULATION)].copy() for f in [ce, cu]]
    assert not cb.job_group.duplicated().any()
    monthly = ce.merge(cu, on=['population', 'year', 'month', 'job_group'], validate='one_to_one', suffixes=('', '_employment'))
    monthly['date'] = pd.to_datetime(dict(year=monthly.year, month=monthly.month, day=1))
    monthly = monthly.set_index(['date', 'job_group']).sort_index()
    jobs = sorted(cb.loc[cb.job_group.ne('Unknown'), 'job_group'].tolist())
    h['calibration_eligible'] = hc
    for k, name in enumerate(['no_worker', 'child_share', 'old_share']): h[name] = x[:, k]
    m['cps_job_group'] = m.job_group.map(lambda j: JOB_PROXY.get(j, j))
    assert set(m.cps_job_group.dropna()).issubset(jobs)
    m['cps_job_is_proxy'] = m.job_group.isin([j for j, v in JOB_PROXY.items() if v is not None])
    m['labour_eligible'] = m.age_years.ge(18).fillna(False) & m.cps_job_group.notna()
    jindex = m.cps_job_group.map({j: i for i, j in enumerate(jobs)}).fillna(0).to_numpy(int)
    cb = cb.set_index('job_group').loc[jobs]
    assert cb.monthly_mean_usd.gt(0).all() and cb.unemployment_rate.between(0, 1).all()
    assert cb.earnings_reliable.all() and cb.unemployment_reliable.all()
    # Lognormal assumption: mean/median = exp(sigma^2/2).
    # Normalize each eligible synthetic job group to mean one; medians are
    # therefore approximate in finite samples. Individual rank stays fixed.
    multiplier = np.ones(len(m))
    if EARNINGS_HETEROGENEITY:
        assert cb.monthly_median_usd.gt(0).all()
        ratio = cb.monthly_mean_usd / cb.monthly_median_usd
        assert ratio.ge(1).all(), 'Lognormal inference requires mean >= median.'
        sigma = np.sqrt(2 * np.log(ratio.to_numpy()))
        rng = np.random.default_rng(EARNINGS_SEED)
        for j in range(len(jobs)):
            mask = m.labour_eligible.to_numpy() & (jindex == j)
            if mask.any():
                draws = rng.lognormal(-sigma[j]**2 / 2, sigma[j], mask.sum())
                multiplier[mask] = draws / draws.mean()
    m['earnings_multiplier'] = multiplier

    banxico = pd.read_parquet(BANXICO_FILE).copy()
    banxico['date'] = pd.to_datetime(banxico.date).dt.to_period('M').dt.to_timestamp()
    assert not banxico.date.duplicated().any()
    base_b = banxico.loc[banxico.date.dt.year.eq(BASELINE_YEAR)]
    assert len(base_b) == 12 and base_b.transactions.gt(0).all()
    # Ratio of annual dollars to annual transactions, NOT mean of monthly means.
    amount_target = base_b.total_usd.sum() / base_b.transactions.sum()
    d = dict(h=h, m=m, source=s, source_x=sx, source_complete=sc, source_y=y,
        H=len(h), M=len(m), hid=hid, core=m.connection_group.eq('Core').to_numpy(), x=x,
        sizes=size, offsets=np.r_[0, np.cumsum(size)[:-1]], adult=m.age_years.ge(18).fillna(False).to_numpy(),
        eligible=m.labour_eligible.to_numpy(), jindex=jindex, jobs=jobs, cps_base=cb, cps_monthly=monthly,
        masks=np.array(masks), targets=targets, weights=weights, amount_target=float(amount_target), banxico=banxico)
    OUT.mkdir(parents=True, exist_ok=True)
    targets.to_csv(OUT / 'mmp_targets.csv', index=False)
    cb.reset_index().to_csv(OUT / 'cps_baseline_used.csv', index=False)
    print(f'{len(h):,} households, {len(m):,} links; Banxico target ${amount_target:.2f}; {targets.fit_used.sum()} receipt moments.', flush=True)
    return d


# %% 3. External employment and capacity. Rates are stocks, NOT layoff hazards.
def make_bank(d, months, seed, dates=None):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    work_draw = streams[0].random((months, d['M']))
    choice_draw = streams[1].random((months, d['M']))
    tie = streams[2].random((months, d['M']))
    wage = np.tile(d['cps_base'].monthly_mean_usd.to_numpy(), (months, 1))
    u = np.tile(d['cps_base'].unemployment_rate.to_numpy(), (months, 1))
    low_quality = np.zeros((months, len(d['jobs'])), dtype=bool)
    if dates is not None:
        assert len(dates) <= months
        for t, date in enumerate(dates, start=months - len(dates)):
            rows = d['cps_monthly'].loc[date].loc[d['jobs']]
            wage[t], u[t] = rows.monthly_mean_usd.to_numpy(), rows.unemployment_rate.to_numpy()
            low_quality[t] = ~(rows.earnings_reliable & rows.unemployment_reliable).to_numpy()
    assert np.isfinite(wage).all() and (wage > 0).all()
    assert np.isfinite(u).all() and ((u >= 0) & (u <= 1)).all()
    # Independent monthly employment-status draws reproduce occupational stock
    # rates in expectation. Joblessness duration is NOT estimated from CPS.
    employed = (work_draw >= u[:, d['jindex']]) & d['eligible'][None, :]
    gross = wage[:, d['jindex']] * d['m'].earnings_multiplier.to_numpy()[None, :] * employed
    capacity = remittable_budget(gross)
    return dict(choice=choice_draw, tie=tie, employed=employed, capacity=capacity,
                wages=wage, gross_earnings=gross, unemployment=u, low_quality=low_quality)


def allocate(willing, capacity, need, previous, tie, d, mode):
    """Budget/remaining need affect positive amounts only, NEVER theta."""
    hid = d['hid']
    if mode == 'uncoordinated':
        transfer = willing * np.minimum(capacity, need[hid])
        bound = willing & (capacity > EPS) & (capacity + EPS < need[hid])
        return transfer, bound, np.zeros(d['M'], bool)
    assert mode == 'coordinated'
    order = np.lexsort((tie, ~previous, hid))
    raw = (capacity * willing)[order]
    # Prefix of offered budgets within each household gives the remaining need.
    # Once a household is covered, all later contributions are zero. No rank loop.
    prefix = np.r_[0., np.cumsum(raw)[:-1]]
    before = prefix - prefix[d['offsets']][hid[order]]
    remaining = np.maximum(need[hid[order]] - before, 0.)
    amounts = np.minimum(raw, remaining)
    amounts[amounts <= EPS] = 0.
    transfer, bound, crowded = np.zeros(d['M']), np.zeros(d['M'], bool), np.zeros(d['M'], bool)
    transfer[order] = amounts
    bound[order] = (raw > EPS) & (raw + EPS < remaining)
    crowded[order] = (raw > EPS) & (remaining <= EPS)
    return transfer, bound, crowded


# %% 4. Minimal monthly ABM: six theta coefficients and one amount scale.
def simulate(params, d, bank, mode, burn, collect=False, initial_active=0., covid_log_need=None):
    T, M = bank['choice'].shape
    assert len(params) == 7 and 0 <= burn < T
    previous = np.random.default_rng(341).random(M) < initial_active
    need_index = d['x'] @ params[2:5]
    base_theta = np.where(d['core'], params[0], params[1]) + need_index[d['hid']]
    expected_sum, actual_sum = np.zeros(d['H']), np.zeros(d['H'])
    amount_sum, count_sum = 0., 0
    logs, hh, agents, availability, earnings, budgets = [], [], [], [], [], []
    for t in range(T):
        shock = 0. if covid_log_need is None else covid_log_need[t]
        shock_h = np.broadcast_to(shock, (d['H'],))
        need = np.exp(params[6] + need_index + shock_h)
        theta = base_theta + params[5] * previous + shock_h[d['hid']]
        probability = expit(theta)
        willing = (bank['choice'][t] < probability) & d['adult']
        capacity = bank['capacity'][t]
        p_positive = probability * (capacity > EPS)
        # Conditional receipt on last month's state and current employment.
        q = -np.expm1(np.bincount(d['hid'], weights=np.log1p(-np.minimum(p_positive, 1-1e-12)), minlength=d['H']))
        transfer, bound, crowded = allocate(willing, capacity, need, previous, bank['tie'][t], d, mode)
        sent = transfer > EPS
        total = np.bincount(d['hid'], weights=transfer, minlength=d['H'])
        n = np.bincount(d['hid'], weights=sent, minlength=d['H']).astype(int)
        assert (transfer >= 0).all() and (transfer <= capacity + EPS).all()
        if mode == 'coordinated': assert (total <= need + 1e-5).all()
        if t >= burn:
            expected_sum += q
            actual_sum += n > 0
            amount_sum += transfer.sum()
            count_sum += sent.sum()
        if collect:
            retained_h = np.bincount(d['hid'], weights=sent & previous, minlength=d['H'])
            prior_h = np.bincount(d['hid'], weights=previous, minlength=d['H']) > 0
            rec = dict(month=t+1, receipt_rate=(n > 0).mean(), senders=int(sent.sum()),
                core_senders=int((sent & d['core']).sum()), kin_senders=int((sent & ~d['core']).sum()),
                total_usd=transfer.sum(), mean_transfer_usd=transfer[sent].mean() if sent.any() else np.nan,
                mean_recipient_usd=total[n > 0].mean() if (n > 0).any() else np.nan,
                willing=int(willing.sum()), willing_no_capacity=int((willing & (capacity <= EPS)).sum()),
                crowded_out=int(crowded.sum()), budget_bound=int(bound.sum()), new_senders=int((sent & ~previous).sum()),
                continuing_senders=int((sent & previous).sum()),
                replacement_households=int((prior_h & (n > 0) & (retained_h == 0)).sum()),
                multi_sender_rate=(n >= 2).mean(), excess_support_rate=(total > need + EPS).mean(),
                unmet_share=np.maximum(need-total, 0).sum()/need.sum(),
                employed_share=bank['employed'][t, d['eligible']].mean())
            if 'gross_earnings' in bank:
                gross = bank['gross_earnings'][t]
                rec['employed_no_surplus'] = int((bank['employed'][t] & (capacity <= EPS)).sum())
                rec['remittance_gross_share_senders'] = transfer.sum()/gross[sent].sum() if sent.any() else np.nan
                rec['remittance_gross_share_all'] = transfer.sum()/gross.sum() if gross.sum() > 0 else np.nan
                earnings.append(gross.copy())
                budgets.append(capacity.copy())
            logs.append(rec)
            hh.append(np.column_stack([total, n, need]))
            agents.append(transfer.copy())
            availability.append(bank['employed'][t].copy())
        previous = sent
    return dict(moments=d['masks'] @ (expected_sum/(T-burn)),
        realized=d['masks'] @ (actual_sum/(T-burn)), household_receipt=actual_sum/(T-burn),
        mean_transfer=amount_sum/max(count_sum, 1), summary=pd.DataFrame(logs), household=hh,
        transfers=agents, employed=availability, gross_earnings=earnings, budgets=budgets, objective_count=count_sum)


# %% 5. Simulated-moment calibration; no fixed need or persistence coefficients.
def fit_model(d, bank, mode, start=INITIAL, fixed_rho=None, max_nfev=MAX_FIT_EVALUATIONS):
    used = d['targets'].fit_used.to_numpy()
    goal = d['targets'].target.to_numpy()[used]
    free = np.arange(7) if fixed_rho is None else np.array([0, 1, 2, 3, 4, 6])
    anchor = start.copy()
    if fixed_rho is not None: anchor[5] = fixed_rho
    cache, history = {}, []
    def residual(values):
        key = tuple(values)
        if key in cache: return cache[key]
        p = anchor.copy(); p[free] = values
        result = simulate(p, d, bank, mode, FIT_BURN_IN)
        r = np.r_[d['weights'][used] * (result['moments'][used]-goal),
                  AMOUNT_OBJECTIVE_WEIGHT * np.log(max(result['mean_transfer'], EPS)/d['amount_target'])]
        history.append(dict(evaluation=len(history)+1, loss=float(r@r),
                            receipt_rmse_pp=float(100*np.linalg.norm(r[:-1])),
                            mean_transfer_usd=result['mean_transfer'], **dict(zip(PARAMETER_NAMES, p))))
        cache[key] = r
        return r
    # Absolute finite differences avoid near-zero numerical derivatives from
    # stochastic sender histories. Common random numbers make comparisons fair.
    def jac(values):
        base = residual(values)
        columns = []
        for j, k in enumerate(free):
            step = FD_STEP[k] if values[j] + FD_STEP[k] <= UPPER[k] else -FD_STEP[k]
            trial = values.copy(); trial[j] += step
            columns.append((residual(trial)-base)/step)
        return np.column_stack(columns)
    fit = least_squares(residual, anchor[free], jac=jac, bounds=(LOWER[free], UPPER[free]),
                        max_nfev=max_nfev, ftol=.002, xtol=.002, gtol=1e-5, x_scale='jac')
    # Keep the best actually evaluated full parameter vector, including search
    # probes; report optimizer status separately from fit quality.
    hist = pd.DataFrame(history)
    best = hist.loc[hist.loss.idxmin(), PARAMETER_NAMES].to_numpy(float)
    return best, hist, dict(success=bool(fit.success), message=str(fit.message), nfev=int(fit.nfev))


def validate(d, parameters):
    monthly, comparisons, stability, first = [], [], [], {}
    for mode, p in parameters.items():
        for seed in SEEDS:
            bank = make_bank(d, MONTHS, seed)
            r = simulate(p, d, bank, mode, BURN_IN, collect=True)
            if seed == SEEDS[0]: first[mode] = r
            monthly.append(r['summary'].assign(mode=mode, seed=seed))
            v = d['targets'].copy()
            v['simulated_receipt'], v['expected_receipt'] = r['realized'], r['moments']
            v['error_pp'] = 100*(v.simulated_receipt-v.target)
            v['mode'], v['seed'] = mode, seed
            comparisons.append(v)
            a, b = r['summary'].iloc[-2*STABILITY_WINDOW:-STABILITY_WINDOW], r['summary'].iloc[-STABILITY_WINDOW:]
            change = 100*(b.receipt_rate.mean()-a.receipt_rate.mean())
            amount_change = b.total_usd.mean()/a.total_usd.mean()-1
            stability.append(dict(mode=mode, seed=seed, receipt_change_pp=change, total_change_relative=amount_change,
                practical_stability=abs(change)<=3 and abs(amount_change)<=.2,
                mean_transfer_usd=r['mean_transfer'], amount_error_relative=r['mean_transfer']/d['amount_target']-1))
        print(f'Validated {mode}.', flush=True)
    return pd.concat(monthly), pd.concat(comparisons), pd.DataFrame(stability), first


# %% 6. Real 2020-2022 CPS paths and same-seed constant-resource counterfactuals.
def historical_runs(d, parameters):
    dates = pd.date_range(HISTORY_START, HISTORY_END, freq='MS')
    T = BURN_IN + len(dates)
    actual_bank = make_bank(d, T, SEEDS[0], dates)
    constant_bank = make_bank(d, T, SEEDS[0])
    shocks = np.zeros(T)
    if COVID_NEED_FILE is not None:
        assert COVID_NEED_COEFFICIENT is not None, 'Nonzero COVID need requires an explicit coefficient.'
        z = pd.read_csv(COVID_NEED_FILE, parse_dates=['date']).set_index('date').covid_need_index.reindex(dates)
        assert z.notna().all() and np.isfinite(z).all()
        shocks[BURN_IN:] = COVID_NEED_COEFFICIENT * z.to_numpy()
    rows, runs = [], {}
    for mode, p in parameters.items():
        for scenario, bank in [('Observed CPS capacity', actual_bank), ('Constant 2019 capacity', constant_bank)]:
            r = simulate(p, d, bank, mode, BURN_IN, collect=True)
            rows.append(r['summary'].iloc[BURN_IN:].assign(date=dates, mode=mode, scenario=scenario))
            if scenario == 'Observed CPS capacity': runs[mode] = r
        if COVID_NEED_FILE is not None:
            r = simulate(p, d, actual_bank, mode, BURN_IN, collect=True, covid_log_need=shocks)
            rows.append(r['summary'].iloc[BURN_IN:].assign(date=dates, mode=mode, scenario='CPS plus supplied need'))
    inputs = []
    for t, date in enumerate(dates, start=BURN_IN):
        for j, job in enumerate(d['jobs']):
            mask = d['eligible'] & (d['jindex'] == j)
            inputs.append(dict(date=date, job_group=job, monthly_gross_earnings_usd=actual_bank['wages'][t,j],
                cps_unemployment_rate=actual_bank['unemployment'][t,j],
                simulated_unemployment_rate=1-actual_bank['employed'][t,mask].mean() if mask.any() else np.nan,
                low_sample_quality=actual_bank['low_quality'][t,j], synthetic_links=int(mask.sum())))
    return pd.concat(rows), runs, pd.DataFrame(inputs), dates


def save_panels(d, result, name, dates=None):
    if not SAVE_PANELS: return
    hh = np.stack(result['household'])
    transfer = np.stack(result['transfers'])
    employed = np.stack(result['employed'])
    start = 0 if dates is None else len(hh)-len(dates)
    hh, transfer, employed = hh[start:], transfer[start:], employed[start:]
    months = np.arange(start+1, start+len(hh)+1)
    hp = pd.DataFrame(dict(month=np.repeat(months, d['H']),
        synthetic_household_id=np.tile(d['h'].synthetic_household_id, len(hh)), received_usd=hh[:,:,0].ravel(),
        n_senders=hh[:,:,1].astype(int).ravel(), perceived_need_usd=hh[:,:,2].ravel()))
    mp = pd.DataFrame(dict(month=np.repeat(months, d['M']),
        synthetic_migrant_id=np.tile(d['m'].synthetic_migrant_id, len(hh)),
        synthetic_household_id=np.tile(d['m'].synthetic_household_id, len(hh)),
        transfer_usd=transfer.ravel(), employed=employed.ravel(), sent=transfer.ravel()>EPS))
    if result['gross_earnings']:
        mp['gross_earnings_usd'] = np.stack(result['gross_earnings'])[start:].ravel()
        mp['remittable_budget_usd'] = np.stack(result['budgets'])[start:].ravel()
    if dates is not None:
        hp['date'], mp['date'] = np.repeat(dates, d['H']), np.repeat(dates, d['M'])
    hp.to_parquet(OUT / f'{name}_household_months.parquet', index=False)
    mp.to_parquet(OUT / f'{name}_migrant_months.parquet', index=False)


# %% 7. Diagnostic charts. Monetary means refer to positive link-transfers.
def plots(d, monthly, validation, first, parameters, profile, histories, history, inputs, warm, comparison):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'uncoordinated':'#2465a0', 'coordinated':'#b75d24'}
    def save(fig, name, note='MMP-like household links; gross CPS earnings ceilings; no national population weighting.'):
        fig.text(.015,.012,note,fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.055,1,.96)); fig.savefig(OUT / name,dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for ax,col,label,factor in zip(axes.flat,['receipt_rate','senders','mean_transfer_usd','excess_support_rate'],
        ['Households receiving (%)','Active sending links','Mean positive transfer (USD)','Households above perceived need (%)'],[100,1,1,100]):
        for (mode,seed),g in monthly.groupby(['mode','seed']):
            ax.plot(g.month,g[col]*factor,color=colors[mode],alpha=.35,label=mode if seed==SEEDS[0] else None)
        ax.axvspan(1,BURN_IN,color='grey',alpha=.08); ax.set(title=label,xlabel='Model month')
    axes[1,0].axhline(d['amount_target'],color='black',ls=':',label='Banxico 2019')
    axes[0,0].legend(); axes[1,0].legend(fontsize=8)
    fig.suptitle('Stationary 2019 baseline: separately calibrated mechanisms'); save(fig,'01_baseline_dynamics.png')

    fig, axes = plt.subplots(1,3,figsize=(16,6),sharex=True)
    for ax,panel in zip(axes,['Workers','Children','Older residents']):
        t = d['targets'].loc[d['targets'].panel.eq(panel)].reset_index()
        y = np.arange(len(t))
        ax.scatter(t.target*100,y,c='black',marker='D',label='MMP')
        for k,mode in enumerate(MODES):
            v = validation.loc[validation['mode'].eq(mode)&validation.panel.eq(panel)].groupby(['network','category']).simulated_receipt.mean()
            ax.scatter([100*v.loc[(r.network,r.category)] for r in t.itertuples()],y+.13*(k+1),label=mode,color=colors[mode])
        ax.set(yticks=y,yticklabels=t.network+' / '+t.category,title=panel,xlabel='Receipt (%)',xlim=(0,100))
        ax.tick_params(axis='y',labelsize=8)
    axes[0].legend(fontsize=8); fig.suptitle('MMP receipt targets and independent-seed simulated averages')
    save(fig,'02_receipt_fit.png','Overlapping need panels; sample-size cutoffs apply. These are calibration checks, not held-out survey validation.')

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for mode,r in first.items():
        amounts = np.stack(r['transfers'])[BURN_IN:]; hh=np.stack(r['household'])[BURN_IN:]
        axes[0].hist(amounts[amounts>EPS],bins=40,density=True,histtype='step',label=mode,color=colors[mode])
        ratio=(hh[:,:,0]/hh[:,:,2]).ravel()
        axes[1].hist(ratio,bins=np.linspace(0,max(3,np.quantile(ratio,.995)),45),density=True,histtype='step',label=mode,color=colors[mode])
    axes[0].axvline(d['amount_target'],ls=':',color='black'); axes[0].set(title='Positive transfer distribution',xlabel='USD',ylabel='Density')
    axes[1].axvline(1,ls=':',color='black'); axes[1].set(title='Household receipts / perceived need',xlabel='Ratio',ylabel='Density')
    axes[0].legend(); axes[1].legend(); save(fig,'03_amount_distributions.png','Banxico constrains the mean transaction size, not the entire transfer distribution or household receipt amounts.')

    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for mode,r in first.items():
        g=r['summary']
        for ax,col,title in zip(axes.flat,['new_senders','continuing_senders','crowded_out','willing_no_capacity'],
                               ['New this month','Continuing senders','Willing but already covered','Willing but no labour capacity']):
            ax.plot(g.month,g[col],label=mode,color=colors[mode]); ax.set(title=title,xlabel='Model month')
    axes[0,0].legend(); save(fig,'04_sending_responsibility.png','New means inactive last month. Employment status is sampled monthly; joblessness durations are not estimated.')

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for mode in MODES:
        g=profile.loc[profile['mode'].eq(mode)].sort_values('rho')
        axes[0].plot(g.rho,g.loss,marker='o',label=mode,color=colors[mode])
        axes[1].plot(g.rho,g.mean_transfer_usd,marker='o',label=mode,color=colors[mode])
        best=histories.loc[histories['mode'].eq(mode)].sort_values('loss').iloc[0]
        axes[0].scatter(parameters[mode][5],best.loss,marker='*',s=130,color=colors[mode])
    axes[0].set(title='Persistence: refit all other coefficients',xlabel='Fixed persistence for sensitivity',ylabel='Calibration objective')
    axes[1].axhline(d['amount_target'],ls=':',color='black'); axes[1].set(title='Amount fit along persistence profile',xlabel='Persistence',ylabel='USD per positive transfer')
    axes[0].legend(); save(fig,'05_persistence_profile.png','Finite-budget numerical profiles are identification diagnostics, not confidence intervals; inspect optimizer flags.')

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for mode in MODES:
        p=parameters[mode]; axes[0].plot(np.arange(6),p[:6],marker='o',label=mode,color=colors[mode])
        g=histories.loc[histories['mode'].eq(mode)]; axes[1].plot(g.evaluation,g.loss,label=mode,color=colors[mode])
    axes[0].set(xticks=np.arange(6),xticklabels=['Core','Kin','No worker','Child share','Old share','Persistence'],ylabel='Fitted coefficient',title='Every theta coefficient fitted')
    axes[0].tick_params(axis='x',rotation=25); axes[0].axhline(0,color='grey',lw=.7); axes[0].legend()
    axes[1].set(xlabel='Parameter evaluation',ylabel='Objective',title='Calibration search'); save(fig,'06_parameters_and_search.png')

    fig, axes=plt.subplots(2,2,figsize=(12,8))
    for job,g in inputs.loc[inputs.job_group.ne('All occupations')].groupby('job_group'):
        axes[0,0].plot(g.date,g.monthly_gross_earnings_usd,label=job)
        axes[0,1].plot(g.date,100*g.cps_unemployment_rate,label=job)
        axes[1,0].scatter(100*g.cps_unemployment_rate,100*g.simulated_unemployment_rate,s=12,label=job)
        axes[1,1].plot(g.date,g.low_sample_quality.astype(int),label=job,alpha=.6)
    axes[0,0].set(title='Observed conditional gross earnings',ylabel='USD / month'); axes[0,0].legend(fontsize=7)
    axes[0,1].set(title='Observed occupational unemployment',ylabel='%'); axes[1,0].plot([0,40],[0,40],ls=':',color='grey')
    axes[1,0].set(xlabel='CPS unemployment (%)',ylabel='Simulated unemployment (%)',title='Rates sampled into agent status')
    axes[1,1].set(title='Input sample flag (1 = sparse)',yticks=[0,1]); fig.autofmt_xdate(); save(fig,'07_cps_inputs.png')

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for (mode,scenario),g in history.groupby(['mode','scenario']):
        for ax,col,title,factor in zip(axes.flat,['receipt_rate','total_usd','mean_transfer_usd','replacement_households'],
            ['Household receipt (%)','Total simulated remittances (USD)','Mean positive transfer (USD)','Receiving households with all-new senders'],[100,1,1,1]):
            ax.plot(g.date,g[col]*factor,color=colors[mode],ls='--' if scenario=='Constant 2019 capacity' else '-',label=f'{mode}: {scenario}')
            ax.set_title(title)
    axes[0,0].legend(fontsize=6); fig.autofmt_xdate(); save(fig,'08_cps_history_counterfactual.png','Capacity-only COVID-era exercise by default: household need is unchanged; these paths are not a fitted COVID explanation.')

    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for mode,r in first.items():
        a=r['summary'].receipt_rate.rolling(12,min_periods=1).mean()*100
        b=warm[mode]['summary'].receipt_rate.rolling(12,min_periods=1).mean()*100
        axes[0].plot(np.arange(1,MONTHS+1),a,color=colors[mode],label=mode+' empty start')
        axes[0].plot(np.arange(1,MONTHS+1),b,color=colors[mode],ls='--',label=mode+' 50% initial flags')
    axes[0].set(title='Initialization sensitivity',xlabel='Model month',ylabel='12-month mean receipt (%)'); axes[0].legend(fontsize=7)
    for mode,g in comparison.groupby('allocation_mode'):
        axes[1].bar(np.arange(len(g)) + (0 if mode=='uncoordinated' else .35),g.mean_transfer_usd,width=.35,label=mode)
    axes[1].set(xticks=np.arange(2)+.175,xticklabels=['Uncoordinated fit','Coordinated fit'],ylabel='Mean transfer USD',title='Same parameters, different allocation')
    axes[1].legend(); save(fig,'09_initialization_and_mechanism.png')

    chosen=[]
    for mask in [d['h'].network_size.eq(1),d['h'].network_type.eq('Kin only'),d['h'].network_type.eq('Core + kin')]:
        possible=np.flatnonzero(mask)
        if len(possible): chosen.append(possible[0])
    fig,axes=plt.subplots(len(chosen),1,figsize=(12,3*len(chosen)),squeeze=False)
    for ax,i in zip(axes.flat,chosen):
        for mode,r in first.items():
            hh=np.stack(r['household']); ax.plot(hh[:,i,0],label=mode,color=colors[mode],alpha=.75)
            ax.axhline(hh[0,i,2],color=colors[mode],ls=':',lw=1)
        ax.set(title=f'Household {d["h"].synthetic_household_id.iloc[i]}: {d["h"].core_count.iloc[i]} core + {d["h"].kin_count.iloc[i]} kin',ylabel='USD',xlabel='Model month (zero indexed)')
        ax.legend(fontsize=8)
    save(fig,'10_household_paths.png','Dotted lines: fitted perceived need under each mechanism. Examples are illustrative, not representative.')


def additional_diagnostics(d, parameters, monthly, history, first):
    """Explain drivers and expose restrictions rather than just showing fit."""
    colors={'uncoordinated':'#2465a0','coordinated':'#b75d24'}
    def save(fig,name,note):
        fig.text(.015,.01,note,fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.05,1,.96)); fig.savefig(OUT/name,dpi=150); plt.close(fig)
    # The common need slopes imply the same demographic effect for core and kin.
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    typical=d['x'][d['h'].calibration_eligible.to_numpy()].mean(axis=0)
    for k,label in enumerate(['No worker (0/1)','Child share (under 12)','Old share (65+)']):
        grid=np.array([0.,1.]) if k==0 else np.linspace(0,1,50)
        xx=np.tile(typical,(len(grid),1)); xx[:,k]=grid
        for mode,p in parameters.items():
            need=xx@p[2:5]
            axes[0,k].plot(grid,100*expit(p[0]+need),color=colors[mode],label=mode+' core')
            axes[0,k].plot(grid,100*expit(p[1]+need),color=colors[mode],ls='--',label=mode+' kin')
            axes[1,k].plot(grid,np.exp(p[6]+need),color=colors[mode],label=mode)
        axes[0,k].set(xlabel=label,ylabel='Willingness (%)')
        axes[1,k].set(xlabel=label,ylabel='Perceived household need (USD)')
    axes[0,0].legend(fontsize=7); axes[1,0].legend(fontsize=8)
    fig.suptitle('Fitted need responses: other household features at their mean, no previous sending')
    save(fig,'11_need_response_curves.png','Associational calibration, not causal need estimates. Budgets and previous contributions do not enter these willingness curves.')

    fig,axes=plt.subplots(1,3,figsize=(14,5))
    after=monthly.loc[monthly.month>BURN_IN]
    for i,mode in enumerate(MODES):
        g=after.loc[after['mode'].eq(mode)]
        axes[0].bar(i,100*g.budget_bound.sum()/max(g.senders.sum(),1),color=colors[mode])
        amounts=np.stack(first[mode]['household'])[BURN_IN:,:,2].mean(axis=0)
        cap=remittable_budget(d['cps_base'].monthly_mean_usd.to_numpy()[d['jindex']] * d['m'].earnings_multiplier.to_numpy())
        ratio=cap[d['eligible']]/amounts[d['hid'][d['eligible']]]
        axes[1].hist(ratio,bins=35,histtype='step',density=True,label=mode,color=colors[mode])
        axes[2].bar(i,100*g.willing_no_capacity.sum()/max(g.willing.sum(),1),color=colors[mode])
    axes[0].set(xticks=[0,1],xticklabels=MODES,ylabel='Positive transfers (%)',title='Budget ceiling binds')
    axes[0].set_ylim(0,max(1,axes[0].get_ylim()[1]))
    axes[1].axvline(1,color='grey',ls=':'); axes[1].set(xlabel='Employed remittable budget / perceived need',title='How generous is the capacity ceiling?'); axes[1].legend(fontsize=8)
    axes[2].set(xticks=[0,1],xticklabels=MODES,ylabel='Willing links (%)',title='Willing but no labour capacity')
    save(fig,'12_capacity_constraints.png','Budget is a share of earnings above the fixed living allowance; unemployment and low pay can prevent sending.')

    # Moments by network size were NOT part of the calibration objective.
    bins=[1,2,4,9,np.inf]; names=['1','2','3-4','5-9','10+']
    def bincode(x): return np.searchsorted(bins,np.asarray(x),side='left')
    sb,hb=bincode(d['source'].network_size),bincode(d['h'].network_size)
    rows=[]
    fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
    for ax,net in zip(axes,['Core only','Kin only','Core + kin']):
        for k,name in enumerate(names):
            sm=d['source_complete'] & d['source'].network_type.eq(net).to_numpy() & (sb==k)
            hm=d['h'].calibration_eligible.to_numpy() & d['h'].network_type.eq(net).to_numpy() & (hb==k)
            target=d['source_y'][sm].mean() if sm.any() else np.nan
            eligible=sm.sum()>=MIN_SOURCE_N and hm.sum()>=MIN_SYNTHETIC_N
            for mode,r in first.items():
                value=r['household_receipt'][hm].mean() if hm.any() else np.nan
                rows.append(dict(network=net,network_size_bin=name,mode=mode,source_n=int(sm.sum()),
                    synthetic_n=int(hm.sum()),mmp_receipt=target,simulated_receipt=value,sufficient_sample=eligible))
        sub=pd.DataFrame(rows).loc[lambda f:f.network.eq(net)]
        for mode in MODES:
            g=sub.loc[sub['mode'].eq(mode)].copy()
            vals=g.simulated_receipt.where(g.sufficient_sample)
            ax.plot(np.arange(5),100*vals,marker='o',color=colors[mode],label=mode)
            if mode==MODES[0]: ax.scatter(np.arange(5),100*g.mmp_receipt.where(g.sufficient_sample),marker='D',color='black',label='MMP')
        ax.set(xticks=np.arange(5),xticklabels=names,title=net,xlabel='Connected migrants',ylabel='Receipt (%)',ylim=(0,100))
    axes[0].legend(fontsize=8)
    pd.DataFrame(rows).to_csv(OUT/'network_size_diagnostics.csv',index=False)
    save(fig,'13_network_size_check.png','Additional moments, not included in fitting. Sparse cells are omitted from the plot and retained with flags in the CSV.')

    fig,axes=plt.subplots(1,3,figsize=(15,5))
    actual=d['banxico'].loc[d['banxico'].date.between(HISTORY_START,HISTORY_END)].copy()
    b=d['banxico'].loc[d['banxico'].date.dt.year.eq(BASELINE_YEAR)]
    actual['mean_transfer']=actual.total_usd/actual.transactions
    controls=[b.total_usd.mean(),b.transactions.mean(),d['amount_target']]
    for ax,acol,scol,control,title in zip(axes,['total_usd','transactions','mean_transfer'],
        ['total_usd','senders','mean_transfer_usd'],controls,['Total remittances','Transactions / positive link transfers','Mean transaction size']):
        ax.plot(actual.date,100*actual[acol]/control,color='black',lw=2,label='Banco de Mexico')
        for mode,g in history.loc[history.scenario.eq('Observed CPS capacity')].groupby('mode'):
            bm=after.loc[after['mode'].eq(mode)]
            denominator=bm.total_usd.sum()/bm.senders.sum() if scol=='mean_transfer_usd' else bm[scol].mean()
            ax.plot(g.date,100*g[scol]/denominator,color=colors[mode],label=mode)
        ax.axhline(100,color='grey',ls=':'); ax.set(title=title,ylabel='Index: 2019 baseline = 100')
    axes[0].legend(fontsize=8); fig.autofmt_xdate()
    save(fig,'14_banxico_growth_context.png','National growth is contextual, not fitted. Capacity-only paths omit heightened need, exchange rates and nonlabour resources.')


# %% 8. Run, export panels and all calibration/diagnostic results.
def run():
    assert MONTHS-BURN_IN >= 2*STABILITY_WINDOW
    d=prepare(); fit_bank=make_bank(d,FIT_MONTHS,FIT_SEED)
    parameters, histories, statuses, profiles = {}, [], {}, []
    for mode in MODES:
        starts=[INITIAL.copy(), INITIAL.copy()]; starts[1][5]=2.5
        candidates=[]
        for i,start in enumerate(starts):
            p,hist,status=fit_model(d,fit_bank,mode,start)
            candidates.append((p,hist,status))
            print(f'{mode} start {i+1}: best loss {hist.loss.min():.6f}',flush=True)
        p,hist,status=min(candidates,key=lambda z:z[1].loss.min())
        parameters[mode],statuses[mode]=p,status
        histories.append(hist.assign(mode=mode))
        for rho in PROFILE_PERSISTENCE:
            pp,ph,ps=fit_model(d,fit_bank,mode,p,fixed_rho=rho,max_nfev=PROFILE_MAX_EVALUATIONS)
            best=ph.loc[ph.loss.idxmin()]
            profiles.append(dict(mode=mode,rho=rho,loss=best.loss,mean_transfer_usd=best.mean_transfer_usd,
                optimizer_success=ps['success'],**dict(zip(PARAMETER_NAMES,pp))))
            print(f'{mode} persistence {rho}: loss {best.loss:.6f}',flush=True)
    histories=pd.concat(histories,ignore_index=True); profile=pd.DataFrame(profiles)
    monthly,validation,stability,first=validate(d,parameters)
    warm={mode:simulate(p,d,make_bank(d,MONTHS,SEEDS[0]),mode,BURN_IN,collect=True,initial_active=.5) for mode,p in parameters.items()}
    comparison=[]
    for fitted_mode,p in parameters.items():
        for allocation_mode in MODES:
            r=simulate(p,d,make_bank(d,MONTHS,SEEDS[0]),allocation_mode,BURN_IN)
            comparison.append(dict(fitted_mode=fitted_mode,allocation_mode=allocation_mode,mean_transfer_usd=r['mean_transfer'],
                                   receipt_rate=r['household_receipt'].mean()))
    comparison=pd.DataFrame(comparison)
    history,runs,inputs,dates=historical_runs(d,parameters)
    for name,frame in dict(monthly_summary=monthly,receipt_validation=validation,stability_checks=stability,
        calibration_history=histories,persistence_profile=profile,mechanism_comparison=comparison,
        cps_history=history,cps_monthly_inputs=inputs).items():
        frame.to_csv(OUT/f'{name}.csv',index=False)
    monthly.to_json(OUT/'monthly_summary.json',orient='records',indent=2)
    d['h'].to_parquet(OUT/'household_baseline.parquet',index=False)
    d['m'].to_parquet(OUT/'migrant_baseline.parquet',index=False)
    for mode in MODES:
        save_panels(d,first[mode],f'baseline_{mode}')
        save_panels(d,runs[mode],f'history_{mode}',dates)
    plots(d,monthly,validation,first,parameters,profile,histories,history,inputs,warm,comparison)
    additional_diagnostics(d,parameters,monthly,history,first)
    result={}
    for mode,p in parameters.items():
        v=validation.loc[validation['mode'].eq(mode)&validation.fit_used]
        st=stability.loc[stability['mode'].eq(mode)]
        result[mode]=dict(parameters=dict(zip(PARAMETER_NAMES,p.tolist())),need_scale_usd=float(np.exp(p[6])),
            optimizer=statuses[mode],max_receipt_error_pp=float(v.error_pp.abs().max()),
            receipt_within_tolerance=bool(v.error_pp.abs().le(RECEIPT_TOLERANCE_PP).all()),
            max_amount_error_relative=float(st.amount_error_relative.abs().max()),
            amount_within_tolerance=bool(st.amount_error_relative.abs().le(AMOUNT_TOLERANCE_REL).all()),
            all_seeds_practically_stable=bool(st.practical_stability.all()),
            initialization_gap_pp=float(100*(warm[mode]['summary'].receipt_rate.iloc[BURN_IN:].mean()-first[mode]['summary'].receipt_rate.iloc[BURN_IN:].mean())),
            bound_parameters=[PARAMETER_NAMES[i] for i in range(7) if min(p[i]-LOWER[i],UPPER[i]-p[i])<.02])
    config=dict(cps_workbook=str(CPS_WORKBOOK),cps_workbook_sha256=hashlib.sha256(CPS_WORKBOOK.read_bytes()).hexdigest(),
        annual_living_allowance_usd=ANNUAL_LIVING_ALLOWANCE_USD,
        monthly_living_allowance_usd=MONTHLY_LIVING_ALLOWANCE_USD,
        remittable_surplus_share=REMITTABLE_SURPLUS_SHARE,
        earnings_heterogeneity=EARNINGS_HETEROGENEITY,earnings_seed=EARNINGS_SEED,
        living_allowance_source='User-supplied MIT table: first column, gross annual requirement; geography/year unspecified',
        cps_population=CPS_POPULATION,job_proxy=JOB_PROXY,baseline_year=BASELINE_YEAR,amount_target_usd=d['amount_target'],
        months=MONTHS,burn_in=BURN_IN,seeds=SEEDS,fit_months=FIT_MONTHS,fit_burn_in=FIT_BURN_IN,fit_seed=FIT_SEED,
        lower=LOWER.tolist(),upper=UPPER.tolist(),initial=INITIAL.tolist(),finite_difference_steps=FD_STEP.tolist(),
        profile_rho=PROFILE_PERSISTENCE,history_start=HISTORY_START,history_end=HISTORY_END,
        covid_need_file=str(COVID_NEED_FILE),covid_need_coefficient=COVID_NEED_COEFFICIENT,
        proxy_links=int(d['m'].cps_job_is_proxy.sum()),labour_eligible_links=int(d['eligible'].sum()))
    report=dict(settings=config,results=result,limitations=[
        'MMP cross-sectional receipt interpreted as monthly prevalence; not nationally representative.',
        'One positive migrant-link-month contribution interpreted as one Banxico transaction.',
        'Fixed MIT living allowance including benchmark taxes; not an individual tax model or historical price estimate.',
        '20% of surplus is assumed; a 20% gross remittance target is not calibrated.',
        'Persistent lognormal earnings inferred from baseline means/medians; historical dispersion held fixed.',
        'Independent monthly employment states, not estimated transition hazards.',
        'Persistence and shared need coefficients may be weakly identified; inspect profiles and bounds.',
        'COVID history is capacity-only unless an explicit need index and coefficient are supplied.'])
    (OUT/'run_settings.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
    return report


if __name__=='__main__':
    run()
