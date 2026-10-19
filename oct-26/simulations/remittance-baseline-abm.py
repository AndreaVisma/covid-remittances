"""Monthly, sequential baseline ABM for the existing HOUSE/PERS synthetic networks.

Run: python oct-26/simulations/remittance-baseline-abm.py
Dependencies: numpy, pandas, scipy, matplotlib, pyarrow. No inputs are changed.
MMP receipt identifies baseline participation targets, NOT monthly frequency or
dollar amounts. See the accompanying notes before interpreting calibration.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import least_squares

# %% 1. Editable run settings. Periods are model months, not dated COVID months.
ROOT = Path('C:/git-projects/covid-remittances')
POPULATION = ROOT / 'oct-26/outputs/synthetic-migrants-house-networks'
TARGET_FILE = POPULATION / 'household_network_targets.csv'
OUT = ROOT / 'oct-26/outputs/remittance-baseline-abm'
MONTHS = 120
BURN_IN = 36
SEEDS = [2026, 2027, 2028]
CALIBRATION_SEED = 901
CALIBRATION_MONTHS = 36
CALIBRATION_BURN_IN = 12
MAX_FIT_EVALUATIONS = 35
MIN_TARGET_HOUSEHOLDS = 30
MIN_SYNTHETIC_HOUSEHOLDS = 10
SAVE_AGENT_MONTHS = True       # First seed only, including nonsenders (zeros).
SAVE_PARQUET = True
SAVE_JSON = True
STABILITY_WINDOW = 24          # Compare two final, nonoverlapping windows.
STABILITY_RECEIPT_PP = 3.0      # Practical diagnostics, not stationarity tests.
STABILITY_AMOUNT_REL = 0.20
FIT_TOLERANCE_PP = 8.0          # Report misses, never force a pass.

# Three FITTED parameters. These are pooled group parameters, not estimates of
# individual-specific willingness. Other behavioural coefficients below are
# explicit assumptions; MMP receipt alone cannot identify them all.
INITIAL_PARAMETERS = {'core_intercept': -0.5, 'kin_intercept': -4.0, 'no_worker': 0.4}
PARAMETER_LOWER = [-7.0, -12.0, -2.0]
PARAMETER_UPPER = [5.0, 1.0, 3.0]
RIDGE_STRENGTH = 0.005         # Weak regularization toward initial values.
PERSISTENCE = 0.8             # Previous senders are more willing AND act first.
CHILD_SHARE_COEFFICIENT = 0.3  # Under-12s, consistent with origin input features.
OLDER_SHARE_COEFFICIENT = 0.3
RESIDUAL_NEED_COEFFICIENT = 0.6
CAPACITY_COEFFICIENT = 0.25
NETWORK_DILUTION = {'Core': 0.6, 'Kin': 0.8}  # Penalty log(same-type network size).

# Provisional monthly monetary assumptions, expressed in scenario USD.
# These are sending BUDGETS, not estimated wages. Not-working adults can have
# other resources; unknown jobs use an explicit fallback. Minors/unknown ages
# cannot send in this version. No US labour shocks or earnings claims are made.
CAPACITY_USD = {'Agriculture': 180., 'Production/construction/transport': 300.,
    'Sales/personal services': 240., 'Professional/administrative': 450.,
    'Not working': 80., 'Working student/pensioner': 150., 'Unknown': 180.}
CONTRIBUTION_FRACTION = {'Core': 1.0, 'Kin': 0.35}
BASE_SUPPORT_USD = 160.
NO_WORKER_NEED_MULTIPLIER = 0.4
OLDER_NEED_MULTIPLIER = 0.2
EPS = 1e-8


# %% 2. Load populations and prepare MMP calibration cells.
def network_labels(core, kin):
    return np.select([(core > 0) & (kin > 0), core > 0, kin > 0],
                     ['Core + kin', 'Core only', 'Kin only'], default='None recorded')


def prepare():
    households = pd.read_parquet(POPULATION / 'households.parquet').sort_values('synthetic_household_id').reset_index(drop=True)
    migrants = pd.read_parquet(POPULATION / 'migrants.parquet').sort_values('synthetic_migrant_id').reset_index(drop=True)
    source = pd.read_csv(TARGET_FILE)
    assert households.synthetic_household_id.is_unique and migrants.synthetic_migrant_id.is_unique
    assert source.mmp_household_id.is_unique, 'Targets must be original households, not replicated synthetic donors.'
    # An unweighted target is coherent with the unweighted synthetic sample.
    # Reject a differently weighted population rather than silently double-weight.
    metadata = json.loads((POPULATION / 'run_settings.json').read_text(encoding='utf-8'))
    assert not metadata['use_mmp_weights'], 'This initial ABM calibrates to unweighted MMP; harmonize weighting first.'
    assert metadata['connection_groups']['Core'] == [1, 2, 3]
    assert metadata['overlap_rule'] == 'maximum_overlap'
    hid = migrants.synthetic_household_id.map(pd.Series(households.index, index=households.synthetic_household_id))
    assert hid.notna().all(), 'Orphan migrant links'
    hid = hid.to_numpy(dtype=int)
    n_h, n_m = len(households), len(migrants)
    sizes = np.bincount(hid, minlength=n_h)
    assert np.array_equal(sizes, households.network_size.to_numpy()) and (sizes > 0).all()
    core = migrants.connection_group.eq('Core').to_numpy()
    assert migrants.connection_group.isin(['Core', 'Kin']).all()
    assert np.array_equal(np.bincount(hid, weights=core, minlength=n_h), households.core_count)

    # Origin demographics use the MMP resident proxy, not the matched ENIGH
    # donor's family. Fill uncertain inputs only for simulation and keep flags;
    # those households are excluded from calibration below.
    resident = households.resident.to_numpy(float)
    child = households.resident_child.to_numpy(float)
    older = households.resident_older.to_numpy(float)
    worker = households.resident_worker.to_numpy(float)
    input_missing = ~np.isfinite(resident + child + older + worker)
    resident = np.where(np.isfinite(resident), resident, households.members.to_numpy(float))
    resident = np.maximum(resident, 1.)
    child, older = np.nan_to_num(child), np.nan_to_num(older)
    no_worker = np.where(np.isfinite(worker), worker == 0, False).astype(float)
    child_share, older_share = child / resident, older / resident
    # A recurring SUPPORT TARGET, not an official needs budget or poverty line.
    # Unmet need does not accumulate; unused migrant budgets are not saved.
    equivalents = 1 + 0.5 * np.maximum(resident - child - 1, 0) + 0.3 * child
    need = BASE_SUPPORT_USD * equivalents * (1 + NO_WORKER_NEED_MULTIPLIER * no_worker + OLDER_NEED_MULTIPLIER * older_share)
    capacity = migrants.job_group.map(CAPACITY_USD).fillna(CAPACITY_USD['Unknown']).to_numpy(float)
    age_eligible = migrants.age_years.ge(18).fillna(False).to_numpy()
    capacity *= age_eligible
    assert (capacity >= 0).all() and (need > 0).all()
    fraction = migrants.connection_group.map(CONTRIBUTION_FRACTION).to_numpy(float)
    assert ((fraction > 0) & (fraction <= 1)).all()
    same_type = np.where(core, households.core_count.to_numpy()[hid], households.kin_count.to_numpy()[hid])
    fixed_theta = (CHILD_SHARE_COEFFICIENT * child_share[hid]
                   + OLDER_SHARE_COEFFICIENT * older_share[hid]
                   + CAPACITY_COEFFICIENT * np.log(np.maximum(capacity, EPS) / 250.)
                   - np.where(core, NETWORK_DILUTION['Core'], NETWORK_DILUTION['Kin']) * np.log(same_type))

    for frame in [source, households]:
        frame['network_type'] = network_labels(frame.core_count, frame.kin_count)
        frame['work_type'] = np.select([frame.resident_worker.eq(0), frame.resident_worker.gt(0)],
                                      ['No worker', 'Worker(s)'], default='Unknown')
        frame['cell'] = frame.network_type + ' | ' + frame.work_type
    source['receipt'] = source.remus.map({1: 1., 2: 0.})
    eligible_source = (source.recorded_fields_complete & source.network_size.gt(0)
                       & source.resident_worker.notna() & source.receipt.notna())
    targets = source.loc[eligible_source].groupby('cell').agg(
        source_n=('receipt', 'size'), source_receipt=('receipt', 'mean')).reset_index()
    eligible_h = households.recorded_fields_complete.to_numpy() & ~input_missing
    synthetic_counts = households.loc[eligible_h].cell.value_counts()
    targets['synthetic_n'] = targets.cell.map(synthetic_counts).fillna(0).astype(int)
    targets['fit_used'] = (targets.source_n >= MIN_TARGET_HOUSEHOLDS) & (targets.synthetic_n >= MIN_SYNTHETIC_HOUSEHOLDS)
    assert targets.fit_used.sum() >= 3, 'Insufficient calibration cells.'
    group_id = households.cell.map(pd.Series(targets.index, index=targets.cell)).fillna(-1).to_numpy(int)
    calibration_id = np.where(eligible_h, group_id, -1)
    households['abm_input_missing'] = input_missing
    households['support_target_usd'] = need
    households['calibration_eligible'] = eligible_h
    migrants['sending_budget_usd'] = capacity
    migrants['sending_eligible'] = capacity > 0
    data = dict(hid=hid, n_h=n_h, n_m=n_m, core=core, sizes=sizes,
                offsets=np.r_[0, np.cumsum(sizes)[:-1]], capacity=capacity, fraction=fraction,
                need=need, fixed_theta=fixed_theta, no_worker=no_worker,
                calibration_id=calibration_id, n_groups=len(targets),
                network_type=households.network_type.to_numpy())
    return households, migrants, source, targets, data


# %% 3. Monthly sequential decisions. This is the ABM, not a household draw.
def simulate(parameters, data, months, seed, burn_in, initial_active=0., collect=False,
             capacity_path=None):
    """Each migrant decides at most once per month, after earlier contributions.

    capacity_path, if supplied, is a (months, migrants) multiplier array for
    mechanism tests; baseline runs never use shocks. All parameters stay fixed.
    """
    d = data
    H, M, hid = d['n_h'], d['n_m'], d['hid']
    rng = np.random.default_rng(seed)
    previous = rng.random(M) < initial_active
    base_theta = np.where(d['core'], parameters[0], parameters[1]) + parameters[2] * d['no_worker'][hid] + d['fixed_theta']
    group_mask = d['calibration_id'] >= 0
    group_counts = np.bincount(d['calibration_id'][group_mask], minlength=d['n_groups'])
    expected_sum = np.zeros(d['n_groups'])
    logs, household_history, transfers_history, expectations = [], [], [], []
    for month in range(months):
        capacity = d['capacity'].copy()
        if capacity_path is not None:
            capacity *= capacity_path[month]
        assert np.isfinite(capacity).all() and (capacity >= 0).all()
        residual = d['need'].copy()
        theta = base_theta + PERSISTENCE * previous
        # For a capacity shock, adjust willingness as well as the budget ceiling.
        theta += CAPACITY_COEFFICIENT * np.log(np.maximum(capacity, EPS) / np.maximum(d['capacity'], EPS))
        p_no_prior_sender = np.where(capacity > EPS, expit(theta), 0.)
        # Exact receipt probability CONDITIONAL on last month's sender state.
        # Along the "all earlier people declined" path residual need is unchanged.
        # The product uses conditional probabilities, not independent final sends.
        log_none = np.bincount(hid, weights=np.log1p(-np.minimum(p_no_prior_sender, 1 - 1e-12)), minlength=H)
        expected_receipt = -np.expm1(log_none)
        if month >= burn_in:
            expected_sum += np.bincount(d['calibration_id'][group_mask], weights=expected_receipt[group_mask], minlength=d['n_groups'])
        # Randomize within incumbent/nonincumbent groups, not by permanent ID.
        # Draw all random numbers even for ineligible people for reproducible
        # common random numbers across parameter fits and mechanism comparisons.
        tie_break = rng.random(M)
        decision_draw = rng.random(M)
        order = np.lexsort((tie_break, ~previous, hid))
        transfer = np.zeros(M)
        for rank in range(int(d['sizes'].max())):
            hh = np.flatnonzero((d['sizes'] > rank) & (residual > EPS))
            if not len(hh):
                break
            agents = order[d['offsets'][hh] + rank]
            eligible = capacity[agents] > EPS
            agents, hh = agents[eligible], hh[eligible]
            probability = expit(theta[agents] + RESIDUAL_NEED_COEFFICIENT * np.log(np.maximum(residual[hh] / d['need'][hh], EPS)))
            accepted = decision_draw[agents] < probability
            agents, hh = agents[accepted], hh[accepted]
            amounts = np.minimum(capacity[agents], d['fraction'][agents] * residual[hh])
            transfer[agents] = amounts
            residual[hh] -= amounts  # One agent per household in each rank.
        sent = transfer > EPS
        total = np.bincount(hid, weights=transfer, minlength=H)
        n_senders = np.bincount(hid, weights=sent, minlength=H).astype(int)
        received = n_senders > 0
        assert (transfer <= capacity + EPS).all() and (transfer >= 0).all()
        assert (residual >= -EPS).all() and np.allclose(total + residual, d['need'])
        if collect:
            retained = np.bincount(hid, weights=sent & previous, minlength=H)
            previous_household = np.bincount(hid, weights=previous, minlength=H) > 0
            turnover = previous_household & received & (retained == 0)
            record = dict(month=month + 1, receipt_rate=received.mean(), expected_receipt_rate=expected_receipt.mean(),
                active_senders=int(sent.sum()), core_senders=int((sent & d['core']).sum()),
                kin_senders=int((sent & ~d['core']).sum()), new_senders=int((sent & ~previous).sum()),
                retained_senders=int((sent & previous).sum()), replacement_households=int(turnover.sum()),
                total_usd=total.sum(), mean_transfer_usd=transfer[sent].mean() if sent.any() else 0.,
                mean_recipient_usd=total[received].mean() if received.any() else 0.,
                unmet_share=residual.sum() / d['need'].sum(), multi_sender_rate=(n_senders >= 2).mean())
            for label in ['Core only', 'Kin only', 'Core + kin']:
                mask = d['network_type'] == label
                record['receipt_' + label] = received[mask].mean() if mask.any() else np.nan
            logs.append(record)
            household_history.append((total.copy(), n_senders.copy(), residual.copy()))
            transfers_history.append(transfer.copy())
            expectations.append(expected_receipt.copy())
        previous = sent
    moments = np.divide(expected_sum, group_counts * (months - burn_in), out=np.full(d['n_groups'], np.nan), where=group_counts > 0)
    return dict(moments=moments, monthly=pd.DataFrame(logs), household=household_history,
                transfers=transfers_history, expectations=expectations)


# %% 4. Fit three parameters, then freeze them for independent validation runs.
def calibrate(data, targets):
    used = targets.fit_used.to_numpy()
    weights = np.sqrt(targets.loc[used, 'source_n'].to_numpy() / targets.loc[used, 'source_n'].sum())
    goal = targets.loc[used, 'source_receipt'].to_numpy()
    initial = np.array(list(INITIAL_PARAMETERS.values()))
    history = []
    def residual(parameters):
        result = simulate(parameters, data, CALIBRATION_MONTHS, CALIBRATION_SEED, CALIBRATION_BURN_IN)
        difference = result['moments'][used] - goal
        history.append(dict(evaluation=len(history) + 1, core_intercept=parameters[0], kin_intercept=parameters[1],
                            no_worker=parameters[2], weighted_rmse_pp=100 * np.sqrt(np.sum((weights * difference) ** 2))))
        if len(history) == 1 or len(history) % 20 == 0:
            print(f'Calibration evaluation {len(history)}: RMSE {history[-1]["weighted_rmse_pp"]:.2f} pp', flush=True)
        return np.r_[weights * difference, np.sqrt(RIDGE_STRENGTH) * (parameters - initial) / 5.]
    # Expected conditional receipt smooths the objective, but simulated history
    # can still create kinks. Check independent seeds rather than trust optimizer
    # success alone. This is simulated-moment calibration, not causal estimation.
    fit = least_squares(residual, initial, bounds=(PARAMETER_LOWER, PARAMETER_UPPER),
                        max_nfev=MAX_FIT_EVALUATIONS, diff_step=0.002, ftol=0.002, xtol=0.002)
    return fit, pd.DataFrame(history)


# %% 5. Save complete trajectories, validation diagnostics and assumptions.
def run():
    assert 0 <= BURN_IN < MONTHS and 0 <= CALIBRATION_BURN_IN < CALIBRATION_MONTHS
    assert MONTHS - BURN_IN >= 2 * STABILITY_WINDOW and len(SEEDS) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    households, migrants, source, targets, data = prepare()
    print(f'Loaded {len(migrants):,} migrants and {len(households):,} linked households.', flush=True)
    fit, history = calibrate(data, targets)
    history.to_csv(OUT / 'calibration_history.csv', index=False)
    parameters = fit.x
    print('Fixed parameters:', dict(zip(INITIAL_PARAMETERS, parameters)), flush=True)
    all_months, comparisons, stability_rows = [], [], []
    first = None
    for seed in SEEDS:
        result = simulate(parameters, data, MONTHS, seed, BURN_IN, collect=True)
        if first is None:
            first = result
        monthly = result['monthly'].assign(seed=seed)
        all_months.append(monthly)
        after = result['household'][BURN_IN:]
        received = np.stack([x[1] > 0 for x in after]).mean(axis=0)
        for g, row in targets.iterrows():
            mask = data['calibration_id'] == g
            comparisons.append(dict(seed=seed, cell=row.cell, source_n=row.source_n, synthetic_n=row.synthetic_n,
                fit_used=row.fit_used, mmp_receipt=row.source_receipt, simulated_receipt=received[mask].mean() if mask.any() else np.nan,
                conditional_expected_receipt=result['moments'][g]))
        a, b = monthly.iloc[-2 * STABILITY_WINDOW:-STABILITY_WINDOW], monthly.iloc[-STABILITY_WINDOW:]
        receipt_change = 100 * (b.receipt_rate.mean() - a.receipt_rate.mean())
        amount_change = (b.total_usd.mean() - a.total_usd.mean()) / max(a.total_usd.mean(), EPS)
        stability_rows.append(dict(seed=seed, receipt_change_pp=receipt_change, total_amount_change_relative=amount_change,
            within_tolerance=abs(receipt_change) <= STABILITY_RECEIPT_PP and abs(amount_change) <= STABILITY_AMOUNT_REL))
        print(f'Seed {seed}: final-window receipt change {receipt_change:.2f} pp; amount change {amount_change:.1%}', flush=True)
    monthly = pd.concat(all_months, ignore_index=True)
    validation = pd.DataFrame(comparisons)
    validation['error_pp'] = 100 * (validation.simulated_receipt - validation.mmp_receipt)
    validation['within_tolerance'] = validation.error_pp.abs() <= FIT_TOLERANCE_PP
    stability = pd.DataFrame(stability_rows)
    # Starting-state sensitivity: a separate run starts with 50% labelled as
    # previous senders. These are initial flags, not observed pre-run transfers.
    warm = simulate(parameters, data, MONTHS, SEEDS[0], BURN_IN, initial_active=0.5, collect=True)
    initialization_gap = 100 * (warm['monthly'].receipt_rate.iloc[BURN_IN:].mean() - first['monthly'].receipt_rate.iloc[BURN_IN:].mean())
    monthly.to_csv(OUT / 'monthly_summary.csv', index=False)
    warm['monthly'].to_csv(OUT / 'alternative_initialization.csv', index=False)
    validation.to_csv(OUT / 'receipt_validation.csv', index=False)
    stability.to_csv(OUT / 'stability_checks.csv', index=False)
    targets.to_csv(OUT / 'mmp_receipt_targets.csv', index=False)

    # Ordinal MMP sizes remain an empirical benchmark, not converted to dollars
    # or thresholded using arbitrary "small/substantial" monetary cutoffs.
    ordinal = source.loc[source.recorded_fields_complete & source.network_size.gt(0)
                         & source.remus.eq(1) & source.remussiz.isin([1, 2, 3])]
    ordinal = ordinal.groupby(['network_type', 'remussiz']).size().rename('n').reset_index()
    ordinal['share'] = ordinal.n / ordinal.groupby('network_type').n.transform('sum')
    ordinal.to_csv(OUT / 'mmp_ordinal_size_targets.csv', index=False)
    total = np.stack([x[0] for x in first['household']])
    counts = np.stack([x[1] for x in first['household']])
    unmet = np.stack([x[2] for x in first['household']])
    amounts = np.stack(first['transfers'])
    household_panel = pd.DataFrame(dict(month=np.repeat(np.arange(1, MONTHS + 1), len(households)),
        synthetic_household_id=np.tile(households.synthetic_household_id, MONTHS),
        received_usd=total.ravel(), n_senders=counts.ravel(), unmet_usd=unmet.ravel()))
    if SAVE_PARQUET:
        household_panel.to_parquet(OUT / 'household_months.parquet', index=False)
        households.to_parquet(OUT / 'household_baseline.parquet', index=False)
        migrants.to_parquet(OUT / 'migrant_baseline.parquet', index=False)
        if SAVE_AGENT_MONTHS:
            pd.DataFrame(dict(month=np.repeat(np.arange(1, MONTHS + 1), len(migrants)),
                synthetic_migrant_id=np.tile(migrants.synthetic_migrant_id, MONTHS),
                synthetic_household_id=np.tile(migrants.synthetic_household_id, MONTHS),
                sent_usd=amounts.ravel(), sent=amounts.ravel() > EPS)).to_parquet(OUT / 'migrant_months.parquet', index=False)
    if SAVE_JSON:
        monthly.to_json(OUT / 'monthly_summary.json', orient='records', indent=2)
        validation.to_json(OUT / 'receipt_validation.json', orient='records', indent=2)
    report = dict(parameters=dict(zip(INITIAL_PARAMETERS, parameters.tolist())), optimizer_success=bool(fit.success),
        optimizer_message=str(fit.message), max_abs_validation_error_pp=float(validation.error_pp.abs().max()),
        all_fitted_cells_within_tolerance=bool(validation.loc[validation.fit_used, 'within_tolerance'].all()),
        all_seeds_stable_by_practical_check=bool(stability.within_tolerance.all()),
        initialization_receipt_gap_pp=float(initialization_gap),
        monetary_calibration='None: budgets and support targets are provisional scenario USD.',
        time_interpretation='MMP cross-sectional receipt treated as stationary monthly prevalence by assumption.',
        population_scope='MMP-like connected households; not nationally representative; agents are household links.')
    config_names = ['MONTHS', 'BURN_IN', 'SEEDS', 'CALIBRATION_SEED', 'CALIBRATION_MONTHS', 'CALIBRATION_BURN_IN',
        'MAX_FIT_EVALUATIONS', 'MIN_TARGET_HOUSEHOLDS', 'MIN_SYNTHETIC_HOUSEHOLDS', 'INITIAL_PARAMETERS',
        'PARAMETER_LOWER', 'PARAMETER_UPPER', 'RIDGE_STRENGTH', 'PERSISTENCE', 'CHILD_SHARE_COEFFICIENT',
        'OLDER_SHARE_COEFFICIENT', 'RESIDUAL_NEED_COEFFICIENT', 'CAPACITY_COEFFICIENT', 'NETWORK_DILUTION',
        'CAPACITY_USD', 'CONTRIBUTION_FRACTION', 'BASE_SUPPORT_USD', 'NO_WORKER_NEED_MULTIPLIER',
        'OLDER_NEED_MULTIPLIER', 'STABILITY_WINDOW', 'STABILITY_RECEIPT_PP', 'STABILITY_AMOUNT_REL', 'FIT_TOLERANCE_PP']
    settings = {name: globals()[name] for name in config_names}
    (OUT / 'run_settings.json').write_text(json.dumps(dict(settings=settings, results=report,
        population=str(POPULATION), targets=str(TARGET_FILE)), indent=2), encoding='utf-8')
    make_plots(monthly, validation, first, warm, households, ordinal, history)
    (OUT / 'REPORT.txt').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


# %% 6. Visualizations: monthly paths, target fit, actual household histories.
def make_plots(monthly, validation, first, warm, households, ordinal, history):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, filename, note='Amounts are provisional scenario USD; no COVID shocks; MMP-like connected households only.'):
        fig.text(0.02, 0.01, note, fontsize=8, color='#555555')
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, title, factor in zip(axes.flat,
        ['receipt_rate', 'active_senders', 'mean_recipient_usd', 'unmet_share'],
        ['Households receiving (%)', 'Active sending links', 'Mean per recipient (scenario USD/month)', 'Unmet share of support targets (%)'],
        [100, 1, 1, 100]):
        for seed, group in monthly.groupby('seed'):
            ax.plot(group.month, group[col] * factor, alpha=0.55, lw=1, label=f'Seed {seed}')
        ax.axvspan(1, BURN_IN, alpha=.08, color='grey', label='Burn-in')
        ax.set(title=title, xlabel='Model month')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Baseline dynamics: fixed parameters and stationary resources')
    save(fig, '01_monthly_dynamics.png')

    avg = validation.groupby('cell').agg(mmp=('mmp_receipt', 'first'), mean=('simulated_receipt', 'mean'),
        low=('simulated_receipt', 'min'), high=('simulated_receipt', 'max')).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(avg))
    ax.scatter(100 * avg.mmp, y - .1, marker='D', color='#1b7837', label='Original MMP receipt')
    ax.errorbar(100 * avg['mean'], y + .1, xerr=100 * np.vstack([avg['mean'] - avg.low, avg.high - avg['mean']]),
                fmt='o', color='#2166ac', capsize=4, label='ABM: mean and range across seeds')
    ax.set(yticks=y, yticklabels=avg.cell, xlabel='Household receipt (%)', xlim=(0, 100))
    ax.legend()
    fig.suptitle('Baseline receipt validation after burn-in')
    save(fig, '02_receipt_validation.png', 'MMP unweighted complete-record targets. Seed ranges are simulation variation, not confidence intervals.')

    d = first['monthly']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].stackplot(d.month, d.core_senders, d.kin_senders, labels=['Core', 'Kin'], colors=['#2166ac', '#d8a535'])
    axes[0, 0].set_title('Who sends: active links by relationship')
    axes[0, 0].legend()
    axes[0, 1].plot(d.month, d.new_senders, label='Inactive last month, sending now')
    axes[0, 1].plot(d.month, d.retained_senders, label='Continuing sender')
    axes[0, 1].set_title('Entry and persistence of responsibility')
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(d.month, d.replacement_households, color='#a6611a')
    axes[1, 0].set_title('Receiving households with entirely new senders')
    for label in ['Core only', 'Kin only', 'Core + kin']:
        axes[1, 1].plot(d.month, d['receipt_' + label] * 100, label=label, alpha=.8)
    axes[1, 1].set_title('Receipt by network type (%)')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Model month')
        ax.axvline(BURN_IN, color='grey', ls=':')
    fig.suptitle('Emergent participation and turnover (first seed)')
    save(fig, '03_sending_responsibility.png', 'Turnover occurs under constant resources; this is not evidence of COVID-induced replacement.')

    total = np.stack([x[0] for x in first['household']])
    unmet = np.stack([x[2] for x in first['household']])
    examples = []
    for mask in [households.network_size.eq(1), households.network_type.eq('Core only') & households.network_size.gt(1),
                 households.network_type.eq('Kin only'), households.network_type.eq('Core + kin')]:
        choices = np.flatnonzero(mask)
        if len(choices):
            examples.append(choices[0])
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.6 * len(examples)), squeeze=False)
    for ax, idx in zip(axes.flat, examples):
        ax.plot(np.arange(1, MONTHS + 1), total[:, idx], label='Received')
        ax.plot(np.arange(1, MONTHS + 1), total[:, idx] + unmet[:, idx], ls='--', color='grey', label='Support target')
        row = households.iloc[idx]
        ax.set(title=f'Household {row.synthetic_household_id}: {row.core_count} core + {row.kin_count} kin',
               ylabel='Scenario USD', xlabel='Model month')
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Individual household histories: variation beneath stable averages')
    save(fig, '04_household_histories.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(first['monthly'].month, first['monthly'].receipt_rate.rolling(12, min_periods=1).mean() * 100, label='No initial senders')
    axes[0].plot(warm['monthly'].month, warm['monthly'].receipt_rate.rolling(12, min_periods=1).mean() * 100, label='50% initially labelled senders')
    axes[0].axvline(BURN_IN, color='grey', ls=':')
    axes[0].set(title='Starting-state sensitivity: 12-month mean', xlabel='Model month', ylabel='Receipt (%)')
    axes[0].legend(fontsize=8)
    axes[1].plot(history.evaluation, history.weighted_rmse_pp)
    axes[1].set(title='Calibration search (not monthly dynamics)', xlabel='Parameter evaluation', ylabel='Weighted receipt RMSE (pp)')
    fig.suptitle('Convergence diagnostics')
    save(fig, '05_stability_and_calibration.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ['Core only', 'Kin only', 'Core + kin']
    pivot = ordinal.pivot(index='network_type', columns='remussiz', values='share').reindex(order).fillna(0)
    bottom = np.zeros(3)
    for code, label, color in [(1, 'Small', '#c6dbef'), (2, 'Intermediate', '#6baed6'), (3, 'Substantial', '#2171b5')]:
        values = pivot[code].to_numpy() if code in pivot else np.zeros(3)
        axes[0].bar(order, values * 100, bottom=bottom, color=color, label=label)
        bottom += values * 100
    axes[0].set(title='MMP: reported importance among recipients', ylabel='Recipients with known size (%)')
    axes[0].legend(fontsize=8)
    means = []
    for label in order:
        x = total[BURN_IN:, households.network_type.eq(label).to_numpy()]
        means.append(x[x > EPS].mean() if (x > EPS).any() else 0.)
    axes[1].bar(order, means, color=['#2166ac', '#d8a535', '#5aae61'])
    axes[1].set(title='ABM: illustrative amounts among recipients', ylabel='Scenario USD per recipient-month')
    fig.suptitle('Amount diagnostics: different measures, NOT a monetary calibration')
    save(fig, '06_amount_context.png', 'MMP ordinal importance has no dollar thresholds. Right panel depends on assumed budgets and contribution rules.')


if __name__ == '__main__':
    run()
