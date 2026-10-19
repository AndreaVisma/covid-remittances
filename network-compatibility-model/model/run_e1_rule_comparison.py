"""E1a under the former stepping-in rule ('all_core'), for comparison with the base rule.

Run after run_e0_baseline.py and run_e1_job_loss.py:
    python network-compatibility-model/model/run_e1_rule_comparison.py

Base rule ('lost_sending', abm.Params default; adopted 19 Oct 2026), kin step in for
core and for other kin in proportion to the routine sending lost:
  logit theta_kin,j = alpha_kin + beta*S_t + kappa*L_h,-j,t
  L = (# eligible core migrants unable) + w * (# other eligible kin migrants unable),
  w = theta_kin0 / theta_core0 (routine 2019 propensities)
Former rule ('all_core'): kappa*U_h,t, U = 1 if all eligible core migrants are unable.

Steps (outputs in outputs/e1-all-core-rule/):
  1. E0 refit of alpha_core, alpha_kin for each kappa under the former rule: e0_kappa_profile.csv
  2. 2019 validation for both rules at kappa 0 and 4 (network type x size, link type):
     e0_validation.csv; stochastic 2019 receipt check (former rule): e0_stochastic_check.csv
  3. E1a with the refitted alphas, same design and tables as outputs/e1-job-loss/
  4. April 2020 count index against kappa for both rules: kappa_ceiling_rules.csv
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1

RULE = 'all_core'
OUT = abm.HERE / 'outputs/e1-all-core-rule'
BASE_PROFILE = abm.HERE / 'outputs/e0-baseline/kappa_profile.csv'
KAPPA_PROFILE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
KAPPAS = [0.0, 1.0, 2.0, 3.0, 4.0]
VALIDATION_KAPPAS = [0.0, 4.0]
CHECK_SEEDS = [11, 12, 13, 14, 15]


def fit_profile(cal):
    rows, start = [], e0.START
    for kappa in KAPPA_PROFILE:
        params, f = cal.fit(kappa, start, rule=RULE)
        start = np.array([params.alpha_core, params.alpha_kin])
        rows.append(dict(kappa=kappa, alpha_core=params.alpha_core, alpha_kin=params.alpha_kin,
                         weighted_rmse_pp=100 * np.sqrt(np.mean(f.fun ** 2)), success=f.success))
        print(f'E0 ({RULE}) kappa {kappa}: rmse {rows[-1]["weighted_rmse_pp"]:.2f} pp')
    return pd.DataFrame(rows)


def params_from(profile, kappa, rule):
    row = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
    return abm.Params(row.alpha_core, row.alpha_kin, kappa=kappa, rule=rule)


def validation(cal, cells_src, cells_sim, profiles):
    out = []
    for rule, profile in profiles.items():
        for kappa in VALIDATION_KAPPAS:
            p_sim = cal.receipt(params_from(profile, kappa, rule))[cal.sim_ok]
            for panel, by in [('network_type', ['network_type']), ('size', ['network_type', 'size_bin']),
                              ('links', ['links'])]:
                t = e0.compare(cells_src, cells_sim, p_sim, by)
                if panel == 'links':
                    t = t.assign(network_type='All')
                t = t.rename(columns={'size_bin': 'category', 'links': 'category'})
                out.append(t.assign(panel=panel, rule=rule, kappa=kappa))
    return pd.concat(out, ignore_index=True)


def stochastic_check(pop, profile):
    inputs = abm.make_inputs(abm.baseline_2019_dates(repeats=2), pop['rate_columns'])
    rows = []
    for kappa in VALIDATION_KAPPAS:
        params = params_from(profile, kappa, RULE)
        sim = [abm.simulate(pop, inputs, params, s)['summary'] for s in CHECK_SEEDS]
        rows.append(dict(kappa=kappa, analytic_receipt_rate=abm.expected_receipt(pop, inputs, params).mean(),
                         simulated_receipt_rate=np.mean([s.receipt_rate.mean() for s in sim]),
                         transfers_per_month=np.mean([s.transfers.mean() for s in sim]),
                         kin_transfers=np.mean([s.kin_transfers.mean() for s in sim]),
                         triggered_households=np.mean([s.triggered_households.mean() for s in sim]),
                         kin_transfers_triggered=np.mean([s.kin_transfers_triggered.mean() for s in sim])))
    return pd.DataFrame(rows)


def ceiling(pop, profiles):
    inputs = abm.make_inputs(pd.DatetimeIndex(['2019-04-01', '2020-04-01']), pop['rate_columns'])
    rows = []
    for rule, profile in profiles.items():
        for kappa in KAPPA_PROFILE:
            f = abm.expected_flows(pop, inputs, params_from(profile, kappa, rule)).transfers
            rows.append(dict(rule=rule, kappa=kappa, index_april_2020=f.iloc[1] / f.iloc[0]))
    return pd.DataFrame(rows)


def main():
    assert abm.Params.rule != RULE
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    h = pop['h']
    sim_ok = e0.target_mask(h)
    cells_sim = e0.household_cells(h).loc[sim_ok]
    cells_src = e0.load_targets(h, sim_ok)
    cal = e0.Calibrator(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']),
                        cells_src, cells_sim, sim_ok)

    profile = fit_profile(cal)
    profile.to_csv(OUT / 'e0_kappa_profile.csv', index=False)
    profiles = {abm.Params.rule: pd.read_csv(BASE_PROFILE), RULE: profile}
    validation(cal, cells_src, cells_sim, profiles).to_csv(OUT / 'e0_validation.csv', index=False)
    check = stochastic_check(pop, profile)
    check.to_csv(OUT / 'e0_stochastic_check.csv', index=False)
    ceiling(pop, profiles).to_csv(OUT / 'kappa_ceiling_rules.csv', index=False)

    fitted = profile.loc[profile.kappa.isin(KAPPAS), ['kappa', 'alpha_core', 'alpha_kin']].to_dict('records')
    e1.run_experiment(fitted, OUT, rule=RULE, settings_extra=dict(
        alphas_from='e0_kappa_profile.csv (refitted under this rule)',
        rule_definition='former rule: logit theta_kin = alpha_kin + beta*S_t + kappa*U; '
                        'U = 1 if all eligible core migrants are unable'))
    pd.set_option('display.width', 220)
    print(profile.round(3).to_string(index=False))
    print(check.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
