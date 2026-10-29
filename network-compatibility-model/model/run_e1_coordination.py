"""Migrant-level comparison model: E0 and E1a under the coordination rule ('coordination').

Run: python network-compatibility-model/model/run_e1_coordination.py

Coordination rule: the household's able migrants, core and kin, share the routine sending lost to
migrants who cannot send. Each able migrant j sends with
  theta_j,t = min(1, theta0_j,t * (1 + kappa * Lambda_h,t / P_h,t))
  Lambda = routine sending of the eligible migrants unable to send (sum of theta0)
  P      = routine sending of the able migrants (sum of theta0)
so kappa is the share of the household's lost expected transfers that its remaining migrants make
up (before the cap at 1), and a household can never receive more than without the shock.
kappa in [0, 1] is the substantive range; values above 1 (over-replacement) are profiled only
to locate the ceiling of the April 2020 index.

Steps (outputs in outputs/e1-coordination/):
  1. E0 refit of alpha_core, alpha_kin for each kappa: e0_kappa_profile.csv
  2. 2019 validation (network type x size, link type): e0_validation.csv;
     stochastic 2019 receipt check: e0_stochastic_check.csv
  3. April 2020 count index and replacement share against kappa: kappa_ceiling_rules.csv
(The rows for the former lost-sending rule were removed in the 29 Oct 2026 cleanup (git history up to a83b60c).)
  4. E1a with the refitted alphas, same design and tables as outputs/e1-job-loss/
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1

RULE = 'coordination'
OUT = abm.HERE / 'outputs/e1-coordination'
KAPPA_PROFILE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
VALIDATION_KAPPAS = [0.0, 1.0]


def params_from(profile, kappa, rule):
    row = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
    return abm.Params(row.alpha_core, row.alpha_kin, kappa=kappa, rule=rule)
CHECK_SEEDS = [11, 12, 13, 14, 15]


def fit_profile(cal, kappas):
    rows, start = [], e0.START
    for kappa in kappas:
        params, f = cal.fit(kappa, start, rule=RULE)
        start = np.array([params.alpha_core, params.alpha_kin])
        rows.append(dict(kappa=kappa, alpha_core=params.alpha_core, alpha_kin=params.alpha_kin,
                         weighted_rmse_pp=100 * np.sqrt(np.mean(f.fun ** 2)), success=f.success))
        print(f'E0 ({RULE}) kappa {kappa}: rmse {rows[-1]["weighted_rmse_pp"]:.2f} pp')
    return pd.DataFrame(rows)


def validation(cal, cells_src, cells_sim, profiles):
    out = []
    for rule, (profile, kappas) in profiles.items():
        for kappa in kappas:
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
                         triggered_households=np.mean([s.triggered_households.mean() for s in sim])))
    return pd.DataFrame(rows)


def ceiling(pop, profiles):
    """April 2020 vs April 2019 (the no-COVID month): count index, and the share of the transfers
    lost at kappa = 0 that the rule recovers."""
    inputs = abm.make_inputs(pd.DatetimeIndex(['2019-04-01', '2020-04-01']), pop['rate_columns'])
    rows = []
    for rule, (profile, kappas) in profiles.items():
        for kappa in kappas:
            f = abm.expected_flows(pop, inputs, params_from(profile, kappa, rule))
            rows.append(dict(rule=rule, kappa=kappa, index_april_2020=f.transfers.iloc[1] / f.transfers.iloc[0],
                             core_extra=f.get('core_transfers_triggered', pd.Series([0, 0])).iloc[1],
                             kin_extra=f.kin_transfers_triggered.iloc[1]))
    t = pd.DataFrame(rows)
    loss0 = 1 - t.loc[t.kappa.eq(0)].groupby('rule').index_april_2020.first()
    t['replaced_share_of_kappa0_loss'] = 1 - (1 - t.index_april_2020) / t.rule.map(loss0)
    return t


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

    profile = fit_profile(cal, sorted(set(KAPPA_PROFILE) | set(KAPPAS)))
    profile.to_csv(OUT / 'e0_kappa_profile.csv', index=False)
    validation(cal, cells_src, cells_sim, {RULE: (profile, VALIDATION_KAPPAS)}
               ).to_csv(OUT / 'e0_validation.csv', index=False)
    check = stochastic_check(pop, profile)
    check.to_csv(OUT / 'e0_stochastic_check.csv', index=False)
    ceil = ceiling(pop, {RULE: (profile, list(profile.kappa))})
    ceil.to_csv(OUT / 'kappa_ceiling_rules.csv', index=False)

    fitted = profile.loc[profile.kappa.isin(KAPPAS), ['kappa', 'alpha_core', 'alpha_kin']].to_dict('records')
    e1.run_experiment(fitted, OUT, rule=RULE, settings_extra=dict(
        alphas_from='e0_kappa_profile.csv (refitted under this rule)',
        rule_definition='coordination: theta_j = min(1, theta0_j * (1 + kappa * Lambda / P)) for every able '
                        'migrant, core or kin; Lambda (P) = routine sending of unable (able) eligible migrants; '
                        'kappa = replacement share'))
    pd.set_option('display.width', 220)
    print(profile.round(3).to_string(index=False))
    print(check.round(4).to_string(index=False))
    print(ceil.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
