"""E2 (first pass): behavioural compatibility map over beta x kappa (model-specification.md §6-§7).

Run: python network-compatibility-model/model/run_e2_map.py   (~15 min, after run_e1_coordination.py)
Migrant-level comparison model: coordination rule, alphas from outputs/e1-coordination/e0_kappa_profile.csv,
kappa = replacement share (0..1 by 0.1, plus 1.5 and 2 for over-replacement). The lost-sending map was
removed in the 29 Oct 2026 cleanup (git history up to a83b60c).

pi = 0 (no benefits, user decision), rho = 0, S_t = reported COVID
deaths / 10,000 (abm.salience_series). For each kappa on the E0 profile (its 2019-refitted
alphas) and each beta on the grid: exact expected transfers 2020-01..2022-12 under observed CPS
employment, relative to the no-COVID run, against Banxico transactions relative to their
2017-19 trend (run_e1_job_loss.banxico).

Compatibility (count criterion only): quarterly log RMSE over 2020Q2..2021Q4 <= TAU_C = 0.04
(user decision 20 Oct 2026). The ENIGH household criterion is not implemented yet, so cells are
'count-compatible', not compatible in the full sense. 2022 is held out (RMSE reported).

Outputs in outputs/e2-map-coordination/: map.csv (one row per cell), quarterly_paths.csv (quarterly index
per cell), run_settings.json.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm
import run_e1_job_loss as e1

TAU_C = 0.04
BETAS = np.round(np.arange(0.0, 0.401, 0.02), 2)
RULE_SETUP = {'coordination': (abm.HERE / 'outputs/e1-coordination/e0_kappa_profile.csv',
                               [round(k, 1) for k in np.arange(0, 1.01, 0.1)] + [1.5, 2.0])}
MONTHS = pd.date_range('2020-01-01', '2022-12-01', freq='MS')
FIT_Q = [str(p) for p in pd.period_range('2020Q2', '2021Q4', freq='Q')]
HOLD_Q = [str(p) for p in pd.period_range('2022Q1', '2022Q4', freq='Q')]


def quarterly(series):
    return series.groupby(series.index.to_period('Q')).sum().rename(index=str)


def log_rmse(model, data, quarters):
    return float(np.sqrt(np.mean(np.log(model[quarters] / data[quarters]) ** 2)))


def main(rule='coordination'):
    profile_path, kappas = RULE_SETUP[rule]
    OUT = abm.HERE / f'outputs/e2-map-{rule}'
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    profile = pd.read_csv(profile_path)
    bx, trend, bx_info = e1.banxico()
    target = quarterly(bx[MONTHS]) / quarterly(trend[MONTHS])
    salience = abm.salience_series(MONTHS)
    covid_inputs = abm.make_inputs(MONTHS, pop['rate_columns'], salience=salience)
    cf_inputs = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in MONTHS]),
                                pop['rate_columns'])
    rows, paths = [], []
    for kappa in kappas:
        r = profile.loc[np.isclose(profile.kappa, kappa)].iloc[0]
        base = dict(alpha_core=r.alpha_core, alpha_kin=r.alpha_kin, kappa=kappa, rule=rule)
        cf = quarterly(abm.expected_flows(pop, cf_inputs, abm.Params(**base)).transfers.set_axis(MONTHS))
        for beta in BETAS:
            f = abm.expected_flows(pop, covid_inputs, abm.Params(**base, beta=beta))
            model = quarterly(f.transfers) / cf
            kin = quarterly(f.kin_transfers)
            rmse = log_rmse(model, target, FIT_Q)
            rows.append(dict(kappa=kappa, beta=beta, alpha_core=r.alpha_core, alpha_kin=r.alpha_kin,
                             rmse_log_quarterly=rmse, count_compatible=rmse <= TAU_C,
                             rmse_log_quarterly_holdout_2022=log_rmse(model, target, HOLD_Q),
                             mean_index_2020=float(model[[q for q in FIT_Q if q.startswith('2020')]].mean()),
                             mean_index_2021=float(model[[q for q in FIT_Q if q.startswith('2021')]].mean()),
                             max_abs_gap=float(np.max(np.abs(np.log(model[FIT_Q] / target[FIT_Q])))),
                             kin_share_2021=float(kin[[q for q in FIT_Q if q.startswith('2021')]].sum()
                                                  / quarterly(f.transfers)[[q for q in FIT_Q if q.startswith('2021')]].sum())))
            paths.append(pd.DataFrame({'quarter': model.index, 'kappa': kappa, 'beta': beta,
                                       'index_model': model.values, 'index_banxico': target[model.index].values}))
        print(f'kappa {kappa}: best rmse {min(x["rmse_log_quarterly"] for x in rows if x["kappa"] == kappa):.3f}')
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / 'map.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(OUT / 'quarterly_paths.csv', index=False)
    best = grid.loc[grid.rmse_log_quarterly.idxmin()]
    settings = dict(experiment='E2 first pass: beta x kappa, count criterion only', pi=0.0, rho=0.0,
                    stepping_in_rule=rule, salience='reported COVID deaths / 10,000 (S)',
                    tau_C=TAU_C, criterion='quarterly log RMSE 2020Q2..2021Q4, model (COVID / no-COVID run) '
                                           'vs Banxico / 2017-19 trend',
                    enigh_criterion='not implemented yet', betas=BETAS.tolist(), kappas=kappas,
                    alphas_from=str(profile_path.relative_to(abm.HERE)), banxico_trend=bx_info,
                    n_cells=len(grid), n_count_compatible=int(grid.count_compatible.sum()),
                    best_cell=dict(kappa=float(best.kappa), beta=float(best.beta),
                                   rmse_log_quarterly=float(best.rmse_log_quarterly)))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    pd.set_option('display.width', 200)
    print(grid.pivot(index='beta', columns='kappa', values='rmse_log_quarterly').round(3).to_string())
    print(json.dumps(settings['best_cell']), 'count-compatible cells:', settings['n_count_compatible'])


if __name__ == '__main__':
    main()
