"""Why kappa changed E1 so little under the former rule ('all_core'). April 2020, beta = 0, pi = 0.

Run after run_e1_rule_comparison.py: python network-compatibility-model/model/diagnose_e1_kappa.py

Writes to outputs/e1-all-core-rule/:
  kappa_reach.csv          where April 2020's lost transfers sit, and how many
                           kin can recover under the former rule (exact, rho = 0)
  kappa_ceiling.csv        April 2020 count index against kappa up to 50
                           (alphas refitted under the former rule; beyond 8, the kappa = 8 fit)
  kappa_rule_variants.csv  April 2020 count index under alternative triggers
                           (stochastic, 200 paired draws; alphas NOT refitted, so
                           only the current rule is calibrated)
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abm

OUT = abm.HERE / 'outputs/e1-all-core-rule'
PROFILE = OUT / 'e0_kappa_profile.csv'
APRIL, BASE = '2020-04-01', '2019-04-01'
CEILING_KAPPAS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 12, 20, 50]
RULES = {'current': 'All eligible core migrants unable',
         'any_core': 'Any eligible core migrant unable',
         'any_link': 'Any other eligible linked migrant unable'}
DRAWS = 200


def params_for(kappa, profile):
    row = profile.loc[profile.kappa.sub(min(kappa, profile.kappa.max())).abs().idxmin()]
    return abm.Params(row.alpha_core, row.alpha_kin, kappa=kappa, rule='all_core')


def flows(pop, date, params):
    return abm.expected_flows(pop, abm.make_inputs(pd.DatetimeIndex([date]), pop['rate_columns']), params).iloc[0]


def reach(pop, profile):
    hid, H, core, el = pop['hid'], pop['H'], pop['core'], pop['eligible']
    p0 = params_for(0, profile)
    a = {d: abm.able_probability(pop, abm.make_inputs(pd.DatetimeIndex([d]), pop['rate_columns']), p0)[0]
         for d in [APRIL, BASE]}
    eligible_kin = np.bincount(hid, weights=~core & el, minlength=H)[hid] > 0
    lost_core = np.where(core, (a[BASE] - a[APRIL]) * expit(p0.alpha_core), 0)
    lost_kin = np.where(core, 0, (a[BASE] - a[APRIL]) * expit(p0.alpha_kin))
    gain = lambda k: (flows(pop, APRIL, params_for(k, profile)).transfers
                      - flows(pop, APRIL, p0).transfers)
    rows = [('All lost transfers', lost_core.sum() + lost_kin.sum()),
            ('Lost core transfers', lost_core.sum()),
            ('Lost kin transfers', lost_kin.sum()),
            ('Lost core transfers in households with eligible kin', lost_core[eligible_kin].sum()),
            ('Recovered by kin at kappa = 4', gain(4)),
            ('Recovered by kin at kappa = 50 (ceiling)', gain(50))]
    out = pd.DataFrame(rows, columns=['quantity', 'transfers_per_month'])
    out['share_of_all_lost'] = out.transfers_per_month / out.transfers_per_month.iloc[0]
    return out


def ceiling(pop, profile):
    rows = []
    for k in CEILING_KAPPAS:
        p = params_for(k, profile)
        a, b = flows(pop, APRIL, p), flows(pop, BASE, p)
        rows.append(dict(kappa=k, index_april_2020=a.transfers / b.transfers,
                         triggered_households=a.triggered_households, kin_transfers_triggered=a.kin_transfers_triggered))
    return pd.DataFrame(rows)


def one_month(pop, date, params, rule, seed):
    """One month from the stationary employment distribution (paired across dates by seed)."""
    hid, H, core, el = pop['hid'], pop['H'], pop['core'], pop['eligible']
    rate = abm.make_inputs(pd.DatetimeIndex([date]), pop['rate_columns']).rates[0, pop['job_index']]
    rng = np.random.default_rng(seed)
    z, u = rng.standard_normal(pop['M']), rng.random(pop['M'])
    able = (norm.cdf(z) < rate) & el
    send_core = able & core & (u < expit(params.alpha_core))
    unable = el & ~able
    if rule == 'current':
        trigger = (pop['has_eligible_core'] & ~(np.bincount(hid, weights=able & core, minlength=H) > 0))[hid]
    elif rule == 'any_core':
        trigger = (np.bincount(hid, weights=core & unable, minlength=H) > 0)[hid]
    else:
        trigger = (np.bincount(hid, weights=unable, minlength=H)[hid] - unable) > 0
    send_kin = able & ~core & (u < expit(params.alpha_kin + params.kappa * trigger))
    return (send_core | send_kin).sum(), send_kin.sum()


def variants(pop, profile):
    rows = []
    for rule, label in RULES.items():
        for k in [0, 4]:
            p = params_for(k, profile)
            x = np.array([[*one_month(pop, APRIL, p, rule, s), *one_month(pop, BASE, p, rule, s)]
                          for s in range(DRAWS)]).sum(0)
            rows.append(dict(rule=rule, label=label, kappa=k, index_april_2020=x[0] / x[2],
                             kin_transfers_april_2020=x[1] / DRAWS, kin_transfers_april_2019=x[3] / DRAWS,
                             transfers_april_2019=x[2] / DRAWS, calibrated=rule == 'current'))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = abm.load_population()
    profile = pd.read_csv(PROFILE)
    tables = dict(kappa_reach=reach(pop, profile), kappa_ceiling=ceiling(pop, profile),
                  kappa_rule_variants=variants(pop, profile))
    pd.set_option('display.width', 200)
    for k, v in tables.items():
        v.to_csv(OUT / f'{k}.csv', index=False)
        print(v.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
