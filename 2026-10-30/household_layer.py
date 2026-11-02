"""Household layer: one habit parameter rho, average-preserving (2 Nov 2026).

Run: python 2026-10-30/household_layer.py [--seeds 200] [--rho R]   (without --rho: calibrate rho first)
Desire of every migrant (core and kin) follows a two-state chain with persistence rho that keeps each migrant's
monthly mean desire at exactly d_t (run_households.simulate, rho argument):
    P(want_t | wanted) = d_t - rho d_{t-1} + rho,   P(want_t | did not) = d_t - rho d_{t-1}.
National results (exact monthly expectations) are therefore those of the base model; rho only sets who sends
from one month to the next. rho is calibrated to ONE target: the ENIGH 2018 share of receiving households that
receive in all six months before an Aug-Nov interview (0.496). Everything else is out of sample:
    1. 2018 distribution of months with receipt (1-6) vs ENIGH (one-month share 0.203, mean 4.08)
    2. change in that distribution 2020 and 2022 vs 2018 (ENIGH: all six -2.5 / +7.8 pp; one month +1.6 / -5.3 pp)
    3. ENIGH-aligned margins 2020 and 2022 (receiving at least once in six months; transfers per receiving household)
    4. who gains and loses, realised paths Apr 2020 - Dec 2021 vs the no-COVID twin, by network type
    5. national check: simulated national index = exact index (base model)
    6. mig174: share of core migrants wanting and able in at least one of L months (MMP heads 0.70-0.93)
Base settings: kappa and beta from outputs/shocks/beta_fit.csv (best fit), base stimulus, cumulative excess deaths.
Outputs (aggregates) in outputs/household-layer/.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

import willing_able as wa
import run_household_model as hm
import run_households as rh
import run_e1_distribution as e1d
import enigh_aligned as ea

OUT = wa.OUT / 'household-layer'
ENIGH18 = dict(share_1=0.203, share_2=0.117, share_3=0.104, share_4=0.047, share_5=0.034, share_6=0.496, mean=4.08)
TARGET, TARGET_SE = 0.496, 0.011


def months_dist(rec, dates):
    return rh.distribution(rh.months_before_interview(rec, dates))


def calibrate(pop, p, kappa, seeds):
    d18 = pd.date_range('2018-01-01', '2018-12-01', freq='MS')
    in18 = wa.abm.make_inputs(d18, pop['rate_columns'])
    zero = np.zeros(12)

    def all_six(rho):
        m = np.concatenate([rh.months_before_interview(rh.simulate(pop, in18, p, kappa, 0.0, None, 0.0, zero, 5000 + s,
                                                                   rho)[0], d18) for s in range(seeds)])
        return rh.distribution(m)

    rows = []
    for rho in (0.0, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96):
        r = all_six(rho)
        rows.append(dict(rho=rho, **r))
        print(f'rho {rho}: all six {r["share_all_six"]:.3f}, one month {r["share_1"]:.3f}, mean {r["mean"]:.2f}', flush=True)
    grid = pd.DataFrame(rows)
    lo = grid.loc[grid.share_all_six < TARGET, 'rho'].max()
    hi = grid.loc[grid.share_all_six >= TARGET, 'rho'].min()
    for _ in range(6):                                         # bisection on the one target
        mid = round((lo + hi) / 2, 4)
        r = all_six(mid)
        rows.append(dict(rho=mid, **r))
        lo, hi = (mid, hi) if r['share_all_six'] < TARGET else (lo, mid)
        print(f'rho {mid}: all six {r["share_all_six"]:.3f}', flush=True)
    grid = pd.DataFrame(rows).sort_values('rho')
    best = grid.iloc[(grid.share_all_six - TARGET).abs().argmin()]
    return float(best.rho), grid


def mig174(pop, rho, seeds=20, Ls=(3, 6, 12, 24, 60)):
    """Share of core migrants wanting AND able in at least one of L months (2019 employment, stationary desire)."""
    p = wa.params_for(wa.load_calibration(), 0.0)
    dates = pd.date_range('2015-01-01', periods=max(Ls), freq='MS')
    rates = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in dates]), pop['rate_columns']).rates
    d = wa.desire(pop, 1, p['a_c'], p['a_k'], p['eta'], g=p['g'])[0]
    core = pop['core']
    phi = wa.abm.employment_phi()
    hit = {L: [] for L in Ls}
    for s in range(seeds):
        rng = np.random.default_rng(17400 + s)
        z = rng.standard_normal(pop['M'])
        wants = rng.random(pop['M']) < d
        ever = np.zeros(pop['M'], bool)
        for t in range(max(Ls)):
            if t > 0:
                z = phi * z + np.sqrt(1 - phi ** 2) * rng.standard_normal(pop['M'])
                wants = rng.random(pop['M']) < np.where(wants, d * (1 - rho) + rho, d * (1 - rho))
            able = norm.cdf(z) < rates[t, pop['job_index']]
            ever |= wants & able
            if t + 1 in hit:
                hit[t + 1].append(ever[core].mean())
    return {L: float(np.mean(v)) for L, v in hit.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--rho', type=float, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pop = wa.load_population()
    cal = wa.load_calibration()
    fits = pd.read_csv(wa.OUT / 'shocks' / 'beta_fit.csv')
    settings = json.loads((wa.OUT / 'shocks' / 'run_settings.json').read_text())
    best = fits.loc[fits.rmse_fit.idxmin()]
    kappa, beta = float(best.kappa), float(best.beta)
    p = wa.params_for(cal, kappa)
    pi, rounds = settings['pi'], settings['stimulus_rounds']

    if args.rho is None:
        rho, grid = calibrate(pop, p, kappa, 100)
        grid.to_csv(OUT / 'calibration_rho.csv', index=False)
    else:
        rho = args.rho
    print(f'rho = {rho}', flush=True)

    # paired simulations: 2018 run, 2020 run (as run_households), and the COVID / no-COVID runs 2019-22
    d18, d20 = (pd.date_range(f'{y}-01-01', f'{y}-12-01', freq='MS') for y in (2018, 2020))
    in18, in20 = (wa.abm.make_inputs(d, pop['rate_columns']) for d in (d18, d20))
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need_c, stim_c = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, rounds)
    need20, stim20 = hm.need_stock(d20), wa.stimulus_indicator(d20, rounds)
    zero12, zero = np.zeros(12), np.zeros(len(hm.DATES))
    window = (hm.DATES >= '2020-04-01') & (hm.DATES <= '2021-12-01')
    net = e1d.household_groups(pop)['network_type'].astype(str).to_numpy()

    results = {}
    for r in (0.0, rho):
        m18, m20, m22, w = [], [], [], {2020: [], 2022: []}
        dm, nat_c, nat_f = [], [], []
        for s in range(args.seeds):
            m18.append(rh.months_before_interview(rh.simulate(pop, in18, p, kappa, 0.0, None, 0.0, zero12, 5000 + s, r)[0], d18))
            m20.append(rh.months_before_interview(rh.simulate(pop, in20, p, kappa, beta, need20, pi, stim20, 5000 + s, r)[0], d20))
            rc, tc = rh.simulate(pop, covid, p, kappa, beta, need_c, pi, stim_c, 9000 + s, r)
            rf, tf = rh.simulate(pop, cf, p, kappa, 0.0, None, 0.0, zero, 9000 + s, r)
            idx22 = {(d.year, d.month): i for i, d in enumerate(hm.DATES)}
            m22.append(np.column_stack([rc[[idx22[(2022, mo - k)] for k in range(1, 7)]].sum(0) for mo in rh.INTERVIEW]))
            for y in w:
                w[y].append((ea.window_measures(rc, tc, hm.DATES, y), ea.window_measures(rf, tf, hm.DATES, y)))
            dm.append(rc[window].sum(0).astype(int) - rf[window].sum(0).astype(int))
            nat_c.append(tc.sum(1)), nat_f.append(tf.sum(1))
        dist = {y: rh.distribution(np.concatenate(m)) for y, m in ((2018, m18), (2020, m20), (2022, m22))}
        aligned = {y: np.log(np.array([a for a, _ in v]).mean(0) / np.array([b for _, b in v]).mean(0)) for y, v in w.items()}
        dm = np.array(dm)                                              # (seeds, H)
        who = []
        for g in ['all'] + sorted(set(net)):
            sel = np.ones(pop['H'], bool) if g == 'all' else net == g
            x = dm[:, sel]
            lost = np.clip(-x, 0, None)
            who.append(dict(network=g, households=int(sel.sum()), share_losing_any=float((x < 0).mean()),
                            share_losing_3plus=float((x <= -3).mean()), share_gaining_any=float((x > 0).mean()),
                            mean_change_months=float(x.mean()),
                            share_lost_months_in_3plus=float(lost[x <= -3].sum() / max(lost.sum(), 1))))
        results[r] = dict(dist=dist, aligned=aligned, who=pd.DataFrame(who), nat=hm.quarterly(pd.Series(np.mean(nat_c, 0), index=hm.DATES)) / hm.quarterly(pd.Series(np.mean(nat_f, 0), index=hm.DATES)))
        print(f'households rho {r}: done', flush=True)

    # exact national index (base) for the identity check
    import run_shocks as rs
    sh = rs.Shocks(pop, cal, rounds, pi)
    _, paths, _ = sh.summary(kappa, beta)
    exact = paths.set_index('quarter').index_transfers

    rows = []
    for r, res in results.items():
        for y, dd in res['dist'].items():
            rows.append(dict(rho=r, year=y, **dd))
    dist = pd.DataFrame(rows)
    dist.to_csv(OUT / 'months_distribution.csv', index=False)
    al = pd.DataFrame([dict(rho=r, year=y, receiving=v[0], tpr=v[1], monthly_receipt=v[2])
                       for r, res in results.items() for y, v in res['aligned'].items()])
    al.to_csv(OUT / 'enigh_aligned.csv', index=False)
    pd.concat([res['who'].assign(rho=r) for r, res in results.items()]).to_csv(OUT / 'who_gains_loses.csv', index=False)
    natc = pd.DataFrame({f'sim_rho_{r}': res['nat'] for r, res in results.items()}).assign(exact=exact)
    natc.to_csv(OUT / 'national_check.csv')
    mg = {r: mig174(pop, r) for r in (0.0, rho)}
    pd.DataFrame(mg).rename_axis('months').to_csv(OUT / 'mig174.csv')
    (OUT / 'run_settings.json').write_text(json.dumps(dict(
        rho=rho, kappa=kappa, beta=beta, seeds=args.seeds, target='ENIGH 2018 all-six share 0.496 (s.e. 0.011)',
        spec='average-preserving habit, same rho for core and kin; national results = base model'), indent=2))

    pd.set_option('display.width', 200)
    print('\nmonths distribution among receiving households\n', dist.round(3).to_string(index=False))
    print('ENIGH 2018', ENIGH18, '| ENIGH all six 2018/2020/2022 .496/.470/.574; one month .203/.219/.150; mean 4.08/3.98/4.41')
    print('\nENIGH-aligned margins (log COVID / no-COVID)\n', al.round(3).to_string(index=False))
    print('ENIGH: 2020 receiving -0.073 tpr +0.075 (s.e. 0.06); 2022 -0.027 / +0.093 (s.e. 0.095)')
    print('\nwho gains and loses\n', pd.concat([res['who'].assign(rho=r) for r, res in results.items()]).round(3).to_string(index=False))
    print('\nnational check (quarterly index)\n', natc.round(4).to_string())
    print('\nmig174 (core migrants wanting and able in >=1 of L months; MMP heads .70/.93/.88/.89/.91)\n', pd.DataFrame(mg).round(3).to_string())


if __name__ == '__main__':
    main()
