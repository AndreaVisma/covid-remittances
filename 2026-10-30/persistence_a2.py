"""Option A2: habit persistence in desire for core AND kin migrants (2 Nov 2026). Scripts.

Specification (decided with the user, 2 Nov): the desire of every migrant follows the two-state chain of option A
    P(want_t | want_t-1) = d_t + rho (1 - d_t),     P(want_t | not want_t-1) = d_t (1 - rho),
with rho = rho_c for core and rho_k for kin migrants; marginal m_t = (1 - rho) d_t + rho m_t-1, m at the first
month = d (stationary start). d_t = expit(theta_t) from willing_able.desire with the kappa = 0 calibration
(outputs/calibration.csv), including beta N*_t. rho_c, rho_k are calibrated on 2018 conditions (ENIGH 2018 months
with receipt among receiving households: all-six and one-month shares) and are NOT changed by the crisis: in the
shock years only job loss changes ability, stepping-in (kappa) responds and need changes d_t. No never-sender type.
The 2019 calibration is unchanged (stationary: m = d whenever d is constant).

Library: persistence_a.py (simulate, ShocksA, marginal and persistence_2018 all take rho_kin); this file only adds
the A2 steps. Outputs (aggregates) in outputs/persistence-a2/.
    python 2026-10-30/persistence_a2.py checks
    python 2026-10-30/persistence_a2.py calibrate
    python 2026-10-30/persistence_a2.py shocks --need E:1      (and --need E:0)
    python 2026-10-30/persistence_a2.py households --need E:1 --kappas 1 0.8
    python 2026-10-30/persistence_a2.py mig174
Seeds: checks 7000+ (identity), 11000+ (marginal), 12345+ (autocorrelation), 13000+/14000+ (exact vs sim);
2018 calibration 5000+ (common random numbers across all (rho_c, rho_k) cells); households 9000+ (COVID run and
its no-COVID twin) and 5000+ (2018 run); mig174 rng 174.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

import willing_able as wa
import run_household_model as hm
import run_households as rh
import run_shocks as rs
import run_e1_distribution as e1d
import persistence_a as pa

OUT = wa.OUT / 'persistence-a2'
GRID = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]
E18 = pa.ENIGH18
ENIGH18_DIST = [0.2029, 0.1167, 0.1036, 0.0474, 0.0339, 0.4955]
KAPPA_CAL = 1.0
QUICK_KAPPAS = [0.0, 0.5, 0.8, 1.0]
BETAS = {'E:1': np.round(np.arange(0.0, 1.001, 0.1), 2), 'E:0': np.round(np.arange(0.0, 4.001, 0.5), 2)}
BETA_MAX = {'E:1': 3.0, 'E:0': 8.0}
ROUNDS = ['2021-01', '2021-03']


def zdist(r):
    return ((r['share_all_six'] - E18['share_all_six']) / E18['share_all_six_se']) ** 2 + \
           ((r['share_1'] - E18['share_1']) / E18['share_1_se']) ** 2


def load_rho():
    j = json.loads((OUT / 'rho.json').read_text())
    return j['rho_c'], j['rho_k']


def ntag(need):
    return need.replace(':', '')


# %% Step 1: checks
def run_checks(pop, cal, seeds=200, rc=0.8, rk=0.9):
    p = wa.params_for(cal, 1.0)
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    stim = wa.stimulus_indicator(hm.DATES, ROUNDS)
    T, core = len(hm.DATES), pop['core']
    kin = ~core
    res = dict(rho_c_check=rc, rho_k_check=rk, seeds=seeds)
    # (a) rho = 0 (both): identical to run_households.simulate, for E:1 and E:0 need and for 2018
    d18, in18 = pa.inputs_2018(pop)
    same = True
    for need_spec, beta in (('E:1', 0.15), ('E:0', 1.0)):
        need = rs.need_series(need_spec, hm.DATES)
        for s in range(3):
            for args in ((covid, p, 1.0, beta, need, 1.0, stim), (in18, p, 1.0, 0.0, None, 0.0, np.zeros(12))):
                r0, t0 = rh.simulate(pop, *args, 7000 + s)
                r1, t1 = pa.simulate(pop, *args, 7000 + s, rho=0.0, rho_kin=0.0)
                same &= np.array_equal(r0, r1) and np.array_equal(t0, t1)
    res['rho0_simulate_identical'] = bool(same)
    d19 = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], g=p['g'])
    res['constant_d_m_equals_d'] = bool(np.allclose(pa.marginal(d19, core, rc, rk), d19))
    # (b) simulated mean desire of core and kin migrants vs m_t, beta = 1.5 (E:1) so d moves strongly
    need = rs.need_series('E:1', hm.DATES)
    beta_b = 1.5
    m = pa.desire_m(pop, T, p, rc, beta_b, need, rk)
    acc = {g: [np.zeros(T), np.zeros(T)] for g in ('core', 'kin')}
    for s in range(seeds):
        _, _, W = pa.simulate(pop, covid, p, 1.0, beta_b, need, 1.0, stim, 11000 + s, rc, True, rk)
        for g, msk in (('core', core), ('kin', kin)):
            x = W[:, msk].mean(1)
            acc[g][0] += x
            acc[g][1] += x ** 2
    rows = []
    for g, msk in (('core', core), ('kin', kin)):
        mean = acc[g][0] / seeds
        se = np.sqrt((acc[g][1] / seeds - mean ** 2) / (seeds - 1))
        mm = m[:, msk].mean(1)
        z = (mean - mm) / se
        res[f'marginal_{g}_max_abs_z'] = float(np.abs(z).max())
        res[f'marginal_{g}_max_abs_diff'] = float(np.abs(mean - mm).max())
        rows.append(pd.DataFrame({'date': hm.DATES, 'group': g, 'sim_mean_desire': mean, 'm_mean': mm, 'se': se,
                                  'z': z}))
    pd.concat(rows).to_csv(OUT / 'check_markov_marginal.csv', index=False)
    # lag-1 autocorrelation at constant d (2019 desire): should be rho_c / rho_k
    for g, msk, r in (('core', core, rc), ('kin', kin, rk)):
        dd = d19[0, msk]
        covs = []
        for s in range(20):
            _, _, W = pa.simulate(pop, covid, p, 1.0, 0.0, None, 0.0, np.zeros(T), 12345 + s, rc, True, rk)
            x0, x1 = W[:-1][:, msk].astype(float), W[1:][:, msk].astype(float)
            covs.append(((x0 - dd) * (x1 - dd)).mean())
        v = (dd * (1 - dd)).mean()
        res[f'lag1_autocorr_{g}_sim'] = float(np.mean(covs) / v)
        res[f'lag1_autocorr_{g}_se'] = float(np.std(covs, ddof=1) / np.sqrt(len(covs)) / v)
    # (c) exact (StateTables with m) vs simulated receipt and transfers, kappa = 1, both need series
    a = wa.ability(pop, covid, 1.0, stim)
    for need_spec, beta, base in (('E:1', 0.3, 13000), ('E:0', 2.0, 14000)):
        need = rs.need_series(need_spec, hm.DATES)
        m = pa.desire_m(pop, T, p, rc, beta, need, rk)
        P, Tr = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, m).pairs(1.0), p['b'])
        ex_r, ex_t = P.sum(1), Tr.sum(1)
        sr, st = [], []
        for s in range(seeds):
            r, t = pa.simulate(pop, covid, p, 1.0, beta, need, 1.0, stim, base + s, rc, rho_kin=rk)
            sr.append(r.sum(1)), st.append(t.sum(1))
        sr, st = np.array(sr), np.array(st)
        zr = (sr.mean(0) - ex_r) / (sr.std(0, ddof=1) / np.sqrt(seeds))
        zt = (st.mean(0) - ex_t) / (st.std(0, ddof=1) / np.sqrt(seeds))
        k = ntag(need_spec)
        res[f'exact_vs_sim_{k}_beta{beta}_receipt_max_abs_z'] = float(np.abs(zr).max())
        res[f'exact_vs_sim_{k}_beta{beta}_transfers_max_abs_z'] = float(np.abs(zt).max())
        res[f'exact_vs_sim_{k}_pass_4se'] = bool(np.abs(zr).max() < 4 and np.abs(zt).max() < 4)
    print(json.dumps(res, indent=2))
    (OUT / 'checks.json').write_text(json.dumps(res, indent=2))
    return res


# %% Step 2: calibrate (rho_c, rho_k) to ENIGH 2018
def run_calibrate(pop, cal, seeds_grid=100, seeds_refine=200, seeds_final=400):
    p = wa.params_for(cal, KAPPA_CAL)
    nt0 = e1d.household_groups(pop)['network_type'].to_numpy()
    cache = {}

    def ev(rc, rk, seeds, stage):
        key = (round(rc, 4), round(rk, 4), seeds)
        if key not in cache:
            m18 = pa.persistence_2018(pop, p, KAPPA_CAL, rc, seeds, rho_kin=rk)
            r = dict(stage=stage, rho_c=rc, rho_k=rk, seeds=seeds, **rh.distribution(m18))
            r['z2'] = zdist(r)
            r['z_all_six'] = (r['share_all_six'] - E18['share_all_six']) / E18['share_all_six_se']
            r['z_one'] = (r['share_1'] - E18['share_1']) / E18['share_1_se']
            cache[key] = r
        return cache[key]

    rows = []
    for rc in GRID:
        for rk in GRID:
            rows.append(ev(rc, rk, seeds_grid, 'grid'))
        print(f'grid rho_c {rc} done', flush=True)
    g = pd.DataFrame(rows)
    print(g.pivot(index='rho_c', columns='rho_k', values='z2').round(1).to_string(), flush=True)
    best = g.loc[g.z2.idxmin()]
    # refine 1: step 0.01 within +-0.04 of the best grid cell (extended to 0.995 if at the top edge)
    c0, k0 = float(best.rho_c), float(best.rho_k)
    for step, half, stage in ((0.01, 0.04, 'refine 0.01'), (0.0025, 0.01, 'refine 0.0025')):
        cs = np.clip(np.round(c0 + np.arange(-half, half + 1e-9, step), 4), 0, 0.995)
        ks = np.clip(np.round(k0 + np.arange(-half, half + 1e-9, step), 4), 0, 0.995)
        rr = [ev(float(c), float(k), seeds_refine, stage) for c in np.unique(cs) for k in np.unique(ks)]
        rows += rr
        rr = pd.DataFrame(rr)
        b = rr.loc[rr.z2.idxmin()]
        c0, k0 = float(b.rho_c), float(b.rho_k)
        print(f'{stage}: best rho_c {c0}, rho_k {k0}, z2 {b.z2:.3f}', flush=True)
    # common rho
    com = [ev(float(r), float(r), seeds_refine, 'common') for r in np.round(np.arange(0.85, 0.9951, 0.005), 4)]
    rows += com
    com = pd.DataFrame(com)
    rbest = float(com.loc[com.z2.idxmin(), 'rho_c'])
    for r in np.round(np.arange(rbest - 0.004, rbest + 0.0041, 0.001), 4):
        rows.append(ev(float(r), float(r), seeds_refine, 'common fine'))
    cm = pd.DataFrame([x for x in rows if x['stage'].startswith('common')])
    rbest = float(cm.loc[cm.z2.idxmin(), 'rho_c'])
    # final distributions, more seeds
    finals, bynet = [], []
    for label, rc, rk in (('A2 calibrated', c0, k0), ('common rho', rbest, rbest), ('base rho = 0', 0.0, 0.0)):
        m18 = pa.persistence_2018(pop, p, KAPPA_CAL, rc, seeds_final, rho_kin=rk)
        r = dict(stage='final', label=label, rho_c=rc, rho_k=rk, seeds=seeds_final, **rh.distribution(m18))
        r['z2'] = zdist(r)
        nt = np.tile(nt0, seeds_final)
        r['share_receivers_kin_only'] = float(np.mean([(nt[m18[:, j] > 0] == 'Kin only').mean() for j in range(4)]))
        finals.append(r)
        for lvl in ['Kin only', 'Core only', 'Core + kin']:
            bynet.append(dict(label=label, rho_c=rc, rho_k=rk, network_type=lvl, **rh.distribution(m18, nt == lvl)))
    finals.append(dict(stage='ENIGH 2018', label='ENIGH 2018', **{f'share_{k}': v for k, v in
                       enumerate(ENIGH18_DIST, 1)}, mean=E18['mean'], share_all_six=E18['share_all_six']))
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(finals)], ignore_index=True)
    out.to_csv(OUT / 'calibration_rho.csv', index=False)
    pd.DataFrame(bynet).to_csv(OUT / 'persistence_2018_by_network.csv', index=False)
    (OUT / 'rho.json').write_text(json.dumps(dict(
        rho_c=c0, rho_k=k0, rho_common=rbest, kappa_used=KAPPA_CAL,
        target='ENIGH 2018 receiving households: all six months 0.4955 (s.e. 0.0108), one month 0.2029 (0.0081); '
               'minimise squared z-distance', seeds=dict(grid=seeds_grid, refine=seeds_refine, final=seeds_final,
                                                         base_seed=5000)), indent=2))
    print(pd.DataFrame(finals).round(3).to_string())
    print(pd.DataFrame(bynet).round(3).to_string())
    return c0, k0, rbest


# %% Step 3: shock years, quick map
def mean_marginals(pop, p, rc, rk, beta, need):
    T = len(hm.DATES)
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['g'])
    m = pa.marginal(d, pop['core'], rc, rk)
    out = {}
    for g, msk in (('core', pop['core']), ('kin', ~pop['core'])):
        for lab, date in (('2019_01', '2019-01-01'), ('2021_12', '2021-12-01'), ('2022_12', '2022-12-01')):
            i = hm.DATES.get_loc(pd.Timestamp(date))
            out[f'm_{g}_{lab}'] = float(m[i, msk].mean())
            if lab != '2019_01':
                out[f'd_{g}_{lab}'] = float(d[i, msk].mean())
    return out


def run_shocks_a2(pop, cal, rc, rk, need, kappas=QUICK_KAPPAS, tag=''):
    out = OUT / f'shocks-{ntag(need)}{tag}'
    out.mkdir(parents=True, exist_ok=True)
    sh = pa.ShocksA(pop, cal, ROUNDS, 1.0, rc, need=need, rho_kin=rk)
    betas = BETAS[need]
    rows, paths, monthly, fits = [], [], [], []
    for kappa in kappas:
        for beta in betas:
            r, pth, m = sh.summary(kappa, float(beta))
            rows.append(r), paths.append(pth)
            if beta == 0:
                monthly.append(m)
        bmax = BETA_MAX[need]
        while True:
            res = minimize_scalar(lambda b: sh.summary(kappa, b)[0]['rmse_fit'], bounds=(0.0, bmax),
                                  method='bounded', options=dict(xatol=1e-3))
            if res.x < 0.98 * bmax:
                break
            bmax *= 2
            print(f'bound binds; raised to {bmax}', flush=True)
        r, pth, m = sh.summary(kappa, float(res.x))
        p = wa.params_for(cal, kappa)
        fits.append(dict(r, rho_c=rc, rho_k=rk, need=need, beta_bound=bmax, **sh.level_gap(kappa),
                         **sh.shortage(kappa, float(res.x)), **mean_marginals(pop, p, rc, rk, float(res.x), sh.need)))
        paths.append(pth.assign(fitted=True)), monthly.append(m.assign(fitted=True))
        print(f'{need} rho {rc}/{rk} kappa {kappa}: beta {res.x:.3f}, rmse {res.fun:.4f}', flush=True)
    grid = pd.DataFrame(rows).assign(rho_c=rc, rho_k=rk, need=need)
    grid.to_csv(out / 'map.csv', index=False)
    fits = pd.DataFrame(fits)
    fits.to_csv(out / 'beta_fit.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / 'quarterly_paths.csv', index=False)
    pd.concat(monthly, ignore_index=True).to_csv(out / 'monthly_paths.csv', index=False)
    (out / 'run_settings.json').write_text(json.dumps(dict(
        option='A2: habit persistence in desire, core and kin (Markov marginal m_t)', rho_c=rc, rho_k=rk,
        stimulus_rounds=ROUNDS, pi=1.0, kappas=kappas, betas=list(map(float, betas)), need=need,
        calibration='calibration.csv'), indent=2))
    print(grid.pivot(index='beta', columns='kappa', values='rmse_fit').round(3).to_string())
    print(fits.round(4).T.to_string())
    return fits


# %% Step 4: household results
def run_households_a2(pop, cal, rc, rk, need_spec, kappas, seeds=200):
    hout = OUT / f'households-{ntag(need_spec)}'
    hout.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(OUT / f'shocks-{ntag(need_spec)}' / 'beta_fit.csv')
    pi = 1.0
    groups = e1d.household_groups(pop)
    hraw = pop['h']
    enigh_grp = {'all': np.ones(pop['H'], bool),
                 'no resident worker': hraw.resident_worker.eq(0).to_numpy(),
                 'resident worker': hraw.resident_worker.gt(0).to_numpy(),
                 'resident 65+': hraw.resident_older.gt(0).to_numpy(),
                 'no resident 65+': hraw.resident_older.eq(0).to_numpy()}
    for lvl in ['Kin only', 'Core only', 'Core + kin']:
        enigh_grp[f'network: {lvl}'] = (groups['network_type'] == lvl).to_numpy()
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = wa.abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need = rs.need_series(need_spec, hm.DATES)
    stim, zero = wa.stimulus_indicator(hm.DATES, ROUNDS), np.zeros(len(hm.DATES))
    d18, in18 = pa.inputs_2018(pop)
    window = (hm.DATES >= rh.WINDOW[0]) & (hm.DATES <= rh.WINDOW[1])
    enigh_al = pd.read_csv(wa.NCM / 'identification/enigh/outputs/pre_excess_over_trend.csv').query(
        "from_year == 2018 and to_year in (2020, 2022) and measure in ('receipt', 'tpr_counts')").set_index(
        ['measure', 'to_year'])
    import enigh_aligned as ea
    al_rows, dist_rows, gl_rows, se_rows = [], [], [], []
    nt = groups['network_type'].to_numpy()
    for kappa in kappas:
        p = wa.params_for(cal, kappa)
        beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
        acc = {y: [] for y in (2020, 2022)}
        months = {k: [] for k in ('2018', '2020', '2022', 'cf2020', 'cf2022')}
        per_seed = []
        gl = {}
        for seed in range(seeds):
            rc_, tc = pa.simulate(pop, covid, p, kappa, beta, need, pi, stim, 9000 + seed, rc, rho_kin=rk)
            rf, tf = pa.simulate(pop, cf, p, kappa, 0.0, None, 0.0, zero, 9000 + seed, rc, rho_kin=rk)
            r18, _ = pa.simulate(pop, in18, p, kappa, 0.0, None, 0.0, np.zeros(12), 5000 + seed, rc, rho_kin=rk)
            for y in acc:
                acc[y].append((ea.window_measures(rc_, tc, hm.DATES, y), ea.window_measures(rf, tf, hm.DATES, y)))
            ms = {'2018': rh.months_before_interview(r18, d18)}
            for y in (2020, 2022):
                ms[str(y)] = pa.months_window(rc_, hm.DATES, y)
                ms[f'cf{y}'] = pa.months_window(rf, hm.DATES, y)
            for k, v in ms.items():
                months[k].append(v)
            ds = {k: rh.distribution(v) for k, v in ms.items()}
            per_seed.append({f'{c}_{k}': ds[k][c] for k in ds for c in ('mean', 'share_1', 'share_all_six')})
            dm = rc_[window].sum(0).astype(int) - rf[window].sum(0).astype(int)
            for lvl in ['all', 'Kin only', 'Core only', 'Core + kin']:
                msk = np.ones(pop['H'], bool) if lvl == 'all' else nt == lvl
                x = dm[msk]
                neg = -x[x < 0].sum()
                v = np.array([(x < 0).mean(), (x <= -3).mean(), (x > 0).mean(), (x >= 3).mean(), (x == 0).mean(),
                              x.mean(), (-x[x <= -3].sum()) / neg if neg else np.nan,
                              rf[window][:, msk].any(0).mean()])
                gl.setdefault(lvl, []).append(v)
        # Monte Carlo s.e. of the model changes (across seeds)
        ps = pd.DataFrame(per_seed)
        for c in ('mean', 'share_1', 'share_all_six'):
            for y in ('2020', '2022'):
                for ref in ('2018', f'cf{y}'):
                    x = ps[f'{c}_{y}'] - ps[f'{c}_{ref}']
                    se_rows.append(dict(kappa=kappa, measure=c, year=y, vs=ref, mean=x.mean(),
                                        mc_se=x.std(ddof=1) / np.sqrt(len(x))))
        for y, v in acc.items():
            c = np.array([a for a, _ in v]).mean(0)
            f = np.array([b for _, b in v]).mean(0)
            lr = np.log(c / f)
            al_rows.append(dict(rho_c=rc, rho_k=rk, need=need_spec, kappa=kappa, beta=beta, year=y, seeds=seeds,
                                model_receiving=lr[0], model_tpr=lr[1], model_monthly_receipt=lr[2],
                                receiving_share_cf=f[0], monthly_receipt_cf=f[2],
                                enigh_receipt=enigh_al.loc[('receipt', y), 'excess'],
                                enigh_receipt_se=enigh_al.loc[('receipt', y), 'excess_se'],
                                enigh_tpr_counts=enigh_al.loc[('tpr_counts', y), 'excess'],
                                enigh_tpr_counts_se=enigh_al.loc[('tpr_counts', y), 'excess_se']))
        mm = {k: np.concatenate(v) for k, v in months.items()}
        for gname, gmask in enigh_grp.items():
            msk = np.tile(gmask, seeds)
            for k, arr in mm.items():
                dist_rows.append(dict(rho=f'{rc}/{rk}', kappa=kappa, group=gname, run=k, **rh.distribution(arr, msk)))
        names = ['share_losing', 'share_losing_3plus', 'share_gaining', 'share_gaining_3plus', 'share_no_change',
                 'mean_change_months', 'share_of_lost_months_in_3plus_losers', 'share_receiving_cf_window']
        for lvl, v in gl.items():
            gl_rows.append(dict(rho_c=rc, rho_k=rk, need=need_spec, kappa=kappa, beta=beta, network_type=lvl,
                                seeds=seeds, **dict(zip(names, np.nanmean(np.array(v), 0)))))
        print(f'households {need_spec} kappa {kappa}: done', flush=True)
    al = pd.DataFrame(al_rows)
    al.to_csv(hout / 'enigh_aligned.csv', index=False)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(hout / 'months_distribution.csv', index=False)
    gl = pd.DataFrame(gl_rows)
    gl.to_csv(hout / 'who_gains_loses_realised.csv', index=False)
    cmp = pa.compare_months_enigh(dist)
    cmp.to_csv(hout / 'months_change_vs_enigh.csv', index=False)
    pd.DataFrame(se_rows).to_csv(hout / 'months_change_mc_se.csv', index=False)
    (hout / 'households_settings.json').write_text(json.dumps(dict(
        rho_c=rc, rho_k=rk, need=need_spec, kappas=kappas, seeds=seeds, seed_bases=dict(covid_and_twin=9000, y2018=5000),
        window=list(rh.WINDOW), interview_shares=rh.INTERVIEW), indent=2))
    print(al.round(3).to_string())
    print(dist[dist.group == 'all'].round(3).to_string())
    print(pd.DataFrame(se_rows).round(4).to_string())
    print(gl.round(3).to_string())


# %% Step 5: mig174 (report only)
def run_mig174(pop, cal, rhos, reps=200, lengths=(3, 6, 12, 24, 60)):
    p = wa.params_for(cal, 0.0)
    core = np.flatnonzero(pop['core'])
    d = wa.desire(pop, 1, p['a_c'], p['a_k'], p['eta'], g=p['g'])[0, core]
    d19 = pd.date_range('2019-01-01', '2019-12-01', freq='MS')
    rates = wa.abm.make_inputs(d19, pop['rate_columns']).rates[:, pop['job_index'][core]]
    phi = wa.abm.employment_phi()
    Lmax = max(lengths)
    rows = []
    for rho in sorted(set(rhos)):
        rng = np.random.default_rng(174)
        n = len(core) * reps
        dd, rr = np.tile(d, reps), np.tile(rates, (1, reps))
        start = rng.integers(0, 12, n)
        z = rng.standard_normal(n)
        wants = rng.random(n) < dd
        any_send, any_want, res = np.zeros(n, bool), np.zeros(n, bool), {}
        for t in range(Lmax):
            if t > 0:
                z = phi * z + np.sqrt(1 - phi ** 2) * rng.standard_normal(n)
                wants = rng.random(n) < np.where(wants, dd + rho * (1 - dd), dd * (1 - rho))
            able = norm.cdf(z) < rr[(start + t) % 12, np.arange(n)]
            any_send |= wants & able
            any_want |= wants
            if t + 1 in lengths:
                res[t + 1] = (any_send.mean(), any_want.mean())
        for L in lengths:
            rows.append(dict(rho_c=rho, months=L, p_any_send=res[L][0], p_any_want=res[L][1], migrants=len(core),
                             reps=reps))
    out = pd.DataFrame(rows)
    mmp = {3: 0.70, 6: 0.93, 12: 0.88, 24: 0.89, 60: 0.91}
    out['mmp_heads'] = out.months.map(mmp)
    out.to_csv(OUT / 'mig174_test.csv', index=False)
    print(out.pivot(index='months', columns='rho_c', values='p_any_send').round(3).to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['checks', 'calibrate', 'shocks', 'households', 'mig174'])
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--need', default='E:1')
    ap.add_argument('--kappas', type=float, nargs='*', default=None)
    ap.add_argument('--rho', type=float, nargs=2, default=None, help='override (rho_c, rho_k)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pop, cal = pa.setup()
    print(f"households {pop['H']}, migrants {pop['M']} (core {int(pop['core'].sum())})", flush=True)
    t0 = time.time()
    if args.step == 'checks':
        run_checks(pop, cal, args.seeds)
    elif args.step == 'calibrate':
        run_calibrate(pop, cal)
    else:
        rc, rk = args.rho if args.rho else load_rho()
        if args.step == 'shocks':
            run_shocks_a2(pop, cal, rc, rk, args.need)
        elif args.step == 'households':
            run_households_a2(pop, cal, rc, rk, args.need, args.kappas or [1.0], args.seeds)
        elif args.step == 'mig174':
            j = json.loads((OUT / 'rho.json').read_text())
            run_mig174(pop, cal, [0.0, rc, j['rho_common'], 0.5, 0.8, 0.9, 0.95])
    print(f'{args.step}: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
