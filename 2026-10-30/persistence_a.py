"""Option A: habit persistence in desire (core migrants only; 2 Nov 2026). Library and scripts.

Specification (decided with the user, 2 Nov): each core migrant's desire is a two-state Markov chain
    P(want_t = 1 | want_t-1 = 1) = d_t + rho (1 - d_t),     P(want_t = 1 | want_t-1 = 0) = d_t (1 - rho),
with d_t = expit(theta_t) the desire of willing_able.desire (kappa = 0 calibration, outputs/calibration.csv) and
rho in [0, 1) the persistence (lag-1 autocorrelation when d is constant). Kin keep independent monthly draws.
rho = 0 is the current model exactly. The marginal monthly desire follows
    m_t = (1 - rho) d_t + rho m_t-1,   m_0 = d_0 (stationary start),
so m = d whenever d is constant (2019 calibration, no-COVID runs, 2018). Migrants stay independent given the
household and desire is independent of ability, so the exact monthly computation (willing_able.StateTables) holds
with m in place of d. The crisis (beta N*_t) raises d_t and so both the probability of starting and of continuing
to send; m responds to d with a geometric lag (mean lag rho / (1 - rho) months).

Run (each step writes aggregates to outputs/persistence-a/):
    python 2026-10-30/persistence_a.py checks        rho = 0 reproduces the current model; Markov marginal; exact vs sim
    python 2026-10-30/persistence_a.py calibrate     rho to ENIGH 2018 (share of receiving households, all six months)
    python 2026-10-30/persistence_a.py mig174        any sending over L months (MMP heads' last trip) vs rho
    python 2026-10-30/persistence_a.py shocks        quick shock map with the Markov marginal (kappa 0/.5/.8/1)
    python 2026-10-30/persistence_a.py households    ENIGH-aligned measures, months 2020/2022 vs 2018, who gains/loses
    python 2026-10-30/persistence_a.py all
Does not modify willing_able.py, run_shocks.py, run_households.py or enigh_aligned.py (imported read-only;
simulate() and Shocks.run/shortage are copied and adapted here).
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import norm

import willing_able as wa
import run_household_model as hm
import run_households as rh
import run_shocks as rs
import run_e1_distribution as e1d

OUT = wa.OUT / 'persistence-a'
RHO_GRID = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
ENIGH18 = dict(mean=4.079, mean_se=0.044, share_all_six=0.4955, share_all_six_se=0.0108,
               share_1=0.2029, share_1_se=0.0081)
MIG174 = {'1-3': (0.70, 23), '4-6': (0.93, None), '7-12': (0.88, None), '13-24': (0.89, None),
          '25-60': (0.91, None), '61+': (0.76, None)}
KAPPA_CAL = 1.0                       # kappa used for the 2018 calibration of rho (base; checked afterwards)
QUICK_KAPPAS = [0.0, 0.5, 0.8, 1.0]
QUICK_BETAS = np.round(np.arange(0.0, 0.401, 0.05), 2)


# %% Library
def marginal(d, core, rho, rho_kin=0.0):
    """(T, M) marginal monthly desire m_t = (1 - rho) d_t + rho m_t-1 for core migrants (m_0 = d_0); kin: m = d
    (rho_kin > 0: same recursion for kin with rho_kin; diagnostic only)."""
    d = np.asarray(d, float)
    if rho == 0 and rho_kin == 0:
        return d
    r = np.where(core, rho, rho_kin)
    m = d.copy()
    for t in range(1, d.shape[0]):
        m[t] = (1 - r) * d[t] + r * m[t - 1]
    return m


def desire_m(pop, T, p, rho, beta=0.0, need=None, rho_kin=0.0):
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['g'])
    return marginal(d, pop['core'], rho, rho_kin)


def simulate(pop, inputs, p, kappa, beta, need, pi, stim, seed, rho=0.0, return_wants=False, rho_kin=0.0):
    """run_households.simulate with Markov desire for core migrants. Same seed and rho = 0 -> identical paths.
    Returns (T, H) receipt (bool) and transfers (int); with return_wants also (T, M) wants (bool).
    rho_kin: persistence for kin (diagnostic only, NOT the agreed specification; default 0 = iid kin)."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    phi = wa.abm.employment_phi()
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['g'])
    nc, nk = pop['n_core'], pop['n_kin']
    rec, trans = np.zeros((T, H), bool), np.zeros((T, H), int)
    W = np.zeros((T, M), bool) if return_wants else None
    z = streams[0].standard_normal(M)
    hsum = lambda v: np.bincount(hid, weights=v, minlength=H).astype(int)
    wants = None
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        employed = norm.cdf(z) < inputs.rates[t, pop['job_index']]
        checked = streams[2].random(M) < pi * stim[t]
        able = (employed | checked) & pop['eligible']
        if t == 0 or (rho == 0 and rho_kin == 0):
            pw = d[t]                                                     # stationary start: m_0 = d_0
        else:
            r = np.where(core, rho, rho_kin)
            pw = np.where(wants, d[t] + r * (1 - d[t]), d[t] * (1 - r))
        wants = streams[3].random(M) < pw
        if return_wants:
            W[t] = wants
        lost = wants & ~able
        u = streams[4].random(M)
        cand = lost * ((u < kappa).astype(int) if kappa <= 1 else 1 + (u < kappa - 1))
        Wc, Wk = hsum(wants & able & core), hsum(wants & able & ~core)
        Xc, Xk, R = hsum(cand * core), hsum(cand * ~core), hsum(able & ~wants)
        xc = np.minimum(Xc, R)
        xk = np.minimum(Xk, R - xc)
        Ec, Ek = wa.relabel(Wc + xc, Wk + xk, nc, nk)
        b0, b_any, b_c, b_k0 = p['b']
        s = wa.expit(b0 + b_any * (Ec > 0) + b_c * np.log1p(Ec) + b_k0 * (nc == 0) * np.log1p(Ek))
        rec[t] = streams[5].random(H) < s
        trans[t] = rec[t] * np.maximum(Ec + Ek, 1)
    return (rec, trans, W) if return_wants else (rec, trans)


class ShocksA(rs.Shocks):
    """run_shocks.Shocks with the Markov marginal m_t in the COVID run (no-COVID run: d constant, so m = d)."""

    def __init__(self, pop, cal, rounds, pi, rho, need='E:1', rho_kin=0.0):
        super().__init__(pop, cal, rounds, pi, need)
        self.rho, self.rho_kin = rho, rho_kin

    def _m(self, p, beta):
        return desire_m(self.pop, len(hm.DATES), p, self.rho, beta, self.need, self.rho_kin)

    def run(self, kappa, beta):
        p = wa.params_for(self.cal, kappa)
        m = self._m(p, beta)
        Pc, Tc = wa.receipt_and_transfers(self.pop, wa.StateTables(self.pop, self.a_covid, m).pairs(kappa), p['b'])
        Pf, Tf = self.counterfactual(kappa)
        s = lambda z: pd.Series(z.sum(1), index=hm.DATES)
        return s(Pc), s(Tc), s(Pf), s(Tf)

    def shortage(self, kappa, beta):
        p = wa.params_for(self.cal, kappa)
        drawn, unfilled, short = wa.StateTables(self.pop, self.a_covid, self._m(p, beta)).shortage(kappa)
        out = {}
        for name, msk in (('2020q2', (hm.DATES >= '2020-04-01') & (hm.DATES <= '2020-06-01')),
                          ('2020_21', (hm.DATES >= '2020-04-01') & (hm.DATES <= '2021-12-01'))):
            out[f'share_households_short_{name}'] = float(short[msk].mean() / self.pop['H'])
            out[f'share_replacements_unfilled_{name}'] = float(unfilled[msk].sum() / max(drawn[msk].sum(), 1e-12))
        return out


def months_window(rec, dates, year):
    """(H, 4) months with receipt in the six months before each Aug..Nov interview of `year`."""
    idx = {(d.year, d.month): i for i, d in enumerate(pd.DatetimeIndex(dates))}
    return np.column_stack([rec[[idx[(year, m - k)] for k in range(1, 7)]].sum(0) for m in rh.INTERVIEW])


def setup():
    pop = wa.load_population()
    cal = wa.load_calibration()
    return pop, cal


def inputs_2018(pop):
    d18 = pd.date_range('2018-01-01', '2018-12-01', freq='MS')
    return d18, wa.abm.make_inputs(d18, pop['rate_columns'])


def persistence_2018(pop, p, kappa, rho, seeds, base_seed=5000, rho_kin=0.0):
    """Interview-weighted months distribution among receiving households, 2018 (as run_households)."""
    d18, in18 = inputs_2018(pop)
    ms = [rh.months_before_interview(simulate(pop, in18, p, kappa, 0.0, None, 0.0, np.zeros(12), base_seed + s,
                                              rho, rho_kin=rho_kin)[0], d18) for s in range(seeds)]
    return np.concatenate(ms)


# %% Step 1: checks
def run_checks(pop, cal, seeds=200):
    p = wa.params_for(cal, 1.0)
    fits = pd.read_csv(wa.OUT / 'shocks' / 'beta_fit.csv')
    beta = float(fits.loc[np.isclose(fits.kappa, 1.0), 'beta'].iloc[0])
    covid = wa.abm.make_inputs(hm.DATES, pop['rate_columns'])
    rounds = ['2021-01', '2021-03']
    need, stim = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, rounds)
    T = len(hm.DATES)
    res = {}
    # (a) rho = 0: marginal is d; simulate identical to run_households.simulate
    d = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta, need, p['g'])
    res['rho0_marginal_equals_desire'] = bool(np.array_equal(marginal(d, pop['core'], 0.0), d))
    d18, in18 = inputs_2018(pop)
    same = True
    for s in range(5):
        for args in ((covid, p, 1.0, beta, need, 1.0, stim), (in18, p, 1.0, 0.0, None, 0.0, np.zeros(12))):
            r0, t0 = rh.simulate(pop, *args, 7000 + s)
            r1, t1 = simulate(pop, *args, 7000 + s, rho=0.0)
            same &= np.array_equal(r0, r1) and np.array_equal(t0, t1)
    res['rho0_simulate_identical'] = bool(same)
    # (a') stationarity: with constant d, m = d for any rho
    d19 = wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], g=p['g'])
    res['constant_d_m_equals_d_rho0.9'] = bool(np.allclose(marginal(d19, pop['core'], 0.9), d19))
    # (b) rho = 0.8 with a strongly moving d (beta = 1.5): simulated mean core desire vs m_t
    rho, beta_b = 0.8, 1.5
    m = desire_m(pop, T, p, rho, beta_b, need)
    core = pop['core']
    acc, acc2 = np.zeros(T), np.zeros(T)
    for s in range(seeds):
        _, _, W = simulate(pop, covid, p, 1.0, beta_b, need, 1.0, stim, 11000 + s, rho, return_wants=True)
        x = W[:, core].mean(1)
        acc += x
        acc2 += x ** 2
    mean = acc / seeds
    se = np.sqrt((acc2 / seeds - mean ** 2) / (seeds - 1))
    m_b = m[:, core].mean(1)
    zb = (mean - m_b) / se
    res['markov_marginal_max_abs_z'] = float(np.abs(zb).max())
    res['markov_marginal_max_abs_diff'] = float(np.abs(mean - m[:, core].mean(1)).max())
    res['m_core_jan2019'], res['m_core_dec2021'], res['d_core_dec2021'] = (
        float(m[0, core].mean()), float(m[hm.DATES.get_loc(pd.Timestamp('2021-12-01')), core].mean()),
        float(wa.desire(pop, T, p['a_c'], p['a_k'], p['eta'], beta_b, need, p['g'])[
            hm.DATES.get_loc(pd.Timestamp('2021-12-01')), core].mean()))
    # lag-1 autocorrelation of core wants in 2019 (constant d): should be rho
    dd = d19[0, core]
    covs = []
    for s in range(20):                                   # 20 seeds x 48 months, constant d (beta = 0)
        _, _, W = simulate(pop, covid, p, 1.0, 0.0, None, 0.0, np.zeros(T), 12345 + s, rho, return_wants=True)
        x0, x1 = W[:-1, core].astype(float), W[1:, core].astype(float)
        covs.append(((x0 - dd) * (x1 - dd)).mean())
    res['lag1_autocorr_core_sim_rho0.8'] = float(np.mean(covs) / (dd * (1 - dd)).mean())
    res['lag1_autocorr_core_sim_se'] = float(np.std(covs, ddof=1) / np.sqrt(len(covs)) / (dd * (1 - dd)).mean())
    # (c) exact (StateTables with m) vs simulated mean receipt and transfers, rho = 0.8, kappa = 1, beta fitted
    a = wa.ability(pop, covid, 1.0, stim)
    m = desire_m(pop, T, p, rho, beta, need)
    P, Tr = wa.receipt_and_transfers(pop, wa.StateTables(pop, a, m).pairs(1.0), p['b'])
    ex_r, ex_t = P.sum(1), Tr.sum(1)
    sr, st = [], []
    for s in range(seeds):
        r, t = simulate(pop, covid, p, 1.0, beta, need, 1.0, stim, 13000 + s, rho)
        sr.append(r.sum(1)), st.append(t.sum(1))
    sr, st = np.array(sr), np.array(st)
    zr = (sr.mean(0) - ex_r) / (sr.std(0, ddof=1) / np.sqrt(seeds))
    zt = (st.mean(0) - ex_t) / (st.std(0, ddof=1) / np.sqrt(seeds))
    res['exact_vs_sim_receipt_max_abs_z'] = float(np.abs(zr).max())
    res['exact_vs_sim_transfers_max_abs_z'] = float(np.abs(zt).max())
    res['exact_vs_sim_pass_4se'] = bool(np.abs(zr).max() < 4 and np.abs(zt).max() < 4)
    res['seeds'] = seeds
    print(json.dumps(res, indent=2))
    (OUT / 'checks.json').write_text(json.dumps(res, indent=2))
    pd.DataFrame({'date': hm.DATES, 'sim_mean_core_desire': mean, 'm_mean_core': m_b, 'se': se, 'z': zb}).to_csv(
        OUT / 'check_markov_marginal.csv', index=False)
    return res


# %% Step 2: calibrate rho to ENIGH 2018
def dist_row(months, mask=None):
    return rh.distribution(months, mask)


def run_calibrate(pop, cal, seeds_grid=100, seeds_final=200):
    p = wa.params_for(cal, KAPPA_CAL)
    groups = e1d.household_groups(pop)
    rows = []
    for rho in RHO_GRID:
        m18 = persistence_2018(pop, p, KAPPA_CAL, rho, seeds_grid)
        nt = np.tile(groups['network_type'].to_numpy(), seeds_grid)
        r = dict(rho=rho, seeds=seeds_grid, **dist_row(m18))
        r['share_receivers_kin_only'] = float(np.mean([(nt[m18[:, j] > 0] == 'Kin only').mean() for j in range(4)]))
        r['kin_only_share_of_one_month'] = float(np.mean([(nt[m18[:, j] == 1] == 'Kin only').mean()
                                                          for j in range(4)]))
        rows.append(r)
        print(f"rho {rho}: all six {r['share_all_six']:.3f}, mean {r['mean']:.2f}, one {r['share_1']:.3f}", flush=True)
    grid = pd.DataFrame(rows)
    # refine: bracket on the grid, then brentq with common random numbers and seeds_final
    f = lambda rho: dist_row(persistence_2018(pop, p, KAPPA_CAL, rho, seeds_final))['share_all_six'] - ENIGH18['share_all_six']
    vals = grid.share_all_six.to_numpy() - ENIGH18['share_all_six']
    k = np.flatnonzero((vals[:-1] < 0) & (vals[1:] >= 0))
    if len(k):
        rho_hat = brentq(f, RHO_GRID[k[0]], RHO_GRID[k[0] + 1], xtol=2e-3)
    else:
        rho_hat = RHO_GRID[int(np.argmin(np.abs(vals)))]
        print('no bracket on the grid; closest grid value used', flush=True)
    m18 = persistence_2018(pop, p, KAPPA_CAL, rho_hat, seeds_final)
    final = dict(rho=rho_hat, seeds=seeds_final, **dist_row(m18))
    nt = np.tile(groups['network_type'].to_numpy(), seeds_final)
    final['share_receivers_kin_only'] = float(np.mean([(nt[m18[:, j] > 0] == 'Kin only').mean() for j in range(4)]))
    final['kin_only_share_of_one_month'] = float(np.mean([(nt[m18[:, j] == 1] == 'Kin only').mean() for j in range(4)]))
    by = []
    for lvl in ['Kin only', 'Core only', 'Core + kin']:
        for rho in (0.0, rho_hat):
            mm = m18 if rho == rho_hat else persistence_2018(pop, p, KAPPA_CAL, 0.0, seeds_final)
            by.append(dict(rho=rho, network_type=lvl, **dist_row(mm, nt == lvl)))
    # excluding kin-only households
    final_nokin = dict(rho=rho_hat, seeds=seeds_final, note='excluding kin-only households', **dist_row(m18, nt != 'Kin only'))
    out = pd.concat([grid.assign(stage='grid'), pd.DataFrame([final]).assign(stage='calibrated'),
                     pd.DataFrame([final_nokin]).assign(stage='calibrated, no kin-only')], ignore_index=True)
    out = out.assign(enigh_mean=ENIGH18['mean'], enigh_share_all_six=ENIGH18['share_all_six'],
                     enigh_share_1=ENIGH18['share_1'])
    out.to_csv(OUT / 'calibration_rho.csv', index=False)
    pd.DataFrame(by).to_csv(OUT / 'persistence_2018_by_network.csv', index=False)
    (OUT / 'rho.json').write_text(json.dumps(dict(rho=rho_hat, kappa_used=KAPPA_CAL, target='ENIGH 2018 share of '
                                                  'receiving households receiving in all six months, 0.4955'), indent=2))
    print(out.round(3).to_string())
    return rho_hat


def run_calibrate_diagnostics(pop, cal, seeds=100):
    """Why A cannot reach 49.6%: the rho -> 1 limit (core only), and a variant with kin persistence too
    (rho_kin = rho; diagnostic, not the agreed specification)."""
    p = wa.params_for(cal, KAPPA_CAL)
    groups = e1d.household_groups(pop)
    nt = np.tile(groups['network_type'].to_numpy(), seeds)
    rows = []
    for variant, rk in (('core only (spec A)', False), ('core and kin (diagnostic)', True)):
        for rho in [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.995, 0.999]:
            m18 = persistence_2018(pop, p, KAPPA_CAL, rho, seeds, rho_kin=rho if rk else 0.0)
            rows.append(dict(variant=variant, rho=rho, seeds=seeds, **dist_row(m18),
                             all_six_kin_only=dist_row(m18, nt == 'Kin only')['share_all_six'],
                             one_month_kin_only=dist_row(m18, nt == 'Kin only')['share_1'],
                             all_six_with_core=dist_row(m18, nt != 'Kin only')['share_all_six'],
                             share_receivers_kin_only=float(np.mean([(nt[m18[:, j] > 0] == 'Kin only').mean()
                                                                     for j in range(4)]))))
            print(variant, rho, round(rows[-1]['share_all_six'], 3), round(rows[-1]['mean'], 2),
                  round(rows[-1]['share_1'], 3), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'calibration_rho_diagnostics.csv', index=False)
    return out


def load_rho():
    return json.loads((OUT / 'rho.json').read_text())['rho']


# %% Step 3: mig174 test
def run_mig174(pop, cal, rho_cal, reps=200, lengths=(3, 6, 12, 24, 60)):
    """Core migrants, 2019 employment (calendar months cycled), own desire d (beta = 0): probability of wanting
    and being able in at least one month over L consecutive months; also of wanting in at least one month."""
    p = wa.params_for(cal, 0.0)
    core = np.flatnonzero(pop['core'])
    d = wa.desire(pop, 1, p['a_c'], p['a_k'], p['eta'], g=p['g'])[0, core]            # (C,)
    d19 = pd.date_range('2019-01-01', '2019-12-01', freq='MS')
    rates = wa.abm.make_inputs(d19, pop['rate_columns']).rates[:, pop['job_index'][core]]   # (12, C)
    phi = wa.abm.employment_phi()
    Lmax = max(lengths)
    rows = []
    for rho in sorted({0.0, rho_cal, 0.5, 0.8, 0.9, 0.927, 0.95}):
        rng = np.random.default_rng(174)
        C = len(core)
        n = C * reps
        dd = np.tile(d, reps)
        rr = np.tile(rates, (1, reps))
        start = rng.integers(0, 12, n)                                       # random calendar start month
        z = rng.standard_normal(n)
        wants = rng.random(n) < dd
        any_send, any_want = np.zeros(n, bool), np.zeros(n, bool)
        res = {}
        for t in range(Lmax):
            if t > 0:
                z = phi * z + np.sqrt(1 - phi ** 2) * rng.standard_normal(n)
                pw = np.where(wants, dd + rho * (1 - dd), dd * (1 - rho))
                wants = rng.random(n) < pw
            able = norm.cdf(z) < rr[(start + t) % 12, np.arange(n)]
            any_send |= wants & able
            any_want |= wants
            if t + 1 in lengths:
                res[t + 1] = (any_send.mean(), any_want.mean())
        for L in lengths:
            rows.append(dict(rho=rho, months=L, p_any_send=res[L][0], p_any_want=res[L][1], migrants=C, reps=reps))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'mig174_test.csv', index=False)
    print(out.pivot(index='months', columns='rho', values='p_any_send').round(3).to_string())
    return out


# %% Step 4: shock years, quick map
def run_shocks_a(pop, cal, rho, kappas=QUICK_KAPPAS, betas=QUICK_BETAS, tag='', rho_kin=0.0):
    out = OUT / ('shocks' + tag)
    out.mkdir(parents=True, exist_ok=True)
    rounds = ['2021-01', '2021-03']
    sh = ShocksA(pop, cal, rounds, 1.0, rho, rho_kin=rho_kin)
    rows, paths, monthly, fits = [], [], [], []
    for kappa in kappas:
        for beta in betas:
            r, pth, m = sh.summary(kappa, float(beta))
            rows.append(r), paths.append(pth)
            if beta == 0:
                monthly.append(m)
        res = minimize_scalar(lambda b: sh.summary(kappa, b)[0]['rmse_fit'], bounds=(0.0, rs.BETA_MAX),
                              method='bounded', options=dict(xatol=1e-3))
        r, pth, m = sh.summary(kappa, float(res.x))
        p = wa.params_for(cal, kappa)
        need = sh.need
        only_core = lambda b: marginal(wa.expit(p['a_c'] + b * need)[:, None], np.array([True]), rho)[:, 0]
        mc = only_core(float(res.x))
        fits.append(dict(r, rho=rho, rho_kin=rho_kin, **sh.level_gap(kappa), **sh.shortage(kappa, float(res.x)),
                         desire_core_2019=float(wa.expit(p['a_c'])),
                         d_core_2021_12=float(wa.expit(p['a_c'] + res.x)),
                         m_core_2021_12=float(mc[hm.DATES.get_loc(pd.Timestamp('2021-12-01'))]),
                         m_core_2022_12=float(mc[-1])))
        paths.append(pth.assign(fitted=True)), monthly.append(m.assign(fitted=True))
        print(f'rho {rho}/{rho_kin} kappa {kappa}: beta {res.x:.3f}, rmse {res.fun:.4f}', flush=True)
    grid = pd.DataFrame(rows).assign(rho=rho, rho_kin=rho_kin)
    grid.to_csv(out / 'map.csv', index=False)
    pd.DataFrame(fits).to_csv(out / 'beta_fit.csv', index=False)
    pd.concat(paths, ignore_index=True).assign(rho=rho).to_csv(out / 'quarterly_paths.csv', index=False)
    pd.concat(monthly, ignore_index=True).assign(rho=rho).to_csv(out / 'monthly_paths.csv', index=False)
    (out / 'run_settings.json').write_text(json.dumps(dict(
        option='A: habit persistence in core desire (Markov marginal m_t)', rho=rho, rho_kin=rho_kin, stimulus_rounds=rounds, pi=1.0,
        kappas=kappas, betas=list(map(float, betas)), need='E:1 cumulative excess deaths', calibration='calibration.csv'),
        indent=2))
    print(grid.pivot(index='beta', columns='kappa', values='rmse_fit').round(3).to_string())
    return pd.DataFrame(fits)


# %% Step 5: household results
def run_households_a(pop, cal, rho, kappas, seeds=200, shocks_dir=None, rho_kin=0.0, tag=''):
    shocks_dir = shocks_dir or OUT / 'shocks'
    hout = OUT / ('households' + tag)
    hout.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(shocks_dir / 'beta_fit.csv')
    rounds, pi = ['2021-01', '2021-03'], 1.0
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
    need, stim, zero = hm.need_stock(hm.DATES), wa.stimulus_indicator(hm.DATES, rounds), np.zeros(len(hm.DATES))
    d18, in18 = inputs_2018(pop)
    window = (hm.DATES >= rh.WINDOW[0]) & (hm.DATES <= rh.WINDOW[1])
    enigh_al = pd.read_csv(rh.wa.NCM / 'identification/enigh/outputs/pre_excess_over_trend.csv').query(
        "from_year == 2018 and to_year in (2020, 2022) and measure in ('receipt', 'tpr_counts')").set_index(
        ['measure', 'to_year'])
    al_rows, dist_rows, gl_rows = [], [], []
    import enigh_aligned as ea
    for rho_run in sorted({0.0, rho}):
        for kappa in kappas:
            p = wa.params_for(cal, kappa)
            if rho_run == rho:
                beta = float(fits.loc[np.isclose(fits.kappa, kappa), 'beta'].iloc[0])
            else:
                f0 = pd.read_csv(wa.OUT / 'shocks' / 'beta_fit.csv')
                beta = float(f0.loc[np.isclose(f0.kappa, kappa), 'beta'].iloc[0])
            acc = {y: [] for y in (2020, 2022)}
            months = {k: [] for k in ('2018', '2020', '2022', 'cf2020', 'cf2022')}
            gl = {}
            nt = groups['network_type'].to_numpy()
            for seed in range(seeds):
                rk = rho_kin if rho_run == rho else 0.0
                rc, tc = simulate(pop, covid, p, kappa, beta, need, pi, stim, 9000 + seed, rho_run, rho_kin=rk)
                rf, tf = simulate(pop, cf, p, kappa, 0.0, None, 0.0, zero, 9000 + seed, rho_run, rho_kin=rk)
                r18, _ = simulate(pop, in18, p, kappa, 0.0, None, 0.0, np.zeros(12), 5000 + seed, rho_run,
                                  rho_kin=rk)
                for y in acc:
                    acc[y].append((ea.window_measures(rc, tc, hm.DATES, y), ea.window_measures(rf, tf, hm.DATES, y)))
                months['2018'].append(rh.months_before_interview(r18, d18))
                for y in (2020, 2022):
                    months[str(y)].append(months_window(rc, hm.DATES, y))
                    months[f'cf{y}'].append(months_window(rf, hm.DATES, y))
                dm = rc[window].sum(0).astype(int) - rf[window].sum(0).astype(int)
                for lvl in ['all', 'Kin only', 'Core only', 'Core + kin']:
                    msk = np.ones(pop['H'], bool) if lvl == 'all' else nt == lvl
                    x = dm[msk]
                    neg = -x[x < 0].sum()
                    v = np.array([(x < 0).mean(), (x <= -3).mean(), (x > 0).mean(), (x >= 3).mean(), (x == 0).mean(),
                                  x.mean(), (-x[x <= -3].sum()) / neg if neg else np.nan,
                                  rf[window][:, msk].any(0).mean()])
                    gl.setdefault(lvl, []).append(v)
            for y, v in acc.items():
                c = np.array([a for a, _ in v]).mean(0)
                f = np.array([b for _, b in v]).mean(0)
                lr = np.log(c / f)
                al_rows.append(dict(rho=rho_run, kappa=kappa, beta=beta, year=y, seeds=seeds,
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
                    dist_rows.append(dict(rho=rho_run, kappa=kappa, group=gname, run=k, **rh.distribution(arr, msk)))
            names = ['share_losing', 'share_losing_3plus', 'share_gaining', 'share_gaining_3plus', 'share_no_change',
                     'mean_change_months', 'share_of_lost_months_in_3plus_losers', 'share_receiving_cf_window']
            for lvl, v in gl.items():
                gl_rows.append(dict(rho=rho_run, kappa=kappa, beta=beta, network_type=lvl, seeds=seeds,
                                    **dict(zip(names, np.nanmean(np.array(v), 0)))))
            print(f'households rho {rho_run} kappa {kappa}: done', flush=True)
    pd.DataFrame(al_rows).to_csv(hout / 'enigh_aligned.csv', index=False)
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(hout / 'months_distribution.csv', index=False)
    pd.DataFrame(gl_rows).to_csv(hout / 'who_gains_loses_realised.csv', index=False)
    compare_months_enigh(dist).to_csv(hout / 'months_change_vs_enigh.csv', index=False)
    (hout / 'households_settings.json').write_text(json.dumps(dict(
        rho=rho, rho_kin=rho_kin, kappas=kappas, seeds=seeds, window=list(rh.WINDOW), interview_shares=rh.INTERVIEW,
        note=('who_gains_loses_realised: per seed and household, months with receipt Apr 2020-Dec 2021 in the COVID '
              'run minus its no-COVID twin (common random numbers); shares averaged over seeds. months_distribution: '
              '2018 from a 2018 run, 2020/2022 from the COVID run (Jan 2019 start), cf = no-COVID twin.')), indent=2))
    print(pd.DataFrame(al_rows).round(3).to_string())
    print(pd.DataFrame(gl_rows).round(3).to_string())


def compare_months_enigh(dist):
    """Changes in the months-with-receipt distribution (receiving households): model 2020 and 2022 (COVID run) minus
    2018 and minus the no-COVID twin, against ENIGH 2020-2018 and 2022-2018 (outputs/enigh_months_received_2022.csv,
    else enigh_months_received.csv for 2020 only)."""
    f22 = wa.OUT / 'enigh_months_received_2022.csv'
    en = pd.read_csv(f22 if f22.exists() else wa.OUT / 'enigh_months_received.csv', dtype={'year': str})
    rows = []
    cols = ['mean', 'share_1', 'share_all_six']
    for (rho, kappa, group), d in dist.groupby(['rho', 'kappa', 'group']):
        d = d.set_index('run')
        for y in ('2020', '2022'):
            row = dict(rho=rho, kappa=kappa, group=group, year=y)
            for c in cols:
                row[f'model_{c}_2018'] = d.loc['2018', c]
                row[f'model_{c}_{y}'] = d.loc[y, c]
                row[f'model_change_{c}_vs_2018'] = d.loc[y, c] - d.loc['2018', c]
                row[f'model_change_{c}_vs_nocovid'] = d.loc[y, c] - d.loc[f'cf{y}', c]
                e = en[(en.year == f'{y}-2018') & (en.group == group)]
                row[f'enigh_change_{c}'] = float(e[c].iloc[0]) if len(e) else np.nan
                row[f'enigh_change_{c}_se'] = float(e[f'{c}_se'].iloc[0]) if len(e) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['checks', 'calibrate', 'diag', 'mig174', 'shocks', 'households', 'all'])
    ap.add_argument('--seeds', type=int, default=200)
    ap.add_argument('--rho', type=float, default=None, help='override the calibrated rho')
    ap.add_argument('--kappas', type=float, nargs='*', default=None, help='households: kappas (default best + 0.8)')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pop, cal = setup()
    t0 = time.time()
    if args.step in ('checks', 'all'):
        run_checks(pop, cal, args.seeds)
    if args.step in ('calibrate', 'all'):
        run_calibrate(pop, cal, 100, args.seeds)
    if args.step == 'diag':
        run_calibrate_diagnostics(pop, cal)
    rho = args.rho if args.rho is not None else (load_rho() if (OUT / 'rho.json').exists() else None)
    if args.step in ('mig174', 'all'):
        run_mig174(pop, cal, rho)
    if args.step in ('shocks', 'all'):
        run_shocks_a(pop, cal, rho)
    if args.step in ('households', 'all'):
        fits = pd.read_csv(OUT / 'shocks' / 'beta_fit.csv')
        best = float(fits.loc[fits.rmse_fit.idxmin(), 'kappa'])
        kappas = args.kappas or sorted({best, 0.8}, reverse=True)
        run_households_a(pop, cal, rho, kappas, args.seeds)
    print(f'{args.step}: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
