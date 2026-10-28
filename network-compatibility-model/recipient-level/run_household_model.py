"""Recipient-level (household) model: E0, E1, E2 map and household distribution (28 Oct 2026).

Run: python network-compatibility-model/recipient-level/run_household_model.py   (~5 min)
     --original-coding: 'Not working' migrants never able to send (abm coding); E0 and E1 only
     --kappa-logit: the first version (28 Oct) with kappa as an unbounded logit shift,
       score + kappa [U_c > 0] log(1 + A_k), U_c = eligible core migrants unable; kappa grid 0..4
     outputs in outputs/household-model[-kappa-logit][-original-coding]/

The household logit H3 of compare_household_logit.py, extended with the two map axes:
    s(a, b) = expit(b0 + b_any [a > 0] + b_c log(1 + a) + b_k log(1 + b) + eta' need_h + beta S_t)
    P(h receives in t) = E[ s(A_c, A_k) + kappa [A_c + A_k > 0] (s(N_c, N_k) - s(A_c, A_k)) ]
over the able counts (A_c, A_k); N_c, N_k = eligible core and kin migrants (nobody out of work).
kappa in [0, 1] is the share of the receipt probability lost to job loss that the household's able
migrants replace: the household analogue of the migrant model's coordination rule (base since
28 Oct 2026). 0 = no stepping-in; 1 = the household receives as if nobody had lost their job, as
long as one linked migrant can send. It cannot over-replace. S_t = reported COVID deaths in Mexico
/ 10,000 (abm.salience_series), 0 in 2019.
Base population: 'Not working' migrants as unknown occupation (all-occupation CPS rate), the best
fit in the comparison (recipient-level/README.md). rho = 0, no benefits.

E0: for each kappa on KAPPAS, the six H3 coefficients by maximum likelihood on 2019 MMP receipt
    (kappa enters 2019 through ordinary unemployment): the profile likelihood of kappa.
E1: beta = 0, exact receipt index (COVID / same calendar month of 2019, the no-COVID run) for
    2019-01..2022-12, quarterly against Banxico transactions relative to their 2017-19 trend
    (run_e1_job_loss.banxico). Mapping to counts: Banxico transfers move with the number of
    receiving households (transfers per receiving household constant). Log RMSE 2020Q2..2021Q4;
    2022 held out.
E2: beta x kappa map, same criterion, tau_C = 0.04; plus a Gaussian log-likelihood of the quarterly
    log residuals with the SD of the 2017-19 quarterly trend residuals (colleague feedback).
Distribution (who lost out), beta = 0, kappa in DIST_KAPPAS: exact expected change in months with
    receipt, Apr 2020 - Dec 2021, per household against its no-COVID twin; and realized paths:
    SEEDS paired runs with the abm employment process (latent Gaussian AR(1), phi from CPS,
    common random numbers for employment and the receipt draw). Outcome: months with receipt
    (the household model has no amounts; decision C pending).
Outputs (aggregates only) in outputs/household-model/.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1
import run_e1_distribution as e1d
import compare_household_logit as hl

TERMS = ['any_core', 'log_core', 'log_kin', 'no_worker', 'any_older']
KAPPAS = {'share': [round(k, 1) for k in np.arange(0, 1.01, 0.1)],
          'logit': [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]}
DIST_KAPPAS = {'share': [0.0, 0.5, 1.0], 'logit': [0.0, 1.0, 2.0]}
BETAS = np.round(np.arange(0.0, 0.401, 0.02), 2)
TAU_C = 0.04
DATES = pd.date_range('2019-01-01', '2022-12-01', freq='MS')
FIT_Q = [str(p) for p in pd.period_range('2020Q2', '2021Q4', freq='Q')]
HOLD_Q = [str(p) for p in pd.period_range('2022Q1', '2022Q4', freq='Q')]
WINDOW = ('2020-04-01', '2021-12-01')
SEEDS = range(2001, 2201)
PMF_TOL = 1e-12


def eligible_counts(pop):
    hsum = lambda w: np.bincount(pop['hid'], weights=w, minlength=pop['H'])
    return hsum(pop['core'] & pop['eligible']), hsum(~pop['core'] & pop['eligible'])


def household_receipt(x, kappa, mode, a, b, n_core, n_kin, need, shift=0.0):
    """Receipt probability for able counts (a, b); n_core, n_kin, need aligned with a."""
    if mode == 'logit':
        return expit(hl.design(a, b, need, TERMS) @ x + shift + kappa * ((n_core - a) > 0) * np.log1p(b))
    s = expit(hl.design(a, b, need, TERMS) @ x + shift)
    s_full = expit(hl.design(n_core, n_kin, need, TERMS) @ x + shift)
    return s + kappa * ((a + b) > 0) * (s_full - s)


# %% Cells: (household, able core a, able kin b) with monthly probabilities
class Cells:
    """All (h, a, b) cells with positive probability in some month, contiguous by household.
    W: (T, n_cells) joint pmf of the able counts per month (rho = 0, exact)."""

    def __init__(self, pop, inputs, mode='share'):
        self.mode = mode
        pmfs = hl.able_pmfs(pop, inputs)
        n_core, n_kin = eligible_counts(pop)
        hh, A, B, W = [], [], [], []
        for h, (pc, pk) in enumerate(pmfs):
            J = pc[:, :, None] * pk[:, None, :]                       # (T, a, b)
            a, b = np.nonzero(J.max(0) > PMF_TOL)
            hh.append(np.full(len(a), h))
            A.append(a)
            B.append(b)
            W.append(J[:, a, b])
        self.hh, a, b = np.concatenate(hh), np.concatenate(A).astype(float), np.concatenate(B).astype(float)
        self.W = np.concatenate(W, axis=1)
        self.H, self.T = pop['H'], len(inputs.dates)
        self.starts = np.searchsorted(self.hh, np.arange(self.H))
        self.X = hl.design(a, b, pop['need_x'][self.hh], TERMS)       # intercept + TERMS
        if mode == 'logit':
            self.trigger = ((n_core[self.hh] - a) > 0) * np.log1p(b)   # [U_c > 0] log(1 + A_k)
        else:
            self.trigger = ((a + b) > 0).astype(float)                 # someone can send
            self.X_full = hl.design(n_core, n_kin, pop['need_x'], TERMS)  # (H, k): nobody out of work
        self.salience = inputs.salience

    def household_sum(self, Z):
        return np.add.reduceat(Z, self.starts, axis=-1)

    def receipt(self, x, kappa=0.0, beta=0.0):
        """(T, H) receipt probabilities."""
        shift = beta * self.salience[:, None]
        if self.mode == 'logit':
            return self.household_sum(self.W * expit((self.X @ x + kappa * self.trigger)[None, :] + shift))
        s = expit((self.X @ x)[None, :] + shift)
        s_full = expit((self.X_full @ x)[None, :] + shift)
        return (self.household_sum(self.W * s * (1 - kappa * self.trigger))
                + kappa * s_full * self.household_sum(self.W * self.trigger))

    def annual(self, x, kappa):
        return self.receipt(x, kappa).mean(0)


def fit_2019(cells, y, mask, kappa=None, start=None):
    """ML for the six H3 coefficients with kappa fixed (kappa None: free, logit mode only).
    Returns (x, kappa, negloglik, se). 2019: S_t = 0, so the annual pmf average suffices."""
    Wbar = cells.W.mean(0)
    if cells.mode == 'share':
        # P_h = sum_c w_c (1 - kappa g_c) s_c + kappa G_h s_full,h,  G_h = sum_c w_c g_c
        G = cells.household_sum(Wbar * cells.trigger)
        keep = Wbar * (1 - kappa * cells.trigger)
        X, Xf = cells.X, cells.X_full

        def prob_grad(z):
            s, sf = expit(X @ z), expit(Xf @ z)
            P = cells.household_sum(keep * s) + kappa * G * sf
            return P, lambda r: X.T @ (r[cells.hh] * keep * s * (1 - s)) + Xf.T @ (r * kappa * G * sf * (1 - sf))
    else:
        X = cells.X if kappa is not None else np.column_stack([cells.X, cells.trigger])
        off = kappa * cells.trigger if kappa is not None else 0.0

        def prob_grad(z):
            s = expit(X @ z + off)
            return cells.household_sum(Wbar * s), lambda r: X.T @ (r[cells.hh] * Wbar * s * (1 - s))

    def nll(z):
        P, grad = prob_grad(z)
        P = np.clip(P, 1e-12, 1 - 1e-12)
        r = np.where(mask, y / P - (1 - y) / (1 - P), 0.0)
        ll = np.sum(np.where(mask, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))
        return -ll, -grad(r)

    z0 = np.r_[-1.0, np.zeros(X.shape[1] - 1)] if start is None else start
    res = minimize(nll, z0, jac=True, method='BFGS', options=dict(gtol=1e-7, maxiter=3000))
    se = hl.numerical_se(lambda z: nll(z)[0], res.x)
    if kappa is None:
        return res.x[:-1], res.x[-1], res.fun, se
    return res.x, kappa, res.fun, se


def quarterly(series):
    return series.groupby(series.index.to_period('Q')).sum().rename(index=str)


def log_rmse(model, data, quarters):
    return float(np.sqrt(np.mean(np.log(model[quarters] / data[quarters]) ** 2)))


def banxico_target():
    bx, trend, info = e1.banxico()
    bx, trend = bx.reindex(DATES), trend.reindex(DATES)
    # quarterly residual SD of the 2017-19 trend fit (pre-period quarters only)
    pre_bx, pre_trend, _ = banxico_pre()
    pre_q = np.log(quarterly(pre_bx) / quarterly(pre_trend))
    info['residual_sd_quarterly'] = float(pre_q.std(ddof=1))
    return quarterly(bx) / quarterly(trend), info


def banxico_pre():
    """Banxico and its trend over the 2017-19 fit window (same fit as run_e1_job_loss.banxico)."""
    saved = e1.DATES
    e1.DATES = pd.date_range(e1.TREND_WINDOW[0], e1.TREND_WINDOW[1], freq='MS')
    try:
        return e1.banxico()
    finally:
        e1.DATES = saved


# %% Distribution: realized paths
def simulate_receipt(pop, inputs, x, kappa, seed, mode='share'):
    """One paired path: (T, H) bool receipt. Employment as abm.simulate (rho = 0); the receipt draw
    u_h,t uses its own stream so the COVID and no-COVID twins share it."""
    T, M, H, hid, core = len(inputs.dates), pop['M'], pop['H'], pop['hid'], pop['core']
    phi = abm.employment_phi()
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    z = streams[0].standard_normal(M)
    n_core, n_kin = eligible_counts(pop)
    out = np.zeros((T, H), bool)
    for t in range(T):
        if t > 0:
            z = phi * z + np.sqrt(1 - phi ** 2) * streams[1].standard_normal(M)
        able = (norm.cdf(z) < inputs.rates[t, pop['job_index']]) & pop['eligible']
        ac = np.bincount(hid, weights=able & core, minlength=H)
        ak = np.bincount(hid, weights=able & ~core, minlength=H)
        p = household_receipt(x, kappa, mode, ac, ak, n_core, n_kin, pop['need_x'])
        out[t] = streams[2].random(H) < p
    return out


def distribution(pop, x_by_kappa, covid, cf, groups, out, mode='share'):
    window = (covid.dates >= WINDOW[0]) & (covid.dates <= WINDOW[1])
    cells_c, cells_f = Cells(pop, covid, mode), Cells(pop, cf, mode)
    bins, hists, rows, ineq = [], [], [], []
    for kappa in DIST_KAPPAS[mode]:
        x = x_by_kappa[kappa]
        exp_c = cells_c.receipt(x, kappa)[window].sum(0)
        exp_f = cells_f.receipt(x, kappa)[window].sum(0)
        lose25, lose_all, lose_half = np.zeros(pop['H']), np.zeros(pop['H']), np.zeros(pop['H'])
        received_cf = np.zeros(pop['H'])
        rcount = np.zeros(len(e1d.REAL_LABELS))
        rhist = np.zeros(len(e1d.REAL_HIST) - 1)
        per_seed = []
        for seed in SEEDS:
            a = simulate_receipt(pop, covid, x, kappa, seed, mode)[window].sum(0)
            b = simulate_receipt(pop, cf, x, kappa, seed, mode)[window].sum(0)
            rec = b > 0
            ch = a[rec] / b[rec] - 1
            received_cf += rec
            lose25 += rec & (a < 0.75 * b)
            lose_half += rec & (a < 0.5 * b)
            lose_all += rec & (a == 0)
            rcount += np.histogram(ch, e1d.REAL_EDGES)[0]
            rhist += np.histogram(ch.clip(e1d.REAL_HIST[0], e1d.REAL_HIST[-1] - 1e-9), e1d.REAL_HIST)[0]
            per_seed.append((float((ch < -0.25).mean()), float((ch <= -0.999).mean()), float(rec.sum()),
                             float(a.sum() / b.sum() - 1)))
        ps = np.array(per_seed)
        bins.append(pd.DataFrame({'kappa': kappa, 'bin': e1d.REAL_LABELS, 'share': rcount / rcount.sum()}))
        hists.append(pd.DataFrame({'kappa': kappa, 'bin_left': e1d.REAL_HIST[:-1], 'bin_right': e1d.REAL_HIST[1:],
                                   'share': rhist / rhist.sum()}))
        hh = groups.assign(exp_c=exp_c, exp_f=exp_f, n_rec=received_cf,
                           lose25=np.divide(lose25, received_cf, out=np.zeros(pop['H']), where=received_cf > 0),
                           lose_all=np.divide(lose_all, received_cf, out=np.zeros(pop['H']), where=received_cf > 0),
                           lose_half=np.divide(lose_half, received_cf, out=np.zeros(pop['H']), where=received_cf > 0))
        hh['change'] = hh.exp_c / hh.exp_f - 1
        ineq.append(dict(kappa=kappa, households=len(hh), aggregate_change_expected=hh.exp_c.sum() / hh.exp_f.sum() - 1,
                         realized_share_lose_25=ps[:, 0].mean(), realized_share_lose_all=ps[:, 1].mean(),
                         realized_receiving_households=ps[:, 2].mean(), realized_aggregate_change=ps[:, 3].mean(),
                         expected_change_p05=float(hh.change.quantile(0.05)),
                         expected_change_median=float(hh.change.median()),
                         expected_change_p95=float(hh.change.quantile(0.95))))
        # group rows: risks weighted by the number of seeds in which the household receives in the twin
        for g in ['network_type', 'size_bin', 'strongest_tie', 'occupation', 'links']:
            for level, d in hh.groupby(g, observed=True):
                w = d.n_rec
                rows.append(dict(kappa=kappa, group=g, level=str(level), households=len(d),
                                 change_expected=d.exp_c.sum() / d.exp_f.sum() - 1,
                                 median_change_expected=d.change.median(),
                                 loss_risk_25=float(np.average(d.lose25, weights=w)) if w.sum() else np.nan,
                                 loss_risk_50=float(np.average(d.lose_half, weights=w)) if w.sum() else np.nan,
                                 lose_all_risk=float(np.average(d.lose_all, weights=w)) if w.sum() else np.nan))
        print(f'distribution kappa {kappa}: done', flush=True)
    tables = dict(realized_bins=pd.concat(bins), realized_histogram=pd.concat(hists), groups=pd.DataFrame(rows),
                  inequality=pd.DataFrame(ineq))
    for k, v in tables.items():
        v.to_csv(out / f'distribution_{k}.csv', index=False)
    return tables


# %% Main
def main(original_coding=False, mode='share'):
    out = HERE / ('outputs/household-model' + ('-kappa-logit' if mode == 'logit' else '')
                  + ('-original-coding' if original_coding else ''))
    out.mkdir(parents=True, exist_ok=True)
    pop = hl.load_population(not_working_as_unknown=not original_coding)
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    kappas = KAPPAS[mode]

    # E0
    base = Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']), mode)
    x_by_kappa, prof, fits = {}, [], []
    start = None
    for kappa in kappas:
        x, _, nll, se = fit_2019(base, y, ok, kappa, start)
        start = x_by_kappa[kappa] = x
        prof.append(dict(kappa=kappa, loglik=-nll, **dict(zip(['intercept'] + TERMS, x)),
                         mean_receipt=base.annual(x, kappa)[ok].mean()))
        fits += [dict(kappa=kappa, parameter=n, estimate=b, se=s) for n, b, s in zip(['intercept'] + TERMS, x, se)]
    prof = pd.DataFrame(prof)
    if mode == 'logit':
        xf, kf, nllf, sef = fit_2019(base, y, ok, None, np.r_[x_by_kappa[0.0], 0.0])
        fits += [dict(kappa='free', parameter=n, estimate=b, se=s)
                 for n, b, s in zip(['intercept'] + TERMS + ['kappa'], np.r_[xf, kf], sef)]
        kappa_free = dict(estimate=float(kf), se=float(sef[-1]), loglik=float(-nllf))
    else:
        top = prof.loc[prof.loglik.idxmax()]
        kappa_free = dict(estimate=float(top.kappa), se=None, loglik=float(top.loglik),
                          note='kappa in [0, 1]: maximum of the profile on the grid')
    prof.to_csv(out / 'e0_kappa_profile.csv', index=False)
    pd.DataFrame(fits).to_csv(out / 'e0_fits.csv', index=False)
    print(prof.round(3).to_string(index=False))
    print('kappa free:', kappa_free, flush=True)

    # E1 and E2
    target, bx_info = banxico_target()
    sigma_q = bx_info['residual_sd_quarterly']
    covid = abm.make_inputs(DATES, pop['rate_columns'], salience=abm.salience_series(DATES))
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in DATES]), pop['rate_columns'])
    cc, cfc = Cells(pop, covid, mode), Cells(pop, cf, mode)
    rows, paths, monthly = [], [], []
    betas = [0.0] if original_coding else BETAS
    for kappa in kappas:
        x = x_by_kappa[kappa]
        rf = pd.Series(cfc.receipt(x, kappa).sum(1), index=DATES)
        for beta in betas:
            rc = pd.Series(cc.receipt(x, kappa, beta).sum(1), index=DATES)
            model = quarterly(rc) / quarterly(rf)
            resid = np.log(model[FIT_Q] / target[FIT_Q])
            rows.append(dict(kappa=kappa, beta=beta, rmse_log_quarterly=log_rmse(model, target, FIT_Q),
                             count_compatible=log_rmse(model, target, FIT_Q) <= TAU_C,
                             loglik_gaussian=float(np.sum(norm.logpdf(resid, scale=sigma_q))),
                             rmse_log_quarterly_holdout_2022=log_rmse(model, target, HOLD_Q),
                             index_2020q2=float(model['2020Q2']),
                             mean_index_2020=float(model[[q for q in FIT_Q if q.startswith('2020')]].mean()),
                             mean_index_2021=float(model[[q for q in FIT_Q if q.startswith('2021')]].mean())))
            paths.append(pd.DataFrame({'quarter': model.index, 'kappa': kappa, 'beta': beta,
                                       'index_model': model.values, 'index_banxico': target[model.index].values}))
            if beta == 0:
                monthly.append(pd.DataFrame({'date': DATES, 'kappa': kappa, 'receipt_covid': rc.values / pop['H'],
                                             'receipt_cf': rf.values / pop['H'], 'index_model': (rc / rf).values}))
    grid = pd.DataFrame(rows)
    grid['delta_loglik'] = grid.loglik_gaussian - grid.loglik_gaussian.max()
    grid.to_csv(out / 'map.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(out / 'quarterly_paths.csv', index=False)
    pd.concat(monthly, ignore_index=True).to_csv(out / 'e1_monthly.csv', index=False)
    print(grid[grid.beta == 0].round(3).to_string(index=False))
    best = grid.loc[grid.rmse_log_quarterly.idxmin()]
    if not original_coding:
        print(grid.pivot(index='beta', columns='kappa', values='rmse_log_quarterly').round(3).to_string())
        print('best', best.to_dict(), 'compatible cells', int(grid.count_compatible.sum()))

    if not original_coding:
        tables = distribution(pop, x_by_kappa, covid, cf, e1d.household_groups(pop), out, mode)
        pd.set_option('display.width', 220)
        print(tables['inequality'].round(3).to_string(index=False))
        print(tables['realized_bins'].round(3).to_string(index=False))

    settings = dict(
        experiment='Recipient-level household model: E0 kappa profile, E1 (beta = 0), E2 map, distribution',
        not_working_as_unknown=not original_coding, households=int(ok.sum()), receiving=int(y[ok].sum()),
        kappa_mode=mode,
        receipt=('expit(b0 + b_any [A_c>0] + b_c log(1+A_c) + b_k log(1+A_k) + eta need + beta S_t '
                 '+ kappa [U_c>0] log(1+A_k)), kappa = logit shift' if mode == 'logit' else
                 's(A_c, A_k) + kappa [A_c + A_k > 0] (s(N_c, N_k) - s(A_c, A_k)), s = expit(b0 + b_any [a>0] '
                 '+ b_c log(1+a) + b_k log(1+b) + eta need + beta S_t); kappa = replacement share in [0, 1]'),
        kappas=kappas, betas=list(map(float, betas)), tau_C=TAU_C, kappa_free=kappa_free,
        criterion='quarterly log RMSE 2020Q2..2021Q4, receipt index (COVID / no-COVID) vs Banxico / 2017-19 trend',
        count_mapping='transfers per receiving household constant (Banxico index = receiving-household index)',
        likelihood=f'Gaussian on quarterly log residuals 2020Q2..2021Q4, SD = 2017-19 quarterly trend residual SD '
                   f'({sigma_q:.4f}); 7 autocorrelated quarters, use ratios only',
        banxico_trend=bx_info, n_cells=len(grid), n_count_compatible=int(grid.count_compatible.sum()),
        best_cell=dict(kappa=float(best.kappa), beta=float(best.beta), rmse=float(best.rmse_log_quarterly)),
        distribution=None if original_coding else dict(
            kappas=DIST_KAPPAS[mode], beta=0.0, seeds=[SEEDS.start, SEEDS.stop - 1], window=list(WINDOW),
            outcome='months with receipt, household vs its no-COVID twin (common random numbers)',
            employment='latent Gaussian AR(1) as abm.simulate, rho = 0'))
    (out / 'run_settings.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main(original_coding='--original-coding' in sys.argv, mode='logit' if '--kappa-logit' in sys.argv else 'share')
