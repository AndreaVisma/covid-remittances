"""Robustness: is pooling the 2010-19 MMP interviews justified? Period indicator P_h = [surveyyr >= 2015] (10 Nov 2026).

Run: python 2026-10-30/robustness/calibrate_period.py         (a few minutes)

The base calibration (calibrate_mmp.py, outputs/calibration.csv) pools all 3,252 households interviewed 2010-19 and
evaluates them under 2019 employment. Receipt rose from 20.2% (2010-14) to 26.1% (2015-19), together with the share of
households with a core migrant. This script asks whether a period difference remains once the networks are accounted
for, adding P_h in two places (one at a time):
    base      outputs/calibration.csv, evaluated again here (loglik check, fit by period).
    R         receipt: s_h = expit(b0 + b_p P_h + b_any [E^c>0] + b_c log(1+E^c) + b_k0 [N^c=0] log(1+E^k));
              Stage 1 fixed at base (a_c, a_k, g); Stage 2 refit (eta_w, eta_o, b0, b_p, b_any, b_c >= 0, b_k0).
    D         desire: theta_j += d_p P_h for every migrant (core and kin). Stage 1 refit with d_p
              (a_c, a_k, g, d_p), then Stage 2 (eta, b) with d_p kept in desire. Not nested in base at Stage 2
              (a_c, a_k, g move too), so the LR test is the Stage 1 one; Stage 2 logliks reported for reference.
    D2        desire, Stage 2 level: a_c, a_k, g fixed at base, d_p joins eta in the outer loop (nested in base).
Standard errors: numerical Hessian (central differences) of the full Stage 2 log-likelihood over the free parameters
(eta, b; b_c dropped when at its bound 0) at the estimate; for Stage 1 of D the same over (a_c, a_k, g, d_p).
Also reported: |estimate| / sqrt(LR), the s.e. implied by the likelihood-ratio statistic (profile-based).
Outputs (aggregates only) in 2026-10-30/outputs/robustness-period/: estimates.csv (one row per variant: parameters,
s.e., loglik, LR test), fit_by_period.csv (observed vs fitted receipt by period, and by period x any core migrant),
implied.csv (receipt with one able sender / no sender and desire, per period), calibration_late_*.csv (2015-19 values
for all households, kappa = 0, in the calibration.csv format, for run_shocks.py --calibration).
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import willing_able as wa                   # noqa: E402
import calibrate_mmp as cm                  # noqa: E402

OUT = wa.OUT / 'robustness-period'
SPLIT = 2015


# %% Model pieces with the period indicator
def desire_p(pop, T, a_c, a_k, eta, g, d_p, P):
    """wa.desire plus d_p P_h on every migrant's desire index."""
    need = pop['need_x'] @ np.asarray(eta, float)
    core_size = np.log(np.maximum(pop['n_core'][pop['hid']], 1))
    base = np.where(pop['core'], a_c + g * core_size, a_k) + need[pop['hid']] + d_p * P[pop['hid']]
    return np.repeat(expit(base)[None, :], T, axis=0)


class Model:
    """Caches the flattened (E^c, E^k) pair weights per desire setting (kappa = 0, mean over 2019 months)."""

    def __init__(self, pop, a, ok, y, P):
        self.pop, self.a, self.ok, self.y, self.P = pop, a, ok, y, P.astype(float)
        self._cache = {}

    def flat(self, a_c, a_k, g, eta, d_p):
        key = tuple(np.round(np.r_[a_c, a_k, g, eta, d_p], 12))
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            d = desire_p(self.pop, self.a.shape[0], a_c, a_k, eta, g, d_p, self.P)
            pairs = wa.StateTables(self.pop, self.a, d).pairs(0.0)
            hh, ec, ek, w = wa.flatten(self.pop, pairs, months='mean')
            self._cache[key] = (hh, ec, ek, w[0])
        return self._cache[key]

    def X(self, hh, ec, ek, with_bp):
        X = wa.design(self.pop, hh, ec, ek)
        return np.column_stack([X[:, :1], self.P[hh], X[:, 1:]]) if with_bp else X

    def receipt(self, hh, w, X, b):
        return np.clip(np.bincount(hh, weights=w * expit(X @ b), minlength=self.pop['H']), 1e-12, 1 - 1e-12)

    def loglik(self, Ph):
        y, ok = self.y, self.ok
        return np.sum(np.where(ok, y * np.log(Ph) + (1 - y) * np.log(1 - Ph), 0.0))


def b_names(with_bp):
    return ['b0', 'b_p', 'b_any', 'b_c', 'b_k0'] if with_bp else list(wa.B_NAMES)


def fit_b(m, hh, w, X, start, with_bp):
    """Inner loop as calibrate_mmp.fit_b (analytic gradient, L-BFGS-B, b_c >= 0), with an optional b_p column."""
    bounds = list(cm.B_BOUNDS)
    starts = [np.asarray(s, float) for s in cm.B_STARTS]
    if with_bp:
        bounds.insert(1, (None, None))
        starts = [np.insert(s, 1, bp) for s in starts for bp in (0.0,)]
    ok, y, H = m.ok, m.y, m.pop['H']

    def nll(b):
        s = expit(X @ b)
        Ph = np.clip(np.bincount(hh, weights=w * s, minlength=H), 1e-12, 1 - 1e-12)
        r = np.where(ok, y / Ph - (1 - y) / (1 - Ph), 0.0)
        ll = np.sum(np.where(ok, y * np.log(Ph) + (1 - y) * np.log(1 - Ph), 0.0))
        return -ll, -(X.T @ (r[hh] * w * s * (1 - s)))

    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    best = None
    for s0 in [start] + starts:
        res = minimize(nll, np.clip(s0, lo, np.inf), jac=True, method='L-BFGS-B', bounds=bounds,
                       options=dict(gtol=1e-8, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


def stage2(m, a_c, a_k, g, eta0, b0, with_bp=False, d_p=0.0, free_dp=False):
    """Outer Nelder-Mead over eta (and d_p if free_dp), inner fit_b. Returns dict(eta, d_p, b, loglik)."""
    state = {'b': np.asarray(b0, float)}

    def profile(x):
        eta, dp = (x[:2], x[2]) if free_dp else (x, d_p)
        hh, ec, ek, w = m.flat(a_c, a_k, g, eta, dp)
        b, ll = fit_b(m, hh, w, m.X(hh, ec, ek, with_bp), state['b'], with_bp)
        state['b'] = b
        state['n'] = state.get('n', 0) + 1
        if state['n'] % 10 == 0:
            print(f"  stage2 eval {state['n']}: x {np.round(x, 4)} loglik {ll:.3f}", flush=True)
        return -ll, b

    x0 = np.r_[eta0, d_p] if free_dp else np.asarray(eta0, float)
    res = minimize(lambda x: profile(x)[0], x0, method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-5, maxiter=400))
    ll, b = profile(res.x)
    eta, dp = (res.x[:2], res.x[2]) if free_dp else (res.x, d_p)
    return dict(eta=np.asarray(eta), d_p=float(dp), b=b, loglik=-ll)


def full_loglik(m, a_c, a_k, g, eta, d_p, b, with_bp):
    hh, ec, ek, w = m.flat(a_c, a_k, g, eta, d_p)
    return m.loglik(m.receipt(hh, w, m.X(hh, ec, ek, with_bp), b))


def hessian_se(f, x, h=1e-3):
    """s.e. from the numerical Hessian of the log-likelihood f at x (central differences)."""
    x = np.asarray(x, float)
    k = len(x)
    hs = h * np.maximum(1.0, np.abs(x))
    f0 = f(x)
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.eye(k)[i] * hs[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / hs[i] ** 2
        for j in range(i):
            ej = np.eye(k)[j] * hs[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * hs[i] * hs[j])
    cov = np.linalg.inv(-H)
    return np.sqrt(np.diag(cov))


def stage2_se(m, a_c, a_k, g, r, with_bp, free_dp=False):
    """s.e. of (eta, [d_p], b) at the Stage 2 estimate r; b_c fixed when at its bound."""
    names = list(wa.ETA_NAMES) + (['d_p'] if free_dp else []) + b_names(with_bp)
    x = np.r_[r['eta'], [r['d_p']] if free_dp else [], r['b']]
    ic = names.index('b_c')
    free = [i for i in range(len(x)) if not (i == ic and x[ic] <= 1e-8)]

    def f(z):
        full = x.copy()
        full[free] = z
        eta = full[:2]
        dp = full[2] if free_dp else r['d_p']
        b = full[(3 if free_dp else 2):]
        return full_loglik(m, a_c, a_k, g, eta, dp, b, with_bp)

    se = np.full(len(x), np.nan)
    se[free] = hessian_se(f, x[free])
    return dict(zip(['se_' + n for n in names], se))


# %% Stage 1 with d_p
def stage1_receipt_p(pop, a, x, P):
    q = a * desire_p(pop, a.shape[0], x[0], x[1], (0.0, 0.0), x[2], x[3], P)
    log_none = np.stack([np.bincount(pop['hid'], weights=np.log1p(-qt), minlength=pop['H']) for qt in q])
    return np.clip((-np.expm1(log_none)).mean(0), 1e-12, 1 - 1e-12)


def stage1_p(m, x0):
    def ll(x):
        return m.loglik(stage1_receipt_p(m.pop, m.a, x, m.P))
    res = minimize(lambda x: -ll(x), x0, method='Nelder-Mead',
                   options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000))
    return res.x, -res.fun, ll


# %% Reporting
def by_period(m, Ph, label):
    pop = m.pop
    late = m.P.astype(bool)
    anycore = pop['n_core'] > 0
    rows = []
    for name, sel in [('all', np.ones(pop['H'], bool))] + [(f'core={c}', anycore == c) for c in (True, False)]:
        for per, ps in (('2010-14', ~late), ('2015-19', late)):
            s = m.ok & sel & ps
            rows.append(dict(model=label, group=name, period=per, households=int(s.sum()),
                             observed=m.y[s].mean(), fitted=Ph[s].mean()))
    return rows


def lr(ll1, ll0, df=1):
    stat = 2 * (ll1 - ll0)
    return dict(lr_stat=stat, lr_p=chi2.sf(max(stat, 0.0), df))


def cal_row(a_c, a_k, g, eta, b, ll, k):
    return dict(kappa=0.0, a_c=a_c, a_k=a_k, g=g, **dict(zip(wa.ETA_NAMES, eta)), **dict(zip(wa.B_NAMES, b)),
                loglik=ll, aic=2 * k - 2 * ll)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='*', default=['R', 'D', 'D2'])
    ap.add_argument('--suffix', default='', help='suffix of the output files (parallel runs)')
    ap.add_argument('--combine', nargs='*', default=None, help='suffixes to merge into the unsuffixed files')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.combine is not None:
        for name in ('estimates', 'fit_by_period', 'implied'):
            df = pd.concat([pd.read_csv(OUT / f'{name}{s}.csv') for s in args.combine]).drop_duplicates()
            key = 'variant' if name == 'estimates' else 'model'
            df = df.drop_duplicates(subset=[c for c in df.columns if c in (key, 'group', 'period')])
            df.to_csv(OUT / f'{name}.csv', index=False)
            with pd.option_context('display.width', 250, 'display.max_columns', 80):
                print(df.round(4).to_string(index=False))
        return
    pop = wa.load_population()
    ok, y = wa.target(pop)
    P = pop['h'].surveyyr.to_numpy() >= SPLIT
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    a = wa.ability(pop, inputs)
    m = Model(pop, a, ok, y, P)

    base = wa.load_calibration().query('kappa == 0').iloc[0]
    bp = wa.params_for(wa.load_calibration(), 0.0)
    a_c, a_k, g, eta_b, b_b = bp['a_c'], bp['a_k'], bp['g'], bp['eta'], bp['b']
    s1_base = pd.read_csv(wa.OUT / 'stage1.csv').iloc[0]

    rows, fit_rows, implied = [], [], []

    # --- base, evaluated again
    hh, ec, ek, w = m.flat(a_c, a_k, g, eta_b, 0.0)
    Ph_base = m.receipt(hh, w, m.X(hh, ec, ek, False), b_b)
    ll_base = m.loglik(Ph_base)
    print(f'base loglik {ll_base:.3f} (calibration.csv {base.loglik:.3f})', flush=True)
    fit_rows += by_period(m, Ph_base, 'base')
    rb = dict(eta=eta_b, d_p=0.0, b=b_b, loglik=ll_base)
    rows.append(dict(variant='base', a_c=a_c, a_k=a_k, g=g, d_p=0.0, **dict(zip(wa.ETA_NAMES, eta_b)),
                     **dict(zip(wa.B_NAMES, b_b)), b_p=0.0, loglik=ll_base, stage1_loglik=s1_base.loglik,
                     **stage2_se(m, a_c, a_k, g, rb, False)))

    if 'R' in args.variants:
        # --- R: indicator in the receipt curve
        rR = stage2(m, a_c, a_k, g, eta_b, np.insert(b_b, 1, 0.0), with_bp=True)
        bR = dict(zip(b_names(True), rR['b']))
        hh, ec, ek, w = m.flat(a_c, a_k, g, rR['eta'], 0.0)
        Ph_R = m.receipt(hh, w, m.X(hh, ec, ek, True), rR['b'])
        seR = stage2_se(m, a_c, a_k, g, rR, True)
        testR = lr(rR['loglik'], ll_base)
        rows.append(dict(variant='R', a_c=a_c, a_k=a_k, g=g, d_p=0.0, **dict(zip(wa.ETA_NAMES, rR['eta'])), **bR,
                         loglik=rR['loglik'], stage1_loglik=s1_base.loglik, **seR, **testR,
                         se_lr_implied=abs(bR['b_p']) / np.sqrt(max(testR['lr_stat'], 1e-12))))
        print(pd.Series(rows[-1]).drop("variant").astype(float).round(4).to_string(), flush=True)
        fit_rows += by_period(m, Ph_R, 'R')
        for per, p in (('2010-14', 0.0), ('2015-19', 1.0)):
            z0 = bR['b0'] + bR['b_p'] * p
            for lab, zz in [('base', None), ('R', z0)]:
                bb = dict(zip(wa.B_NAMES, b_b)) if lab == 'base' else bR
                z = bb['b0'] if lab == 'base' else zz
                implied.append(dict(model=lab, period=per, receipt_no_sender=expit(z),
                                    receipt_one_core_sender=expit(z + bb['b_any'] + bb['b_c'] * np.log(2)),
                                    receipt_one_kin_sender_no_core=expit(z + bb['b_k0'] * np.log(2)),
                                    desire_only_core=expit(a_c), desire_kin=expit(a_k)))
        late_R = cal_row(a_c, a_k, g, rR['eta'], [bR['b0'] + bR['b_p'], bR['b_any'], bR['b_c'], bR['b_k0']],
                         rR['loglik'], 7)
        pd.DataFrame([late_R]).to_csv(OUT / 'calibration_late_R.csv', index=False)

    if 'D' in args.variants:
        # --- D: indicator in desire, Stage 1 refit
        x1, ll1, f1 = stage1_p(m, np.r_[s1_base.a_c, s1_base.a_k, s1_base.g, 0.0])
        se1 = hessian_se(f1, x1, h=1e-4)
        test1 = lr(ll1, s1_base.loglik)
        print(f'Stage 1 with d_p: {np.round(x1, 4)} se {np.round(se1, 4)} loglik {ll1:.3f} vs {s1_base.loglik:.3f}',
              flush=True)
        rD = stage2(m, x1[0], x1[1], x1[2], eta_b, b_b, with_bp=False, d_p=x1[3])
        seD = stage2_se(m, x1[0], x1[1], x1[2], rD, False)
        hh, ec, ek, w = m.flat(x1[0], x1[1], x1[2], rD['eta'], x1[3])
        Ph_D = m.receipt(hh, w, m.X(hh, ec, ek, False), rD['b'])
        rows.append(dict(variant='D', a_c=x1[0], a_k=x1[1], g=x1[2], d_p=x1[3], **dict(zip(wa.ETA_NAMES, rD['eta'])),
                         **dict(zip(wa.B_NAMES, rD['b'])), b_p=0.0, loglik=rD['loglik'], stage1_loglik=ll1,
                         **seD, **dict(zip(['se1_a_c', 'se1_a_k', 'se1_g', 'se1_d_p'], se1)),
                         **{k + '_stage1': v for k, v in test1.items()}, **lr(rD['loglik'], ll_base),
                         se_lr_implied=abs(x1[3]) / np.sqrt(max(test1['lr_stat'], 1e-12))))
        print(pd.Series(rows[-1]).drop("variant").astype(float).round(4).to_string(), flush=True)
        fit_rows += by_period(m, Ph_D, 'D')
        bD = dict(zip(wa.B_NAMES, rD['b']))
        for per, p in (('2010-14', 0.0), ('2015-19', 1.0)):
            implied.append(dict(model='D', period=per, receipt_no_sender=expit(bD['b0']),
                                receipt_one_core_sender=expit(bD['b0'] + bD['b_any'] + bD['b_c'] * np.log(2)),
                                receipt_one_kin_sender_no_core=expit(bD['b0'] + bD['b_k0'] * np.log(2)),
                                desire_only_core=expit(x1[0] + x1[3] * p), desire_kin=expit(x1[1] + x1[3] * p)))
        pd.DataFrame([cal_row(x1[0] + x1[3], x1[1] + x1[3], x1[2], rD['eta'], rD['b'], rD['loglik'], 7)]).to_csv(
            OUT / 'calibration_late_D.csv', index=False)

    if 'D2' in args.variants:
        # --- D2: d_p in the Stage 2 outer loop, a_c, a_k, g at base (nested in base)
        rD2 = stage2(m, a_c, a_k, g, eta_b, b_b, with_bp=False, d_p=0.0, free_dp=True)
        seD2 = stage2_se(m, a_c, a_k, g, rD2, False, free_dp=True)
        testD2 = lr(rD2['loglik'], ll_base)
        hh, ec, ek, w = m.flat(a_c, a_k, g, rD2['eta'], rD2['d_p'])
        Ph_D2 = m.receipt(hh, w, m.X(hh, ec, ek, False), rD2['b'])
        rows.append(dict(variant='D2', a_c=a_c, a_k=a_k, g=g, d_p=rD2['d_p'], **dict(zip(wa.ETA_NAMES, rD2['eta'])),
                         **dict(zip(wa.B_NAMES, rD2['b'])), b_p=0.0, loglik=rD2['loglik'], stage1_loglik=s1_base.loglik,
                         **seD2, **testD2,
                         se_lr_implied=abs(rD2['d_p']) / np.sqrt(max(testD2['lr_stat'], 1e-12))))
        print(pd.Series(rows[-1]).drop("variant").astype(float).round(4).to_string(), flush=True)
        fit_rows += by_period(m, Ph_D2, 'D2')
        bD2 = dict(zip(wa.B_NAMES, rD2['b']))
        for per, p in (('2010-14', 0.0), ('2015-19', 1.0)):
            implied.append(dict(model='D2', period=per, receipt_no_sender=expit(bD2['b0']),
                                receipt_one_core_sender=expit(bD2['b0'] + bD2['b_any'] + bD2['b_c'] * np.log(2)),
                                receipt_one_kin_sender_no_core=expit(bD2['b0'] + bD2['b_k0'] * np.log(2)),
                                desire_only_core=expit(a_c + rD2['d_p'] * p), desire_kin=expit(a_k + rD2['d_p'] * p)))
        pd.DataFrame([cal_row(a_c + rD2['d_p'], a_k + rD2['d_p'], g, rD2['eta'], rD2['b'], rD2['loglik'], 7)]).to_csv(
            OUT / 'calibration_late_D2.csv', index=False)

    est = pd.DataFrame(rows)
    est.to_csv(OUT / f'estimates{args.suffix}.csv', index=False)
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(OUT / f'fit_by_period{args.suffix}.csv', index=False)
    imp = pd.DataFrame(implied)
    imp.to_csv(OUT / f'implied{args.suffix}.csv', index=False)
    with pd.option_context('display.width', 200):
        print(fit.round(4).to_string(index=False))
        print(imp.round(4).to_string(index=False))
    (OUT / f'settings{args.suffix}.json').write_text(json.dumps(dict(
        script='2026-10-30/robustness/calibrate_period.py', split=f'P_h = [surveyyr >= {SPLIT}]',
        households=int(ok.sum()), households_late=int((ok & P).sum()),
        employment='2019 CPS rates for all households (as base), kappa = 0, rho = 0',
        variants=dict(R='b_p P_h in the receipt index; Stage 1 at base; Stage 2 refit',
                      D='d_p P_h in every migrant desire; Stage 1 refit with d_p, then Stage 2',
                      D2='d_p P_h in desire; a_c, a_k, g at base; d_p in the Stage 2 outer loop'),
        se='numerical Hessian of the log-likelihood (central differences); b_c fixed when at bound 0'), indent=2))


if __name__ == '__main__':
    main()
