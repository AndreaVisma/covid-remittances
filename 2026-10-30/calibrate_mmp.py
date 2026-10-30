"""Calibration of the willing-and-able model to the 2019 MMP data (plan Step 8; 30 Oct 2026).

Run: python 2026-10-30/calibrate_mmp.py            (about 10 minutes)
     --kappas 0 0.5 1   calibrate at these kappas instead (default: kappa = 0 only; stepping-in is a response to the
                        shock, so normal times are calibrated without it and kappa enters only the shock runs; user,
                        30 Oct)
     --shift 0.5        also refit Stage 2 with a_c, a_k shifted by +-0.5 (the sensitivity of Step 8)
     --tag _k0          suffix for the output files (parallel runs, one per kappa)

Stage 1 (a_c, a_k, g): migrant-level model with independent senders, household receives if at least one
    linked migrant sends: P_h = mean_t [1 - prod_j (1 - a_j,t d_j)], d_j = expit(a_c + g log N^c_h) for core,
    expit(a_k) for kin; 2019 employment, maximum likelihood on reported receipt (remus) of the e0.target_mask
    households. The fit with g = 0 is checked against the migrant-level M1 fit in recipient-level/outputs/fits.csv.
Stage 2 (b0, b_any, b_c, b_k0, eta_w, eta_o), at kappa = 0: a_c, a_k, g fixed from Stage 1; the receipt
    coefficients by maximum likelihood (analytic gradient, L-BFGS-B with b_c >= 0, receipt grows with core
    senders; inner loop, best of the warm start and B_STARTS, because the likelihood has several local optima)
    given the desire parameters eta_w, eta_o
    (outer loop, Nelder-Mead: eta changes desire and so the state distribution). P_h = mean over 2019 months of
    E[s_h(E^c, E^k)]. No stimulus and no crisis in 2019 (pi S_t = 0, N* = 0).
Outputs (aggregates only) in 2026-10-30/outputs/: stage1.csv, stage1_by_core.csv (observed vs fitted receipt by
    number of core migrants), calibration.csv (one row per kappa: a_c, a_k, g, eta, b, loglik, AIC),
    calibration_shift.csv (with --shift), calibration_settings.json.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import willing_able as wa

KAPPAS = [0.0]
M1_FITS = wa.NCM / 'recipient-level/outputs/fits.csv'


# %% Stage 1
def stage1_receipt(pop, a, x):
    """(H,) P_h with independent senders for x = (a_c, a_k, g)."""
    q = a * wa.desire(pop, 1, x[0], x[1], g=x[2])                           # (T, M)
    log_none = np.stack([np.bincount(pop['hid'], weights=np.log1p(-qt), minlength=pop['H']) for qt in q])
    return np.clip((-np.expm1(log_none)).mean(0), 1e-12, 1 - 1e-12)


def stage1(pop, a, ok, y, fix_g=None):
    """ML for (a_c, a_k, g) with independent senders (g fixed if fix_g is given); returns (a_c, a_k, g), loglik."""
    def nll(x):
        P = stage1_receipt(pop, a, x if fix_g is None else np.r_[x, fix_g])
        return -np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))

    x0 = np.array([-0.2, -3.3] if fix_g is not None else [-0.2, -3.3, 0.0])
    res = minimize(nll, x0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
    return (res.x if fix_g is None else np.r_[res.x, fix_g]), -res.fun


# %% Stage 2
B_BOUNDS = [(None, None), (None, None), (0.0, None), (None, None)]   # b0, b_any, b_c >= 0, b_k0 (user, 30 Oct)
B_STARTS = [np.r_[-3.9, 4.0, 0.5, 1.3], np.r_[-4.5, 8.5, 0.0, 9.0], np.r_[-4.6, 14.0, 8.0, 9.5],
            np.r_[-4.0, 6.0, 1.0, 5.0], np.r_[-4.6, 11.0, 2.0, 10.0]]


def fit_b(X, hh, w, ok, y, H, start):
    """Inner loop: ML for the receipt coefficients given the pair weights w (n,) and design X (n, 4); the best of
    the warm start and B_STARTS."""
    def nll(b):
        s = expit(X @ b)
        P = np.clip(np.bincount(hh, weights=w * s, minlength=H), 1e-12, 1 - 1e-12)
        r = np.where(ok, y / P - (1 - y) / (1 - P), 0.0)
        ll = np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))
        grad = X.T @ (r[hh] * w * s * (1 - s))
        return -ll, -grad

    best = None
    for s0 in [start] + B_STARTS:
        s0 = np.clip(np.asarray(s0, float), [lo if lo is not None else -np.inf for lo, _ in B_BOUNDS], np.inf)
        res = minimize(nll, s0, jac=True, method='L-BFGS-B', bounds=B_BOUNDS, options=dict(gtol=1e-8, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


def stage2(pop, a, ok, y, a_c, a_k, g, kappa, start=None):
    """Outer loop over eta (desire), inner over b. Returns dict with eta, b, loglik."""
    T = a.shape[0]
    best = {'b': np.r_[-3.9, 4.0, 0.5, 1.3] if start is None else start['b']}

    def profile(eta):
        d = wa.desire(pop, T, a_c, a_k, eta, g=g)
        pairs = wa.StateTables(pop, a, d).pairs(kappa)
        hh, ec, ek, w = wa.flatten(pop, pairs, months='mean')
        X = wa.design(pop, hh, ec, ek)
        b, ll = fit_b(X, hh, w[0], ok, y, pop['H'], best['b'])
        best['b'] = b                                                        # warm start
        return -ll, b

    eta0 = np.zeros(2) if start is None else start['eta']
    res = minimize(lambda e: profile(e)[0], eta0, method='Nelder-Mead',
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=200))
    ll, b = profile(res.x)
    return dict(eta=res.x, b=b, loglik=-ll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappas', type=float, nargs='*', default=KAPPAS)
    ap.add_argument('--shift', type=float, default=None)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    wa.OUT.mkdir(parents=True, exist_ok=True)

    pop = wa.load_population()
    ok, y = wa.target(pop)
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    a = wa.ability(pop, inputs)                                              # 2019: no stimulus

    (a_c, a_k, g), ll1 = stage1(pop, a, ok, y)
    x0, ll0 = stage1(pop, a, ok, y, fix_g=0.0)                              # check: reproduces M1
    m1 = pd.read_csv(M1_FITS).query("model == 'M1'").set_index('parameter').estimate
    s1 = pd.DataFrame([dict(a_c=a_c, a_k=a_k, g=g, loglik=ll1, d_core_single=expit(a_c), d_kin=expit(a_k),
                            g0_a_c=x0[0], g0_a_k=x0[1], g0_loglik=ll0,
                            m1_alpha_core=m1['alpha_core'], m1_alpha_kin=m1['alpha_kin'])])
    s1.to_csv(wa.OUT / f'stage1{args.tag}.csv', index=False)
    print(s1.round(4).to_string(index=False), flush=True)
    fit = pd.DataFrame(dict(n_core=np.minimum(pop['n_core'], 4), observed=y,
                            fitted=stage1_receipt(pop, a, (a_c, a_k, g)), fitted_g0=stage1_receipt(pop, a, x0)))[ok]
    by = fit.groupby('n_core').agg(households=('observed', 'size'), observed=('observed', 'mean'),
                                   fitted=('fitted', 'mean'), fitted_g0=('fitted_g0', 'mean')).reset_index()
    by.to_csv(wa.OUT / f'stage1_by_core{args.tag}.csv', index=False)
    print(by.round(3).to_string(index=False), flush=True)

    def run(ac, ak, path):
        rows, start = [], None
        for kappa in sorted(args.kappas):
            r = stage2(pop, a, ok, y, ac, ak, g, kappa, start)
            start = r
            k = len(wa.B_NAMES) + len(wa.ETA_NAMES)
            rows.append(dict(kappa=kappa, a_c=ac, a_k=ak, g=g, **dict(zip(wa.ETA_NAMES, r['eta'])),
                             **dict(zip(wa.B_NAMES, r['b'])), loglik=r['loglik'], aic=2 * k - 2 * r['loglik']))
            print(f"kappa {kappa}: loglik {r['loglik']:.2f}", flush=True)
        out = pd.DataFrame(rows)
        out.to_csv(path, index=False)
        return out

    cal = run(a_c, a_k, wa.OUT / f'calibration{args.tag}.csv')
    print(cal.round(3).to_string(index=False))
    if args.shift:
        shifted = [run(a_c + s, a_k + s, wa.OUT / f'calibration_shift_{s:+g}{args.tag}.csv')
                   for s in (args.shift, -args.shift)]
    settings = dict(
        plan='2026-10-30/model_plan.tex, Step 8', households=int(ok.sum()), receiving=int(y[ok].sum()),
        stage1='independent senders, P = mean_t[1 - prod(1 - a d)], d = expit(a_c + g log N^c) core, '
               'expit(a_k) kin; ML (Nelder-Mead)',
        stage2='a_c, a_k, g fixed; eta by Nelder-Mead (outer), b = (b0, b_any, b_c, b_k0) by L-BFGS-B with '
               'analytic gradient (inner), per kappa; b_k removed (kin matter only without core)',
        kappas=sorted(args.kappas), shift=args.shift, employment='2019 CPS rates, rho = 0, no stimulus',
        receipt_target='MMP remus == 1, mean monthly receipt probability (mapping assumption)')
    (wa.OUT / f'calibration_settings{args.tag}.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
