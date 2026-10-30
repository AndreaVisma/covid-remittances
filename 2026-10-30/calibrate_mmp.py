"""Calibration of the willing-and-able model to the 2019 MMP data (plan Step 8; 30 Oct 2026).

Run: python 2026-10-30/calibrate_mmp.py            (not run yet; expected tens of minutes)
     --kappas 0 0.5 1   subset of the kappa grid (default 0, 0.1, ..., 1)
     --shift 0.5        also refit Stage 2 with a_c, a_k shifted by +-0.5 (the sensitivity of Step 8)

Stage 1 (a_c, a_k): migrant-level model with independent senders, household receives if at least one
    linked migrant sends: P_h = mean_t [1 - prod_j (1 - a_j,t expit(a_type(j)))], 2019 employment, maximum
    likelihood on reported receipt (remus) of the e0.target_mask households. Checked against the migrant-level
    M1 fit in recipient-level/outputs/fits.csv.
Stage 2 (b0, b_any, b_c, b_k, b_k0, eta_w, eta_o), for each kappa: a_c, a_k fixed from Stage 1; the receipt
    coefficients by maximum likelihood (analytic gradient, L-BFGS-B with b_k >= 0, inner loop) given the desire
    parameters eta_w, eta_o
    (outer loop, Nelder-Mead: eta changes desire and so the state distribution). P_h = mean over 2019 months of
    E[s_h(E^c, E^k)]. No stimulus and no crisis in 2019 (pi S_t = 0, N* = 0).
Outputs (aggregates only) in 2026-10-30/outputs/: stage1.csv, calibration.csv (one row per kappa: a_c, a_k,
    eta, b, loglik, AIC), calibration_shift.csv (with --shift), calibration_settings.json.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import willing_able as wa

KAPPAS = [round(k, 1) for k in np.arange(0, 1.01, 0.1)]
M1_FITS = wa.NCM / 'recipient-level/outputs/fits.csv'


# %% Stage 1
def stage1(pop, a, ok, y):
    """ML for (a_c, a_k) with independent senders; returns estimates, loglik."""
    core = pop['core']

    def nll(x):
        q = a * expit(np.where(core, x[0], x[1]))[None, :]                   # (T, M)
        log_none = np.stack([np.bincount(pop['hid'], weights=np.log1p(-qt), minlength=pop['H']) for qt in q])
        P = np.clip((-np.expm1(log_none)).mean(0), 1e-12, 1 - 1e-12)
        return -np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))

    res = minimize(nll, np.array([-0.2, -3.3]), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8))
    return res.x, -res.fun


# %% Stage 2
B_BOUNDS = [(None, None), (None, None), (None, None), (0.0, None), (None, None)]   # b_k >= 0 (user, 30 Oct)


def fit_b(X, hh, w, ok, y, H, start):
    """Inner loop: ML for the receipt coefficients given the pair weights w (n,) and design X (n, 5).
    b_k (kin senders in households with core migrants) is constrained to be >= 0, so that a kin migrant
    who steps in never lowers receipt (the unconstrained 2019 estimate was -0.16, not significant)."""
    def nll(b):
        s = expit(X @ b)
        P = np.clip(np.bincount(hh, weights=w * s, minlength=H), 1e-12, 1 - 1e-12)
        r = np.where(ok, y / P - (1 - y) / (1 - P), 0.0)
        ll = np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))
        grad = X.T @ (r[hh] * w * s * (1 - s))
        return -ll, -grad

    start = np.asarray(start, float).copy()
    start[3] = max(start[3], 0.0)
    res = minimize(nll, start, jac=True, method='L-BFGS-B', bounds=B_BOUNDS,
                   options=dict(gtol=1e-8, maxiter=5000))
    return res.x, -res.fun


def stage2(pop, a, ok, y, a_c, a_k, kappa, start=None):
    """Outer loop over eta (desire), inner over b. Returns dict with eta, b, loglik."""
    T = a.shape[0]
    best = {'b': np.r_[-3.9, 4.0, 0.5, 0.0, 1.3] if start is None else start['b']}

    def profile(eta):
        d = wa.desire(pop, T, a_c, a_k, eta)
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
    args = ap.parse_args()
    wa.OUT.mkdir(parents=True, exist_ok=True)

    pop = wa.load_population()
    ok, y = wa.target(pop)
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    a = wa.ability(pop, inputs)                                              # 2019: no stimulus

    (a_c, a_k), ll1 = stage1(pop, a, ok, y)
    m1 = pd.read_csv(M1_FITS).query("model == 'M1'").set_index('parameter').estimate
    s1 = pd.DataFrame([dict(a_c=a_c, a_k=a_k, loglik=ll1, d_core=expit(a_c), d_kin=expit(a_k),
                            m1_alpha_core=m1['alpha_core'], m1_alpha_kin=m1['alpha_kin'])])
    s1.to_csv(wa.OUT / 'stage1.csv', index=False)
    print(s1.round(4).to_string(index=False), flush=True)

    def run(ac, ak, path):
        rows, start = [], None
        for kappa in sorted(args.kappas):
            r = stage2(pop, a, ok, y, ac, ak, kappa, start)
            start = r
            k = len(wa.B_NAMES) + len(wa.ETA_NAMES)
            rows.append(dict(kappa=kappa, a_c=ac, a_k=ak, **dict(zip(wa.ETA_NAMES, r['eta'])),
                             **dict(zip(wa.B_NAMES, r['b'])), loglik=r['loglik'], aic=2 * k - 2 * r['loglik']))
            print(f"kappa {kappa}: loglik {r['loglik']:.2f}", flush=True)
        out = pd.DataFrame(rows)
        out.to_csv(path, index=False)
        return out

    cal = run(a_c, a_k, wa.OUT / 'calibration.csv')
    print(cal.round(3).to_string(index=False))
    if args.shift:
        shifted = [run(a_c + s, a_k + s, wa.OUT / f'calibration_shift_{s:+g}.csv')
                   for s in (args.shift, -args.shift)]
    settings = dict(
        plan='2026-10-30/model_plan.tex, Step 8', households=int(ok.sum()), receiving=int(y[ok].sum()),
        stage1='independent senders, P = mean_t[1 - prod(1 - a expit(a_type))], ML (Nelder-Mead)',
        stage2='a_c, a_k fixed; eta by Nelder-Mead (outer), b by L-BFGS-B with analytic gradient and b_k >= 0 '
               '(inner), per kappa',
        kappas=sorted(args.kappas), shift=args.shift, employment='2019 CPS rates, rho = 0, no stimulus',
        receipt_target='MMP remus == 1, mean monthly receipt probability (mapping assumption)')
    (wa.OUT / 'calibration_settings.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
