"""Calibration of the willing-and-able model to the 2019 MMP data (plan Step 8; 30 Oct 2026; simplified
functional forms 13 Nov 2026, see willing_able.py and handoff §0).

Run: python 2026-10-30/calibrate_mmp.py            (about 20 minutes: Stage 2 is fitted twice, for the b test)
One b for every sender since 13 Nov 2026 (user); the separate-b fit is kept only as the test.
     --kappas 0 0.5 1   calibrate at these kappas instead (default: kappa = 0 only; stepping-in is a response to the
                        shock, so normal times are calibrated without it and kappa enters only the shock runs; user,
                        30 Oct)
     --shift 0.5        also refit Stage 2 with a_c, a_k shifted by +-0.5 (the sensitivity of Step 8)
     --tag _k0          suffix for the output files (parallel runs, one per kappa)

Stage 1 (a_c, a_k, delta): migrant-level model with independent senders, household receives if at least one
    linked migrant sends: P_h = mean_t [1 - prod_j (1 - a_j,t d_j)], d_j = expit(a_role + delta M_j), M_j = core
    migrants of the household other than j; 2019 employment, maximum likelihood on reported receipt (remus) of the
    e0.target_mask households. Checks: delta = 0 reproduces the migrant-level M1 fit in
    recipient-level/outputs/fits.csv; test delta_c = delta_k (separate coordination coefficients, LR test).
Stage 2 (b0, b, eta_w, eta_o), at kappa = 0: a_c, a_k, delta fixed from Stage 1; receipt
    s = expit(b0 + b (E^c + E^k)) by maximum likelihood (analytic gradient, L-BFGS-B with b >= 0,
    receipt grows with senders; inner loop, best of the warm start and B_STARTS, because the likelihood has
    several local optima) given the desire parameters eta_w, eta_o
    (outer loop, Nelder-Mead: eta changes desire and so the state distribution). P_h = mean over 2019 months of
    E[s_h(E^c, E^k)]. No stimulus and no crisis in 2019 (pi S_t = 0, N* = 0). Also fitted with separate b_c, b_k
    (columns sep_b_*): LR test of b_c = b_k in calibration_simple.csv (little power: see handoff §0a).
Outputs (aggregates only) in 2026-10-30/outputs/, all with the suffix _simple (the files without it hold the
    earlier forms): stage1_simple.csv (incl. the delta test), stage1_by_core_simple.csv (observed vs fitted receipt
    by number of core migrants), calibration_simple.csv (one row per kappa: a_c, a_k, delta, eta, b, loglik, AIC;
    columns sep_* and lr_b / p_b_equal for the b_c = b_k test), calibration_shift_*_simple.csv (with
    --shift), calibration_settings_simple.json.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

import willing_able as wa

KAPPAS = [0.0]
SPEC = '_simple'                                   # suffix of every output file (simplified forms, 13 Nov 2026)
M1_FITS = wa.NCM / 'recipient-level/outputs/fits.csv'


# %% Stage 1
def stage1_receipt(pop, a, x):
    """(H,) P_h with independent senders for x = (a_c, a_k, delta) or (a_c, a_k, delta_c, delta_k)."""
    q = a * wa.desire(pop, 1, x[0], x[1], delta=x[2], delta_k=x[3] if len(x) > 3 else None)   # (T, M)
    log_none = np.stack([np.bincount(pop['hid'], weights=np.log1p(-qt), minlength=pop['H']) for qt in q])
    return np.clip((-np.expm1(log_none)).mean(0), 1e-12, 1 - 1e-12)


def stage1(pop, a, ok, y, fix_delta=None, separate=False):
    """ML with independent senders. Default: (a_c, a_k, delta). fix_delta: delta fixed (check against M1).
    separate: (a_c, a_k, delta_c, delta_k), for the test delta_c = delta_k. Returns parameters, loglik."""
    def nll(x):
        P = stage1_receipt(pop, a, x if fix_delta is None else np.r_[x, fix_delta])
        return -np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))

    x0 = np.array([-0.2, -3.3] + ([] if fix_delta is not None else [0.0, 0.0] if separate else [0.0]))
    res = minimize(nll, x0, method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-8, maxiter=8000))
    return (res.x if fix_delta is None else np.r_[res.x, fix_delta]), -res.fun


# %% Stage 2
B_BOUNDS = [(None, None), (0.0, None)]                             # b0, b >= 0
B_STARTS = [np.r_[-5.0, 8.2], np.r_[-3.9, 4.0], np.r_[-4.6, 12.0], np.r_[-4.0, 6.0]]


def fit_b(X, hh, w, ok, y, H, start, by_role=False):
    """Inner loop: ML for the receipt coefficients given the pair weights w (n,) and design X (n, 2), or X (n, 3)
    with by_role (separate b_c, b_k; test only); the best of the warm start and B_STARTS."""
    if by_role:
        bounds = B_BOUNDS + B_BOUNDS[1:]
        starts = [s if len(s) == 3 else np.r_[s, s[1]] for s in [start] + B_STARTS]
    else:
        bounds, starts = B_BOUNDS, [s[:2] for s in [start] + B_STARTS]

    def nll(b):
        s = expit(X @ b)
        P = np.clip(np.bincount(hh, weights=w * s, minlength=H), 1e-12, 1 - 1e-12)
        r = np.where(ok, y / P - (1 - y) / (1 - P), 0.0)
        ll = np.sum(np.where(ok, y * np.log(P) + (1 - y) * np.log(1 - P), 0.0))
        grad = X.T @ (r[hh] * w * s * (1 - s))
        return -ll, -grad

    best = None
    for s0 in starts:
        s0 = np.clip(np.asarray(s0, float), [lo if lo is not None else -np.inf for lo, _ in bounds], np.inf)
        res = minimize(nll, s0, jac=True, method='L-BFGS-B', bounds=bounds, options=dict(gtol=1e-8, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res
    return best.x, -best.fun


def stage2(pop, a, ok, y, a_c, a_k, delta, kappa, start=None, by_role=False):
    """Outer loop over eta (desire), inner over b. Returns dict with eta, b (B_NAMES order), loglik."""
    T = a.shape[0]
    best = {'b': B_STARTS[0] if start is None else start['b']}

    def profile(eta):
        d = wa.desire(pop, T, a_c, a_k, eta, delta=delta)
        pairs = wa.StateTables(pop, a, d).pairs(kappa)
        hh, ec, ek, w = wa.flatten(pop, pairs, months='mean')
        X = (wa.design_by_role if by_role else wa.design)(pop, hh, ec, ek)
        b, ll = fit_b(X, hh, w[0], ok, y, pop['H'], best['b'], by_role)
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
    tag = SPEC + args.tag

    pop = wa.load_population()
    ok, y = wa.target(pop)
    inputs = wa.abm.make_inputs(wa.abm.baseline_2019_dates(), pop['rate_columns'])
    a = wa.ability(pop, inputs)                                              # 2019: no stimulus

    (a_c, a_k, delta), ll1 = stage1(pop, a, ok, y)
    x0, ll0 = stage1(pop, a, ok, y, fix_delta=0.0)                          # check: reproduces M1
    xs, lls = stage1(pop, a, ok, y, separate=True)                          # test: delta_c = delta_k
    m1 = pd.read_csv(M1_FITS).query("model == 'M1'").set_index('parameter').estimate
    lr_d = 2 * (lls - ll1)
    s1 = pd.DataFrame([dict(a_c=a_c, a_k=a_k, delta=delta, loglik=ll1, d_core_single=expit(a_c),
                            d_kin_no_core=expit(a_k), d_kin_one_core=expit(a_k + delta),
                            delta0_a_c=x0[0], delta0_a_k=x0[1], delta0_loglik=ll0,
                            sep_a_c=xs[0], sep_a_k=xs[1], sep_delta_c=xs[2], sep_delta_k=xs[3], sep_loglik=lls,
                            lr_delta=lr_d, p_delta_equal=chi2.sf(max(lr_d, 0.0), 1),
                            m1_alpha_core=m1['alpha_core'], m1_alpha_kin=m1['alpha_kin'])])
    s1.to_csv(wa.OUT / f'stage1{tag}.csv', index=False)
    print(s1.round(4).T.to_string(), flush=True)
    fit = pd.DataFrame(dict(n_core=np.minimum(pop['n_core'], 4), observed=y,
                            fitted=stage1_receipt(pop, a, (a_c, a_k, delta)),
                            fitted_delta0=stage1_receipt(pop, a, x0)))[ok]
    by = fit.groupby('n_core').agg(households=('observed', 'size'), observed=('observed', 'mean'),
                                   fitted=('fitted', 'mean'), fitted_delta0=('fitted_delta0', 'mean')).reset_index()
    by.to_csv(wa.OUT / f'stage1_by_core{tag}.csv', index=False)
    print(by.round(3).to_string(index=False), flush=True)

    def run(ac, ak, path, test=True):
        rows, start = [], None
        for kappa in sorted(args.kappas):
            r = stage2(pop, a, ok, y, ac, ak, delta, kappa, start)
            start = r
            k = len(wa.B_NAMES) + len(wa.ETA_NAMES)
            b = r['b']
            row = dict(kappa=kappa, a_c=ac, a_k=ak, delta=delta, **dict(zip(wa.ETA_NAMES, r['eta'])),
                       **dict(zip(wa.B_NAMES, b)), loglik=r['loglik'], aic=2 * k - 2 * r['loglik'],
                       receipt_none=expit(b[0]), receipt_one_sender=expit(b[0] + b[1]),
                       receipt_two_senders=expit(b[0] + 2 * b[1]))
            if test:                                                     # separate b_c, b_k: test of b_c = b_k
                rs = stage2(pop, a, ok, y, ac, ak, delta, kappa, r, by_role=True)
                lr_b = 2 * (rs['loglik'] - r['loglik'])
                row.update(**{f'sep_{n}': v for n, v in zip(wa.ETA_NAMES, rs['eta'])},
                           **{f'sep_{n}': v for n, v in zip(wa.B_ROLE_NAMES, rs['b'])}, sep_loglik=rs['loglik'],
                           lr_b=lr_b, p_b_equal=chi2.sf(max(lr_b, 0.0), 1))
            rows.append(row)
            print(f"kappa {kappa}: loglik {r['loglik']:.2f}", flush=True)
        out = pd.DataFrame(rows)
        out.to_csv(path, index=False)
        return out

    cal = run(a_c, a_k, wa.OUT / f'calibration{tag}.csv')
    print(cal.round(3).T.to_string())
    if args.shift:
        shifted = [run(a_c + s, a_k + s, wa.OUT / f'calibration_shift_{s:+g}{tag}.csv', test=False)
                   for s in (args.shift, -args.shift)]
    settings = dict(
        plan='2026-10-30/model_plan.tex, Step 8', households=int(ok.sum()), receiving=int(y[ok].sum()),
        forms='simplified 13 Nov 2026: desire expit(a_role + delta M_j + eta need), M_j = other core migrants of '
              'the household; receipt expit(b0 + b E), E = all effective senders (one b, 13 Nov 2026)',
        stage1='independent senders, P = mean_t[1 - prod(1 - a d)]; ML (Nelder-Mead); LR test delta_c = delta_k',
        stage2='a_c, a_k, delta fixed; eta by Nelder-Mead (outer), (b0, b) by L-BFGS-B with analytic gradient '
               '(inner), b >= 0, per kappa; separate b_c, b_k refitted for the LR test of b_c = b_k',
        kappas=sorted(args.kappas), shift=args.shift, employment='2019 CPS rates, rho = 0, no stimulus',
        receipt_target='MMP remus == 1, mean monthly receipt probability (mapping assumption)')
    (wa.OUT / f'calibration_settings{tag}.json').write_text(json.dumps(settings, indent=2))


if __name__ == '__main__':
    main()
