"""Comparison model: the colleague's original specification, no stepping-in (30 Oct 2026).

Run (repo root): python network-compatibility-model/recipient-level/robustness/colleague_spec.py   (~2-4 min)

Two levels, no stepping-in, no count mapping (the index is on expected receiving households, read as counts):
 (1) Migrant propensity. Each CORE migrant j is keen (sends) in month t with probability
        pi_send = expit(A + B employed_j,t + C Covid_t),   employed_j,t ~ Bernoulli(e_o(j),t)
     (CPS rate of the migrant's occupation, abm.able_probability with Params(0, 0), eligibility as in the
     base), so p_j,t = elig_j [e expit(A + B + C Covid_t) + (1 - e) expit(A + C Covid_t)].
     B = employment effect: B -> inf with A -> -inf is the pure employment gate; finite B lets out-of-work
     migrants still send (out-of-work / employed keen ratio expit(A) / expit(A + B) at Covid = 0).
 (2) Household receipt. P(h receives in t) = E[expit(alpha + beta_k Keen_h,t + delta FarFam_h [+ eta' need_h])],
     Keen = number of keen core migrants (Poisson-binomial over the household's core migrants, exact),
     FarFam = number of kin (non-core) links, a fixed count, not employment-gated.
 Covid_t: 0 in 2019 and in the no-COVID run; version E = hm.need_stock (cumulative excess deaths, 1 in Dec 2021),
 version S = abm.salience_series (monthly reported COVID deaths, the flow).
Variants (cumulative): (i) literal; (ii) + stimulus: in Jan and Mar 2021 e -> e + (1 - e) pi_t with
 pi_t = hm.stimulus_pi (COVID run only, as in the base); (iii) = (ii) + household need features (no_worker,
 any_older = pop['need_x'], as in hm.design_features) in the receipt logit.
Calibration: 2019 MMP receipt (remus == 1 on e0.target_mask), 2019 CPS employment, Covid = 0, maximum likelihood
 of alpha, beta_k, delta, A (+ eta) for each B on B_GRID; the profile log-likelihood over B is the analogue of the
 kappa profile. A free first (bounded to A_BOUNDS); A pinned (mean 2019 core keen probability = expit(M1 alpha_core))
 is also fitted and reported. (i) and (ii) share the 2019 fit (no stimulus in 2019).
Index: COVID run / no-COVID run (same calendar month of 2019 employment), quarterly, vs hm.banxico_target().
 C >= 0 fitted per B, variant, Covid version and A mode by bounded scalar minimisation of the quarterly log RMSE
 over hm.FIT_Q; 2022 held out; spikes via hm.spikes on the monthly index.
Outputs (aggregates only): outputs/robustness/colleague_spec/{map.csv (incl. the 2019 profile), summary.csv, quarterly_paths.csv,
 run_settings.json, colleague_spec.png}; README.md is written by hand from these.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RL = HERE.parent
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent / 'model'))
import run_household_model as hm
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1
import compare_household_logit as hl

TASK = 'colleague_spec'
OUT = RL / 'outputs/robustness' / TASK
B_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0]
A_BOUNDS = (-15.0, 15.0)
C_MAX = 8.0
VARIANTS = {'i': dict(stimulus=False, need=False, label='(i) literal'),
            'ii': dict(stimulus=True, need=False, label='(ii) + stimulus'),
            'iii': dict(stimulus=True, need=True, label='(iii) + stimulus + household need')}
COVID_VERSIONS = {'E': 'cumulative excess deaths (hm.need_stock, 1 in Dec 2021)',
                  'S': 'monthly reported COVID deaths (abm.salience_series, flow)'}
ENIGH = dict(receiving_households_vs_trend=-0.02, se=0.10, transfers_per_receiving_household=0.09)


# %% Keen counts
class Keen:
    """Core migrants padded per household: e (T, H, n) with mask; pmf of the keen count (T, H, n + 1)."""

    def __init__(self, pop, inputs, pi_t=None):
        e = abm.able_probability(pop, inputs, abm.Params(0.0, 0.0))
        if pi_t is not None:
            e = e + (1 - e) * np.asarray(pi_t, float)[:, None] * pop['eligible'][None, :]
        core = np.flatnonzero(pop['core'])
        hid = pop['hid'][core]
        order = np.argsort(hid, kind='stable')
        core, hid = core[order], hid[order]
        n = np.bincount(hid, minlength=pop['H'])
        slot = np.arange(len(core)) - np.searchsorted(hid, hid)
        self.n_max = int(n.max())
        self.T, self.H = e.shape[0], pop['H']
        # households grouped by their number of core migrants (speed: most have none or one)
        self.groups = []
        for k in range(1, self.n_max + 1):
            hh = np.flatnonzero(n == k)
            if not len(hh):
                continue
            E = np.zeros((self.T, len(hh), k))
            Mk = np.zeros((len(hh), k))
            pos = np.full(self.H, -1)
            pos[hh] = np.arange(len(hh))
            sel = n[hid] == k
            E[:, pos[hid[sel]], slot[sel]] = e[:, core[sel]]
            Mk[pos[hid[sel]], slot[sel]] = pop['eligible'][core[sel]]
            self.groups.append((hh, E, Mk))
        self.n_core = n

    def _p(self, E, Mk, A, B, C, covid):
        lin = A + (0.0 if covid is None else C * np.asarray(covid, float)[:, None, None])
        return Mk * (E * expit(lin + B) + (1 - E) * expit(lin))

    def pmf(self, A, B, C=0.0, covid=None):
        out = np.zeros((self.T, self.H, self.n_max + 1))
        out[..., 0] = 1.0
        for hh, E, Mk in self.groups:
            P = self._p(E, Mk, A, B, C, covid)
            g = np.zeros(P.shape[:2] + (P.shape[2] + 1,))
            g[..., 0] = 1.0
            for j in range(P.shape[2]):
                q = P[..., j:j + 1]
                nxt = g * (1 - q)
                nxt[..., 1:] += g[..., :-1] * q
                g = nxt
            out[:, hh, :g.shape[2]] = g
        return out

    def mean_core_p(self, A, B):
        """Mean keen probability over core migrants and months."""
        num = den = 0.0
        for hh, E, Mk in self.groups:
            num += self._p(E, Mk, A, B, 0.0, None).sum()
            den += Mk.sum() * self.T
        return float(num / den)


def receipt(pmf, theta, far, need_x):
    """(T, H) receipt probabilities; theta = [alpha, beta_k, delta, A, (eta_w, eta_o)]."""
    k = np.arange(pmf.shape[-1])
    lin = theta[0] + theta[2] * far
    if len(theta) > 4:
        lin = lin + need_x @ np.asarray(theta[4:6])
    s = expit(lin[:, None] + theta[1] * k[None, :])          # (H, n + 1)
    return np.einsum('thk,hk->th', pmf, s)


# %% 2019 fit
def fit_2019(keen, far, need_x, y, ok, B, need, A_fixed=None, start=None):
    """ML on 2019 receipt. A free (bounded) unless A_fixed. Returns (theta, loglik)."""
    k_need = 2 if need else 0

    def unpack(z):
        if A_fixed is None:
            return z
        return np.r_[z[:3], A_fixed, z[3:]]

    def nll(z):
        th = unpack(z)
        P = receipt(keen.pmf(th[3], B), th, far, need_x).mean(0)
        P = np.clip(P[ok], 1e-12, 1 - 1e-12)
        return -np.sum(y[ok] * np.log(P) + (1 - y[ok]) * np.log(1 - P))

    if start is None:
        start = np.r_[-2.0, 1.0, 0.0, 0.0, np.zeros(k_need)]
    z0 = start if A_fixed is None else np.r_[start[:3], start[4:]]
    bounds = [(-20, 20)] * 3 + ([A_BOUNDS] if A_fixed is None else []) + [(-10, 10)] * k_need
    best = None
    # extra starts for free A: employed keen probability near the M1 core value, and a lower one
    for z_init in [z0, np.r_[z0[:3], -0.23 - B, z0[4:]] if A_fixed is None else None,
                   np.r_[z0[:3], -2.0 - B, z0[4:]] if A_fixed is None else None]:
        if z_init is None:
            continue
        res = minimize(nll, z_init, method='L-BFGS-B', bounds=bounds,
                       options=dict(ftol=1e-13, gtol=1e-8, maxiter=2000))
        if best is None or res.fun < best.fun:
            best = res
    return unpack(best.x), -float(best.fun), nll


def a_curvature(nll_free, theta):
    """Profile curvature in A: SE of A from the numerical Hessian (inf if flat / not positive definite)."""
    se = hl.numerical_se(nll_free, np.asarray(theta, float), step=1e-3)
    return float(se[3]) if np.isfinite(se[3]) else np.inf


def pin_A(keen19, B, target):
    """A such that the mean 2019 core keen probability equals target."""
    f = lambda A: keen19.mean_core_p(A, B) - target
    return brentq(f, -40, 40)


# %% Main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hl.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    far = np.bincount(pop['hid'], weights=~pop['core'], minlength=pop['H']).astype(float)
    need_x = pop['need_x']
    target, _ = hm.banxico_target()
    bx_m, trend_m, _ = e1.banxico()
    spike_bx = hm.spikes((bx_m / trend_m).reindex(hm.DATES))
    a_core = hm.m1_alphas()[0]
    p_target = float(expit(a_core))
    base_prof = pd.read_csv(RL / 'outputs/household-model/e0_kappa_profile.csv')
    base_ll0 = float(base_prof.loc[np.isclose(base_prof.kappa, 0), 'loglik'].iloc[0])
    base_k = 7

    keen19 = Keen(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    covid_in = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf_in = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    keen_cf = Keen(pop, cf_in)
    keen_c = {False: Keen(pop, covid_in), True: Keen(pop, covid_in, hm.stimulus_pi(hm.DATES))}
    covid = {'E': hm.need_stock(hm.DATES), 'S': abm.salience_series(hm.DATES)}
    ser = lambda z: pd.Series(z.sum(1), index=hm.DATES)

    # 2019 fits: need features (False / True) x B x A mode
    fits, profile = {}, []
    for need in [False, True]:
        start, B_prev = None, None
        for B in B_GRID:
            if start is not None:   # warm start keeping A + B and an unsaturated beta_k
                start = start.copy()
                start[3] -= B - B_prev
                start[1] = min(start[1], 8.0)
            th, ll, nll = fit_2019(keen19, far, need_x, y, ok, B, need, start=start)
            start, B_prev = th, B
            free_nll = lambda z: nll(z)
            se_A = a_curvature(free_nll, th)
            A_pin = pin_A(keen19, B, p_target)
            th_p, ll_p, _ = fit_2019(keen19, far, need_x, y, ok, B, need, A_fixed=A_pin, start=th)
            k_free = 4 + 2 * need + 1          # + B (profiled)
            for mode, t, l, k in [('free', th, ll, k_free), ('pinned', th_p, ll_p, k_free - 1)]:
                fits[(need, B, mode)] = t
                profile.append(dict(need_features=need, B=B, A_mode=mode, loglik_2019=l, n_params=k,
                                    aic_2019=2 * k - 2 * l, alpha=t[0], beta_k=t[1], delta=t[2], A=t[3],
                                    eta_worker=t[4] if need else np.nan, eta_older=t[5] if need else np.nan,
                                    se_A=se_A if mode == 'free' else np.nan,
                                    A_at_bound=bool(mode == 'free' and min(abs(t[3] - b) for b in A_BOUNDS) < 1e-3),
                                    keen_employed_2019=float(expit(t[3] + B)), keen_out_of_work_2019=float(expit(t[3])),
                                    ratio_out_of_work_to_employed=float(expit(t[3]) / expit(t[3] + B)),
                                    mean_core_keen_2019=keen19.mean_core_p(t[3], B),
                                    mean_receipt_2019=float(receipt(keen19.pmf(t[3], B), t, far, need_x).mean(0)[ok].mean())))
            print(f'need {need} B {B}: free ll {ll:.2f} A {th[3]:.2f} (se {se_A:.2f}) beta_k {th[1]:.2f} | '
                  f'pinned ll {ll_p:.2f} A {A_pin:.2f}', flush=True)
    profile = pd.DataFrame(profile)
    profile['base_loglik_kappa0'] = base_ll0
    profile['base_aic_kappa0'] = 2 * base_k - 2 * base_ll0

    # Map: variant x Covid version x B x A mode, C fitted
    rows, paths = [], []
    for v, spec in VARIANTS.items():
        kc = keen_c[spec['stimulus']]
        for cv, cser in covid.items():
            for B in B_GRID:
                for mode in ['free', 'pinned']:
                    th = fits[(spec['need'], B, mode)]
                    A = th[3]
                    Pf = ser(receipt(keen_cf.pmf(A, B), th, far, need_x))
                    Pf_q = hm.quarterly(Pf)

                    def monthly(C):
                        return ser(receipt(kc.pmf(A, B, C, cser), th, far, need_x))

                    f = lambda C: hm.log_rmse(hm.quarterly(monthly(C)) / Pf_q, target, hm.FIT_Q)
                    res = minimize_scalar(f, bounds=(0.0, C_MAX), method='bounded', options=dict(xatol=1e-4))
                    C = float(res.x) if res.fun < f(0.0) else 0.0
                    Pc = monthly(C)
                    q = hm.quarterly(Pc) / Pf_q
                    q0 = hm.quarterly(monthly(0.0)) / Pf_q
                    sp = hm.spikes(Pc / Pf)
                    rec = profile[(profile.need_features == spec['need']) & (profile.B == B) &
                                  (profile.A_mode == mode)].iloc[0]
                    rows.append(dict(
                        variant=v, covid_version=cv, B=B, A_mode=mode, loglik_2019=rec.loglik_2019,
                        aic_2019=rec.aic_2019, n_params_2019=rec.n_params, se_A=rec.se_A, A_at_bound=rec.A_at_bound,
                        mean_core_keen_2019=rec.mean_core_keen_2019, mean_receipt_2019=rec.mean_receipt_2019,
                        alpha=th[0], beta_k=th[1], delta=th[2], A=A,
                        eta_worker=rec.eta_worker, eta_older=rec.eta_older,
                        ratio_out_of_work_to_employed=rec.ratio_out_of_work_to_employed,
                        keen_employed_2019=rec.keen_employed_2019, C=C, C_at_bound=C > C_MAX - 1e-3,
                        rmse_fit=hm.log_rmse(q, target, hm.FIT_Q),
                        rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q),
                        index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                        mean_index_2021=float(q[[x for x in hm.FIT_Q if x.startswith('2021')]].mean()),
                        mean_index_2022=float(q[hm.HOLD_Q].mean()), index_2022q3=float(q['2022Q3']),
                        index_2020q2_C0=float(q0['2020Q2']), rmse_fit_C0=hm.log_rmse(q0, target, hm.FIT_Q),
                        spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                        spike_rmse=float(np.sqrt(np.mean([(sp[m] - spike_bx[m]) ** 2 for m in hm.SPIKE_MONTHS]))),
                        keen_employed_dec2021=float(expit(A + B + C * cser[hm.DATES.get_loc('2021-12-01')])),
                        keen_out_of_work_dec2021=float(expit(A + C * cser[hm.DATES.get_loc('2021-12-01')]))))
                    paths.append(pd.DataFrame({'variant': v, 'covid_version': cv, 'B': B, 'A_mode': mode, 'C': C,
                                               'quarter': q.index, 'index_model': q.values,
                                               'index_model_C0': q0.values,
                                               'index_banxico': target[q.index].values}))
            print(f'variant {v} {cv}: done', flush=True)
    grid = pd.DataFrame(rows)
    grid['delta_loglik_2019'] = grid.loglik_2019 - grid.groupby(['variant', 'A_mode']).loglik_2019.transform('max')
    grid['is_best_B'] = grid.rmse_fit == grid.groupby(['variant', 'covid_version', 'A_mode']).rmse_fit.transform('min')
    grid['is_ml_B'] = grid.delta_loglik_2019 == 0
    grid.to_csv(OUT / 'map.csv', index=False)
    paths = pd.concat(paths, ignore_index=True)

    summary = grid[grid.is_best_B].copy()
    ml = grid[grid.is_ml_B][['variant', 'covid_version', 'A_mode', 'B', 'rmse_fit', 'index_2020q2']].rename(
        columns={'B': 'B_ml_2019', 'rmse_fit': 'rmse_fit_at_ml_B', 'index_2020q2': 'index_2020q2_at_ml_B'})
    summary = summary.merge(ml, on=['variant', 'covid_version', 'A_mode'])
    summary.to_csv(OUT / 'summary.csv', index=False)
    best_paths = paths.merge(summary[['variant', 'covid_version', 'A_mode', 'B']], on=['variant', 'covid_version',
                                                                                        'A_mode', 'B'])
    paths.to_csv(OUT / 'quarterly_paths.csv', index=False)

    base_paths = pd.read_csv(RL / 'outputs/household-model/quarterly_paths.csv')
    base_cell = base_paths[np.isclose(base_paths.kappa, 1.0) & np.isclose(base_paths.beta, 0.65)]
    base_noss = base_paths[np.isclose(base_paths.kappa, 0.0) & np.isclose(base_paths.beta, 0.0)]
    base_receipt_2022 = float(base_cell.set_index('quarter').loc[hm.HOLD_Q, 'index_receipt'].mean())
    settings = dict(
        task=TASK, B_grid=B_GRID, A_bounds=A_BOUNDS, C_max=C_MAX, variants=VARIANTS, covid_versions=COVID_VERSIONS,
        keen='p_j,t = elig_j [e expit(A + B + C Covid_t) + (1 - e) expit(A + C Covid_t)], core migrants only',
        receipt='E[expit(alpha + beta_k Keen + delta FarFam (+ eta need))], Keen Poisson-binomial (exact), '
                'FarFam = number of kin links (not gated)',
        count_mapping='none: index on expected receiving households, compared with Banxico counts directly',
        calibration='ML on 2019 MMP receipt (remus == 1, e0.target_mask), 2019 CPS employment, Covid = 0',
        A_pinned=f'mean 2019 core keen probability = expit(M1 alpha_core) = {p_target:.4f}',
        households=int(ok.sum()), receiving=int(y[ok].sum()),
        base_receipt_logit_kappa0=dict(loglik=base_ll0, n_params=base_k, aic=2 * base_k - 2 * base_ll0),
        base_reference=dict(cell='kappa 1, beta 0.65', rmse_fit=hm.log_rmse(base_cell.set_index('quarter').index_model,
                                                                           target, hm.FIT_Q),
                            index_2020q2_no_stepping_in=float(base_noss.set_index('quarter').loc['2020Q2', 'index_model']),
                            receipt_index_mean_2022=base_receipt_2022),
        spike_banxico={m: float(x) for m, x in spike_bx.items()}, enigh=ENIGH,
        fit=dict(criterion='quarterly log RMSE over hm.FIT_Q, C in [0, C_max] bounded scalar, 2022 held out'))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=float))
    pd.set_option('display.width', 250)
    cols = ['variant', 'covid_version', 'A_mode', 'B', 'B_ml_2019', 'ratio_out_of_work_to_employed', 'C', 'rmse_fit',
            'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022', 'spike_2021_01', 'spike_2021_03',
            'loglik_2019', 'aic_2019']
    print(profile.round(3).to_string(index=False))
    print(summary[cols].round(4).to_string(index=False))
    print(settings['base_reference'], spike_bx)
    figure(grid, profile, best_paths, base_cell, target, settings)


def figure(grid, profile, best_paths, base_cell, target, settings):
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location('model_make_figures', RL.parent / 'model/make_figures.py')
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    colors = {'i': mf.INK2, 'ii': mf.BLUE, 'iii': mf.ORANGE}
    ls = {'E': '-', 'S': '--'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    for need, c, lab in [(False, mf.BLUE, '(i)/(ii) no household need'), (True, mf.ORANGE, '(iii) + household need')]:
        d = profile[(profile.need_features == need) & (profile.A_mode == 'free')]
        ax.plot(d.B, d.loglik_2019 - d.loglik_2019.max(), color=c, lw=2, marker='o', ms=3.5, label=lab)
    ax.set_title('2019 profile log-likelihood over B (A free)', fontsize=10, loc='left', color=mf.INK)
    ax.set_xlabel('B (employment effect on keen logit)')
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    for (v, cv), d in grid[grid.A_mode == 'free'].groupby(['variant', 'covid_version']):
        ax.plot(d.B, d.rmse_fit, color=colors[v], ls=ls[cv], lw=1.8, label=f'{v}, Covid {cv}')
    ax.axhline(settings['base_reference']['rmse_fit'], color=mf.MUTED, lw=1, ls=':')
    ax.text(10, settings['base_reference']['rmse_fit'] - 0.0004, 'base (κ = 1, β = 0.65)', ha='right', va='top', fontsize=8, color=mf.INK2)
    ax.set_title('Quarterly log RMSE 2020Q2–2021Q4 (C fitted)', fontsize=10, loc='left', color=mf.INK)
    ax.set_xlabel('B')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[2]
    quarters = list(target.index)
    x = np.arange(len(quarters))
    ax.axhline(1, color=mf.MUTED, lw=1)
    ax.axvspan(quarters.index('2022Q1') - 0.5, len(quarters) - 0.5, color=mf.HOLDOUT_WASH, zorder=0)
    for (v, cv), d in best_paths[best_paths.A_mode == 'free'].groupby(['variant', 'covid_version']):
        d = d.set_index('quarter').reindex(quarters)
        ax.plot(x, d.index_model, color=colors[v], ls=ls[cv], lw=1.6, label=f'{v} {cv}, B = {d.B.iloc[0]:g}')
    b = base_cell.set_index('quarter').reindex(quarters)
    ax.plot(x, b.index_model, color=mf.GRAY_LIGHT, lw=2.2, label='base (κ = 1, β = 0.65)')
    ax.plot(x, target.values, color=mf.INK, lw=2, label='Banxico vs trend')
    ax.set_xticks(x[::4])
    ax.set_xticklabels(quarters[::4], fontsize=8)
    ax.set_title('Best B per variant (A free)', fontsize=10, loc='left', color=mf.INK)
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left')
    for a in axes:
        a.grid(color=mf.GRID, lw=0.6)
        for s in ['top', 'right']:
            a.spines[s].set_visible(False)
    fig.suptitle("Colleague's specification: keen migrants x employment effect B, need on receipt, no stepping-in",
                 x=0.01, ha='left', fontsize=12.5, fontweight='semibold', color=mf.INK)
    fig.tight_layout()
    fig.savefig(OUT / f'{TASK}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
