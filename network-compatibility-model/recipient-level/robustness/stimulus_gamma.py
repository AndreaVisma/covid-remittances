"""Robustness 6c (29 Oct 2026): stimulus as a rise in every migrant's sending probability (gamma).

Run from the repo root: python network-compatibility-model/recipient-level/robustness/stimulus_gamma.py

Base (run_household_model.py): in a stimulus month a non-employed migrant can send with probability pi
(pi = 1 in Jan and Mar 2021). Alternative: every migrant's sending probability rises in the stimulus months,
    theta_j,t = expit(alpha_j + beta N*_t + gamma D_t),   D_t = [t in Jan 2021, Mar 2021],
entering through the count multiplier r_h,t(beta, gamma) / r_h,t(0, 0) (more transfers per receiving
household, receipt unchanged, like option 6a). Third channel: gamma as a household receipt logit shift in
the stimulus months (s = expit(x'b + gamma D_t)).
Variants (pi = stimulus pi in Jan/Mar 2021; Apr 2020 pi = 0 throughout):
    base                pi = 1, gamma = 0                    (reproduces need_fit.csv)
    pi0                 pi = 0, gamma = 0                    (no stimulus at all; reference)
    gamma_senders       pi = 0, gamma on senders
    pi1_gamma_senders   pi = 1, gamma on senders
    gamma_receipt       pi = 0, gamma on the receipt logit
    pi1_gamma_receipt   pi = 1, gamma on the receipt logit
Fit per kappa (alternating, to a fixed point): beta = argmin quarterly log RMSE 2020Q2..2021Q4 given gamma,
gamma = argmin spike RMSE (Jan/Mar 2021 spikes vs Banxico) given beta. Separability: a pi profile
(pi in PI_GRID, gamma_senders fitted) at every kappa.
Outputs (aggregates only): outputs/robustness/stimulus_gamma/.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
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

OUT = RL / 'outputs/robustness/stimulus_gamma'
REPORT_KAPPAS = [0.0, 0.5, 0.8, 1.0]
GAMMA_BOUNDS = {'senders': (-3.0, 10.0), 'receipt': (-3.0, 6.0)}
PI_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
VARIANTS = {'base': (1.0, None), 'pi0': (0.0, None),
            'gamma_senders': (0.0, 'senders'), 'pi1_gamma_senders': (1.0, 'senders'),
            'gamma_receipt': (0.0, 'receipt'), 'pi1_gamma_receipt': (1.0, 'receipt')}
T0 = time.time()


def log(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def set_pi(pi):
    """Stimulus pi in Jan/Mar 2021 (Apr 2020 stays 0); keys unchanged so stimulus_pi's check passes."""
    hm.PI_STIMULUS = {'2020-04': 0.0, '2021-01': float(pi), '2021-03': float(pi)}


class GSenders(hm.Senders):
    """r_h,t(beta, gamma) with theta_j,t = expit(alpha_j + beta N*_t + gamma D_t)."""

    def __init__(self, pop, inputs, need, D):
        super().__init__(pop, inputs, need)
        self.D = D

    def r(self, beta=0.0, gamma=0.0):
        if gamma == 0.0:
            return super().r(beta)
        q = self.e * expit(self.alpha[None, :] + beta * self.need[:, None] + gamma * self.D[:, None])
        hsum = lambda Z: np.stack([np.bincount(self.hid, weights=z, minlength=self.H) for z in Z])
        p_any = -np.expm1(hsum(np.log1p(-q)))
        return np.divide(hsum(q), p_any, out=np.ones_like(p_any), where=p_any > 1e-12)


def receipt_rows(cells, x, kappa, rows, shift):
    """Receipt (len(rows), H) in the given months with a logit shift (same formula as Cells.receipt)."""
    W = cells.W[rows]
    s = expit(cells.X @ x + shift)
    sf = expit(cells.X_full @ x + shift)
    return (cells.household_sum(W * s * (1 - kappa * cells.trigger))
            + kappa * sf * cells.household_sum(W * cells.trigger))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hl.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    kappas = hm.KAPPAS

    # E0 exactly as the base (warm-started chain over KAPPAS)
    base2019 = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    x_by_kappa, start = {}, None
    for kappa in kappas:
        x, _, _, _ = hm.fit_2019(base2019, y, ok, kappa, start)
        start = x_by_kappa[kappa] = x
    log('E0 done')

    target, _ = hm.banxico_target()
    bx_m, trend_m, _ = e1.banxico()
    spike_bx = hm.spikes((bx_m / trend_m).reindex(hm.DATES))
    covid = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need = hm.need_stock(hm.DATES)
    ym = hm.DATES.strftime('%Y-%m')
    D = np.isin(ym, hm.SPIKE_MONTHS).astype(float)
    rows_D = np.flatnonzero(D)

    set_pi(0.0)
    cfc = hm.Cells(pop, cf)
    snd_f = GSenders(pop, cf, np.zeros(len(hm.DATES)), np.zeros(len(hm.DATES)))
    r_f = snd_f.r(0.0)
    cells_by_pi, snd_by_pi = {}, {}
    for pi in PI_GRID:
        set_pi(pi)
        cells_by_pi[pi] = hm.Cells(pop, covid)
        snd_by_pi[pi] = GSenders(pop, covid, need, D)
        log(f'cells pi = {pi} built')
    set_pi(1.0)
    Pf_by_kappa = {k: cfc.receipt(x_by_kappa[k], k) for k in kappas}
    Pf_sum = {k: pd.Series(P.sum(1), index=hm.DATES) for k, P in Pf_by_kappa.items()}
    rf_lower = {k: pd.Series((P * r_f).sum(1), index=hm.DATES) for k, P in Pf_by_kappa.items()}
    r0_by_pi = {pi: s.r(0.0) for pi, s in snd_by_pi.items()}

    def model_paths(pi, channel, kappa, beta, gamma, P0):
        snd = snd_by_pi[pi]
        g_s = gamma if channel == 'senders' else 0.0
        r_c = snd.r(beta, g_s)
        Pc = P0
        if channel == 'receipt' and gamma != 0.0:
            Pc = P0.copy()
            Pc[rows_D] = receipt_rows(cells_by_pi[pi], x_by_kappa[kappa], kappa, rows_D, gamma)
        rc = pd.Series((Pc * r_c / r0_by_pi[pi]).sum(1), index=hm.DATES)
        tc = pd.Series((Pc * r_c).sum(1), index=hm.DATES)
        return rc / Pf_sum[kappa], tc / rf_lower[kappa], (rc, tc)

    def qindex(rc_tc, kappa):
        rc, tc = rc_tc
        return hm.quarterly(rc) / hm.quarterly(Pf_sum[kappa]), hm.quarterly(tc) / hm.quarterly(rf_lower[kappa])

    def spike_rmse(sp):
        return float(np.sqrt(np.mean([(sp[m] - spike_bx[m]) ** 2 for m in hm.SPIKE_MONTHS])))

    def fit(pi, channel, kappa, P0):
        f_q = lambda b, g: hm.log_rmse(qindex(model_paths(pi, channel, kappa, b, g, P0)[2], kappa)[0],
                                       target, hm.FIT_Q)
        f_s = lambda b, g: spike_rmse(hm.spikes(model_paths(pi, channel, kappa, b, g, P0)[0].to_numpy()))
        gamma, beta, it = 0.0, None, 0
        for it in range(1, 31):
            b_new = float(minimize_scalar(lambda b: f_q(b, gamma), bounds=(0.0, hm.BETA_MAX), method='bounded',
                                          options=dict(xatol=1e-4)).x)
            if channel is None:
                beta = b_new
                break
            g_new = float(minimize_scalar(lambda g: f_s(b_new, g), bounds=GAMMA_BOUNDS[channel],
                                          method='bounded', options=dict(xatol=1e-5)).x)
            done = beta is not None and abs(b_new - beta) < 1e-4 and abs(g_new - gamma) < 1e-4
            beta, gamma = b_new, g_new
            if done:
                break
        m, lo, rt = model_paths(pi, channel, kappa, beta, gamma, P0)
        q, ql = qindex(rt, kappa)
        sp, spl = hm.spikes(m.to_numpy()), hm.spikes(lo.to_numpy())
        stats = dict(beta=beta, gamma=gamma, iterations=it,
                     rmse_fit=hm.log_rmse(q, target, hm.FIT_Q),
                     rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q),
                     index_2020q2=float(q['2020Q2']), index_2021q1=float(q['2021Q1']),
                     index_2021q4=float(q['2021Q4']), mean_index_2022=float(q[hm.HOLD_Q].mean()),
                     spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'], spike_rmse=spike_rmse(sp),
                     rmse_fit_lower_bound=hm.log_rmse(ql, target, hm.FIT_Q),
                     spike_rmse_lower_bound=spike_rmse(spl),
                     gamma_at_bound=channel is not None and min(abs(gamma - v) for v in GAMMA_BOUNDS[channel]) < 1e-3)
        return stats, q, ql

    # Variants over all kappas
    res, qpaths = [], {}
    for name, (pi, channel) in VARIANTS.items():
        for kappa in kappas:
            P0 = cells_by_pi[pi].receipt(x_by_kappa[kappa], kappa)
            st, q, ql = fit(pi, channel, kappa, P0)
            res.append(dict(variant=name, pi=pi, channel=channel or 'none', kappa=kappa, **st))
            qpaths[(name, kappa)] = (q, ql)
        log(f'variant {name} done')
    full = pd.DataFrame(res)

    # Reproduction check vs need_fit.csv
    nf = pd.read_csv(RL / 'outputs/household-model/need_fit.csv').set_index('kappa')
    b = full[full.variant == 'base'].set_index('kappa')
    repro = {c_mine: float(np.max(np.abs(b[c_mine].to_numpy() - nf.loc[b.index, c_base].to_numpy())))
             for c_mine, c_base in [('beta', 'beta'), ('rmse_fit', 'rmse_log_quarterly'),
                                    ('rmse_holdout_2022', 'rmse_log_quarterly_holdout_2022'),
                                    ('index_2020q2', 'index_2020q2'), ('index_2021q4', 'index_2021q4'),
                                    ('mean_index_2022', 'mean_index_2022')]}
    log('reproduction max abs diff vs need_fit.csv:', repro)

    # pi profile: gamma_senders fitted for each pi
    prof = []
    for pi in PI_GRID:
        for kappa in kappas:
            P0 = cells_by_pi[pi].receipt(x_by_kappa[kappa], kappa)
            st, _, _ = fit(pi, 'senders', kappa, P0)
            prof.append(dict(pi=pi, kappa=kappa, **{k: st[k] for k in
                             ['beta', 'gamma', 'rmse_fit', 'spike_rmse', 'spike_2021_01', 'spike_2021_03']}))
        log(f'pi profile {pi} done')
    prof = pd.DataFrame(prof)

    # Summary: report kappas + best kappa by rmse_fit and by spike_rmse; base values for comparison
    metrics = ['beta', 'gamma', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
               'spike_2021_01', 'spike_2021_03', 'spike_rmse', 'rmse_fit_lower_bound']
    base_by_k = full[full.variant == 'base'].set_index('kappa')
    out_rows, best = [], {}
    for name, d in full.groupby('variant', sort=False):
        kb, ks = float(d.loc[d.rmse_fit.idxmin(), 'kappa']), float(d.loc[d.spike_rmse.idxmin(), 'kappa'])
        best[name] = dict(kappa_min_rmse_fit=kb, kappa_min_spike_rmse=ks)
        for kappa in sorted(set(REPORT_KAPPAS) | {kb, ks}):
            r = d[d.kappa == kappa].iloc[0].to_dict()
            r['row'] = '+'.join([t for t, c in [('report', kappa in REPORT_KAPPAS), ('best_rmse_fit', kappa == kb),
                                                 ('best_spike_rmse', kappa == ks)] if c])
            r['kappa_min_rmse_fit'], r['kappa_min_spike_rmse'] = kb, ks
            r.update({f'base_{m}': base_by_k.loc[kappa, m] for m in metrics})
            out_rows.append(r)
    summary = pd.DataFrame(out_rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(full.drop(columns=['pi', 'channel']).round(4).to_string(index=False))
    print(prof.round(4).to_string(index=False))
    print(best)

    qp = []
    for _, r in summary.iterrows():
        q, ql = qpaths[(r.variant, r.kappa)]
        qp.append(pd.DataFrame({'variant': r.variant, 'kappa': r.kappa, 'beta': r.beta, 'gamma': r.gamma,
                                'quarter': q.index, 'index_model': q.values, 'index_model_lower_bound': ql.values,
                                'index_banxico': target[q.index].values}))
    pd.concat(qp, ignore_index=True).drop_duplicates(['variant', 'kappa', 'quarter']).to_csv(
        OUT / 'quarterly_paths.csv', index=False)

    pbest = prof.loc[prof.groupby('pi').spike_rmse.idxmin()]
    settings = dict(
        task='stimulus_gamma (6c)', date='2026-10-29',
        theta='expit(alpha_j + beta N*_t + gamma D_t), D_t = [Jan 2021, Mar 2021]; Apr 2020 pi = 0 in all variants',
        variants={k: dict(pi=v[0], channel=v[1] or 'none') for k, v in VARIANTS.items()},
        channels=dict(senders='gamma in theta, count multiplier r(beta, gamma) / r(0, 0); receipt unchanged',
                      receipt='gamma as household receipt logit shift in D_t months (s and s_full)'),
        fit=('per kappa, alternating to a fixed point (|change| < 1e-4): beta = argmin quarterly log RMSE '
             '2020Q2..2021Q4 given gamma (bounded 0..3); gamma = argmin spike RMSE (Jan/Mar 2021, log index minus '
             'mean of neighbouring months, vs Banxico) given beta'),
        gamma_bounds=GAMMA_BOUNDS, spike_banxico=spike_bx,
        reproduction_max_abs_diff_vs_need_fit=repro, best_kappa=best,
        all_kappas=full.round(6).to_dict('records'),
        pi_profile=dict(note='gamma_senders fitted at each pi (alternating fit as above)',
                        rows=prof.round(6).to_dict('records'),
                        best_kappa_by_spike_per_pi=pbest.round(6).to_dict('records')))
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=float))
    figure(full, qpaths, target, spike_bx, best)
    log('done')


def figure(full, qpaths, target, spike_bx, best):
    import matplotlib.pyplot as plt
    import importlib.util
    spec = importlib.util.spec_from_file_location('model_make_figures', RL.parent / 'model/make_figures.py')
    mf = importlib.util.module_from_spec(spec)   # the style module recipient-level/make_figures.py imports as mf
    spec.loader.exec_module(mf)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    show = [('base', 'Base: π = 1', mf.K4), ('gamma_senders', 'γ on senders, π = 0', mf.ORANGE),
            ('gamma_receipt', 'γ on receipt, π = 0', mf.AQUA), ('pi0', 'No stimulus (π = 0, γ = 0)', mf.MUTED)]
    ax = axes[0]
    for name, lab, col in show:
        d = full[full.variant == name].sort_values('kappa')
        ax.plot(d.kappa, d.spike_2021_01, color=col, label=lab)
        ax.plot(d.kappa, d.spike_2021_03, color=col, ls='--')
    ax.axhline(spike_bx['2021-01'], color=mf.INK, lw=1)
    ax.axhline(spike_bx['2021-03'], color=mf.INK, lw=1, ls='--')
    ax.set_xlabel('κ (replacement share)')
    ax.set_ylabel('Spike (log points): Jan solid, Mar dashed')
    ax.legend(loc='best', fontsize=8)
    ax = axes[1]
    for name, lab, col in show:
        d = full[full.variant == name].sort_values('kappa')
        ax.plot(d.kappa, d.rmse_fit, color=col, label=lab)
    ax.set_xlabel('κ (replacement share)')
    ax.set_ylabel('Quarterly log RMSE 2020Q2–2021Q4')
    fig.suptitle('Stimulus as γ (all migrants) vs π (non-employed): spikes and fit by κ', x=0.01, ha='left',
                 fontsize=12.5, fontweight='semibold', color=mf.INK)
    fig.savefig(OUT / 'stimulus_gamma.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    if '--figure-only' in sys.argv:   # redraw from run_settings.json
        st = json.loads((OUT / 'run_settings.json').read_text())
        figure(pd.DataFrame(st['all_kappas']), None, None, st['spike_banxico'], st['best_kappa'])
    else:
        main()
