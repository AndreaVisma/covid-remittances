"""Robustness (29 Oct 2026), option 6a: savings margin for out-of-work migrants.

Run (repo root): python network-compatibility-model/recipient-level/robustness/savings_margin.py

An eligible migrant out of work and without benefits can still send from savings. In the household
model 'able' is binary, so savings enter as a probability of being able: with probability s an
out-of-work migrant is able in any month (the closest analogue of 'sends a share s of routine
sending'). Ability becomes
    P(able) = e + (1 - e) max(pi_t, s)      (eligible migrants; s = 0 is the base model)
consistently everywhere ability enters, by replacing hm.stimulus_pi with max(pi_t, s) inside this
process only: the able-count pmfs of hm.Cells (hl.able_pmfs, 2019 calibration, COVID and no-COVID
runs) and the Senders e of the count mapping. The 7 receipt coefficients are refitted per kappa
under s (2019 unemployment also has savings). The migrant-level M1 alphas used for theta_j in the
count mapping are kept at their base values (fitted with s = 0); they enter only r(beta) / r(0).
beta is fitted continuously per kappa as in need_fit.csv. Outputs (aggregates only) in
outputs/robustness/savings_margin/.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
RL = HERE.parent
sys.path.insert(0, str(RL))
sys.path.insert(0, str(RL.parent / 'model'))
import run_household_model as hm
import abm
import run_e0_baseline as e0
import run_e1_job_loss as e1
import compare_household_logit as hl

TASK = 'savings_margin'
OUT = RL / 'outputs/robustness' / TASK
SAVINGS = [0.0, 0.25, 0.5]
REPORT_KAPPAS = [0.0, 0.5, 0.8, 1.0]
_stimulus_pi = hm.stimulus_pi


def set_savings(s):
    hm.stimulus_pi = lambda dates: np.maximum(_stimulus_pi(dates), s)


def run_variant(s, pop, y, ok, target, spike_bx):
    set_savings(s)
    base = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    covid = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    cc, cfc = hm.Cells(pop, covid), hm.Cells(pop, cf)
    need = hm.need_stock(hm.DATES)
    snd_c, snd_f = hm.Senders(pop, covid, need), hm.Senders(pop, cf, np.zeros(len(hm.DATES)))
    r_c0, r_f = snd_c.r(0.0), snd_f.r(0.0)
    ser = lambda z: pd.Series(z.sum(1), index=hm.DATES)
    rows, paths, start = [], [], None
    for kappa in hm.KAPPAS:
        # hm.fit_2019 without its numerical SEs (saves CPU)
        x, nll = fit_no_se(base, y, ok, kappa, start)
        start = x
        Pf, Pc = cfc.receipt(x, kappa), cc.receipt(x, kappa)
        Pf_sum = hm.quarterly(ser(Pf))

        def idx(b):
            return hm.quarterly(ser(Pc * snd_c.r(b) / r_c0)) / Pf_sum
        res = minimize_scalar(lambda b: hm.log_rmse(idx(b), target, hm.FIT_Q), bounds=(0.0, hm.BETA_MAX),
                              method='bounded', options=dict(xatol=1e-4))
        b = float(res.x)
        r_b = snd_c.r(b)
        rc, rf, tc, tf = ser(Pc * r_b / r_c0), ser(Pf), ser(Pc * r_b), ser(Pf * r_f)
        q, low = hm.quarterly(rc) / hm.quarterly(rf), hm.quarterly(tc) / hm.quarterly(tf)
        sp = hm.spikes(rc / rf)
        rows.append(dict(savings_s=s, kappa=kappa, beta=b, rmse_fit=float(res.fun),
                         rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q),
                         index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                         mean_index_2022=float(q[hm.HOLD_Q].mean()),
                         spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                         spike_rmse=float(np.sqrt(np.mean([(sp[m] - spike_bx[m]) ** 2 for m in hm.SPIKE_MONTHS]))),
                         rmse_fit_lower_bound=hm.log_rmse(low, target, hm.FIT_Q),
                         loglik_2019=-nll, mean_receipt_2019=float(base.annual(x, kappa)[ok].mean()),
                         **{f'coef_{n}': v for n, v in zip(['intercept'] + hm.TERMS, x)}))
        paths.append(pd.DataFrame({'savings_s': s, 'kappa': kappa, 'beta': b, 'quarter': q.index,
                                   'index_model': q.values, 'index_model_lower_bound': low.values,
                                   'index_banxico': target[q.index].values}))
        print(f's {s} kappa {kappa}: beta {b:.4f} rmse {res.fun:.4f}', flush=True)
    set_savings(0.0)
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True)


def fit_no_se(cells, y, mask, kappa, start):
    saved = hl.numerical_se
    hl.numerical_se = lambda f, x: np.full(len(x), np.nan)
    try:
        x, _, nll, _ = hm.fit_2019(cells, y, mask, kappa, start)
    finally:
        hl.numerical_se = saved
    return x, nll


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hl.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    target, _ = hm.banxico_target()
    bx_m, trend_m, _ = e1.banxico()
    spike_bx = hm.spikes((bx_m / trend_m).reindex(hm.DATES))

    full, paths = [], []
    for s in SAVINGS:
        r, p = run_variant(s, pop, y, ok, target, spike_bx)
        full.append(r)
        paths.append(p)
    full = pd.concat(full, ignore_index=True)

    # check: s = 0 reproduces the base need_fit.csv
    ref = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')
    s0 = full[full.savings_s == 0].reset_index(drop=True)
    cols = {'beta': 'beta', 'rmse_fit': 'rmse_log_quarterly', 'rmse_holdout_2022': 'rmse_log_quarterly_holdout_2022',
            'index_2020q2': 'index_2020q2', 'index_2021q4': 'index_2021q4', 'mean_index_2022': 'mean_index_2022'}
    check = {k: float(np.abs(s0[k].to_numpy() - ref[v].to_numpy()).max()) for k, v in cols.items()}
    print('max abs diff vs base need_fit.csv:', check)

    # summary: report kappas plus the best kappa per variant, with base (s = 0) values for comparison
    base_vals = s0.set_index('kappa')
    metric = ['beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
              'spike_2021_01', 'spike_2021_03', 'spike_rmse', 'rmse_fit_lower_bound']
    rows, best = [], {}
    for s, d in full.groupby('savings_s'):
        kb = float(d.loc[d.rmse_fit.idxmin(), 'kappa'])
        ks = float(d.loc[d.spike_rmse.idxmin(), 'kappa'])
        best[s] = dict(kappa_min_rmse_fit=kb, kappa_min_spike_rmse=ks)
        for kappa in sorted(set(REPORT_KAPPAS) | {kb}):
            r = d[np.isclose(d.kappa, kappa)].iloc[0]
            row = dict(variant=f's={s}', savings_s=s, kappa=kappa, is_best_kappa=kappa == kb,
                       kappa_min_rmse_fit=kb, kappa_min_spike_rmse=ks, **{m: r[m] for m in metric})
            row.update({f'base_{m}': base_vals.loc[kappa, m] for m in metric})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'summary.csv', index=False)
    full.to_csv(OUT / 'all_kappas.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(OUT / 'quarterly_paths.csv', index=False)
    settings = dict(
        task=TASK, savings_s=SAVINGS, kappas=hm.KAPPAS, report_kappas=REPORT_KAPPAS,
        ability='P(able) = e + (1 - e) max(pi_t, s), eligible migrants; pi_t = hm.PI_STIMULUS',
        interpretation=('able is binary in the household model: savings enter as a probability s that an '
                        'out-of-work migrant is able in a month (closest analogue of sending a share s of '
                        'routine sending)'),
        applied_to=['hl.able_pmfs via hm.Cells (2019 calibration, COVID, no-COVID)', 'hm.Senders.e (count mapping)'],
        refit='7 receipt coefficients by ML on 2019 MMP per kappa and s (no SEs computed)',
        not_refit='migrant-level M1 alphas (theta_j in r(beta)), kept at base values',
        beta='fitted continuously per kappa on [0, 3], quarterly log RMSE 2020Q2..2021Q4',
        spike_banxico={m: float(v) for m, v in spike_bx.items()},
        base_reproduction_max_abs_diff=check, best=best)
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    pd.set_option('display.width', 250)
    print(summary[['variant', 'kappa'] + metric].round(4).to_string(index=False))
    print(best)
    figure(full, spike_bx)


def figure(full, spike_bx):
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location('model_make_figures', RL.parent / 'model/make_figures.py')
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    colors = {0.0: mf.INK2, 0.25: mf.BLUE, 0.5: mf.ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for s, d in full.groupby('savings_s'):
        kw = dict(color=colors[s], lw=2, label=f's = {s:g}' + (' (base)' if s == 0 else ''))
        axes[0].plot(d.kappa, d.rmse_fit, **kw)
        axes[1].plot(d.kappa, d.beta, **kw)
        axes[2].plot(d.kappa, d.spike_rmse, **kw)
    for ax, t in zip(axes, ['Quarterly log RMSE 2020Q2-2021Q4', 'Fitted beta', 'Spike RMSE (Jan, Mar 2021)']):
        ax.set_title(t, fontsize=10, loc='left', color=mf.INK)
        ax.set_xlabel('kappa (replacement share)')
        ax.grid(color=mf.GRID, lw=0.6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Savings margin: out-of-work migrants able with probability s', x=0.01, ha='left',
                 fontsize=12.5, fontweight='semibold', color=mf.INK)
    fig.tight_layout()
    fig.savefig(OUT / f'{TASK}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
