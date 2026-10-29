"""Robustness 6g (29 Oct 2026): the count mapping. Base vs lower bound vs mixed.

Run from the repo root: python network-compatibility-model/recipient-level/robustness/lower_bound_mapping.py

Mappings (r_h,t(beta) = E[able senders | someone sends], hm.Senders), mixed with weight w:
    counts_t(w) = sum_h P_h,t [ w r_h,t(beta) + (1 - w) r_h,t(beta) / r_h,t(0) ]
w = 0 is the base mapping, w = 1 the lower bound (hm.count_paths logic; the no-COVID run uses the same
w with its own r, beta = 0, so its base term is sum_h P_h,t).
(i)   lower-bound columns of map.csv summarised by kappa;
(ii)  checks the reproduction of the map.csv lower-bound columns at the grid points (max abs diff < 1e-8)
      and refits beta continuously per kappa under the lower bound;
(iii) the same refit for w in {0.25, 0.5, 0.75} (and 0, 1 for reference).
Outputs (aggregates only): outputs/robustness/lower_bound_mapping/.
"""
from pathlib import Path
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

OUT = RL / 'outputs/robustness/lower_bound_mapping'
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
LB_COLS = ['rmse_log_quarterly_lower_bound', 'index_2020q2_lower_bound', 'mean_index_2021_lower_bound',
           'spike_2021-01_lower_bound', 'spike_2021-03_lower_bound', 'spike_rmse_lower_bound']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid_ref = pd.read_csv(RL / 'outputs/household-model/map.csv')
    nf_ref = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')

    # E0 as in hm.main (warm-started chain over kappa, so the coefficients match exactly)
    pop = hm.hl.load_population()
    h = pop['h']
    ok = hm.e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    base = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    x_by_kappa, start = {}, None
    for kappa in hm.KAPPAS:
        x, _, _, _ = hm.fit_2019(base, y, ok, kappa, start)
        start = x_by_kappa[kappa] = x
    print('E0 done', flush=True)

    target, _ = hm.banxico_target()
    bx_m, trend_m, _ = hm.e1.banxico()
    spike_bx = hm.spikes((bx_m / trend_m).reindex(hm.DATES))
    covid = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    cc, cfc = hm.Cells(pop, covid), hm.Cells(pop, cf)
    snd_c = hm.Senders(pop, covid, hm.need_stock(hm.DATES))
    snd_f = hm.Senders(pop, cf, np.zeros(len(hm.DATES)))
    r_c0, r_f = snd_c.r(0.0), snd_f.r(0.0)
    cache = {}

    def r_c(beta):
        if beta not in cache:
            cache[beta] = snd_c.r(beta)
        return cache[beta]

    def monthly(Pc, Pf, beta, w):
        rc = r_c(beta)
        c = pd.Series((Pc * (w * rc + (1 - w) * rc / r_c0)).sum(1), index=hm.DATES)
        f = pd.Series((Pf * (w * r_f + (1 - w))).sum(1), index=hm.DATES)
        return c, f

    def qindex(Pc, Pf, beta, w):
        c, f = monthly(Pc, Pf, beta, w)
        return hm.quarterly(c) / hm.quarterly(f), c / f

    def summary(Pc, Pf, beta, w):
        q, m = qindex(Pc, Pf, beta, w)
        sp = hm.spikes(m)
        return dict(rmse_fit=hm.log_rmse(q, target, hm.FIT_Q),
                    rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q),
                    index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                    mean_index_2021=float(q[[p for p in hm.FIT_Q if p.startswith('2021')]].mean()),
                    mean_index_2022=float(q[hm.HOLD_Q].mean()),
                    spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                    spike_rmse=float(np.sqrt(np.mean([(sp[k] - spike_bx[k]) ** 2 for k in hm.SPIKE_MONTHS])))), q

    repro, fits, paths, at_base = [], [], [], []
    for kappa in hm.KAPPAS:
        x = x_by_kappa[kappa]
        Pf, Pc = cfc.receipt(x, kappa), cc.receipt(x, kappa)
        # (ii) reproduction of map.csv lower-bound columns
        for beta in hm.BETAS:
            s, _ = summary(Pc, Pf, beta, 1.0)
            ref = grid_ref[(grid_ref.kappa == kappa) & np.isclose(grid_ref.beta, beta)].iloc[0]
            mine = [s['rmse_fit'], s['index_2020q2'], s['mean_index_2021'], s['spike_2021_01'], s['spike_2021_03'],
                    s['spike_rmse']]
            repro.append(dict(kappa=kappa, beta=beta,
                              max_abs_diff=float(np.max(np.abs(np.array(mine) - ref[LB_COLS].to_numpy(float))))))
        # lower bound at the base-fitted (continuous) beta
        b_base = float(nf_ref.loc[np.isclose(nf_ref.kappa, kappa), 'beta'].iloc[0])
        s, _ = summary(Pc, Pf, b_base, 1.0)
        s0, _ = summary(Pc, Pf, b_base, 0.0)
        at_base.append(dict(kappa=kappa, beta_base=b_base, rmse_base=s0['rmse_fit'],
                            **{f'{k}_lower_bound': v for k, v in s.items()}))
        # (ii)/(iii) continuous refit per mapping weight
        for w in WEIGHTS:
            res = minimize_scalar(lambda b: hm.log_rmse(qindex(Pc, Pf, b, w)[0], target, hm.FIT_Q),
                                  bounds=(0.0, hm.BETA_MAX), method='bounded', options=dict(xatol=1e-4))
            bhat = float(res.x)
            s, q = summary(Pc, Pf, bhat, w)
            fits.append(dict(w=w, kappa=kappa, beta=bhat, beta_at_bound=bhat > hm.BETA_MAX - 1e-3, **s))
            paths.append(pd.DataFrame({'w': w, 'kappa': kappa, 'beta': bhat, 'quarter': q.index,
                                       'index_model': q.values, 'index_banxico': target[q.index].values}))
            if w == 1.0:
                sb, qb = summary(Pc, Pf, b_base, 1.0)
                paths.append(pd.DataFrame({'w': 'lower_bound_at_base_beta', 'kappa': kappa, 'beta': b_base,
                                           'quarter': qb.index, 'index_model': qb.values,
                                           'index_banxico': target[qb.index].values}))
            cache.clear()
        print(f'kappa {kappa}: done', flush=True)

    repro = pd.DataFrame(repro)
    print('reproduction max abs diff:', repro.max_abs_diff.max())
    assert repro.max_abs_diff.max() < 1e-8, 'lower-bound columns of map.csv not reproduced'
    repro.to_csv(OUT / 'reproduction_check.csv', index=False)

    # (i) grid summary
    g = grid_ref
    rows = []
    for kappa, d in g.groupby('kappa'):
        best_lb = d.loc[d.rmse_log_quarterly_lower_bound.idxmin()]
        best_base = d.loc[d.rmse_log_quarterly.idxmin()]
        rows.append(dict(kappa=kappa, grid_beta_base=best_base.beta, grid_rmse_base=best_base.rmse_log_quarterly,
                         rmse_lower_bound_at_grid_base_beta=best_base.rmse_log_quarterly_lower_bound,
                         index_2020q2_lower_bound_at_grid_base_beta=best_base.index_2020q2_lower_bound,
                         grid_beta_lower_bound=best_lb.beta, grid_rmse_lower_bound=best_lb.rmse_log_quarterly_lower_bound,
                         index_2020q2_lower_bound_at_lb_beta=best_lb.index_2020q2_lower_bound,
                         spike_2021_01_lower_bound_at_lb_beta=best_lb['spike_2021-01_lower_bound'],
                         spike_2021_03_lower_bound_at_lb_beta=best_lb['spike_2021-03_lower_bound'],
                         lb_beta_at_grid_edge=best_lb.beta >= g.beta.max() - 1e-9))
    gs = pd.DataFrame(rows).merge(pd.DataFrame(at_base), on='kappa')
    gs.to_csv(OUT / 'grid_summary.csv', index=False)

    fits = pd.DataFrame(fits)
    fits.to_csv(OUT / 'fits.csv', index=False)
    paths = pd.concat(paths, ignore_index=True)
    paths.to_csv(OUT / 'quarterly_paths.csv', index=False)
    pd.set_option('display.width', 250)
    print(gs.round(4).to_string(index=False))
    print(fits.round(4).to_string(index=False))
    print('Banxico spikes', spike_bx, '2020Q2', float(target['2020Q2']), '2021Q4', float(target['2021Q4']),
          '2022 mean', float(target[hm.HOLD_Q].mean()))
    pd.DataFrame([dict(index_2020q2=float(target['2020Q2']), index_2021q4=float(target['2021Q4']),
                       mean_index_2022=float(target[hm.HOLD_Q].mean()),
                       spike_2021_01=spike_bx['2021-01'], spike_2021_03=spike_bx['2021-03'])]).to_csv(
        OUT / 'banxico_moments.csv', index=False)

    # figure at the best kappa of the base fit
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    kb = float(fits[fits.w == 0.0].sort_values('rmse_fit').kappa.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    p = paths[paths.kappa == kb]
    qs = p.quarter.unique()
    xs = np.arange(len(qs))
    ax.plot(xs, target[qs].values, color='black', lw=2.2, label='Banxico / 2017-19 trend')
    styles = [(0.0, '#1f77b4', '-', 'base (w = 0)'), (0.5, '#9467bd', '-', 'mixed w = 0.5'),
              (1.0, '#d62728', '-', 'lower bound (w = 1)'),
              ('lower_bound_at_base_beta', '#d62728', ':', 'lower bound at base beta')]
    for w, col, ls, lab in styles:
        d = p[p.w.astype(str) == str(w)]
        ax.plot(xs, d.index_model.values, color=col, ls=ls, lw=1.6,
                label=f'{lab}, beta = {d.beta.iloc[0]:.2f}')
    ax.axvspan(xs[list(qs).index('2020Q2')] - 0.5, xs[list(qs).index('2021Q4')] + 0.5, color='0.92', zorder=0)
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xticks(xs[::2])
    ax.set_xticklabels(qs[::2], rotation=45)
    ax.set_ylabel('index (COVID / no-COVID)')
    ax.set_title(f'Count mapping, kappa = {kb:g} (shaded: fit window; beta refit per mapping)')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / 'paths_by_mapping.png', dpi=150)


if __name__ == '__main__':
    main()
