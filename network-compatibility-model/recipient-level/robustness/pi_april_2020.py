"""Robustness 6b (29 Oct 2026): stimulus pi in April 2020 (first round).

Run from the repo root: python network-compatibility-model/recipient-level/robustness/pi_april_2020.py

Base model run_household_model.py sets pi = 0 in Apr 2020 (round 1 excluded ITIN filers and
mixed-status couples; the April moment is distorted by the March 2020 rush at the weak peso).
Variants: pi(Apr 2020) in {0 (base), 0.5, 1}; pi = 1 in Jan and Mar 2021 throughout. Only the
values of hm.PI_STIMULUS are patched (stimulus_pi asserts its keys). The 2019 receipt fit and the
no-COVID run use 2019 dates, so they do not depend on pi; only the COVID run is rebuilt.
For each variant and kappa: beta fitted continuously (as need_fit), quarterly fit/holdout, spikes,
monthly index Mar-Jun 2020, April spike (log Apr - mean log Mar, May), Mar+Apr pooled index.
Outputs (aggregates only): outputs/robustness/pi_april_2020/.
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

OUT = RL / 'outputs/robustness/pi_april_2020'
VARIANTS = {'pi_apr_0_base': 0.0, 'pi_apr_0.5': 0.5, 'pi_apr_1': 1.0}
REPORT_KAPPAS = [0.0, 0.5, 0.8, 1.0]
MONTHS = ['2020-03', '2020-04', '2020-05', '2020-06']


def log_spike(s, m):
    l = np.log(s)
    i = list(s.index).index(m)
    return float(l.iloc[i] - 0.5 * (l.iloc[i - 1] + l.iloc[i + 1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pop = hm.hl.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)

    # E0 (independent of pi: 2019 dates, no stimulus months)
    base = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    x_by_kappa, start = {}, None
    for kappa in hm.KAPPAS:
        start = x_by_kappa[kappa] = hm.fit_2019(base, y, ok, kappa, start)[0]
    print('E0 done', flush=True)

    target, _ = hm.banxico_target()
    bx_m, trend_m, _ = e1.banxico()
    bxi = (bx_m / trend_m).reindex(hm.DATES)
    bxi.index = hm.DATES.strftime('%Y-%m')
    bx_ratio = lambda ms: float(bx_m.reindex(hm.DATES)[pd.to_datetime(ms)].sum()
                                / trend_m.reindex(hm.DATES)[pd.to_datetime(ms)].sum())
    spike_bx = {m: log_spike(bxi, m) for m in hm.SPIKE_MONTHS + ['2020-04']}
    banxico_row = dict(variant='banxico', **{f'index_{m}': float(bxi[m]) for m in MONTHS},
                       spike_2020_04=spike_bx['2020-04'], spike_2021_01=spike_bx['2021-01'],
                       spike_2021_03=spike_bx['2021-03'], index_2020_mar_apr_pooled=bx_ratio(['2020-03', '2020-04']),
                       index_2020q2=float(target['2020Q2']), index_2021q4=float(target['2021Q4']),
                       mean_index_2022=float(target[hm.HOLD_Q].mean()))
    print('Banxico', {k: round(v, 4) if isinstance(v, float) else v for k, v in banxico_row.items()}, flush=True)

    covid = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    need = hm.need_stock(hm.DATES)
    cfc = hm.Cells(pop, cf)
    snd_f = hm.Senders(pop, cf, np.zeros(len(hm.DATES)))
    r_f = snd_f.r(0.0)
    a_core = hm.m1_alphas()[0]

    saved = dict(hm.PI_STIMULUS)
    rows, qpaths, mpaths = [], [], []
    try:
        for vname, pi_apr in VARIANTS.items():
            hm.PI_STIMULUS['2020-04'] = pi_apr
            cc = hm.Cells(pop, covid)
            snd_c = hm.Senders(pop, covid, need)
            r_c0 = snd_c.r(0.0)
            for kappa in hm.KAPPAS:
                x = x_by_kappa[kappa]
                Pf, Pc = cfc.receipt(x, kappa), cc.receipt(x, kappa)
                f_m = pd.Series(Pf.sum(1), index=hm.DATES)
                tf_m = pd.Series((Pf * r_f).sum(1), index=hm.DATES)

                def paths(b):
                    r_c = snd_c.r(b)
                    return (pd.Series((Pc * r_c / r_c0).sum(1), index=hm.DATES),
                            pd.Series((Pc * r_c).sum(1), index=hm.DATES))

                res = minimize_scalar(lambda b: hm.log_rmse(hm.quarterly(paths(b)[0]) / hm.quarterly(f_m), target,
                                                            hm.FIT_Q),
                                      bounds=(0.0, hm.BETA_MAX), method='bounded', options=dict(xatol=1e-4))
                bhat = float(res.x)
                c_m, tc_m = paths(bhat)
                q = hm.quarterly(c_m) / hm.quarterly(f_m)
                ql = hm.quarterly(tc_m) / hm.quarterly(tf_m)
                mi = c_m / f_m
                mi.index = hm.DATES.strftime('%Y-%m')
                sp = {m: log_spike(mi, m) for m in hm.SPIKE_MONTHS + ['2020-04']}
                ma = ['2020-03', '2020-04']
                rows.append(dict(
                    variant=vname, pi_apr_2020=pi_apr, kappa=kappa, beta=bhat, rmse_fit=float(res.fun),
                    rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q), index_2020q2=float(q['2020Q2']),
                    index_2021q4=float(q['2021Q4']), mean_index_2022=float(q[hm.HOLD_Q].mean()),
                    spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                    spike_rmse=float(np.sqrt(np.mean([(sp[m] - spike_bx[m]) ** 2 for m in hm.SPIKE_MONTHS]))),
                    rmse_fit_lower_bound=hm.log_rmse(ql, target, hm.FIT_Q),
                    **{f'index_{m}': float(mi[m]) for m in MONTHS},
                    spike_2020_04=sp['2020-04'],
                    index_2020_mar_apr_pooled=float(c_m[pd.to_datetime(ma)].sum() / f_m[pd.to_datetime(ma)].sum()),
                    theta_core_2021_12=float(hm.expit(a_core + bhat))))
                qpaths.append(pd.DataFrame({'variant': vname, 'pi_apr_2020': pi_apr, 'kappa': kappa, 'beta': bhat,
                                            'quarter': q.index, 'index_model': q.values,
                                            'index_model_lower_bound': ql.values,
                                            'index_banxico': target[q.index].values}))
                mpaths.append(pd.DataFrame({'variant': vname, 'pi_apr_2020': pi_apr, 'kappa': kappa,
                                            'month': mi.index, 'index_model': mi.values,
                                            'index_banxico': bxi.values}))
            print(vname, 'done', flush=True)
    finally:
        hm.PI_STIMULUS.clear()
        hm.PI_STIMULUS.update(saved)

    allk = pd.DataFrame(rows)
    # reproduction check: base variant vs outputs/household-model/need_fit.csv
    nf = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')
    b0 = allk[allk.variant == 'pi_apr_0_base'].reset_index(drop=True)
    cols = {'beta': 'beta', 'rmse_fit': 'rmse_log_quarterly', 'rmse_holdout_2022': 'rmse_log_quarterly_holdout_2022',
            'index_2020q2': 'index_2020q2', 'index_2021q4': 'index_2021q4', 'mean_index_2022': 'mean_index_2022'}
    repro = float(max(np.abs(b0[a].to_numpy() - nf[b].to_numpy()).max() for a, b in cols.items()))
    print('reproduction max abs diff vs need_fit.csv:', repro, flush=True)

    # summary: report kappas + best kappa per variant, with base values
    base_vals = b0.set_index('kappa')
    metric_cols = ['beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
                   'spike_2021_01', 'spike_2021_03', 'spike_rmse', 'rmse_fit_lower_bound', 'index_2020-04',
                   'spike_2020_04', 'index_2020_mar_apr_pooled']
    summ, best = [], {}
    for vname, d in allk.groupby('variant', sort=False):
        kb, ks = float(d.loc[d.rmse_fit.idxmin(), 'kappa']), float(d.loc[d.spike_rmse.idxmin(), 'kappa'])
        best[vname] = dict(kappa_min_rmse_fit=kb, kappa_min_spike_rmse=ks)
        for i, kappa in enumerate(REPORT_KAPPAS + [kb]):
            r = d[d.kappa == kappa].iloc[0].to_dict()
            r['row'] = 'best_rmse_fit' if i == len(REPORT_KAPPAS) else 'kappa'
            r['kappa_min_rmse_fit'], r['kappa_min_spike_rmse'] = kb, ks
            r.update({f'base_{c}': float(base_vals.loc[kappa, c]) for c in metric_cols})
            summ.append(r)
    summ = pd.DataFrame(summ)
    summ = pd.concat([summ, pd.DataFrame([dict(banxico_row, row='banxico')])], ignore_index=True)
    summ.to_csv(OUT / 'summary.csv', index=False)
    allk.to_csv(OUT / 'all_kappas.csv', index=False)
    qp = pd.concat(qpaths, ignore_index=True)
    qp[qp.kappa.isin(REPORT_KAPPAS)].to_csv(OUT / 'quarterly_paths.csv', index=False)
    mp = pd.concat(mpaths, ignore_index=True)
    mp[mp.kappa.isin(REPORT_KAPPAS)].to_csv(OUT / 'monthly_paths.csv', index=False)

    pd.set_option('display.width', 250)
    show = ['variant', 'kappa', 'beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4',
            'spike_2021_01', 'spike_2021_03', 'spike_rmse', 'rmse_fit_lower_bound',
            'index_2020-03', 'index_2020-04', 'index_2020-05', 'index_2020-06', 'spike_2020_04',
            'index_2020_mar_apr_pooled']
    print(summ[show].round(4).to_string(index=False))
    print(best)

    (OUT / 'run_settings.json').write_text(json.dumps(dict(
        task='pi_april_2020 (6b)', variants=VARIANTS, pi_other={'2021-01': 1.0, '2021-03': 1.0},
        reproduction_max_abs_diff_vs_need_fit=repro, best=best, banxico=banxico_row,
        april_spike='log index Apr 2020 - mean(log Mar 2020, log May 2020), monthly',
        mar_apr_pooled='(model counts Mar+Apr 2020) / (no-COVID Mar+Apr); Banxico: (Mar+Apr) / (trend Mar+Apr)',
        beta='fitted continuously per kappa and variant, bounded (0, 3), quarterly log RMSE 2020Q2-2021Q4',
        e0='2019 receipt coefficients independent of pi (2019 dates), fitted once'), indent=2))

    figure(mp, summ)


def figure(mp, summ):
    import importlib.util
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location('rl_make_figures', RL / 'make_figures.py')
    rlf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlf)
    mf = rlf.mf
    months = [f'2020-{m:02d}' for m in range(1, 9)]
    colors = {'pi_apr_0_base': mf.K0, 'pi_apr_0.5': mf.K2, 'pi_apr_1': mf.K4}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kappa in zip(axes, [0.8, 1.0]):
        d = mp[(mp.kappa == kappa) & mp.month.isin(months)]
        bx = d[d.variant == 'pi_apr_0_base']
        ax.plot(range(len(months)), bx.index_banxico, color=mf.INK, lw=2, marker='o', ms=4, label='Banxico / trend')
        for v, c in colors.items():
            dv = d[d.variant == v]
            ax.plot(range(len(months)), dv.index_model, color=c, lw=1.8, marker='o', ms=3,
                    label=f"model, pi(Apr 2020) = {dv.variant.iloc[0].split('_')[2]}")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([pd.Timestamp(m).strftime('%b') for m in months])
        ax.set_title(f'kappa = {kappa} (beta refitted)', fontsize=10.5, color=mf.INK)
        ax.axhline(1, color=mf.GRAY_LIGHT, lw=0.8, zorder=0)
    axes[0].set_ylabel('Monthly index (COVID / no-COVID)')
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle('Stimulus in April 2020: monthly index Jan-Aug 2020', x=0.01, ha='left', fontsize=12.5,
                 fontweight='semibold', color=mf.INK)
    fig.savefig(OUT.parent / 'pi_april_2020.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
