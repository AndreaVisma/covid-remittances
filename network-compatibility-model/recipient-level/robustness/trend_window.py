"""Robustness: Banxico trend window (29 Oct 2026).

Run: python network-compatibility-model/recipient-level/robustness/trend_window.py   (~3-5 min)

The model side is the base household model (run_household_model.py, option 6a need, stimulus pi,
'Not working' as unknown); only the target changes: Banxico transactions relative to a log-linear
trend with calendar-month effects fitted on different pre-COVID windows (base 2017-19), plus a
log-linear trend without month effects (2017-19). For each target, beta is refitted continuously for
every kappa (as need_fit.csv) and the best kappa is the one with the lowest fit RMSE. The 2021Q4
decomposition (raw Banxico vs the same quarter of 2019 = trend + job loss + need + unexplained, log
points, as recipient-level/make_figures.py decomposition()) is recomputed for each window at the best
kappa and its continuous beta. The 200-seed distribution step is skipped.
Check: at the base window the wrapper reproduces outputs/household-model/need_fit.csv (max abs diff
< 1e-8) and the base decomposition at the grid-best cell (kappa 1, beta 0.65).
Outputs (aggregates only): outputs/robustness/trend_window/.
"""
from pathlib import Path
import importlib.util
import json
import sys
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
import run_e1_job_loss as e1
import compare_household_logit as hl
import run_e0_baseline as e0

OUT = RL / 'outputs/robustness/trend_window'
VARIANTS = [  # name, window, month effects
    ('2017-19 (base)', ('2017-01-01', '2019-12-01'), True),
    ('2015-19', ('2015-01-01', '2019-12-01'), True),
    ('2013-19', ('2013-01-01', '2019-12-01'), True),
    ('2018-19', ('2018-01-01', '2019-12-01'), True),
    ('2010-19', ('2010-01-01', '2019-12-01'), True),
    ('2017-19, no month effects', ('2017-01-01', '2019-12-01'), False),
]
REPORT_KAPPAS = [0.0, 0.5, 0.8, 1.0]
ALL_Q = [str(p) for p in pd.period_range('2019Q1', '2022Q4', freq='Q')]


def banxico(window, month_effects=True):
    """As run_e1_job_loss.banxico() with a given window; optionally without month effects."""
    bx = pd.read_parquet(abm.BANXICO)
    bx['date'] = pd.to_datetime(bx.date)
    t = bx.set_index('date').transactions

    def design(idx):
        cols = [(idx.year - 2019) * 12 + idx.month]
        cols += [(idx.month == m).astype(float) for m in range(1, 13)] if month_effects else [np.ones(len(idx))]
        return np.column_stack(cols)
    pre = t.loc[window[0]:window[1]]
    coef, *_ = np.linalg.lstsq(design(pre.index), np.log(pre.to_numpy()), rcond=None)
    resid = np.log(pre.to_numpy()) - design(pre.index) @ coef
    pre_q = np.log(hm.quarterly(pre) / hm.quarterly(pd.Series(np.exp(design(pre.index) @ coef), index=pre.index)))
    trend = pd.Series(np.exp(design(hm.DATES) @ coef), index=hm.DATES)
    info = dict(window=list(window), month_effects=month_effects, growth_per_year=float(np.expm1(12 * coef[0])),
                residual_sd_monthly=float(resid.std(ddof=len(coef))),
                residual_sd_quarterly=float(pre_q.std(ddof=1)))
    return t.reindex(hm.DATES), trend, info


def rel19(s):
    s = hm.quarterly(s)
    base = s[[q for q in s.index if q.startswith('2019')]]
    return s / base.set_axis([1, 2, 3, 4]).loc[pd.PeriodIndex(s.index, freq='Q').quarter].to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # check the generalized trend against run_e1_job_loss.banxico()
    bx0, tr0, _ = e1.banxico()
    _, tr1, _ = banxico(e1.TREND_WINDOW)
    assert np.max(np.abs(tr1.to_numpy() / tr0.reindex(hm.DATES).to_numpy() - 1)) < 1e-12

    # model side (target-free): E0 fits, receipt paths, senders
    pop = hl.load_population()
    h = pop['h']
    ok = e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    base = hm.Cells(pop, abm.make_inputs(abm.baseline_2019_dates(), pop['rate_columns']))
    x_by_kappa, start = {}, None
    for kappa in hm.KAPPAS:
        x, _, _, _ = hm.fit_2019(base, y, ok, kappa, start)
        start = x_by_kappa[kappa] = x
    print('E0 done', flush=True)
    covid = abm.make_inputs(hm.DATES, pop['rate_columns'])
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in hm.DATES]), pop['rate_columns'])
    cc, cfc = hm.Cells(pop, covid), hm.Cells(pop, cf)
    need = hm.need_stock(hm.DATES)
    snd_c, snd_f = hm.Senders(pop, covid, need), hm.Senders(pop, cf, np.zeros(len(hm.DATES)))
    r_c0, r_f = snd_c.r(0.0), snd_f.r(0.0)
    P = {k: (cc.receipt(x_by_kappa[k], k), cfc.receipt(x_by_kappa[k], k)) for k in hm.KAPPAS}
    r_cache = {}

    def r_c(beta):
        key = round(float(beta), 12)
        if key not in r_cache:
            r_cache[key] = snd_c.r(beta)
        return r_cache[key]

    S = lambda z: pd.Series(z.sum(1), index=hm.DATES)

    def monthly(kappa, beta, lower=False):
        Pc, Pf = P[kappa]
        rc = r_c(beta)
        return S(Pc * rc) / S(Pf * r_f) if lower else S(Pc * rc / r_c0) / S(Pf)

    def qindex(kappa, beta, lower=False):
        Pc, Pf = P[kappa]
        rc = r_c(beta)
        if lower:
            return hm.quarterly(S(Pc * rc)) / hm.quarterly(S(Pf * r_f))
        return hm.quarterly(S(Pc * rc / r_c0)) / hm.quarterly(S(Pf))

    def fit_beta(kappa, target, lower=False):
        res = minimize_scalar(lambda b: hm.log_rmse(qindex(kappa, b, lower), target, hm.FIT_Q), bounds=(0.0, hm.BETA_MAX),
                              method='bounded', options=dict(xatol=1e-4))
        return float(res.x), float(res.fun)

    a_core = hm.m1_alphas()[0]
    summary, paths, infos, decomp_rows = [], [], {}, []
    for name, window, month_eff in VARIANTS:
        bx, trend, info = banxico(window, month_eff)
        infos[name] = info
        target = hm.quarterly(bx) / hm.quarterly(trend)
        spike_bx = hm.spikes((bx / trend).to_numpy())
        fits = {}
        for kappa in hm.KAPPAS:
            fits[kappa] = fit_beta(kappa, target)
        nf = pd.DataFrame([dict(kappa=k, beta=b, rmse_log_quarterly=r) for k, (b, r) in fits.items()])
        if name.endswith('(base)'):  # reproduce need_fit.csv
            ref = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')
            rows_chk = []
            for k in hm.KAPPAS:
                b = fits[k][0]
                q = qindex(k, b)
                rows_chk.append([b, fits[k][1], hm.log_rmse(q, target, hm.HOLD_Q), q['2020Q2'], q['2021Q4'],
                                 q[hm.HOLD_Q].mean(), expit(a_core + b)])
            cols = ['beta', 'rmse_log_quarterly', 'rmse_log_quarterly_holdout_2022', 'index_2020q2', 'index_2021q4',
                    'mean_index_2022', 'theta_core_2021_12']
            diff = float(np.max(np.abs(np.array(rows_chk) - ref[cols].to_numpy())))
            print(f'reproduction of need_fit.csv: max abs diff {diff:.2e}', flush=True)
            assert diff < 1e-8, diff
            infos[name]['need_fit_max_abs_diff'] = diff
        best_kappa = float(nf.loc[nf.rmse_log_quarterly.idxmin(), 'kappa'])
        # decomposition at the best kappa and its continuous beta
        raw, tr = rel19(bx), rel19(trend)
        for kappa in sorted(set(REPORT_KAPPAS + [best_kappa])):
            beta, rmse = fits[kappa]
            q, q0, ql = qindex(kappa, beta), qindex(kappa, 0.0), qindex(kappa, beta, lower=True)
            bl, rl = fit_beta(kappa, target, lower=True)
            m, ml = monthly(kappa, beta), monthly(kappa, beta, lower=True)
            sp, spl = hm.spikes(m.to_numpy()), hm.spikes(ml.to_numpy())
            lt, l0, lb = np.log(target), np.log(q0), np.log(q)
            dec = pd.DataFrame({'raw': np.log(raw), 'trend': np.log(tr), 'job_loss': l0, 'need': lb - l0})
            dec['unexplained'] = dec.raw - dec.trend - dec.job_loss - dec.need
            row = dict(variant=name, kappa=kappa, is_best_kappa=kappa == best_kappa,
                       trend_growth_per_year=info['growth_per_year'], beta=beta, rmse_fit=rmse,
                       rmse_fit_beta0=hm.log_rmse(q0, target, hm.FIT_Q),
                       rmse_holdout_2022=hm.log_rmse(q, target, hm.HOLD_Q),
                       index_2020q2=float(q['2020Q2']), index_2021q4=float(q['2021Q4']),
                       mean_index_2022=float(q[hm.HOLD_Q].mean()),
                       index_2021q4_beta0=float(q0['2021Q4']),
                       gap_closed_2021q4=float((lb - l0)['2021Q4'] / (lt - l0)['2021Q4']),
                       gap_closed_rmse=1 - rmse / hm.log_rmse(q0, target, hm.FIT_Q),
                       theta_core_2021_12=float(expit(a_core + beta)),
                       spike_2021_01=sp['2021-01'], spike_2021_03=sp['2021-03'],
                       spike_banxico_2021_01=spike_bx['2021-01'], spike_banxico_2021_03=spike_bx['2021-03'],
                       spike_rmse=float(np.sqrt(np.mean([(sp[k] - spike_bx[k]) ** 2 for k in hm.SPIKE_MONTHS]))),
                       rmse_lower_bound=hm.log_rmse(ql, target, hm.FIT_Q),
                       spike_rmse_lower_bound=float(np.sqrt(np.mean([(spl[k] - spike_bx[k]) ** 2 for k in hm.SPIKE_MONTHS]))),
                       beta_lower_bound_refit=bl, rmse_lower_bound_refit=rl)
            for qq in ['2020Q2', '2021Q4']:
                for c in dec.columns:
                    row[f'decomp_{qq}_{c}'] = float(dec.loc[qq, c])
            row.update({f'target_{qq}': float(target[qq]) for qq in ALL_Q if qq >= '2020Q1'})
            summary.append(row)
            paths.append(pd.DataFrame({'variant': name, 'kappa': kappa, 'beta': beta, 'quarter': ALL_Q,
                                       'index_banxico': target[ALL_Q].values, 'index_model': q[ALL_Q].values,
                                       'index_model_beta0': q0[ALL_Q].values,
                                       'index_model_lower_bound': ql[ALL_Q].values,
                                       **{f'decomp_{c}': dec.loc[ALL_Q, c].values for c in dec.columns}}))
        if name.endswith('(base)'):  # base decomposition at the grid-best cell (kappa 1, beta 0.65)
            ref = pd.read_csv(RL / 'outputs/decomposition.csv').query("model == 'Household model'").set_index('quarter')
            k_, b_ = float(ref.kappa.iloc[0]), float(ref.beta.iloc[0])
            l0, lb = np.log(qindex(k_, 0.0)), np.log(qindex(k_, b_))
            un = np.log(raw) - np.log(tr) - lb
            d = float(np.max(np.abs(un[ref.index] - ref.unexplained)))
            print(f'base decomposition (grid cell kappa {k_}, beta {b_}): max abs diff in unexplained {d:.1e} '
                  '(file rounded to 5 dp)', flush=True)
            infos[name]['decomposition_max_abs_diff'] = d
        print(name, info['growth_per_year'], 'best kappa', best_kappa, fits[best_kappa], flush=True)

    summary = pd.DataFrame(summary)
    base = summary[summary.variant == VARIANTS[0][0]].set_index('kappa')
    for c in ['beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2', 'index_2021q4', 'mean_index_2022',
              'spike_rmse', 'rmse_lower_bound', 'gap_closed_2021q4', 'decomp_2021Q4_unexplained']:
        summary[f'base_{c}'] = summary.kappa.map(base[c])
    summary.to_csv(OUT / 'summary.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(OUT / 'quarterly_paths.csv', index=False)
    settings = dict(task='trend_window', model='base household model (run_household_model.py), unchanged',
                    target='Banxico transactions / log-linear trend (with or without calendar-month effects) fitted '
                           'on the window, quarterly ratio of sums; fit 2020Q2..2021Q4, 2022 held out',
                    beta='fitted continuously per kappa (bounded, [0, 3]) against each target, as need_fit.csv',
                    best_kappa='lowest fit RMSE over kappa 0..1 (step 0.1)',
                    gap_closed_2021q4='(log model(beta) - log model(0)) / (log Banxico - log model(0)) in 2021Q4',
                    gap_closed_rmse='1 - RMSE(beta) / RMSE(beta = 0)',
                    lower_bound='rmse_lower_bound at the base-mapping beta; *_refit: beta refitted under lower bound',
                    decomposition='raw Banxico vs same quarter of 2019 = trend + job loss (beta = 0) + need '
                                  '(beta vs beta = 0) + unexplained, log points, at each row kappa and its beta',
                    spikes='Banxico spikes from each variant monthly index (bx / trend)',
                    distribution='skipped', variants=infos)
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2))
    pd.set_option('display.width', 250)
    cols = ['variant', 'kappa', 'trend_growth_per_year', 'beta', 'rmse_fit', 'rmse_holdout_2022', 'index_2020q2',
            'target_2020Q2', 'index_2021q4', 'target_2021Q4', 'gap_closed_2021q4', 'spike_rmse', 'rmse_lower_bound',
            'decomp_2021Q4_trend', 'decomp_2021Q4_need', 'decomp_2021Q4_unexplained']
    print(summary[cols].round(4).to_string(index=False))
    figure(summary)


def figure(summary):
    spec = importlib.util.spec_from_file_location('rl_make_figures', RL / 'make_figures.py')
    rmf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rmf)
    mf, plt = rmf.mf, rmf.plt
    mf.style()
    best = summary[summary.is_best_kappa].reset_index(drop=True)
    labels = [f"{v}\n({g:+.1%}/yr, κ={k:g})" for v, g, k in zip(best.variant, best.trend_growth_per_year, best.kappa)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(best))
    ax = axes[0]
    for kappa, c in [(0.0, mf.ORANGE), (1.0, mf.BLUE)]:
        d = summary[summary.kappa == kappa].set_index('variant').loc[best.variant]
        ax.plot(x, d.beta, marker='o', color=c, label=f'κ = {kappa:g}')
    ax.set_xticks(x, labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('β fitted (logit rise of sending at end-2021)')
    ax.legend(fontsize=8)
    ax = axes[1]
    parts = [('decomp_2021Q4_trend', 'Trend', mf.GRAY_LIGHT), ('decomp_2021Q4_job_loss', 'Job loss', mf.ORANGE),
             ('decomp_2021Q4_need', 'Need', mf.BLUE), ('decomp_2021Q4_unexplained', 'Unexplained', mf.AQUA)]
    pos, neg = np.zeros(len(best)), np.zeros(len(best))
    for key, nm, c in parts:
        v = best[key].to_numpy() * 100
        ax.bar(x, v, bottom=np.where(v >= 0, pos, neg), width=0.7, color=c, edgecolor=mf.SURFACE, label=nm)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(x, best.decomp_2021Q4_raw * 100, 'o', color=mf.INK, label='Banxico vs 2019Q4')
    ax.axhline(0, color=mf.AXIS, lw=1)
    ax.set_xticks(x, labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('2021 Q4, log points × 100')
    ax.legend(fontsize=7.5, ncol=1, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.suptitle('Trend window: β and the 2021 Q4 decomposition (household model, best κ per window)', x=0.01,
                 ha='left', fontsize=12, fontweight='semibold', color=mf.INK)
    fig.savefig(OUT.parent / 'trend_window.png', dpi=mf.DPI, bbox_inches='tight')


if __name__ == '__main__':
    main()
