"""Robustness 6f: need series and persistence around the base need specification (29 Oct 2026).

Run from the repo root: python network-compatibility-model/recipient-level/robustness/need_series_decay.py  (~5-10 min)

Base (run_household_model.py): need option 6a, theta_j,t = expit(alpha_j + beta N*_t) for all migrants,
N*_t cumulative excess deaths (series E, lambda = 1), base count mapping. This script reuses
need_margins.Experiment with need_margins.KAPPAS patched to {0, 0.7, 0.8, 1} in this process only
(need_margins outputs are not rewritten) and refits the slope of each row at each kappa
(bounded scalar search, xatol 1e-4, as run_household_model need_fit). The base row is evaluated at the
beta in outputs/household-model/need_fit.csv, so it matches that file exactly; the script asserts it.

Rows: base (6a, E, 1); 6a S cumulative; 6a L cumulative; 6a E lambda 0.95 / 0.9 / 0 (flow); 6b E cumulative;
6a lower-bound mapping (E cumulative); household shift H (E cumulative); placebo R ramp (lambda 1) and step
(lambda 0) under 6a.
Implied size: theta at Dec 2021 = expit(alpha + beta N*_Dec2021) with M1 alpha_core, alpha_kin
(outputs/fits.csv); H leaves theta unchanged and shifts the receipt logit instead.
Outputs (aggregates only): outputs/robustness/need_series_decay/{summary.csv, quarterly_paths.csv,
README.md, quarterly_paths.png}.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL))
import need_margins as nm

nm.KAPPAS = [0.0, 0.7, 0.8, 1.0]       # patched in this process only
KAPPAS = nm.KAPPAS
OUT = RL / 'outputs/robustness/need_series_decay'
DATES = nm.DATES

ROWS = [  # (label, margin, series, lambda)
    ('base: 6a, E cumulative', '6a', 'E', 1.0),
    ('6a, S cumulative', '6a', 'S', 1.0),
    ('6a, L cumulative', '6a', 'L', 1.0),
    ('6a, E lambda 0.95', '6a', 'E', 0.95),
    ('6a, E lambda 0.9', '6a', 'E', 0.9),
    ('6a, E flow (lambda 0)', '6a', 'E', 0.0),
    ('6b kin only, E cumulative', '6b', 'E', 1.0),
    ('6a lower-bound mapping, E cumulative', '6a_lb', 'E', 1.0),
    ('H household shift, E cumulative', 'H', 'E', 1.0),
    ('placebo: 6a, R ramp (lambda 1)', '6a', 'R', 1.0),
    ('placebo: 6a, R step (lambda 0)', '6a', 'R', 0.0),
]


def fit(ex, margin, kappa, N):
    f = lambda b: ex.summary(margin, kappa, b, N)[0]['rmse_fit']
    res = minimize_scalar(f, bounds=(0.0, nm.BETA_MAX), method='bounded', options=dict(xatol=1e-4))
    return float(res.x) if res.fun < f(0.0) else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m1 = pd.read_csv(RL / 'outputs/fits.csv').query("model == 'M1'").set_index('parameter').estimate
    a_core, a_kin = m1['alpha_core'], m1['alpha_kin']
    need_fit = pd.read_csv(RL / 'outputs/household-model/need_fit.csv')
    ex = nm.Experiment()
    bx = ex.target
    dec21 = np.where(DATES == pd.Timestamp('2021-12-01'))[0][0]
    zero = np.zeros(len(DATES))
    none = {k: ex.summary('H', k, 0.0, zero)[0] for k in KAPPAS}
    rows, paths = [], []
    for label, margin, code, lam in ROWS:
        N = nm.need_stock(code, lam)
        for kappa in KAPPAS:
            if label.startswith('base'):
                ref = need_fit[np.isclose(need_fit.kappa, kappa)].iloc[0]
                beta = float(ref.beta)
            else:
                beta = fit(ex, margin, kappa, N)
            s, q = ex.summary(margin, kappa, beta, N)
            if label.startswith('base'):
                for a, b in [(s['rmse_fit'], ref.rmse_log_quarterly),
                             (s['rmse_holdout_2022'], ref.rmse_log_quarterly_holdout_2022),
                             (s['index_2021q4'], ref.index_2021q4), (s['mean_index_2022'], ref.mean_index_2022)]:
                    assert abs(a - b) < 1e-8, (kappa, a, b)
            gap = np.log(bx['2021Q4']) - np.log(none[kappa]['index_2021q4'])
            closed = (np.log(s['index_2021q4']) - np.log(none[kappa]['index_2021q4'])) / gap
            shift = beta * N[dec21]
            six = margin.startswith('6')
            rows.append(dict(row=label, margin=margin, series=code, lam=lam, kappa=kappa, slope=beta,
                             rmse_fit=s['rmse_fit'], rmse_holdout_2022=s['rmse_holdout_2022'],
                             index_2020q2=s['index_2020q2'], index_2021q4=s['index_2021q4'],
                             mean_index_2022=s['mean_index_2022'], spike_2021_01=s['spike_2021_01'],
                             spike_2021_03=s['spike_2021_03'], spike_rmse=s['spike_rmse'],
                             gap_closed_2021q4=closed, N_dec2021=N[dec21], logit_shift_dec2021=shift,
                             theta_core_2019=expit(a_core), theta_kin_2019=expit(a_kin),
                             theta_core_2021_12=expit(a_core + shift * (six and not margin.startswith('6b'))),
                             theta_kin_2021_12=expit(a_kin + shift * six),
                             beta_at_bound=beta > nm.BETA_MAX - 0.01))
            paths.append(pd.DataFrame(dict(row=label, kappa=kappa, quarter=q.index, index_model=q.values,
                                           index_banxico=bx[q.index].values)))
            print(f'{label:40s} kappa={kappa:.1f} beta={beta:.3f} rmse={s["rmse_fit"]:.4f}', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'summary.csv', index=False)
    qp = pd.concat(paths, ignore_index=True)
    qp.to_csv(OUT / 'quarterly_paths.csv', index=False)

    # cross-check against need_margins fits.csv where it has the cell
    old = pd.read_csv(RL / 'outputs/need-margins/fits.csv')
    m = res.merge(old, on=['margin', 'series', 'lam', 'kappa'], suffixes=('', '_nm'))
    xdiff = float((m.rmse_fit - m.rmse_fit_nm).abs().max())
    bdiff = float((m.slope - m.beta).abs().max())
    print(f'vs need_margins fits.csv: {len(m)} cells, max |rmse diff| {xdiff:.1e}, max |beta diff| {bdiff:.1e}')

    figure(qp)
    readme(res, none, ex, len(m), xdiff, bdiff)


def figure(qp, kappa=1.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [('base: 6a, E cumulative', 'base: 6a, E cumulative', '#2a78d6', '-'),
            ('6a, S cumulative', 'S cumulative', '#eb6834', '--'),
            ('6a, E lambda 0.9', 'E, decay 0.9', '#1baf7a', '-.'),
            ('6a, E flow (lambda 0)', 'E flow', '#eda100', ':'),
            ('placebo: 6a, R ramp (lambda 1)', 'placebo ramp', '#e87ba4', (0, (5, 1, 1, 1)))]
    d = qp[np.isclose(qp.kappa, kappa)]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    b = d[d.row == show[0][0]]
    x = np.arange(len(b))
    ax.plot(x, b.index_banxico, color='#222222', lw=2.2, marker='o', ms=4, label='Banxico (over 2017-19 trend)')
    for key, lab, col, ls in show:
        r = d[d.row == key]
        ax.plot(x, r.index_model.values, color=col, lw=2, ls=ls, label=lab)
    ax.axhline(1, color='#bbbbbb', lw=0.8, zorder=0)
    qs = b.quarter.tolist()
    fit_end = qs.index('2021Q4')
    ax.axvspan(fit_end + 0.5, len(qs) - 0.5, color='#eeeeee', zorder=-1)
    ax.text(fit_end + 0.6, ax.get_ylim()[0] + 0.005, '2022 holdout', fontsize=8, color='#555555')
    ax.set_xticks(x[::2]); ax.set_xticklabels(qs[::2], rotation=45, fontsize=8)
    ax.set_ylabel('count index (COVID / no-COVID)')
    ax.set_title(f'Need series and persistence under 6a, kappa = {kappa:g}: quarterly paths', fontsize=10)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#eeeeee', lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'quarterly_paths.png')


def readme(res, none, ex, ncheck, xdiff, bdiff):
    def tab(k):
        d = res[np.isclose(res.kappa, k)]
        lines = ['| row | slope | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | 2022 mean | spikes Jan / Mar | spike RMSE | gap closed |',
                 '|---|---|---|---|---|---|---|---|---|---|']
        for r in d.itertuples():
            lines.append(f'| {r.row} | {r.slope:.2f} | {r.rmse_fit:.4f} | {r.rmse_holdout_2022:.4f} | {r.index_2020q2:.3f} | '
                         f'{r.index_2021q4:.3f} | {r.mean_index_2022:.3f} | {100*r.spike_2021_01:+.1f} / {100*r.spike_2021_03:+.1f} | '
                         f'{100*r.spike_rmse:.1f} | {100*r.gap_closed_2021q4:.0f}% |')
        n = none[k]
        lines.append(f'| *no need (beta = 0)* | 0 | {n["rmse_fit"]:.4f} | {n["rmse_holdout_2022"]:.4f} | {n["index_2020q2"]:.3f} | '
                     f'{n["index_2021q4"]:.3f} | {n["mean_index_2022"]:.3f} | {100*n["spike_2021_01"]:+.1f} / {100*n["spike_2021_03"]:+.1f} | '
                     f'{100*n["spike_rmse"]:.1f} | 0% |')
        return '\n'.join(lines)

    sz = res[res.row.isin(['base: 6a, E cumulative', '6b kin only, E cumulative', 'H household shift, E cumulative'])]
    size = ['| row | kappa | slope | core theta 2019 -> Dec 2021 | kin theta 2019 -> Dec 2021 |', '|---|---|---|---|---|']
    for r in sz[sz.kappa.isin([0.8, 1.0])].itertuples():
        if r.margin == 'H':
            size.append(f'| {r.row} | {r.kappa:g} | {r.slope:.2f} | unchanged (receipt logit +{r.slope:.2f}) | unchanged |')
        else:
            size.append(f'| {r.row} | {r.kappa:g} | {r.slope:.2f} | {100*r.theta_core_2019:.1f}% -> {100*r.theta_core_2021_12:.1f}% | '
                        f'{100*r.theta_kin_2019:.1f}% -> {100*r.theta_kin_2021_12:.1f}% |')
    sp = ex.spike_bx
    txt = f"""# Robustness 6f: need series and persistence

Script: `recipient-level/robustness/need_series_decay.py`. Reuses `need_margins.Experiment` with `KAPPAS` patched to
{{0, 0.7, 0.8, 1}} in-process; slopes refitted per row and kappa (bounded search, xatol 1e-4). The base row uses the
slope in `outputs/household-model/need_fit.csv` and reproduces that file (asserted). Cross-check with
`outputs/need-margins/fits.csv` on the {ncheck} shared cells: max |RMSE diff| {xdiff:.1e}, max |slope diff| {bdiff:.1e}.

Banxico: 2021Q4 {ex.target['2021Q4']:.3f}, 2022 mean {ex.target[nm.HOLD_Q].mean():.3f}, spikes
{100*sp['2021-01']:+.1f} / {100*sp['2021-03']:+.1f} log points. Gap closed = share of the no-need model's 2021Q4 log gap.
Spikes and spike RMSE in log points. Full table (all kappa, implied thetas) in `summary.csv`, paths in
`quarterly_paths.csv`, figure `quarterly_paths.png` (kappa = 1).

## kappa = 1 (base best)

{tab(1.0)}

## kappa = 0.8

{tab(0.8)}

## kappa = 0.7 and 0

See `summary.csv`.

## Implied size at Dec 2021 (M1 alpha_core {nm.pd.read_csv(RL / 'outputs/fits.csv').query("model == 'M1' and parameter == 'alpha_core'").estimate.iloc[0]:.4f}, alpha_kin {nm.pd.read_csv(RL / 'outputs/fits.csv').query("model == 'M1' and parameter == 'alpha_kin'").estimate.iloc[0]:.3f})

{chr(10).join(size)}
"""
    (OUT / 'README.md').write_text(txt, encoding='utf-8')


if __name__ == '__main__':
    main()
