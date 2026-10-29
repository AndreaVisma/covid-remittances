"""Figures for the recipient-level (household) receipt logit, H3 fit.

Run: python network-compatibility-model/recipient-level/make_figures.py
(after compare_household_logit.py). Style and colours from model/make_figures.py.

S1 (concept): the fitted curve P = expit(score) with illustrative households (made-up compositions,
   not MMP households) and arrows for one migrant losing their job.
S2 (data; --not-working-as-unknown draws S2 for that variant fit): every sample household at its expected 2019 score (able counts averaged over 2019 CPS
   employment), drawn on the curve and sized by the number of households with that score, with
   observed MMP receipt by score band (aggregates, bands with >= 30 households) and a histogram.
S3-S8 (--model, after run_household_model.py): the household model through E0-E2 and the
   distribution: S3 kappa in 2019, S4 receipt index vs Banxico (beta = 0), S5 beta x kappa map,
   S6 compatible paths, S7 realized losses, S8 who loses, S9 count decomposition, S10 USD accounting
   (counts x observed average amount, split into US inflation and the real amount; amounts are
   accounting only, not modelled: user decision, 29 Oct 2026).
Score = intercept + any_core * [A_c > 0] + log_core * log(1 + A_c) + log_kin * log(1 + A_k)
        + no_worker * (no worker - mean) + any_older * (older resident - mean).
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))
import compare_household_logit as hl
import run_e1_job_loss as e1
# The model's make_figures.py shares this file's name, so load it by path.
_spec = importlib.util.spec_from_file_location('model_make_figures', HERE.parent / 'model/make_figures.py')
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

FIG = HERE / 'figures'
NET_COLORS = {'Core only': mf.BLUE, 'Core + kin': mf.ORANGE, 'Kin only': mf.AQUA}
MIN_BAND_N = 30
CAPTIONS = {}


def h3_coefficients(outputs=HERE / 'outputs'):
    f = pd.read_csv(outputs / 'fits.csv').query("model == 'H3'")
    return dict(zip(f.parameter, f.estimate))


def score(b, core, kin, no_worker, older):
    """no_worker, older: centred values."""
    return (b['intercept'] + b['any_core'] * (core > 0) + b['log_core'] * np.log1p(core)
            + b['log_kin'] * np.log1p(kin) + b['no_worker'] * no_worker + b['any_older'] * older)


def save(fig, name, title, caption):
    fig.suptitle(title, x=0.01, ha='left', fontsize=12.5, fontweight='semibold', color=mf.INK)
    fig.savefig(FIG / f'{name}.png', dpi=mf.DPI, bbox_inches='tight')
    plt.close(fig)
    CAPTIONS[name] = (title, caption)
    print('saved', name)


def curve(ax, x):
    ax.plot(x, expit(x), color=mf.GRAY_LIGHT, lw=2.5, zorder=1, solid_capstyle='round')
    mf.pct_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Probability of receiving in a month')


# %% S1 concept
def s1_concept(b, need_mean):
    w, o = need_mean
    # label, network type, (core, kin, no worker, older), label position (data), job loss (core, kin) or None
    examples = [
        ('Two cousins in the US;\nworkers at home', 'Kin only', (0, 2, 0, 0), (-4.4, 0.21), None),
        ('Two cousins; no worker,\na grandparent at home', 'Kin only', (0, 2, 1, 1), (-1.15, 0.05), None),
        ('One son in the US; workers at home', 'Core only', (1, 0, 0, 0), (0.4, 0.36),
         ((0, 0), 'he loses his job')),
        ('Three children and two cousins;\nno worker, a grandparent', 'Core + kin', (3, 2, 1, 1), (-0.9, 0.88),
         ((2, 2), 'one child loses their job')),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.linspace(-4.5, 2.5, 300)
    curve(ax, x)
    for label, net, (c, k, nw, ol), pos, loss in examples:
        s0 = score(b, c, k, nw - w, ol - o)
        text = f'{label}: {expit(s0):.0%}'
        if loss is not None:
            (c1, k1), what = loss
            s1 = score(b, c1, k1, nw - w, ol - o)
            ax.annotate('', (s1, expit(s1)), (s0, expit(s0)),
                        arrowprops=dict(arrowstyle='-|>', color=mf.RED, lw=1.6, shrinkA=6, shrinkB=5,
                                        connectionstyle='arc3,rad=0.25'))
            ax.scatter(s1, expit(s1), s=40, facecolor=mf.SURFACE, edgecolor=mf.RED, linewidth=1.5, zorder=5)
            text += f'\n{what}: {expit(s1):.0%}'
        mf.ring_dot(ax, s0, expit(s0), NET_COLORS[net], size=70, zorder=6)
        ax.annotate(text, (s0, expit(s0)), xytext=pos, textcoords='data', fontsize=8, color=mf.INK2,
                    va='center', arrowprops=dict(arrowstyle='-', color=mf.MUTED, lw=0.7, shrinkA=2, shrinkB=6))
    ax.set_xlabel('Household score (log-odds of receiving)')
    ax.set_xlim(x[0], x[-1])
    handles = [Line2D([], [], marker='o', ls='', color=c, markeredgecolor=mf.SURFACE, markersize=8, label=n)
               for n, c in NET_COLORS.items()]
    handles.append(Line2D([], [], color=mf.RED, lw=1.6, marker='>', markersize=5, label='One migrant loses their job'))
    ax.legend(handles=handles, loc='upper left', frameon=False)
    terms = (f"score = {b['intercept']:.2f} + {b['any_core']:.2f}·[any able core migrant] "
             f"+ {b['log_core']:.2f}·log(1 + able core) + {b['log_kin']:.2f}·log(1 + able kin)\n"
             f"            + {b['no_worker']:.2f}·no worker at home + {b['any_older']:.2f}·older resident   "
             f"(need terms centred)")
    ax.text(0.0, -0.17, terms, transform=ax.transAxes, fontsize=7.5, color=mf.MUTED, va='top')
    save(fig, 'S1_receipt_curve_examples', 'Where households sit on the receipt curve',
         'Fitted household logit (H3, MMP 2019, maximum likelihood). Illustrative compositions, not MMP '
         'households, with all migrants working. The first able core migrant moves a household from the '
         'flat bottom of the curve to its upper half; further core migrants and kin add little. Red '
         'arrows: one migrant losing their job. Losing an only core migrant is a large drop; losing one '
         'of three is small.')


# %% S2 data
def s2_households(b, pop, suffix='', note=''):
    h = pop['h']
    ok = hl.e0.target_mask(h)
    y = h.remus.eq(1).to_numpy(float)
    inputs = hl.abm.make_inputs(hl.abm.baseline_2019_dates(), pop['rate_columns'])
    pmfs = hl.able_pmfs(pop, inputs)
    need = pop['need_x']
    exp_score, fitted = np.zeros(pop['H']), np.zeros(pop['H'])
    for i, (pc, pk) in enumerate(pmfs):
        pc, pk = pc.mean(0), pk.mean(0)          # 2019 average pmfs
        a, k = np.arange(len(pc)), np.arange(len(pk))
        S = score(b, a[:, None], k[None, :], need[i, 0], need[i, 1])
        W = np.outer(pc, pk)
        exp_score[i], fitted[i] = np.sum(W * S), np.sum(W * expit(S))
    d = pd.DataFrame({'score': exp_score, 'fitted': fitted, 'y': y, 'net': h.network_type})[ok]

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.1], hspace=0.08))
    x = np.linspace(d.score.min() - 0.3, d.score.max() + 0.3, 300)
    curve(ax, x)
    g = d.assign(sx=d.score.round(2)).groupby(['net', 'sx']).size().rename('n').reset_index()
    handles = []
    for net, c in NET_COLORS.items():
        t = g[g.net == net]
        ax.scatter(t.sx, expit(t.sx), s=6 + t.n, color=c, alpha=0.55, edgecolor=mf.SURFACE,
                   linewidth=0.6, zorder=3)
        handles.append(Line2D([], [], marker='o', ls='', color=c, alpha=0.7, markersize=8,
                              label=f'{net} ({int(t.n.sum()):,} households)'))
    edges = np.r_[-np.inf, np.arange(-3.0, 2.01, 0.5), np.inf]
    bands = d.assign(band=pd.cut(d.score, edges)).groupby('band', observed=True).agg(
        n=('y', 'size'), mmp=('y', 'mean'), sx=('score', 'mean'), fit=('fitted', 'mean'))
    bands = bands[bands.n >= MIN_BAND_N]
    lo, hi = mf.wilson(bands.mmp, bands.n)
    ax.errorbar(bands.sx, bands.mmp, yerr=[bands.mmp - lo, hi - bands.mmp], fmt='none', ecolor=mf.INK,
                elinewidth=1, capsize=0, zorder=6)
    ax.scatter(bands.sx, bands.mmp, s=36, marker='D', color=mf.INK, edgecolor=mf.SURFACE, linewidth=1.2,
               zorder=7)
    handles.append(Line2D([], [], marker='D', ls='', color=mf.INK, markersize=6,
                          label='MMP receipt by score band (95% CI)'))
    ax.legend(handles=handles, loc='upper left', frameon=False, title='Bubble area = households',
              title_fontsize=8, alignment='left')

    bins = np.arange(np.floor(d.score.min() * 4) / 4, d.score.max() + 0.25, 0.25)
    bottom = np.zeros(len(bins) - 1)
    for net, c in NET_COLORS.items():
        n, _ = np.histogram(d.score[d.net == net], bins)
        axh.bar(bins[:-1], n, width=0.23, align='edge', bottom=bottom, color=c, edgecolor=mf.SURFACE, linewidth=0.8)
        bottom += n
    axh.set_ylabel('Households')
    axh.set_xlabel('Expected household score in 2019 (log-odds of receiving)')
    axh.grid(axis='x', visible=False)
    gap = np.abs(bands.fit - expit(bands.sx)).max()
    save(fig, f'S2_receipt_curve_households{suffix}', 'All sample households on the fitted receipt curve' + note,
         f'Fitted household logit (H3). Each bubble is a group of households with the same expected 2019 '
         f'score (able migrant counts averaged over 2019 CPS employment), drawn on the curve and sized '
         f'by the number of households; colour is network type. Diamonds: observed MMP receipt in score '
         f'bands of 0.5 with at least {MIN_BAND_N} households, 95% Wilson intervals. Kin-only households '
         f'cluster at the flat bottom, households with core migrants in the upper half; almost none sit '
         f'on the steep middle. The curve at the expected score differs from the mean fitted probability '
         f'by at most {gap:.1%} per band. {len(d):,} households.')


# %% S3-S8: the household model through E0-E2 and the distribution (run_household_model.py)
HM = HERE / 'outputs/household-model'
HM_KAPPA_COLORS = {0.0: mf.K0, 0.5: mf.K2, 1.0: mf.K4}


def quarter_ticks(ax, quarters):
    x = np.arange(len(quarters))
    ax.set_xticks(x, [q.replace('Q', ' Q') if q.endswith('Q1') else q[-2:] for q in quarters])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    return x


def holdout(ax, quarters):
    hold = quarters.index('2022Q1')
    ax.axvspan(hold - 0.5, len(quarters) - 0.5, color=mf.HOLDOUT_WASH, lw=0, zorder=0)
    return hold


def s3_kappa_2019(prof, fits, settings):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    d = prof.loglik - prof.loglik.max()
    ax.plot(prof.kappa, d, color=mf.BLUE, zorder=3)
    ax.scatter(prof.kappa, d, s=22, color=mf.BLUE, edgecolor=mf.SURFACE, lw=1, zorder=4)
    ax.axhline(-1.92, color=mf.MUTED, lw=1, ls='--')
    ax.text(prof.kappa.max(), -1.92, '95% profile limit', ha='right', va='bottom', fontsize=8, color=mf.INK2)
    kf = settings['kappa_free']
    ax.set_title('Profile log-likelihood', fontsize=10)
    ax.text(0.98, 0.95, f'maximum on the grid: κ = {kf["estimate"]:g}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=mf.INK2)
    ax.set_xlabel('κ (share of the receipt lost to job loss that other migrants replace)')
    ax.set_ylabel('Log-likelihood − maximum')
    ax = axes[1]
    for col, label, c in [('any_core', 'Any able core migrant', mf.BLUE), ('log_kin', 'log(1 + able kin)', mf.AQUA),
                          ('log_core', 'log(1 + able core)', mf.ORANGE)]:
        ax.plot(prof.kappa, prof[col], color=c, label=label)
        mf.end_label(ax, prof.kappa.iloc[-1], prof[col].iloc[-1], f'{prof[col].iloc[-1]:.2f}')
    ax.set_title('Fitted coefficients, refitted for each κ', fontsize=10)
    ax.set_xlabel('κ')
    ax.legend(loc='center left', bbox_to_anchor=(0, 0.62))
    ax.set_xlim(None, prof.kappa.max() * 1.12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, 'S3_kappa_2019', 'In 2019 the household data cannot tell the replacement share κ apart',
         'Household model: H3 receipt s(able core, able kin) plus stepping-in, P = s + κ·[someone can send]·(s if '
         'nobody were out of work − s), κ ∈ [0, 1] the share of the receipt lost to job loss that other migrants '
         'replace. "Not working" migrants as unknown occupation, 2019 MMP receipt, maximum likelihood with κ fixed on '
         'a grid. Left: log-likelihood relative to its maximum; dashed line: the 95% limit of a one-parameter profile '
         '(−1.92). The whole range κ = 0–1 lies within 0.14 points: ordinary unemployment in 2019 is too rare to '
         'identify κ. Right: the other coefficients refitted at each κ.')


def s4_index(paths, migrant_q, grid):
    quarters = sorted(paths.quarter.unique())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = quarter_ticks(ax, quarters)
    hold = holdout(ax, quarters)
    ax.axhline(1, color=mf.MUTED, lw=1, zorder=1)
    mq = migrant_q.set_index('quarter')
    for k, ls in [(0.0, ':'), (1.0, '--')]:
        d = migrant_q[migrant_q.kappa == k].set_index('quarter').reindex(quarters)
        ax.plot(x, d.index_model, color=mf.GRAY_LIGHT, lw=1.5, ls=ls, zorder=2,
                label=f'Migrant model, coordination, κ = {k:g} (transfers)')
    for k, c in HM_KAPPA_COLORS.items():
        d = paths[(paths.kappa == k) & (paths.beta == 0)].set_index('quarter').loc[quarters]
        r = grid[(grid.kappa == k) & (grid.beta == 0)].iloc[0]
        ax.plot(x, d.index_model, color=c, zorder=3,
                label=f'Household model, κ = {k:g} (RMSE {r.rmse_log_quarterly:.3f})')
    b = paths[(paths.kappa == 0) & (paths.beta == 0)].set_index('quarter').loc[quarters].index_banxico
    ax.plot(x, b, color=mf.INK, zorder=4, label='Banxico vs its 2017–19 trend')
    k0 = paths[(paths.kappa == 0) & (paths.beta == 0)].set_index('quarter').loc[quarters].index_model
    i = quarters.index('2020Q2')
    for yv, c, dy in [(k0.iloc[i], mf.K0, -14), (b.iloc[i], mf.INK, 10)]:
        mf.ring_dot(ax, i, yv, c)
        ax.annotate(f'{yv:.3f}', (i, yv), xytext=(0, dy), textcoords='offset points', ha='center', fontsize=8.5,
                    color=mf.INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel('Relative to no-COVID path (quarterly)')
    lo = min(k0.min(), b.min(), migrant_q[migrant_q.kappa == 0].index_model.min())
    ax.set_ylim(lo - 0.03, 1.2)
    ax.text(hold - 0.3, lo - 0.025, '2022 held out', fontsize=8.5, color=mf.INK2, va='bottom')
    ax.legend(loc='upper left', ncol=2, fontsize=8)
    r0 = grid[(grid.kappa == 0) & (grid.beta == 0)].iloc[0]
    save(fig, 'S4_receipt_index', 'Households lose less receipt than the migrant model loses transfers; 2021 is still missed',
         'β = 0, ρ = 0, no benefits (E1a on the household model). Household model: expected number of receiving '
         'households, COVID run / no-COVID run (2019 CPS rates of the same month), exact; read as transfers under '
         'a constant number of transfers per receiving household. κ = replacement share; coefficients refitted to 2019 for '
         'each κ. Gray: the migrant-level model, coordination rule (κ = replacement share; transfer counts, original '
         '"Not working" coding, E1a). Banxico: transactions '
         f'relative to their 2017–19 trend. 2020 Q2 at κ = 0: {r0.index_2020q2:.3f}. 2022 held out.')


def s5_map(grid, settings):
    from matplotlib.colors import LinearSegmentedColormap
    tau = settings['tau_C']
    piv = grid.pivot(index='beta', columns='kappa', values='rmse_log_quarterly').sort_index(ascending=False)
    ok = grid.pivot(index='beta', columns='kappa', values='count_compatible').sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(piv.values, cmap=LinearSegmentedColormap.from_list('seq', mf.SEQ_BLUE), aspect='auto', vmin=0,
                   vmax=min(0.10, piv.values.max()))
    ax.grid(False)
    for r, c in zip(*np.where(ok.values.astype(bool))):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=mf.INK, lw=1.5))
    b = np.unravel_index(np.argmin(piv.values), piv.shape)
    ax.scatter(b[1], b[0], s=50, color=mf.SURFACE, edgecolor=mf.INK, lw=1.5, zorder=5)
    right = b[1] > piv.shape[1] / 2
    ax.annotate(f'best {piv.values[b]:.3f}', (b[1], b[0]), xytext=(-8 if right else 8, 0), textcoords='offset points',
                va='center', ha='right' if right else 'left',
                fontsize=8.5, color=mf.INK, bbox=dict(boxstyle='round,pad=0.2', fc=mf.SURFACE, ec='none', alpha=0.85))
    ax.set_xticks(range(piv.shape[1]), [f'{k:g}' for k in piv.columns])
    ax.set_yticks(range(0, piv.shape[0], 2), [f'{v:.2f}' for v in piv.index[::2]])
    ax.set_xlabel('κ (replacement share of the receipt lost to job loss)')
    ax.set_ylabel('β (logit rise in each migrant’s sending probability, end-2021)')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Quarterly log RMSE, 2020 Q2 – 2021 Q4', color=mf.INK2)
    cb.outline.set_visible(False)
    cb.ax.axhline(tau, color=mf.INK, lw=1.5)
    n = int(ok.values.sum())
    best = grid.loc[grid.rmse_log_quarterly.idxmin()]
    save(fig, 'S5_compatibility_map', f'Household model: {n} of {ok.size} (β, κ) cells fit within τ_C = {tau}',
         f'E2 on the household model (stimulus π as in the base, ρ = 0). Need: more senders per receiving household, '
         f'θ = expit(α + β N*_t), N*_t = cumulative excess deaths in Mexico (1 in Dec 2021). Fit error of the quarterly '
         f'count index against Banxico relative to its 2017–19 trend. Outlined: count-compatible cells (RMSE ≤ '
         f'{tau}). Dot: best cell (κ = {best.kappa:g}, β = {best.beta:.2f}). Coefficients refitted to 2019 for each '
         f'κ. β is on the migrant sending scale (M1 α), and need leaves receipt unchanged.')


def s6_paths(paths, grid):
    quarters = sorted(paths.quarter.unique())
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = quarter_ticks(ax, quarters)
    hold = holdout(ax, quarters)
    ax.axhline(1, color=mf.AXIS, lw=1)
    ok = grid[grid.count_compatible]
    get = lambda k, b: paths[(paths.kappa == k) & (paths.beta == b)].set_index('quarter').loc[quarters]
    for _, c in ok.iterrows():
        ax.plot(x, get(c.kappa, c.beta).index_model, color=mf.K0, lw=1, alpha=0.7, zorder=2)
    best = grid.loc[grid.rmse_log_quarterly.idxmin()]
    d = get(best.kappa, best.beta)
    ax.plot(x, d.index_model, color=mf.K4, zorder=3,
            label=f'Best cell: κ = {best.kappa:g}, β = {best.beta:.2f} (RMSE {best.rmse_log_quarterly:.3f})')
    z = grid[grid.beta == 0].sort_values('rmse_log_quarterly').iloc[0]
    ax.plot(x, get(z.kappa, 0.0).index_model, color=mf.MUTED, lw=1.5, zorder=2,
            label=f'Best with β = 0: κ = {z.kappa:g} (RMSE {z.rmse_log_quarterly:.3f})')
    ax.plot(x, d.index_banxico, color=mf.INK, zorder=4, label='Banxico vs its 2017–19 trend')
    if len(ok) > 1:
        ax.plot([], [], color=mf.K0, lw=1, label=f'Other count-compatible cells ({len(ok) - 1})')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel('Relative to no-COVID path (quarterly)')
    ax.text(hold - 0.3, ax.get_ylim()[0] + 0.003, '2022 held out', fontsize=8.5, color=mf.INK2, va='bottom')
    ax.legend(loc='upper left')
    save(fig, 'S6_compatible_paths', 'Household model: the best cell still follows the death waves',
         'E2 on the household model. Quarterly receipt index of the best cell, the other count-compatible cells '
         'and the best β = 0 cell, against Banxico relative to its 2017–19 trend. 2022 held out.')


def s7_realized(rb, ineq):
    losses = ['Lost everything', 'Lost 50–99%', 'Lost 25–50%', 'Lost 10–25%']
    gains = ['Gained 0–10%', 'Gained 10–25%', 'Gained > 25%']
    kappas = list(HM_KAPPA_COLORS)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axvline(0, color=mf.AXIS, lw=1)
    for i, k in enumerate(kappas):
        d = rb[rb.kappa == k].set_index('bin').share * 100
        y = len(kappas) - 1 - i
        left = 0.0
        for b, c in list(zip(losses + ['Lost 0–10%'], mf.LOSS_RAMP + ['#f3c2c1']))[::-1]:
            ax.barh(y, -d[b], left=-left, height=0.5, color=c, edgecolor=mf.SURFACE, lw=1.5)
            left += d[b]
        right = 0.0
        for b, c in zip(gains, mf.GAIN_RAMP):
            ax.barh(y, d[b], left=right, height=0.5, color=c, edgecolor=mf.SURFACE, lw=1.5)
            right += d[b]
        ax.annotate(f'lost: {left:.0f}% (everything: {d["Lost everything"]:.1f}%)', (-left, y), xytext=(-6, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=8.5, color=mf.INK2)
        ax.annotate(f'gained: {right:.0f}% · unchanged: {d["No change"]:.0f}%', (right, y), xytext=(6, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=8.5, color=mf.INK2)
    ax.set_yticks(range(len(kappas)), [f'κ = {k:g}' for k in kappas[::-1]])
    ax.grid(axis='y', visible=False)
    lim = rb[rb.bin.str.startswith('Lost')].groupby('kappa').share.sum().max() * 100
    ax.set_xlim(-lim * 2.1, lim * 1.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{abs(v):.0f}%'))
    ax.set_xlabel('Share of receiving households in a typical COVID path  (← fewer months · more months →)')
    handles = [Patch(color=c, label=b) for b, c in zip(losses + ['Lost 0–10%'], mf.LOSS_RAMP + ['#f3c2c1'])] + \
              [Patch(color=c, label=b) for b, c in zip(gains, mf.GAIN_RAMP)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8)
    k0 = ineq[ineq.kappa == 0].iloc[0]
    save(fig, 'S7_realized_losses', 'Household model: 3–8% of receiving households lose months; almost none lose every month',
         'β = 0, ρ = 0, no benefits. Each household that receives in its no-COVID twin is compared with it (common '
         'random numbers for employment and the receipt draw), Apr 2020 – Dec 2021: change in the number of months '
         'with receipt, pooled over 200 paths. Unlike the migrant model (H1, change in USD), every household draws its '
         'receipt each month, so luck moves months both ways; the paired draw removes most of it. About '
         f'{k0.realized_receiving_households:,.0f} households receive in a path.')


def s8_groups(g):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    labels, ys, y = [], [], 0
    for col, title, levels in mf.GROUP_ROWS:
        labels.append((y, title, True))
        y += 1
        for lv in levels:
            if ((g.group == col) & (g.level == lv)).any():
                labels.append((y, lv, False))
                ys.append((col, lv, y))
                y += 1
        y += 0.4
    panels = [('change_expected', 'Expected change in months with receipt'),
              ('lose_all_risk', 'Risk of losing every month (share of paths)')]
    offsets = dict(zip(HM_KAPPA_COLORS, [-0.2, 0, 0.2]))
    for ax, (metric, title) in zip(axes, panels):
        if metric == 'change_expected':
            ax.axvline(0, color=mf.AXIS, lw=1)
        for k, c in HM_KAPPA_COLORS.items():
            d = g[g.kappa == k].set_index(['group', 'level'])
            xs = [d.loc[(cc, lv), metric] * 100 for cc, lv, _ in ys]
            ax.scatter(xs, [yy + offsets[k] for _, _, yy in ys], s=34, color=c, edgecolor=mf.SURFACE, lw=1.2,
                       zorder=3, label=f'κ = {k:g}')
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
        ax.grid(axis='y', visible=False)
        if metric != 'change_expected':
            ax.set_xlim(0, None)
    axes[0].set_yticks([l[0] for l in labels], [l[1] if l[2] else mf.OCC_SHORT.get(l[1], l[1]) for l in labels])
    for tick, l in zip(axes[0].get_yticklabels(), labels):
        if l[2]:
            tick.set_fontweight('semibold')
            tick.set_color(mf.INK)
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right')
    save(fig, 'S8_who_loses', 'Household model: single links and core-only networks lose most, whatever κ',
         'β = 0, ρ = 0, no benefits; 200 paired paths, Apr 2020 – Dec 2021. Left: expected change in months with '
         'receipt by group (exact, receipt-weighted). Right: share of paths in which a household that receives in '
         'its no-COVID twin receives in no month, averaged over the group. Main occupation: the most common job '
         'group among the household’s eligible migrants ("Not working" migrants count as Other/none).')


DECOMP_PARTS = [('trend', 'Pre-COVID trend (+6.3% a year, 2017–19)', mf.GRAY_LIGHT),
                ('job_loss', 'US job loss, net of stepping-in (κ)', mf.BLUE),
                ('need', 'Need: more senders, cumulative excess deaths (β)', mf.ORANGE),
                ('unexplained', 'Unexplained', mf.AQUA)]


def decomposition(paths, label):
    """Raw Banxico vs the same quarter of 2019 = trend x job loss x need x unexplained, in log points.
    Job loss: the model at beta = 0 and the best cell's kappa; need: best cell / that path."""
    bx, trend, _ = e1.banxico()
    q = lambda s: s.groupby(s.index.to_period('Q')).sum()
    rel19 = lambda s: (s / s[s.index.year == 2019].set_axis([1, 2, 3, 4]).loc[s.index.quarter].to_numpy()).rename(index=str)
    raw, tr = rel19(q(bx)), rel19(q(trend))
    grid = paths.groupby(['kappa', 'beta']).apply(
        lambda d: np.sqrt(np.mean(np.log(d.set_index('quarter').index_model[FIT_Q]
                                         / d.set_index('quarter').index_banxico[FIT_Q]) ** 2)), include_groups=False)
    kappa, beta = grid.idxmin()
    get = lambda k, b: paths[(paths.kappa == k) & (paths.beta == b)].set_index('quarter').index_model
    best, jl = get(kappa, beta), get(kappa, 0.0)
    quarters = [s for s in best.index if s >= '2020Q1']
    d = pd.DataFrame({'quarter': quarters, 'raw': np.log(raw[quarters]).values, 'trend': np.log(tr[quarters]).values,
                      'job_loss': np.log(jl[quarters]).values, 'need': np.log(best[quarters] / jl[quarters]).values})
    d['unexplained'] = d.raw - d.trend - d.job_loss - d.need
    return d.assign(model=label, kappa=kappa, beta=beta)


FIT_Q = [str(p) for p in pd.period_range('2020Q2', '2021Q4', freq='Q')]


def s9_decomposition(parts):
    fig, axes = plt.subplots(1, len(parts), figsize=(11, 4.6), sharey=True)
    for ax, d in zip(axes, parts):
        quarters = list(d.quarter)
        x = quarter_ticks(ax, quarters)
        holdout(ax, quarters)
        ax.axhline(0, color=mf.AXIS, lw=1)
        pos, neg = np.zeros(len(d)), np.zeros(len(d))
        for key, name, c in DECOMP_PARTS:
            v = d[key].to_numpy() * 100
            bottom = np.where(v >= 0, pos, neg)
            ax.bar(x, v, bottom=bottom, width=0.7, color=c, edgecolor=mf.SURFACE, lw=1, label=name, zorder=2)
            pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
        ax.plot(x, d.raw * 100, color=mf.INK, marker='o', markersize=4, zorder=4, label='Banxico, vs same quarter of 2019')
        for i in [quarters.index('2020Q2'), quarters.index('2021Q4')]:
            ax.annotate(f'{np.expm1(d.raw.iloc[i]):+.0%}', (i, max(d.raw.iloc[i] * 100, pos[i])), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8.5, color=mf.INK2)
        ax.set_title(f'{d.model.iloc[0]} (κ = {d.kappa.iloc[0]:g}, β = {d.beta.iloc[0]:.2f})', fontsize=10)
        ax.tick_params(axis='x', labelsize=7.5)
    axes[0].set_ylabel('Log points × 100 (≈ %)')
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:+.0f}'))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    a = parts[0].set_index('quarter').loc['2021Q4']
    save(fig, 'S9_decomposition', 'Why transfers grew despite job loss: trend, stepping-in and need; the household model leaves a small remainder',
         'Banxico transactions relative to the same quarter of 2019 (line), split in log points: the 2017–19 trend '
         'continued (log-linear with month effects); the model’s job-loss path at β = 0 with stepping-in κ (COVID / '
         'no-COVID run); need, i.e. the senders that cumulative excess deaths bring in (best map cell relative to its '
         'β = 0 path; migrant model: its own need term); and what '
         'remains. κ and β are each model’s best map cell (RMSE against Banxico vs trend, 2020 Q2–2021 Q4). Left: '
         'migrant-level model, coordination rule (κ = replacement share, original "Not working" coding, transfer '
         'counts). Right: household model (κ = replacement share, receipt). 2022 held out (shaded). '
         f'2021 Q4, migrant model: raw {a.raw * 100:+.0f}, trend {a.trend * 100:+.0f}, job loss {a.job_loss * 100:+.1f}, '
         f'need {a.need * 100:+.1f}, unexplained {a.unexplained * 100:+.0f} log points.')


USD_PARTS = DECOMP_PARTS[:-1] + [('unexplained', 'Unexplained (counts)', mf.AQUA),
                                  ('inflation', 'US inflation (CPI-U), on the amount', '#e87ba4'),
                                  ('real_amount', 'Real amount per transfer', '#6250d6')]


def usd_accounting(d):
    """Banxico USD vs the same quarter of 2019 = counts (decomposition d) x average amount per transfer,
    the amount split into US CPI-U inflation and the real amount. Log points."""
    bx = pd.read_parquet(hl.abm.BANXICO)
    bx = bx.assign(date=pd.to_datetime(bx.date)).set_index('date')[['total_usd', 'transactions']]
    cpi = pd.read_csv(HERE.parent / 'data-prep/outputs/us_cpi_monthly.csv', parse_dates=['date']).set_index('date').cpi_u
    q = bx.groupby(bx.index.to_period('Q')).sum().join(cpi.groupby(cpi.index.to_period('Q')).mean())
    rel19 = lambda s: np.log(s / s[s.index.year == 2019].set_axis([1, 2, 3, 4]).loc[s.index.quarter].to_numpy()).rename(index=str)
    avg = rel19(q.total_usd / q.transactions)
    out = d.assign(raw_usd=rel19(q.total_usd)[d.quarter].values, amount=avg[d.quarter].values,
                   inflation=rel19(q.cpi_u)[d.quarter].values)
    out['real_amount'] = out.amount - out.inflation
    assert np.allclose(out.raw + out.amount, out.raw_usd)
    return out


def s10_usd(u):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    quarters = list(u.quarter)
    x = quarter_ticks(ax, quarters)
    holdout(ax, quarters)
    ax.axhline(0, color=mf.AXIS, lw=1)
    pos, neg = np.zeros(len(u)), np.zeros(len(u))
    for key, name, c in USD_PARTS:
        v = u[key].to_numpy() * 100
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=bottom, width=0.7, color=c, edgecolor=mf.SURFACE, lw=1, label=name, zorder=2)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(x, u.raw_usd * 100, color=mf.INK, marker='o', markersize=4, zorder=4, label='Banxico USD, vs same quarter of 2019')
    ax.set_ylabel('Log points × 100 (≈ %)')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:+.0f}'))
    ax.tick_params(axis='x', labelsize=7.5)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    a = u.set_index('quarter').loc['2021Q4']
    b = u.set_index('quarter').loc['2022Q4']
    save(fig, 'S10_usd_accounting', 'USD received: counts drive the lasting rise; the real amount rise of 2020–21 fades by 2022',
         'Banxico USD relative to the same quarter of 2019 (line) = transfer counts × average amount per transfer, '
         'in log points. Counts are split as in S9 (household model, best map cell κ = '
         f'{u.kappa.iloc[0]:g}, β = {u.beta.iloc[0]:.2f}); the observed average amount is split into US CPI-U '
         'inflation and the real amount (2019 USD). Amounts are accounting, not modelled. 2022 held out (shaded). '
         f'2021 Q4: USD {a.raw_usd * 100:+.0f}, counts {a.raw * 100:+.0f}, inflation {a.inflation * 100:+.1f}, '
         f'real amount {a.real_amount * 100:+.1f} log points; 2022 Q4: real amount {b.real_amount * 100:+.1f}.')


def model_figures():
    settings = json.loads((HM / 'run_settings.json').read_text())
    grid = pd.read_csv(HM / 'map.csv')
    paths = pd.read_csv(HM / 'quarterly_paths.csv')
    s3_kappa_2019(pd.read_csv(HM / 'e0_kappa_profile.csv'), pd.read_csv(HM / 'e0_fits.csv'), settings)
    s4_index(paths, pd.read_csv(HERE.parent / 'outputs/e1-coordination/aggregate_quarterly.csv'), grid)
    s5_map(grid, settings)
    s6_paths(paths, grid)
    s7_realized(pd.read_csv(HM / 'distribution_realized_bins.csv'), pd.read_csv(HM / 'distribution_inequality.csv'))
    s8_groups(pd.read_csv(HM / 'distribution_groups.csv', dtype={'level': str}))
    parts = [decomposition(pd.read_csv(HERE.parent / 'outputs/e2-map-coordination/quarterly_paths.csv'),
                           'Migrant-level model'),
             decomposition(paths, 'Household model')]
    pd.concat(parts, ignore_index=True).round(5).to_csv(HERE / 'outputs/decomposition.csv', index=False)
    s9_decomposition(parts)
    u = usd_accounting(parts[1])
    u.round(5).to_csv(HERE / 'outputs/usd_accounting.csv', index=False)
    s10_usd(u)


def write_captions():
    path = FIG / 'captions.md'
    old = {}
    if path.exists():                    # keep captions of figures not redrawn in this call
        for block in path.read_text(encoding='utf-8').split('\n## ')[1:]:
            head, _, body = block.partition('\n\n')
            name, _, title = head.partition(': ')
            old[name] = (title, body.strip())
    old.update(CAPTIONS)
    lines = ['# Recipient-level figures', '']
    for name in sorted(old):
        title, caption = old[name]
        lines += [f'## {name}: {title}', '', caption, '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    FIG.mkdir(exist_ok=True)
    mf.style()
    if '--model' in sys.argv:
        model_figures()
        return write_captions()
    b = h3_coefficients()
    pop = hl.abm.load_population()
    raw = np.column_stack([pop['h'].resident_worker.eq(0), pop['h'].resident_older.gt(0)]).astype(float)
    known = (pop['h'].resident.gt(0) & pop['h'][['resident_worker', 'resident_older']].notna().all(axis=1)).to_numpy()
    need_mean = raw[known].mean(0)
    s1_concept(b, need_mean)
    s2_households(b, pop)
    write_captions()


if __name__ == '__main__':
    main()
