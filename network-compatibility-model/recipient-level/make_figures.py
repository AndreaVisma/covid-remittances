"""Figures for the recipient-level (household) receipt logit, H3 fit.

Run: python network-compatibility-model/recipient-level/make_figures.py
(after compare_household_logit.py). Style and colours from model/make_figures.py.

S1 (concept): the fitted curve P = expit(score) with illustrative households (made-up compositions,
   not MMP households) and arrows for one migrant losing their job.
S2 (data; --not-working-as-unknown draws S2 for that variant fit): every sample household at its expected 2019 score (able counts averaged over 2019 CPS
   employment), drawn on the curve and sized by the number of households with that score, with
   observed MMP receipt by score band (aggregates, bands with >= 30 households) and a histogram.
Score = intercept + any_core * [A_c > 0] + log_core * log(1 + A_c) + log_kin * log(1 + A_k)
        + no_worker * (no worker - mean) + any_older * (older resident - mean).
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'model'))
import compare_household_logit as hl
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


def main():
    FIG.mkdir(exist_ok=True)
    mf.style()
    if '--not-working-as-unknown' in sys.argv:      # S2 only, variant fit
        b = h3_coefficients(HERE / 'outputs/not-working-unknown')
        s2_households(b, hl.load_population(True), '_not_working_unknown', ', "Not working" as unknown occupation')
        return
    b = h3_coefficients()
    pop = hl.abm.load_population()
    raw = np.column_stack([pop['h'].resident_worker.eq(0), pop['h'].resident_older.gt(0)]).astype(float)
    known = (pop['h'].resident.gt(0) & pop['h'][['resident_worker', 'resident_older']].notna().all(axis=1)).to_numpy()
    need_mean = raw[known].mean(0)
    s1_concept(b, need_mean)
    s2_households(b, pop)
    lines = ['# Recipient-level figures', '']
    for name, (title, caption) in CAPTIONS.items():
        lines += [f'## {name}: {title}', '', caption, '']
    (FIG / 'captions.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
