"""Data figures (MMP networks, CPS employment) and the shared figure style.

Run: python network-compatibility-model/model/make_figures.py   (A1-A4, B1)
Also imported by recipient-level/make_figures.py for the palette and helpers. The figures of the former
migrant-level runs (B2-B3, C, D, H, M, P, R) were removed in the 29 Oct 2026 cleanup (git history up to
a83b60c).
Plan: network-compatibility-model/figures-plan.md. Captions: figures/captions.md.
Figures show aggregates only (no MMP microdata rows).

Colour roles (dataviz reference palette, light mode; validated):
  link types   Core #2a78d6, Close kin #eb6834, Distant kin #1baf7a (all-pairs pass)
  occupations  #e87ba4, #008300, #6250d6, #e34948 (adjacent pass); Not working and
               Unknown in neutral grays
  data vs model  MMP = primary ink + 95% Wilson interval, model = blue
  error        diverging blue (model below MMP) / red (model above)
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'data-prep'))
import abm

BASE = abm.HERE
FIG = BASE / 'figures'
OLD_POP = abm.ROOT / 'oct-26/outputs/synthetic-migrants-house-networks'

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
RED, GRAY_LIGHT = '#e34948', '#c3c2b7'
LINK_COLORS = {'Core': BLUE, 'Close kin': ORANGE, 'Distant kin': AQUA}
OCC_ORDER = ['Agriculture', 'Production/construction/transport', 'Sales/personal services',
             'Professional/administrative', 'Not working', 'Other/unknown']
OCC_COLORS = dict(zip(OCC_ORDER, ['#e87ba4', '#008300', '#6250d6', '#e34948', MUTED, GRAY_LIGHT]))
OCC_SHORT = {'Agriculture': 'Agriculture', 'Production/construction/transport': 'Production/constr./transport',
             'Sales/personal services': 'Sales/personal services',
             'Professional/administrative': 'Professional/admin.', 'Not working': 'Not working',
             'Other/unknown': 'Other/unknown'}
DPI = 200
CAPTIONS = {}


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
        'axes.labelsize': 9, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'grid.linestyle': '-',
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'legend.fontsize': 8.5, 'legend.labelcolor': INK2, 'font.size': 9, 'lines.linewidth': 2,
        'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round'})


def wilson(p, n, z=1.96):
    p, n = np.asarray(p, float), np.asarray(n, float)
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def pct_axis(ax, axis='y'):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == 'y' else ax.xaxis).set_major_formatter(fmt)


def save(fig, name, title, caption, title_y=None):
    """title_y: figure-fraction height of the title; tall figures pass a value
    close to 1 so the title does not float far above the plot."""
    fig.suptitle(title, x=0.01, ha='left', fontsize=12.5, fontweight='semibold', color=INK,
                 **({} if title_y is None else dict(y=title_y)))
    fig.savefig(FIG / f'{name}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    CAPTIONS[name] = (title, caption)
    print('saved', name)


def ring_dot(ax, x, y, color, size=40, zorder=4, **kw):
    ax.scatter(x, y, s=size, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=zorder, **kw)


# %% A. Population
def strongest_tie(h):
    return np.select([h.core_count.gt(0), h.close_kin_count.gt(0)], ['Core', 'Close kin'], 'Distant kin')


def a1_network_sizes(h):
    h = h.assign(tie=strongest_tie(h), size=h.network_size.clip(upper=10))
    t = h.groupby(['size', 'tie']).size().unstack(fill_value=0).reindex(columns=list(LINK_COLORS), fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bottom = np.zeros(len(t))
    x = np.arange(len(t))
    for tie, color in LINK_COLORS.items():
        ax.bar(x, t[tie], bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=1.5, label=tie)
        bottom += t[tie].to_numpy()
    for xi, total in zip(x, bottom):
        ax.text(xi, total + 15, f'{int(total):,}', ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x, [str(s) if s < 10 else '10+' for s in t.index])
    ax.set_xlabel('Links per household (network size)')
    ax.set_ylabel('Households')
    ax.grid(axis='x', visible=False)
    ax.legend(title='Closest tie in the US', title_fontsize=8.5, loc='upper right')
    save(fig, 'A1_network_sizes', 'Most households are linked to one or two US migrants',
         f'Households in the rebuilt population ({len(h):,} connected MMP households, Mexico interviews '
         '2010–19) by number of links, coloured by the closest tie: core (head, spouse or child in the US), '
         'close kin (parents, siblings, other roster relatives) or distant kin only (aggregated link).')


def a2_old_vs_new(h):
    old = pd.read_parquet(OLD_POP / 'households.parquet')
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for frame, color, label, z in [(old, MUTED, 'Old population (resampled, all kin individual)', 2),
                                   (h, BLUE, 'Rebuilt population (distant kin aggregated)', 3)]:
        s = frame.network_size.sort_values().to_numpy()
        share = np.cumsum(s) / s.sum()
        ax.step(s, share, where='post', color=color, lw=2, zorder=z, label=label)
        ring_dot(ax, s.max(), 1, color, size=30)
        ax.annotate(f'max {s.max()}', (s.max(), 1), xytext=(0, 9), textcoords='offset points',
                    ha='center', fontsize=8.5, color=INK2)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 169], ['1', '2', '5', '10', '20', '50', '100', '169'])
    ax.minorticks_off()
    pct_axis(ax)
    ax.set_xlabel('Household network size (log scale)')
    ax.set_ylabel('Cumulative share of all links')
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc='lower right')
    save(fig, 'A2_old_vs_rebuilt', 'The rebuild removes the long tail of distant relatives',
         'Cumulative share of links held by households up to a given network size. Old population: '
         '1,019 households resampled with replacement until 10,000 migrants, every counted cousin, '
         'nephew and in-law a separate link (54% of links in households with 20+ links). Rebuilt: every '
         'connected household once, distant kin collapsed into one link per household.')


def a3_close_vs_distant(summary):
    close = summary.loc[summary.dimension.eq('close kin (no core)') & summary.group.ne('0')]
    distant = summary.loc[summary.dimension.eq('distant relatives (distant only)') & summary.group.ne('0')]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), sharey=True)
    for ax, d, color, xlabel, title in [
            (axes[0], close, ORANGE, 'Close kin in the US', 'Households without core migrants'),
            (axes[1], distant, AQUA, 'Distant relatives in the US', 'Households with only distant kin')]:
        x = np.arange(len(d))
        lo, hi = wilson(d.receipt, d.n)
        ax.bar(x, d.receipt, width=0.55, color=color, zorder=2)
        ax.errorbar(x, d.receipt, yerr=[d.receipt - lo, hi - d.receipt], fmt='none', ecolor=INK2,
                    elinewidth=1, capsize=3, zorder=3)
        for xi, r, top in zip(x, d.receipt, hi):
            ax.text(xi, top + 0.004, f'{r:.1%}', ha='center', va='bottom', fontsize=8, color=INK)
        ax.set_xticks(x, [f'{g}\nn={n:,}' for g, n in zip(d.group.astype(str), d.n)])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, 0.25)
        ax.set_yticks(np.arange(0, 0.251, 0.05))
    pct_axis(axes[0])
    axes[0].set_ylabel('MMP households receiving US remittances')
    save(fig, 'A3_close_vs_distant_kin', 'Receipt rises with close kin, not with the number of distant relatives',
         'Share of MMP households (Mexico, 2010–19) that receive US remittances, with 95% Wilson intervals. '
         'Left: households without core migrants, by number of close kin in the US. Right: households whose '
         'only ties are distant relatives (uncles/aunts, cousins, nieces/nephews, in-laws), by how many are '
         'in the US. This is why each close relative is a separate link and distant relatives form one '
         'aggregated link.')


def a4_occupations(m):
    m = m.assign(link=m.link_type.map({'Core': 'Core', 'Close kin': 'Close kin',
                                        'Distant kin (aggregated)': 'Distant kin'}),
                 occ=m.job_group.where(m.job_group.isin(OCC_ORDER[:5]), 'Other/unknown'))
    share = pd.crosstab(m.link, m.occ, normalize='index').reindex(index=list(LINK_COLORS), columns=OCC_ORDER, fill_value=0)
    imputed = m.groupby('link').profile_imputed.mean().reindex(share.index)
    n = m.groupby('link').size().reindex(share.index)
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    y = np.arange(len(share))[::-1]
    left = np.zeros(len(share))
    for occ in OCC_ORDER:
        ax.barh(y, share[occ], left=left, height=0.5, color=OCC_COLORS[occ], edgecolor=SURFACE,
                linewidth=1.5, label=OCC_SHORT[occ])
        for yi, l, w in zip(y, left, share[occ]):
            if w >= 0.08:
                txt = 'white' if occ not in ['Other/unknown'] else INK
                ax.text(l + w / 2, yi, f'{w:.0%}', ha='center', va='center', fontsize=8, color=txt)
        left += share[occ].to_numpy()
    for yi, link in zip(y, share.index):
        ax.text(1.01, yi, f'{imputed[link]:.0%} imputed\n(n={n[link]:,})', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, share.index)
    ax.set_xlim(0, 1)
    pct_axis(ax, 'x')
    ax.grid(visible=False)
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    save(fig, 'A4_occupations_by_link', 'Kin occupations mirror core occupations because they are imputed',
         'Occupation of linked migrants by link type (last US occupation for roster migrants). Questionnaire-'
         'only links (most close kin, all distant links) copy a whole profile from a roster donor, so their '
         'occupational mix is borrowed. Differences in occupational exposure across households are therefore '
         'partly imputation noise; correlated job loss is treated as a sensitivity dimension only.')


# %% B. Inputs
def b1_employment():
    r = pd.read_csv(abm.CPS_RATES)
    r['date'] = pd.to_datetime(dict(year=r.year, month=r.month, day=1))
    r = r.loc[r.year.between(2019, 2022)]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.axvspan(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-12-31'), color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(pd.Timestamp('2019-07-01'), 0.655, '2019: calibration year', ha='center', fontsize=8, color=INK2)
    for occ in OCC_ORDER[:4]:
        d = r.loc[r.job.eq(occ)]
        ax.plot(d.date, d.rate_model, color=OCC_COLORS[occ], lw=2, label=OCC_SHORT[occ])
    sales = r.loc[r.job.eq('Sales/personal services')].set_index('date').rate_model
    ax.annotate(f'April 2020: {sales.min():.0%} employed\n(sales/personal services)', (sales.idxmin(), sales.min()),
                xytext=(40, -6), textcoords='offset points', fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    pct_axis(ax)
    ax.set_ylim(0.64, 1.0)
    ax.set_ylabel('Employed / labour force')
    ax.legend(ncol=2, loc='lower right')
    pop = json.loads(abm.CPS_SETTINGS.read_text())['population']
    save(fig, 'B1_cps_employment', 'COVID hit migrant employment hard and unevenly across occupations',
         f'Monthly employment rate (employed / employed + unemployed) by occupation of current or last job, '
         f'CPS, ages 18–64, {pop}. Raw monthly rates, smoothed (3-month centred) only where cells are small '
         '(some agriculture months). Agriculture is seasonal in every year. These rates drive migrants\' '
         'monthly ability to send.')




# %% Shared with recipient-level/make_figures.py
K0, K4 = '#86b6ef', '#1c5cab'                     # ordinal blue: kappa = 0 (light) -> 4 (dark); validated
HOLDOUT_WASH = '#f0efec'
NETWORKS = ['Core only', 'Core + kin', 'Kin only']


def save_bare(fig, name, title, caption):
    """E1 figures go on slides that carry the headline, so no suptitle; captions.md keeps it."""
    fig.savefig(FIG / f'{name}.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    CAPTIONS[name] = (title, caption)
    print('saved', name)


def end_label(ax, x, y, text, dy=0, color=INK2):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords='offset points', va='center', fontsize=8.5, color=color)



SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # palette ramp, light->dark

LOSS_RAMP = ['#8a2322', '#c03837', '#dc5e5d', '#eb9594']     # dark -> light, validated ordinal (red)
GAIN_RAMP = ['#86b6ef', '#3987e5', '#184f95']                # light -> dark, validated ordinal (blue)
K2 = '#3987e5'                                               # kappa = 2 between K0 and K4 (validated ramp)
KAPPA_COLORS = {0.0: K0, 2.0: K2, 4.0: K4}
GROUP_ROWS = [('network_type', 'Network type', ['Core only', 'Core + kin', 'Kin only']),
              ('size_bin', 'Network size', ['1', '2', '3-4', '5-9', '10+']),
              ('strongest_tie', 'Strongest tie', ['Core', 'Close kin', 'Distant kin only']),
              ('occupation', 'Main migrant occupation', ['Agriculture', 'Production/construction/transport',
                                                         'Sales/personal services', 'Professional/administrative',
                                                         'Other/none'])]



def main():
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    h = pd.read_parquet(abm.POPULATION / 'households.parquet')
    m = pd.read_parquet(abm.POPULATION / 'migrants.parquet')
    summary = pd.read_csv(BASE / 'data-prep/outputs/population_summary.csv')
    a1_network_sizes(h)
    if OLD_POP.exists():                # the pre-rebuild population (gitignored, local only)
        a2_old_vs_new(h)
    a3_close_vs_distant(summary)
    a4_occupations(m)
    b1_employment()
    write_captions()


def write_captions():
    """Rewrites captions.md from the figures made in this run; keep earlier entries."""
    path = FIG / 'captions.md'
    old = {}
    if path.exists():
        for block in path.read_text(encoding='utf-8').split('\n## ')[1:]:
            head, _, body = block.partition('\n\n')
            name, _, title = head.partition('.png: ')
            old[name] = (title, body.strip())
    old.update(CAPTIONS)
    lines =['# Figure captions', '', 'Generated by `model/make_figures.py`. Plan: `figures-plan.md`.', '']
    for name, (title, cap) in old.items():
        lines += [f'## {name}.png: {title}', '', cap, '']
    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
