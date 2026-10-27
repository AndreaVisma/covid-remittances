"""PNG figures for the rebuilt population, model inputs and E0 calibration.

Run: python network-compatibility-model/model/make_figures.py
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
import remittance_amounts as ra

BASE = abm.HERE
FIG = BASE / 'figures'
E0 = BASE / 'outputs/e0-baseline'
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


def b3_salience():
    """Salience S_t = reported COVID deaths in Mexico (10,000 a month), primary ink; excess deaths blue."""
    idx = pd.read_csv(BASE / 'data-prep/outputs/salience_index_monthly.csv', index_col='date', parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), color='#f0efec', lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=1)
    ax.plot(idx.index, idx.S, color=INK, label='S_t: reported COVID deaths (base)')
    ax.plot(idx.index, idx.S_excess_deaths, color=BLUE, lw=1.5, label='Excess all-cause deaths (variant)')
    for when in ['2020-07-01', '2021-01-01', '2021-08-01', '2022-02-01']:
        y = idx.loc[when, 'S']
        ring_dot(ax, pd.Timestamp(when), y, INK, size=30)
        ax.annotate(f'{y:.1f}', (pd.Timestamp(when), y), xytext=(7, 2), textcoords='offset points', fontsize=8.5,
                    color=INK2)
    ax.text(pd.Timestamp('2022-03-20'), idx.S_excess_deaths.max() * 0.95, 'held out', fontsize=8, color=MUTED)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.set_ylabel('Deaths per month (10,000s)')
    ax.legend(loc='upper left')
    save(fig, 'B3_salience', 'National salience S_t: reported COVID deaths in Mexico',
         'S_t = reported COVID-19 deaths in Mexico per month (OWID), in units of 10,000; 0 in 2019. Variant: '
         'excess all-cause deaths (World Mortality Dataset; 2015–19 same-week mean with a linear trend; negative '
         'values set to 0). Reported deaths are about 45% of excess deaths. 2022 is held out.')


def b2_amounts():
    m = ra.load_mmp()
    _, pos, _ = ra.amount_sample(m)
    a = json.loads(abm.AMOUNTS.read_text())
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bins = np.logspace(np.log10(20), np.log10(pos.usd_2019.max() * 1.05), 36)
    counts, edges, _ = ax.hist(pos.usd_2019, bins=bins, color=GRAY_LIGHT, edgecolor=SURFACE, linewidth=1,
                               label=f'MMP reported monthly amounts (n={len(pos):,}, 2019 USD)')
    x = np.logspace(np.log10(20), np.log10(edges[-1]), 400)
    dens = np.exp(-(np.log(x) - a['mu']) ** 2 / (2 * a['sigma'] ** 2)) / (x * a['sigma'] * np.sqrt(2 * np.pi))
    width = np.diff(np.log(edges)).mean()
    ax.plot(x, dens * x * width * len(pos), color=BLUE, lw=2,
            label=f'Model routine amounts: lognormal, mean ${a["mean_usd_2019"]:.0f}, σ {a["sigma"]:.2f}')
    top = counts.max()
    # (value, label, side): labels sit above the bars, Banxico left of its line, MMP right.
    refs = [(a['enigh_2018_validation']['recipient_monthly_usd_median'], 'ENIGH recipient median', 'right', 1.32),
            (a['mean_usd_2019'], 'Banxico mean per transaction', 'left', 1.13),
            (a['mmp_median_usd_2019'], 'MMP median', 'right', 1.32)]
    for v, label, side, height in refs:
        ax.axvline(v, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
        x = v * (1.04 if side == 'right' else 0.96)
        ax.text(x, top * height, f'{label}\n${v:.0f}', fontsize=7.5, color=INK2, va='top',
                ha='left' if side == 'right' else 'right')
    ax.set_ylim(0, top * 1.35)
    ax.set_xscale('log')
    ax.set_xticks([25, 50, 100, 250, 500, 1000, 2500, 5000], ['$25', '$50', '$100', '$250', '$500', '$1k', '$2.5k', '$5k'])
    ax.minorticks_off()
    ax.set_xlabel('USD per sending month (log scale)')
    ax.set_ylabel('Migrants')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.2), ncol=1)
    save(fig, 'B2_amounts', 'Each migrant sends a fixed routine amount drawn from a lognormal',
         'Gray: MMP self-reported average monthly remittances on the last US trip (heads and other migrants, '
         'trips 2000–19, deflated to 2019 USD, top 1% trimmed). Blue: the model\'s routine-amount distribution, '
         'with its spread (σ) from MMP within 5-year periods and its level set to the Banxico 2019 mean per '
         'transaction. MMP reports sit higher than Banxico (recall, proxy reports, trip-start deflation); '
         'ENIGH averages over months and under-reports.')


# %% C. Calibration
def e0_cells():
    fit = pd.read_csv(E0 / 'fit_network_type.csv').query('kappa == 0')
    comp = pd.read_csv(E0 / 'validation_composition.csv').query('kappa == 0')
    rows = [fit.assign(panel='Network type (targeted)', label=fit.network_type)]
    names = {'links': 'Link type', 'survey_period': 'Survey period', 'workers': 'Resident workers',
             'children': 'Resident children', 'older': 'Older residents'}
    for panel, title in names.items():
        d = comp.loc[comp.panel.eq(panel)]
        label = d.category if panel in ['links', 'survey_period'] else d.network_type + ' · ' + d.category
        rows.append(d.assign(panel=title, label=label))
    return pd.concat(rows, ignore_index=True)


def c1_ladder(cells):
    cells = cells.reset_index(drop=True)
    n = len(cells) + cells.panel.nunique()
    height = 0.26 * n + 1.2
    fig, ax = plt.subplots(figsize=(9.0, height))
    fig.subplots_adjust(left=0.36, right=0.98, top=1 - 0.75 / height)
    y, ticks, labels, ypos = n, [], [], []
    for panel, d in cells.groupby('panel', sort=False):
        y -= 1
        ax.text(-0.01, y, panel, transform=ax.get_yaxis_transform(), ha='right', va='center',
                fontsize=9, fontweight='semibold', color=INK)
        for _, r in d.iterrows():
            y -= 1
            ypos.append(y)
            ticks.append(y)
            labels.append(f'{r.label}  (n={int(r.source_n):,})')
    cells['y'] = ypos
    lo, hi = wilson(cells.mmp_receipt, cells.source_n)
    weak = ~cells.reliable.astype(bool)
    for i, r in cells.iterrows():
        alpha = 0.35 if weak[i] else 1
        ax.plot([lo[i], hi[i]], [r.y, r.y], color=INK2, lw=1.2, alpha=alpha, zorder=2)
        ax.plot([r.mmp_receipt, r.model_receipt], [r.y, r.y], color=GRID, lw=3, zorder=1)
        ring_dot(ax, r.mmp_receipt, r.y, INK, size=34, alpha=alpha)
        ring_dot(ax, r.model_receipt, r.y, BLUE, size=34, alpha=alpha, zorder=5)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(0, 1)
    pct_axis(ax, 'x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Share of households receiving US remittances (per month, model; MMP "receives")')
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color=INK, label='MMP, with 95% interval'),
                       Line2D([], [], marker='o', ls='', color=BLUE, label='Model (κ = 0)')],
              loc='lower right', bbox_to_anchor=(1, 1.0), ncol=2)
    save(fig, 'C1_calibration_ladder', 'The 2019 baseline matches network types, link types and survey periods',
         title_y=1 - 0.15 / height, caption=
         'Baseline receipt by group of households: MMP observed share with 95% Wilson interval (black) and '
         'model expected monthly receipt probability under 2019 employment (blue), κ = 0. Only the three '
         'network-type rows are targeted (α_core, α_kin); all others are untargeted checks. The model has '
         'no household-composition terms, so composition differences (older residents, children) are not '
         'reproduced. Faded rows: fewer than 30 households.')


def c2_size(size):
    size = size.query('kappa == 0')
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharey=True)
    for ax, net in zip(axes, ['Core only', 'Core + kin', 'Kin only']):
        d = size.loc[size.network_type.eq(net)].reset_index(drop=True)
        x = np.arange(len(d))
        lo, hi = wilson(d.mmp_receipt, d.source_n)
        weak = ~d.reliable.astype(bool)
        ax.plot(x, d.model_receipt, color=BLUE, lw=2, zorder=3)
        for i in x:
            a = 0.35 if weak[i] else 1
            ax.plot([i, i], [lo[i], hi[i]], color=INK2, lw=1.2, alpha=a, zorder=2)
            ring_dot(ax, i, d.mmp_receipt[i], INK, size=34, alpha=a)
            ring_dot(ax, i, d.model_receipt[i], BLUE, size=26, alpha=a, zorder=5)
        ax.set_xticks(x, [f'{s}\nn={n:,}' for s, n in zip(d.size_bin, d.source_n)], fontsize=7.5)
        ax.set_title(net, fontsize=10)
        ax.set_xlabel('Links per household')
        ax.grid(axis='x', visible=False)
    axes[0].set_ylim(0, 1)
    pct_axis(axes[0])
    axes[0].set_ylabel('Households receiving')
    axes[2].legend(handles=[Line2D([], [], marker='o', ls='', color=INK, label='MMP, 95% interval'),
                            Line2D([], [], marker='o', color=BLUE, label='Model (κ = 0)')], loc='upper left')
    fig.text(0.01, -0.1, 'Old population (first E0 run, all kin as separate links): kin-only 10–19 links '
             '8.1% model vs 8.7% MMP, 20+ links 18.5% vs 8.8%; core + kin 20+ links 77% vs 56%.',
             fontsize=8, color=INK2, ha='left')
    save(fig, 'C2_receipt_by_size', 'Receipt rises with network size as in the data; core networks hold the remaining misfits',
         'Receipt by network size and type: MMP share with 95% Wilson interval (black) and model (blue), κ = 0. '
         'Kin-only households match within about one point at every size. Core networks show the remaining '
         'misfits (core + kin with 2 links −12 pp, with 5–9 links +11 pp). Faded points: fewer than 30 '
         'households. Old population for comparison (first E0 run, distant relatives as individual links): '
         'kin-only 10–19 links 8.1% model vs 8.7% MMP, 20+ links 18.5% vs 8.8%; core + kin 20+ links 77% vs 56%.')


def c3_errors(cells, size):
    s = size.query('kappa == 0').assign(panel='Network size', label=lambda d: d.network_type + ' · ' + d.size_bin.astype(str))
    d = pd.concat([cells, s], ignore_index=True)
    d = d.loc[d.reliable.astype(bool)].sort_values('error_pp').reset_index(drop=True)
    height = 0.24 * len(d) + 1.2
    fig, ax = plt.subplots(figsize=(9.0, height))
    fig.subplots_adjust(left=0.4, right=0.98, top=1 - 0.6 / height)
    y = np.arange(len(d))
    colors = np.where(d.error_pp < 0, BLUE, RED)
    ax.barh(y, d.error_pp, height=0.6, color=colors, zorder=2)
    ax.axvline(0, color=AXIS, lw=1)
    for yi, e in zip(y, d.error_pp):
        ax.text(e + (0.4 if e >= 0 else -0.4), yi, f'{e:+.1f}', va='center', ha='left' if e >= 0 else 'right',
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, d.panel + ': ' + d.label.astype(str))
    ax.tick_params(axis='y', length=0, labelsize=7.5)
    ax.grid(axis='y', visible=False)
    lim = np.ceil(d.error_pp.abs().max() / 5) * 5 + 3
    ax.set_xlim(-lim, lim)
    ax.set_xlabel('Model − MMP (percentage points)')
    ax.legend(handles=[Patch(color=BLUE, label='Model below MMP'), Patch(color=RED, label='Model above MMP')],
              loc='lower right')
    save(fig, 'C3_errors', 'Largest baseline errors: household composition and some core-network sizes',
         title_y=1 - 0.15 / height, caption=
         'Model minus MMP receipt (percentage points) for every group with at least 30 households, κ = 0, '
         'sorted. Targeted rows (network type) are within ±1 pp by construction.')


def c4_kappa():
    prof = pd.read_csv(E0 / 'kappa_profile.csv')
    k19 = pd.read_csv(E0 / 'kappa_2019.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True)
    panels = [(axes[0, 0], prof.kappa, prof.weighted_rmse_pp, 'Weighted fit error (pp)', '{:.2f}'),
              (axes[0, 1], k19.kappa, k19.kin_transfers_triggered, 'Kin transfers triggered per month', '{:.1f}'),
              (axes[1, 0], prof.kappa, prof.alpha_core, 'Fitted α_core', '{:.2f}'),
              (axes[1, 1], prof.kappa, prof.alpha_kin, 'Fitted α_kin', '{:.2f}')]
    for ax, x, y, title, fmt in panels:
        ax.plot(x, y, color=BLUE, lw=2)
        ring_dot(ax, x.iloc[-1], y.iloc[-1], BLUE, size=30)
        ax.annotate(fmt.format(y.iloc[-1]), (x.iloc[-1], y.iloc[-1]), xytext=(-6, 8), textcoords='offset points',
                    ha='right', fontsize=8, color=INK2)
        ax.annotate(fmt.format(y.iloc[0]), (x.iloc[0], y.iloc[0]), xytext=(4, 8), textcoords='offset points',
                    fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.25)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 1].set_ylim(bottom=0)
    # Same ±0.5 logit window for both α so small movements are not exaggerated.
    for ax, col in [(axes[1, 0], 'alpha_core'), (axes[1, 1], 'alpha_kin')]:
        mid = prof[col].mean()
        ax.set_ylim(mid - 0.5, mid + 0.5)
    for ax in axes[1]:
        ax.set_xlabel('κ')
    save(fig, 'C4_kappa_2019', 'The 2019 data barely constrain κ; the COVID period must',
         'κ: kin stepping-in, in proportion to the routine sending lost to linked migrants who cannot send. '
         'For each κ, α_core and α_kin are '
         'refitted to 2019 receipt by network type (α panels on the same ±0.5 logit window). Fit error changes by less '
         'than one percentage point across κ = 0–8, and the fitted α barely move, because few migrants are '
         'unable to send in a normal year. Top right: kin transfers in households where some linked migrant '
         'cannot send (stochastic 2019 runs, 5 seeds, out of ~1,100 transfers a month).')


def c5_tiles():
    en = pd.read_csv(E0 / 'enigh_style_check.csv').query('kappa == 0').iloc[0]
    st = json.loads((E0 / 'run_settings.json').read_text())
    k19 = pd.read_csv(E0 / 'kappa_2019.csv').query('kappa == 0').iloc[0]
    tiles = [('Receive at least once in 6 months', f'{en.share_receiving_6m:.0%}', 'connected households, model'),
             ('Median monthly USD among receivers', f'${en.median_usd_month:.0f}',
              f'ENIGH 2018: ${st["enigh_2018_recipient_monthly_usd"]["median"]:.0f}'),
             ('Mean monthly USD among receivers', f'${en.mean_usd_month:.0f}',
              f'ENIGH 2018: ${st["enigh_2018_recipient_monthly_usd"]["mean"]:.0f}'),
             ('National transactions per model transfer', f'{k19.banxico_scale:,.0f}', 'Banxico 2019 scale factor')]
    import textwrap
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.0))
    fig.subplots_adjust(left=0.01, right=0.99, wspace=0.06, top=0.82, bottom=0.02)
    for ax, (label, value, sub) in zip(axes, tiles):
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0.0, 0.0), 1, 1, transform=ax.transAxes, facecolor=SURFACE,
                                   edgecolor=GRID, lw=1))
        ax.text(0.08, 0.88, textwrap.fill(label, 24), transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')
        ax.text(0.08, 0.30, value, transform=ax.transAxes, fontsize=20, fontweight='semibold', color=INK)
        ax.text(0.08, 0.10, sub, transform=ax.transAxes, fontsize=8, color=INK2)
    save(fig, 'C5_household_amounts', 'Household amounts are in a plausible range',
         'Model 2019 baseline, κ = 0, six-month window (stochastic runs, 5 seeds), against ENIGH 2018 '
         'recipient households (monthly USD from quarterly MXN). ENIGH captures only ~8% of the Banxico '
         'total, so model amounts above ENIGH are expected.')


# %% D. E1a: job loss only
E1 = BASE / 'outputs/e1-job-loss'
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


def index_panel(ax, x, model0, model4, banxico, holdout_from, labels_at):
    ax.axvspan(holdout_from - 0.5, x[-1] + 0.5, color=HOLDOUT_WASH, lw=0, zorder=0)
    ax.axhline(1, color=MUTED, lw=1, zorder=1)
    ax.fill_between(x, model0, model4, color=K4, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, model0, color=K0, zorder=3, label='Model, κ = 0')
    ax.plot(x, model4, color=K4, zorder=3, label='Model, κ = 4')
    ax.plot(x, banxico, color=INK, zorder=4, label='Banxico vs its 2017–19 trend')
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper left', ncol=3)
    ax.text(holdout_from - 0.3, 0.825, '2022 held out', fontsize=8.5, color=INK2, va='bottom')


def d1_quarterly(q):
    k0, k4 = q.query('kappa == 0'), q.query('kappa == 4')
    x = np.arange(len(k0))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    index_panel(ax, x, k0.index_model.to_numpy(), k4.index_model.to_numpy(), k0.index_banxico.to_numpy(),
                holdout_from=int(np.where(k0.quarter.str.startswith('2022'))[0][0]), labels_at=None)
    ax.set_xticks(x, [s.replace('Q', ' Q') if s.endswith('Q1') else s[-2:] for s in k0.quarter])
    trough = int(np.argmin(k0.index_model))
    ring_dot(ax, trough, k0.index_model.iloc[trough], K0)
    ax.annotate(f'{k0.index_model.iloc[trough]:.2f}', (trough, k0.index_model.iloc[trough]), xytext=(0, -14),
                textcoords='offset points', ha='center', fontsize=8.5, color=INK2)
    for i in [trough, len(x) - 5, len(x) - 1]:
        ring_dot(ax, i, k0.index_banxico.iloc[i], INK)
        ax.annotate(f'{k0.index_banxico.iloc[i]:.2f}', (i, k0.index_banxico.iloc[i]), xytext=(0, 9),
                    textcoords='offset points', ha='center', fontsize=8.5, color=INK2)
    ax.set_ylim(0.82, 1.2)
    ax.set_ylabel('Transfers relative to no-COVID path\n(quarterly)')
    save_bare(fig, 'D1_count_index_quarterly', 'Job loss with stepping-in matches 2020; the 2021 rise above trend is missed',
              'E1a (β = 0, ρ = 0, no benefits). Model: quarterly transfer counts relative to the no-COVID run (2019 CPS '
              'rates continued; exact expected counts, κ = 0 and κ = 4, E0 alphas for each). Banxico: transactions '
              'relative to their 2017–19 trend (log-linear, +6.3% a year, with month effects). 1 = no-COVID path. '
              '2022 is held out.')


def d1_monthly(m):
    k0, k4 = m.query('kappa == 0'), m.query('kappa == 4')
    dates = pd.to_datetime(k0.date)
    x = np.arange(len(k0))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    index_panel(ax, x, k0.index_model.to_numpy(), k4.index_model.to_numpy(), k0.index_banxico.to_numpy(),
                holdout_from=int(np.where(dates.dt.year.eq(2022))[0][0]), labels_at=None)
    ax.set_xticks(x[::6], [f'{d:%b %Y}' if d.month == 1 else f'{d:%b}' for d in dates[::6]])
    mar = int(np.where(dates.eq('2020-03-01'))[0][0])
    ring_dot(ax, mar, k0.index_banxico.iloc[mar], INK)
    ax.annotate('Mar 2020 record month,\nexcluded from fit', (mar, k0.index_banxico.iloc[mar]), xytext=(-8, 6),
                textcoords='offset points', ha='right', fontsize=8, color=INK2)
    apr = mar + 1
    for y, c, dy in [(k0.index_model.iloc[apr], K0, -14), (k0.index_banxico.iloc[apr], INK, -14)]:
        ring_dot(ax, apr, y, c)
        ax.annotate(f'{y:.2f}', (apr, y), xytext=(10, dy + 4), textcoords='offset points', fontsize=8.5, color=INK2)
    ax.set_ylim(0.82, 1.2)
    ax.set_ylabel('Transfers relative to no-COVID path\n(monthly)')
    save_bare(fig, 'D1b_count_index_monthly', 'Monthly: the model bottoms out in April 2020',
              'As D1, monthly. The model low is April 2020 (0.85 at κ = 0), when Banxico fell to 0.91 of its trend. '
              'March 2020, Banxico’s record month (1.09 of trend), is excluded from the fit window (2020-04 to '
              '2021-12).')


def d2_decomposition(m):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, kappa in zip(axes, [0, 4]):
        d = m.query('kappa == @kappa')
        dates = pd.to_datetime(d.date)
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(dates, d.core_transfers_covid - d.core_transfers_cf, color=BLUE, label='Core migrants')
        ax.plot(dates, d.kin_transfers_covid - d.kin_transfers_cf, color=ORANGE, label='Kin migrants')
        ax.set_title(f'κ = {kappa}', fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        apr = d.set_index(dates).loc['2020-04-01']
        for col, c, dy in [('core', BLUE, -10), ('kin', ORANGE, -12 if kappa == 0 else 10)]:
            y = apr[f'{col}_transfers_covid'] - apr[f'{col}_transfers_cf']
            ring_dot(ax, pd.Timestamp('2020-04-01'), y, c)
            ax.annotate(f'{y:+.0f}', (pd.Timestamp('2020-04-01'), y), xytext=(8, dy), textcoords='offset points',
                        fontsize=8.5, color=INK2, va='center')
    axes[0].set_ylabel('Transfers per month vs counterfactual\n(model units; 2019 ≈ 1,100 a month)')
    axes[0].legend(loc='lower right')
    save_bare(fig, 'D2_core_kin_decomposition', 'Core migrants carry the loss; at κ = 4 kin offset much of it',
              'E1a. Expected monthly transfers minus the no-COVID counterfactual, by sender type. At κ = 0 kin '
              'transfers also fall (kin lose jobs too); at κ = 4 kin step in for core and kin migrants who cannot '
              'send (base lost-sending rule).')


def d3_reach(reach, ceil):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    labels = {'All lost transfers': ('All lost transfers', MUTED),
              'Lost core transfers': ('… of which core', BLUE),
              'Lost kin transfers': ('… of which kin', ORANGE),
              'Lost core transfers in households with eligible kin': ('Core losses where kin exist', BLUE),
              'Recovered by kin at kappa = 4': ('Recovered by kin, κ = 4', ORANGE),
              'Recovered by kin at kappa = 50 (ceiling)': ('Recovered by kin, κ → ∞', ORANGE)}
    r = reach.iloc[::-1]
    ax.set_axisbelow(True)
    y = np.arange(len(r))
    for yi, (_, row) in zip(y, r.iterrows()):
        name, c = labels[row.quantity]
        ax.barh(yi, row.transfers_per_month, height=0.55, color=c)
        ax.annotate(f'{row.transfers_per_month:.0f}', (row.transfers_per_month, yi), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(y, [labels[q][0] for q in r.quantity])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Transfers per month, April 2020 vs counterfactual (model units)')
    ax.set_title('Where the April 2020 losses are', fontsize=10)
    ax.legend(handles=[Patch(color=BLUE, label='Core'), Patch(color=ORANGE, label='Kin'),
                       Patch(color=MUTED, label='All')], loc='lower right')
    ax.set_xlim(0, reach.transfers_per_month.max() * 1.15)
    bx.axhline(1, color=MUTED, lw=1)
    bx.plot(ceil.kappa, ceil.index_april_2020, color=K4, marker='o', ms=4, mec=SURFACE, mew=1.2)
    bx.set_xscale('symlog', linthresh=1)
    bx.set_xticks([0, 1, 2, 4, 8, 20, 50], ['0', '1', '2', '4', '8', '20', '50'])
    top = ceil.index_april_2020.max()
    bx.annotate(f'ceiling {top:.2f}', (50, top), xytext=(0, 9), textcoords='offset points', ha='right',
                fontsize=8.5, color=INK2)
    bx.annotate(f'{ceil.index_april_2020.iloc[0]:.2f}', (0, ceil.index_april_2020.iloc[0]), xytext=(6, -8),
                textcoords='offset points', fontsize=8.5, color=INK2)
    bx.text(0.02, 1.004, 'No-COVID counterfactual = 1', fontsize=8, color=MUTED, va='bottom')
    bx.set_ylim(0.84, 1.02)
    bx.set_xlabel('κ (log scale)')
    bx.set_ylabel('April 2020 transfers / April 2019')
    bx.set_title('More κ saturates quickly', fontsize=10)
    save_bare(fig, 'D3_kappa_reach', 'Under the former rule, stepping-in could reach only a third of the losses',
              'Former stepping-in rule (all eligible core unable), E1a, April 2020, exact expected counts (ρ = 0). '
              'Left: lost transfers against the counterfactual. Only '
              'core losses in households that also have eligible kin can be replaced, and only when all of the '
              'household’s eligible core migrants are unable. Right: April 2020 count index against κ (alphas '
              'refitted under the former rule; beyond κ = 8 the κ = 8 fit). This diagnosis led to the base rule.')


def d4_variants(v):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    order = ['current', 'any_core', 'any_link']
    names = {'current': 'All eligible core unable (former rule)', 'any_core': 'Any eligible core unable',
             'any_link': 'Any other linked migrant unable'}
    ax.axvline(1, color=MUTED, lw=1)
    for i, rule in enumerate(order[::-1]):
        d = v.query('rule == @rule').set_index('kappa').index_april_2020
        ax.plot([d[0], d[4]], [i, i], color=GRID, lw=3, solid_capstyle='round', zorder=1)
        ring_dot(ax, d[0], i, K0, size=60)
        ring_dot(ax, d[4], i, K4, size=60)
        ax.annotate(f'{d[4]:.2f}', (d[4], i), xytext=(8, 0), textcoords='offset points', va='center',
                    fontsize=8.5, color=INK2)
    ax.set_yticks(range(3), [names[r] + ('' if r == 'current' else '\n(α not refitted)') for r in order[::-1]])
    ax.grid(axis='y', visible=False)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel('April 2020 transfers / April 2019')
    ax.text(1.003, 2.45, 'No-COVID = 1', fontsize=8, color=MUTED)
    ax.legend(handles=[Line2D([], [], color=K0, marker='o', ls='', ms=7, label='κ = 0'),
                       Line2D([], [], color=K4, marker='o', ls='', ms=7, label='κ = 4')], loc='upper right')
    save_bare(fig, 'D4_trigger_variants', 'Who triggers stepping-in matters more than how strongly',
              'Diagnostic only. April 2020 count index under three triggers for kin stepping-in, κ = 0 and 4, '
              '200 paired stochastic draws. Only the former rule is calibrated; the variants keep its alphas, so '
              'their 2019 baselines are too high (any linked migrant: 1,344 transfers a month vs 1,091). The base '
              'model now uses a proportional network-wide trigger (lost sending; R figures).')


def d5_receipt_paths(paths):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        d = {k: paths[paths.group.eq('network_type') & paths.level.eq(net) & paths.kappa.eq(k)] for k in [0, 4]}
        same = np.allclose(d[0].receipt_ratio, d[4].receipt_ratio, atol=3e-3)
        for kappa, c in [(0, K0), (4, K4)]:
            ax.plot(pd.to_datetime(d[kappa].date), d[kappa].receipt_ratio, color=c, label=f'κ = {kappa}')
            lo = d[kappa].loc[d[kappa].receipt_ratio.idxmin()]
            if kappa == 0 or not same:
                ring_dot(ax, pd.to_datetime(lo.date), lo.receipt_ratio, c, size=30)
                ax.annotate(f'{lo.receipt_ratio:.2f}', (pd.to_datetime(lo.date), lo.receipt_ratio),
                            xytext=(7, -2) if kappa == 0 else (8, 4), textcoords='offset points', fontsize=8.5,
                            color=INK2)
        ax.axhline(1, color=MUTED, lw=1)
        if same:
            ax.text(0.97, 0.30, 'κ = 0 and κ = 4 coincide:\nno kin to step in', transform=ax.transAxes, ha='right',
                    fontsize=8, color=MUTED)
        ax.set_title(net, fontsize=10)
        ax.set_xlim(pd.Timestamp('2019-07-01'), pd.Timestamp('2022-06-01'))
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    axes[0].set_ylabel('Receipt probability / counterfactual')
    axes[1].legend(loc='lower right')
    save_bare(fig, 'D5_receipt_by_network_type', 'Without stepping-in, the April dip lasts until autumn 2021',
              'E1a. Mean monthly receipt probability (exact) relative to the no-COVID counterfactual, by network type. '
              'At κ = 4 (base lost-sending rule) core + kin households barely dip and kin-only households recover '
              'faster; core-only households have nobody to step in.')


def d6_size(out):
    sizes = ['1', '2', '3-4', '5-9', '10+']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [('interruption_rate', 'Interrupted months\n(share of counterfactual receipt months)'),
              ('usd_drop', 'Drop in USD received, Apr 2020 – Dec 2021')]
    for ax, (col, title) in zip(axes, panels):
        for kappa, c, dx in [(0, K0, -0.08), (4, K4, 0.08)]:
            d = out.query('group == "size_bin" and kappa == @kappa').set_index('level').loc[sizes]
            ax.plot(np.arange(5) + dx, d[col], color=c, lw=0, marker='o', ms=7, mec=SURFACE, mew=1.5,
                    label=f'κ = {kappa}')
        ax.set_xticks(range(5), sizes)
        ax.set_xlabel('Network size (linked migrants)')
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_ylim(min(0, out.query('group == "size_bin"')[col].min()) * 1.15, None)
        pct_axis(ax)
    n = out.query('group == "size_bin" and kappa == 0').set_index('level').loc[sizes].households
    axes[0].text(4, 0.003, f'n = {n.iloc[-1]}', ha='center', fontsize=8, color=MUTED)
    axes[1].text(0.02, 0.03, 'below 0: more USD than in the counterfactual', transform=axes[1].transAxes,
                 fontsize=8, color=MUTED)
    axes[0].legend(loc='upper right')
    save_bare(fig, 'D6_outcomes_by_size', 'Bigger networks are interrupted less; with stepping-in they lose fewer dollars',
              'E1a, 50 paired stochastic runs (common random numbers), Apr 2020 – Dec 2021. Left: months in which the '
              'counterfactual household receives but the COVID household does not. Right: 1 − USD received / '
              'counterfactual USD (negative: more than without COVID). κ = 4 under the base lost-sending rule. '
              'Size 10+ has only 30 households.')


# %% R. E1a: base stepping-in rule ('lost_sending') vs former rule ('all_core')
E1A = BASE / 'outputs/e1-all-core-rule'                         # former rule, run_e1_rule_comparison.py
RULE_COLORS = {'lost_sending': BLUE, 'all_core': ORANGE}       # validated pair (adjacent CVD ΔE 24.7)
RULE_NAMES = {'lost_sending': 'Lost sending (base)', 'all_core': 'All core unable (former)'}
K0_GRAY = MUTED


def rule_lines(ax, x, frames, col, k0_label='κ = 0 (both rules)'):
    """κ = 0 (identical under both rules) in gray, κ = 4 per rule in its color."""
    base = frames['all_core']
    ax.plot(x, base.query('kappa == 0')[col].to_numpy(), color=K0_GRAY, lw=1.5, label=k0_label)
    for rule, d in frames.items():
        ax.plot(x, d.query('kappa == 4')[col].to_numpy(), color=RULE_COLORS[rule], label=f'{RULE_NAMES[rule]}, κ = 4')


def r1_index(qs):
    k0 = qs['all_core'].query('kappa == 0')
    x = np.arange(len(k0))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    hold = int(np.where(k0.quarter.str.startswith('2022'))[0][0])
    ax.axvspan(hold - 0.5, x[-1] + 0.5, color=HOLDOUT_WASH, lw=0, zorder=0)
    ax.axhline(1, color=AXIS, lw=1)
    rule_lines(ax, x, qs, 'index_model')
    ax.plot(x, k0.index_banxico.to_numpy(), color=INK, label='Banxico vs its 2017–19 trend')
    q2 = int(np.where(k0.quarter.eq('2020Q2'))[0][0])
    for rule, d in qs.items():
        y = d.query('kappa == 4').index_model.iloc[q2]
        ring_dot(ax, q2, y, RULE_COLORS[rule])
        ax.annotate(f'{y:.2f}', (q2, y), xytext=(10, 0), textcoords='offset points', va='center', fontsize=8.5,
                    color=INK2)
    y = k0.index_model.iloc[q2]
    ring_dot(ax, q2, y, K0_GRAY)
    ax.annotate(f'{y:.2f}', (q2, y), xytext=(0, -14), textcoords='offset points', ha='center', fontsize=8.5,
                color=INK2)
    ax.set_xticks(x, [s.replace('Q', ' Q') if s.endswith('Q1') else s[-2:] for s in k0.quarter])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_ylim(0.82, 1.2)
    ax.text(hold - 0.3, 0.825, '2022 held out', fontsize=8.5, color=INK2, va='bottom')
    ax.set_ylabel('Transfers relative to no-COVID path\n(quarterly)')
    ax.legend(loc='upper left', ncol=2)
    save_bare(fig, 'R1_count_index_rules', 'Kin-for-kin stepping-in fills the 2020 dip, but not the 2021 rise',
              'E1a (β = 0, ρ = 0, no benefits), quarterly counts relative to the no-COVID path (Banxico: relative to '
              'its 2017–19 trend). κ = 0 is the '
              'same under both rules. κ = 4 under the base lost-sending rule (kin respond to the routine sending '
              'lost to any unable linked migrant, core = 1, kin = θ_kin/θ_core) and the former rule (kin step in '
              'only when all eligible core migrants are unable); alphas refitted to 2019 for each rule and κ.')


def r2_profiles(base_profile, new_profile, ceil, banxico_april):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.9))
    for rule, prof in [('all_core', base_profile.query('kappa <= 8')), ('lost_sending', new_profile)]:
        ax.plot(prof.kappa, prof.weighted_rmse_pp, color=RULE_COLORS[rule], marker='o', ms=4, mec=SURFACE,
                mew=1.2, label=RULE_NAMES[rule])
        end = prof.iloc[-1]
        ax.annotate(f'{end.weighted_rmse_pp:.2f}', (end.kappa, end.weighted_rmse_pp), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color=INK2)
        c = ceil.query('rule == @rule')
        bx.plot(c.kappa, c.index_april_2020, color=RULE_COLORS[rule], marker='o', ms=4, mec=SURFACE, mew=1.2)
        end = c.iloc[-1]
        bx.annotate(f'{end.index_april_2020:.2f}', (end.kappa, end.index_april_2020), xytext=(6, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('κ')
    ax.set_title('2019 fit error by network type (pp)', fontsize=10)
    ax.legend(loc='upper left')
    bx.axhline(1, color=AXIS, lw=1)
    bx.axhline(banxico_april, color=INK, lw=1)
    bx.annotate(f'Banxico April 2020 vs trend: {banxico_april:.2f}', (0, banxico_april), xytext=(0, 5),
                textcoords='offset points', fontsize=8.5, color=INK2)
    bx.set_xlabel('κ')
    bx.set_title('April 2020 transfers / April 2019', fontsize=10)
    save_bare(fig, 'R2_kappa_profiles_rules', '2019 still barely constrains κ; the new rule can exceed full replacement',
              'Left: weighted error of the 2019 fit to MMP receipt by network type after refitting α_core and α_kin '
              'at each κ. Right: April 2020 count index against κ (exact, ρ = 0), each point with its own refitted '
              'alphas.')


def r3_size_fit(val):
    sizes = ['1', '2', '3-4', '5-9', '10+']
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        d = val.query('panel == "size" and network_type == @net')
        src = d.query('rule == "all_core" and kappa == 0').set_index('category').reindex(sizes).dropna(subset=['source_n'])
        x = np.array([sizes.index(c) for c in src.index])
        lo, hi = wilson(src.mmp_receipt, src.source_n)
        ok = src.reliable.astype(bool).to_numpy()
        ax.vlines(x, lo, hi, color=INK, lw=1)
        ax.scatter(x[ok], src.mmp_receipt[ok], s=36, color=INK, zorder=4, label='MMP (95% interval)')
        ax.scatter(x[~ok], src.mmp_receipt[~ok], s=36, facecolor=SURFACE, edgecolor=INK, zorder=4)
        k0 = d.query('rule == "all_core" and kappa == 0').set_index('category').loc[src.index]
        ax.plot(x, k0.model_receipt, color=K0_GRAY, lw=1.5, label='Model, κ = 0 (both rules)')
        for rule in ['all_core', 'lost_sending']:
            m = d.query('rule == @rule and kappa == 4').set_index('category').loc[src.index]
            ax.plot(x, m.model_receipt, color=RULE_COLORS[rule], label=f'{RULE_NAMES[rule]}, κ = 4')
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_title(net, fontsize=10)
        ax.set_xlabel('Network size')
        pct_axis(ax)
    axes[0].set_ylabel('Monthly receipt probability, 2019')
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    save_bare(fig, 'R3_2019_size_fit_rules', 'Both rules keep the 2019 network-size fit',
              'MMP receipt by network size (dots, 95% Wilson intervals; open dots: fewer than 30 households) and '
              'model receipt after refitting to network-type targets: κ = 0 and κ = 4 under each rule.')


def r4_paths(paths):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, net in zip(axes, NETWORKS):
        sel = {r: p[p.group.eq('network_type') & p.level.eq(net)] for r, p in paths.items()}
        dates = pd.to_datetime(sel['all_core'].query('kappa == 0').date)
        ax.axhline(1, color=AXIS, lw=1)
        rule_lines(ax, dates, sel, 'receipt_ratio')
        k4 = {r: d.query('kappa == 4').receipt_ratio.to_numpy() for r, d in sel.items()}
        k0 = sel['all_core'].query('kappa == 0').receipt_ratio.to_numpy()
        # Paths within 0.3% count as the same line (alphas differ slightly across rules and κ).
        same_rules = np.allclose(k4['all_core'], k4['lost_sending'], atol=3e-3)
        base_is_k0 = np.allclose(k4['all_core'], k0, atol=3e-3)
        notes = (['all lines coincide:', 'no kin to step in'] if same_rules and base_is_k0
                 else ['former rule at κ = 4 = κ = 0'] if base_is_k0 else [])
        for rule in (['lost_sending'] if same_rules else ['all_core', 'lost_sending']):
            lo = k4[rule].min()
            ax.annotate(f'{lo:.2f}', (pd.Timestamp('2020-04-01'), lo), xytext=(8, -4 if rule == 'all_core' else 6),
                        textcoords='offset points', fontsize=8.5, color=INK2)
        if notes:
            ax.text(0.97, 0.30, '\n'.join(notes), transform=ax.transAxes, ha='right', fontsize=8, color=MUTED)
        ax.set_title(net, fontsize=10)
        ax.set_xlim(pd.Timestamp('2019-07-01'), pd.Timestamp('2022-06-01'))
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    axes[0].set_ylabel('Receipt probability / counterfactual')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    save_bare(fig, 'R4_receipt_by_network_type_rules', 'Under the base rule, kin-only households also gain from stepping-in',
              'E1a, mean monthly receipt probability (exact) relative to the no-COVID counterfactual, by network type. '
              'Values: April 2020 low at κ = 4 under each rule.')


def r5_size(outs):
    sizes = ['1', '2', '3-4', '5-9', '10+']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [('interruption_rate', 'Interrupted months\n(share of counterfactual receipt months)'),
              ('usd_drop', 'Drop in USD received, Apr 2020 – Dec 2021')]
    series = [('κ = 0 (both rules)', outs['all_core'], 0, K0_GRAY, -0.16)] + [
        (f'{RULE_NAMES[r]}, κ = 4', outs[r], 4, RULE_COLORS[r], dx) for r, dx in [('all_core', 0), ('lost_sending', 0.16)]]
    for ax, (col, title) in zip(axes, panels):
        for label, out, kappa, c, dx in series:
            d = out.query('group == "size_bin" and kappa == @kappa').set_index('level').loc[sizes]
            ax.plot(np.arange(5) + dx, d[col], lw=0, marker='o', ms=7, color=c, mec=SURFACE, mew=1.5, label=label)
        ax.set_xticks(range(5), sizes)
        ax.set_xlabel('Network size (linked migrants)')
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color=AXIS, lw=1)
        lo = min(0, min(o.query('group == "size_bin"')[col].min() for _, o, *_ in series))
        ax.set_ylim(lo * 1.15, None)
        pct_axis(ax)
    axes[1].text(0.02, 0.03, 'below 0: more USD than in the counterfactual', transform=axes[1].transAxes,
                 fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    save_bare(fig, 'R5_outcomes_by_size_rules', 'Under the base rule, large networks are protected, even over-protected at κ = 4',
              'E1a, 50 paired stochastic runs, Apr 2020 – Dec 2021, by network size: interrupted months and drop in '
              'USD received relative to the counterfactual (negative: households receive more than without COVID). '
              'Size 10+ has only 30 households.')


def r6_kin(ms):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axhline(0, color=AXIS, lw=1)
    d0 = ms['all_core'].query('kappa == 0')
    dates = pd.to_datetime(d0.date)
    ax.plot(dates, d0.kin_transfers_covid - d0.kin_transfers_cf, color=K0_GRAY, lw=1.5, label='κ = 0 (both rules)')
    for rule, m in ms.items():
        d = m.query('kappa == 4')
        y = d.kin_transfers_covid - d.kin_transfers_cf
        ax.plot(dates, y, color=RULE_COLORS[rule], label=f'{RULE_NAMES[rule]}, κ = 4')
        i = int(np.argmax(y.to_numpy()))
        ring_dot(ax, dates.iloc[i], y.iloc[i], RULE_COLORS[rule])
        ax.annotate(f'{y.iloc[i]:+.0f}', (dates.iloc[i], y.iloc[i]), xytext=(8, 0), textcoords='offset points',
                    va='center', fontsize=8.5, color=INK2)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.set_ylabel('Kin transfers per month vs counterfactual\n(model units)')
    ax.legend(loc='upper right')
    save_bare(fig, 'R6_kin_transfers_rules', 'How many extra kin transfers each rule produces',
              'E1a, expected kin transfers minus the no-COVID counterfactual. The April 2020 loss of core transfers is '
              'about 130 a month under both rules.')


def rule_figures():
    both = lambda name: {'lost_sending': pd.read_csv(E1 / name), 'all_core': pd.read_csv(E1A / name)}
    q, m = both('aggregate_quarterly.csv'), both('aggregate_monthly.csv')
    r1_index(q)
    bm = m['all_core'].query('kappa == 0').set_index('date')
    r2_profiles(pd.read_csv(E1A / 'e0_kappa_profile.csv'), pd.read_csv(E0 / 'kappa_profile.csv'),
                pd.read_csv(E1A / 'kappa_ceiling_rules.csv'), bm.loc['2020-04-01', 'index_banxico'])
    r3_size_fit(pd.read_csv(E1A / 'e0_validation.csv', dtype={'category': str}))
    r4_paths(both('receipt_ratio_paths.csv'))
    r5_size(both('household_outcomes.csv'))
    r6_kin(m)
    write_captions()


# %% M. E2 compatibility map (first pass, count criterion only)
E2 = BASE / 'outputs/e2-map'
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # palette ramp, light->dark


def m1_map(grid, settings):
    from matplotlib.colors import LinearSegmentedColormap
    tau = settings['tau_C']
    piv = grid.pivot(index='beta', columns='kappa', values='rmse_log_quarterly').sort_index(ascending=False)
    ok = grid.pivot(index='beta', columns='kappa', values='count_compatible').sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    cmap = LinearSegmentedColormap.from_list('seq', SEQ_BLUE)
    im = ax.imshow(piv.values, cmap=cmap, aspect='auto', vmin=0, vmax=min(0.15, piv.values.max()))
    ax.grid(False)
    rows, cols = np.where(ok.values.astype(bool))
    for r, c in zip(rows, cols):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.5))
    b = np.unravel_index(np.argmin(piv.values), piv.shape)
    ax.scatter(b[1], b[0], s=50, color=SURFACE, edgecolor=INK, lw=1.5, zorder=5)
    ax.annotate(f'best {piv.values[b]:.3f}', (b[1], b[0]), xytext=(8, 0), textcoords='offset points', va='center',
                fontsize=8.5, color=INK, bbox=dict(boxstyle='round,pad=0.2', fc=SURFACE, ec='none', alpha=0.85))
    ax.set_xticks(range(piv.shape[1]), [f'{k:g}' for k in piv.columns])
    ax.set_yticks(range(0, piv.shape[0], 2), [f'{v:.2f}' for v in piv.index[::2]])
    ax.set_xlabel('κ (kin stepping-in)')
    ax.set_ylabel('β (need response per 10,000 COVID deaths)')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Quarterly log RMSE, 2020 Q2 – 2021 Q4', color=INK2)
    cb.outline.set_visible(False)
    cb.ax.axhline(tau, color=INK, lw=1.5)
    n = int(ok.values.sum())
    save_bare(fig, 'M1_compatibility_map', f'{n} of {ok.size} (β, κ) cells fit within τ_C = {tau}',
              f'E2 first pass (π = 0, ρ = 0, base lost-sending rule, S_t = reported COVID deaths / 10,000). Fit error '
              f'of the quarterly transfer index (model: COVID run / no-COVID run) against Banxico relative to its '
              f'2017–19 trend. Outlined: count-compatible cells (RMSE ≤ τ_C = {tau}; the ENIGH household criterion is not '
              f'applied yet). Dot: best cell. α refitted to 2019 for each κ. Line on the colour bar: τ_C.')


def m2_paths(paths, grid, settings):
    tau = settings['tau_C']
    quarters = sorted(paths.quarter.unique())
    x = np.arange(len(quarters))
    fig, ax = plt.subplots(figsize=(10, 4.4))
    hold = quarters.index('2022Q1')
    ax.axvspan(hold - 0.5, x[-1] + 0.5, color=HOLDOUT_WASH, lw=0, zorder=0)
    ax.axhline(1, color=AXIS, lw=1)
    ok = grid[grid.count_compatible]
    for _, c in ok.iterrows():
        d = paths[(paths.kappa == c.kappa) & (paths.beta == c.beta)].set_index('quarter').loc[quarters]
        ax.plot(x, d.index_model, color=K0, lw=1, alpha=0.7, zorder=2)
    best = grid.loc[grid.rmse_log_quarterly.idxmin()]
    d = paths[(paths.kappa == best.kappa) & (paths.beta == best.beta)].set_index('quarter').loc[quarters]
    ax.plot(x, d.index_model, color=K4, zorder=3, label=f'Best cell: κ = {best.kappa:g}, β = {best.beta:.2f}')
    z = paths[(paths.kappa == 4.0) & (paths.beta == 0.0)].set_index('quarter').loc[quarters]
    ax.plot(x, z.index_model, color=MUTED, lw=1.5, zorder=2, label='β = 0, κ = 4 (E1a)')
    ax.plot(x, d.index_banxico, color=INK, zorder=4, label='Banxico vs its 2017–19 trend')
    if len(ok):
        ax.plot([], [], color=K0, lw=1, label=f'Other count-compatible cells ({len(ok) - 1})')
    ax.set_xticks(x, [q.replace('Q', ' Q') if q.endswith('Q1') else q[-2:] for q in quarters])
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.set_ylabel('Transfers relative to no-COVID path\n(quarterly)')
    ax.text(hold - 0.3, ax.get_ylim()[0] + 0.003, '2022 held out', fontsize=8.5, color=INK2, va='bottom')
    ax.legend(loc='upper left')
    save_bare(fig, 'M2_compatible_paths', 'The best cell follows the COVID-death waves; Banxico rises steadily',
              f'E2 first pass. Quarterly transfer index of the best cell and the other count-compatible cells '
              f'(RMSE ≤ τ_C = {tau}; {int(grid.count_compatible.sum())} cells), the β = 0 path at κ = 4, and Banxico '
              'relative to its 2017–19 trend. '
              '2022 is held out.')


def e2_figures():
    settings = json.loads((E2 / 'run_settings.json').read_text())
    grid = pd.read_csv(E2 / 'map.csv')
    m1_map(grid, settings)
    m2_paths(pd.read_csv(E2 / 'quarterly_paths.csv'), grid, settings)
    write_captions()


# %% H. Household distribution, beta = 0 (run_e1_distribution.py)
EH = BASE / 'outputs/e1-distribution'
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


def h1_realized(rb, ri):
    losses = ['Lost everything', 'Lost 50–99%', 'Lost 25–50%', 'Lost 10–25%']
    small = ['Lost 0–10%']
    gains = ['Gained 0–10%', 'Gained 10–25%', 'Gained > 25%']
    kappas = [0.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axvline(0, color=AXIS, lw=1)
    for i, k in enumerate(kappas):
        d = rb[rb.kappa == k].set_index('bin').share * 100
        y = len(kappas) - 1 - i
        left = 0.0
        # mildest loss next to zero, total loss at the far end
        for b, c in list(zip(losses + small, LOSS_RAMP + ['#f3c2c1']))[::-1]:
            w = d[b]
            ax.barh(y, -w, left=-left, height=0.5, color=c, edgecolor=SURFACE, lw=1.5)
            left += w
        lost_any = left
        right = 0.0
        for b, c in zip(gains, GAIN_RAMP):
            ax.barh(y, d[b], left=right, height=0.5, color=c, edgecolor=SURFACE, lw=1.5)
            right += d[b]
        ax.annotate(f'lost: {lost_any:.0f}%', (-lost_any, y), xytext=(-6, 0), textcoords='offset points',
                    ha='right', va='center', fontsize=8.5, color=INK2)
        ax.annotate(f'gained: {right:.1f}% · unchanged: {d["No change"]:.0f}%', (right, y), xytext=(6, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(kappas)), [f'κ = {k:g}' for k in kappas[::-1]])
    ax.grid(axis='y', visible=False)
    ax.set_xlim(-30, 22)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{abs(v):.0f}%'))
    ax.set_xlabel('Share of receiving households in a typical COVID path  (← lost · gained →)')
    handles = [Patch(color=c, label=b) for b, c in zip(losses + small, LOSS_RAMP + ['#f3c2c1'])] + \
              [Patch(color=c, label=b) for b, c in zip(gains, GAIN_RAMP)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=8)
    k0 = ri[ri.kappa == 0].iloc[0]
    save_bare(fig, 'H1_realized_losses', 'In a typical COVID path, one in five households loses; 1.5% lose everything',
              'β = 0, ρ = 0, no benefits (E1a). Each receiving household is compared with its own no-COVID twin '
              '(common random numbers), Apr 2020 – Dec 2021: change in USD received, pooled over 200 simulated paths. '
              f'About {k0.receiving_households:,.0f} of 3,130 households receive anything in a path. Stepping-in (κ) '
              'reduces moderate losses and creates some gains, but not the total losses: those households have '
              'nobody who can step in. Gains come from occupations whose employment rose above 2019.')


def h2_expected(hist, ineq):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k in [0.0, 4.0]:
        d = hist[hist.kappa == k]
        x = np.r_[d.bin_left.values, d.bin_right.values[-1]] * 100
        y = np.r_[d.share_households.values, d.share_households.values[-1]] * 100
        ax.step(x, y, where='post', color=KAPPA_COLORS[k], label=f'κ = {k:g}')
    ax.axvline(0, color=AXIS, lw=1)
    for k in [0.0, 4.0]:
        a = ineq[ineq.kappa == k].iloc[0]
        ax.axvline(a.aggregate_change * 100, color=KAPPA_COLORS[k], lw=1, ymax=0.9)
    ax.set_xlim(-20, 25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:+.0f}%'))
    pct_axis(ax)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xlabel('Expected change in USD received, Apr 2020 – Dec 2021 (household vs its no-COVID twin)')
    ax.set_ylabel('Share of households')
    ax.legend(loc='upper right')
    save_bare(fig, 'H2_expected_change', 'Ex ante, almost every household faces a similar small loss',
              'β = 0. Expected change per household (average over 200 paired paths), 2.5-point bins; thin lines: '
              'aggregate change (−4.4% at κ = 0, −0.6% at κ = 4). With ρ = 0 and kin occupations copied from core '
              'donors, households differ ex ante only through their migrants’ occupations and, with κ > 0, their network: '
              'the inequality in H1 is realized, not expected.')


def h3_groups(g):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    labels, ys, y = [], [], 0
    for col, title, levels in GROUP_ROWS:
        labels.append((y, title, True))
        y += 1
        for lv in levels:
            labels.append((y, lv, False))
            ys.append((col, lv, y))
            y += 1
        y += 0.4
    panels = [('change_usd_weighted', 'Expected change in USD received'),
              ('loss_risk_25', 'Risk of losing > 25% (share of paths)')]
    for ax, (metric, title) in zip(axes, panels):
        if metric == 'change_usd_weighted':
            ax.axvline(0, color=AXIS, lw=1)
        for k, dx in [(0.0, -0.15), (4.0, 0.15)]:
            d = g[g.kappa == k].set_index(['group', 'level'])
            xs = [d.loc[(c, lv), metric] * 100 for c, lv, _ in ys]
            ax.scatter(xs, [yy + dx for _, _, yy in ys], s=34, color=KAPPA_COLORS[k], edgecolor=SURFACE, lw=1.2,
                       zorder=3, label=f'κ = {k:g}')
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax.grid(axis='y', visible=False)
        if metric == 'loss_risk_25':
            ax.set_xlim(0, None)
    axes[0].set_yticks([l[0] for l in labels],
                       [l[1] if l[2] else OCC_SHORT.get(l[1], l[1]) for l in labels])
    for tick, l in zip(axes[0].get_yticklabels(), labels):
        if l[2]:
            tick.set_fontweight('semibold')
            tick.set_color(INK)
    axes[0].invert_yaxis()
    n10 = g[(g.kappa == 0) & (g.group == 'size_bin') & (g.level == '10+')].households.iloc[0]
    axes[1].legend(loc='lower right')
    save_bare(fig, 'H3_who_loses', 'Who loses: sales and service workers’ households, and those with nobody to step in',
              'β = 0, ρ = 0, no benefits; 200 paired paths, Apr 2020 – Dec 2021. Left: expected change in USD received '
              'by group (USD-weighted). Right: share of paths in which a household loses more than 25%, averaged over '
              f'the group. Main occupation: the most common job group among the household’s eligible migrants. Network '
              f'size 10+ has only {n10} households.')


def dist_figures():
    h1_realized(pd.read_csv(EH / 'realized_bins.csv'), pd.read_csv(EH / 'realized_inequality.csv'))
    h2_expected(pd.read_csv(EH / 'change_histogram.csv'), pd.read_csv(EH / 'inequality.csv'))
    h3_groups(pd.read_csv(EH / 'groups.csv', dtype={'level': str}))
    write_captions()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    if '--theta' in sys.argv:
        return p1_sending_probability()
    if '--bubbles' in sys.argv:
        return p2_receipt_bubbles()
    if '--dist' in sys.argv:
        return dist_figures()
    if '--e2' in sys.argv:
        return e2_figures()
    if '--e1' in sys.argv:
        return e1_figures()
    if '--e1-rules' in sys.argv:
        return rule_figures()
    h = pd.read_parquet(abm.POPULATION / 'households.parquet')
    m = pd.read_parquet(abm.POPULATION / 'migrants.parquet')
    summary = pd.read_csv(BASE / 'data-prep/outputs/population_summary.csv')
    a1_network_sizes(h)
    a2_old_vs_new(h)
    a3_close_vs_distant(summary)
    a4_occupations(m)
    b1_employment()
    b2_amounts()
    b3_salience()
    cells = e0_cells()
    size = pd.read_csv(E0 / 'validation_network_size.csv')
    c1_ladder(cells.copy())
    c2_size(size)
    c3_errors(cells.copy(), size)
    c4_kappa()
    c5_tiles()
    e1_figures()
    e2_figures()


def e1_figures():
    d1_quarterly(pd.read_csv(E1 / 'aggregate_quarterly.csv'))
    m = pd.read_csv(E1 / 'aggregate_monthly.csv')
    d1_monthly(m)
    d2_decomposition(m)
    d3_reach(pd.read_csv(E1A / 'kappa_reach.csv'), pd.read_csv(E1A / 'kappa_ceiling.csv'))
    d4_variants(pd.read_csv(E1A / 'kappa_rule_variants.csv'))
    d5_receipt_paths(pd.read_csv(E1 / 'receipt_ratio_paths.csv'))
    d6_size(pd.read_csv(E1 / 'household_outcomes.csv'))
    write_captions()


def p1_sending_probability():
    """Sending probability of an able core and an able kin migrant under the coordination rule,
    along kappa (2019-refitted alphas, interpolated) for three betas, at the Jan 2021 death peak."""
    profile = pd.read_csv(BASE / 'outputs/e1-coordination/e0_kappa_profile.csv').query('kappa <= 1')
    best = json.loads((BASE / 'outputs/e2-map-coordination/run_settings.json').read_text())['best_cell']
    s_peak = float(abm.salience_series(pd.DatetimeIndex(['2021-01-01']))[0])
    kappa = np.linspace(0, 1, 201)
    a_core = np.interp(kappa, profile.kappa, profile.alpha_core)
    a_kin = np.interp(kappa, profile.kappa, profile.alpha_kin)
    betas = [(0.0, BLUE), (best['beta'], ORANGE), (0.10, AQUA)]
    marks = [0.0, 0.25, 0.5, 0.75, 1.0]

    def theta(k, b, who):
        """Two example households: core = three core migrants, one unable (Lambda/P = 1/2);
        kin = one core migrant unable, one kin migrant able (Lambda/P = theta_core0 / theta_kin0)."""
        ac, ak = np.interp(k, profile.kappa, profile.alpha_core), np.interp(k, profile.kappa, profile.alpha_kin)
        tc, tk = abm.expit(ac + b * s_peak), abm.expit(ak + b * s_peak)
        return np.minimum(1, tc * (1 + k / 2)) if who == 'core' else np.minimum(1, tk * (1 + k * tc / tk))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
    panels = [('core', 'Core migrant', 'Three core migrants, one has lost their job'),
              ('kin', 'Kin migrant', 'One core migrant has lost their job, one kin migrant is able')]
    for ax, (who, head, sub) in zip(axes, panels):
        routine = abm.expit(a_core[0] if who == 'core' else a_kin[0])
        ax.axhline(routine, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(1.0, routine, '2019 routine', color=MUTED, fontsize=7.5, ha='right', va='bottom')
        for b, color in betas:
            y = theta(kappa, b, who)
            ax.plot(kappa, y, color=GRAY_LIGHT, lw=2, zorder=2)
            ring_dot(ax, marks, theta(np.array(marks), b, who), color, size=46, zorder=4)
            ax.text(1.03, y[-1], f'β = {b:g}', color=INK2, fontsize=8, va='center')
        yb = theta(np.array([best['kappa']]), best['beta'], who)[0]
        ax.scatter(best['kappa'], yb, s=150, facecolor='none', edgecolor=INK, linewidth=1.3, zorder=5)
        ax.annotate(f'best fit: κ = {best["kappa"]:g}, β = {best["beta"]:g}\nθ = {yb:.0%}', (best['kappa'], yb),
                    xytext=(-14, 22), textcoords='offset points',
                    ha='right', fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
        ax.set_title(head, loc='left', fontsize=10, fontweight='semibold', pad=20)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va='bottom')
        ax.set_xlim(-0.03, 1.02)
        ax.set_xticks(marks)
        ax.set_xlabel('κ, replacement share')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(0, 1.04)
    axes[0].set_ylabel('Monthly probability of sending', labelpad=4)
    pct_axis(axes[0])
    handles = [Line2D([], [], marker='o', ls='-', color=GRAY_LIGHT, markerfacecolor=c, markeredgecolor=SURFACE,
                      markersize=7, label=f'β = {b:g}') for b, c in betas]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.25, top=0.80)
    save(fig, 'P1_sending_probability', 'How β and κ move the probability of sending', title_y=0.99, caption=
         f'Monthly sending probability of an able migrant under the coordination rule, '
         f'θ = min(1, θ⁰(1 + κΛ/P)) with logit θ⁰ = α + βS, at the January 2021 death peak '
         f'(S = {s_peak:.2f}, i.e. {s_peak * 1e4:,.0f} reported COVID-19 deaths). Grey curves: all κ in [0, 1], '
         f'with α refitted to 2019 for each κ (interpolated between the 0.1 grid). Dots: κ = 0, 0.25, 0.5, 0.75, 1. '
         f'Ring: best cell of the E2 map under this rule (quarterly log RMSE 0.043). Left: household with three core '
         f'migrants, one unable to send (Λ/P = 1/2). Right: household with one core migrant unable and one kin '
         f'migrant able (Λ/P = θ⁰_core/θ⁰_kin). Dashed line: 2019 routine probability (β = 0, κ = 0).')
    write_captions()


def p2_receipt_bubbles():
    """Exact monthly receipt probability (rho = 0) by core x kin migrant counts under the coordination
    rule: no-COVID level, and the COVID change with no response (beta = kappa = 0) and at the best cell."""
    profile = pd.read_csv(BASE / 'outputs/e1-coordination/e0_kappa_profile.csv')
    best = json.loads((BASE / 'outputs/e2-map-coordination/run_settings.json').read_text())['best_cell']
    pop = abm.load_population()
    h = pop['h']
    months = pd.date_range('2020-04-01', '2021-12-01', freq='MS')
    covid = abm.make_inputs(months, pop['rate_columns'], salience=abm.salience_series(months))
    cf = abm.make_inputs(pd.DatetimeIndex([pd.Timestamp(2019, d.month, 1) for d in months]), pop['rate_columns'])

    def params(kappa, beta):
        return abm.Params(alpha_core=np.interp(kappa, profile.kappa, profile.alpha_core),
                          alpha_kin=np.interp(kappa, profile.kappa, profile.alpha_kin),
                          beta=beta, kappa=kappa, rule='coordination')

    p0, pb = params(0.0, 0.0), params(best['kappa'], best['beta'])
    core_bins, kin_bins = ['0', '1', '2', '3+'], ['0', '1', '2', '3–4', '5+']
    d = pd.DataFrame(dict(
        core=pd.cut(h.core_count, [-1, 0, 1, 2, np.inf], labels=core_bins),
        kin=pd.cut(h.kin_count, [-1, 0, 1, 2, 4, np.inf], labels=kin_bins),
        level=abm.expected_receipt(pop, cf, pb).mean(0),
        none=(abm.expected_receipt(pop, covid, p0) - abm.expected_receipt(pop, cf, p0)).mean(0),
        best=(abm.expected_receipt(pop, covid, pb) - abm.expected_receipt(pop, cf, pb)).mean(0)))
    g = d.groupby(['core', 'kin'], observed=False).agg(n=('level', 'size'), level=('level', 'mean'),
                                                        none=('none', 'mean'), best=('best', 'mean')).reset_index()
    g['x'] = g.core.cat.codes
    g['y'] = g.kin.cat.codes
    min_n = 10
    shown = g[g.n >= min_n]
    hidden = g[g.n < min_n]

    s_max = 700
    d_max = shown[['none', 'best']].abs().to_numpy().max()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    panels = [('level', 'No-COVID receipt probability', 'Monthly, 2019 employment, best cell'),
              ('none', 'COVID change: no response', 'β = 0, κ = 0 (job losses only)'),
              ('best', 'COVID change: best cell', f'β = {best["beta"]:g}, κ = {best["kappa"]:g}')]
    for ax, (col, head, sub) in zip(axes, panels):
        v = shown[col].to_numpy()
        if col == 'level':
            size, face, edge = v * s_max, BLUE, BLUE
            labels = [f'{x:.0%}' for x in v]
        else:
            size = np.abs(v) / d_max * s_max
            face = edge = np.where(v < 0, ORANGE, AQUA)
            labels = [f'{x * 100:+.1f}' for x in v]
        ax.scatter(shown.x, shown.y, s=size, facecolor=face, edgecolor=edge, alpha=0.35, linewidth=0, zorder=3)
        ax.scatter(shown.x, shown.y, s=size, facecolor='none', edgecolor=edge, linewidth=1.2, zorder=4)
        for x, y, lab, sz in zip(shown.x, shown.y, labels, size):
            small = sz < 350                      # label beside the bubble, not on its outline
            ax.annotate(lab, (x, y), xytext=(np.sqrt(sz) / 2 + 3 if small else 0, 0), textcoords='offset points',
                        ha='left' if small else 'center', va='center', fontsize=7.5, color=INK, zorder=5)
        for x, y in zip(hidden.x, hidden.y):
            ax.text(x, y, '–', ha='center', va='center', fontsize=9, color=MUTED)
        if col == 'level':
            for x, y, n in zip(shown.x, shown.y, shown.n):
                ax.text(x, y - 0.4, f'n = {n}', ha='center', va='top', fontsize=6.5, color=MUTED)
        ax.set_title(head, loc='left', fontsize=10, fontweight='semibold', pad=18)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va='bottom')
        ax.set_xticks(range(len(core_bins)), core_bins)
        ax.set_yticks(range(len(kin_bins)), kin_bins)
        ax.set_xlim(-0.6, len(core_bins) - 0.4)
        ax.set_ylim(-0.75, len(kin_bins) - 0.4)
        ax.set_xlabel('Core migrants (household members)')
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Kin migrants (wider family)')
    axes[2].text(1.0, -0.2, 'Labels in panels 2–3: percentage points', transform=axes[2].transAxes,
                 ha='right', fontsize=7.5, color=MUTED)
    handles = [Patch(facecolor=ORANGE, edgecolor=ORANGE, alpha=0.6, label='Receives less than without COVID'),
               Patch(facecolor=AQUA, edgecolor=AQUA, alpha=0.6, label='Receives more than without COVID')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.62, -0.05))
    fig.subplots_adjust(wspace=0.12, top=0.80, bottom=0.17)
    save(fig, 'P2_receipt_by_core_kin',
         'Receipt depends mostly on core migrants; with need and coordination, wider family turns losses into gains',
         title_y=0.99, caption=
         f'Exact expected monthly probability that a household receives a transfer (ρ = 0, coordination rule), '
         f'averaged over households in each cell, Apr 2020 – Dec 2021. Left: no-COVID counterfactual (2019 CPS '
         f'rates of the same month, S = 0) at the best cell, bubble area ∝ probability. Middle and right: COVID '
         f'run minus counterfactual, bubble area ∝ |change| on a common scale; α refitted to 2019 for each κ. '
         f'Best cell of the E2 map (quarterly log RMSE {best["rmse_log_quarterly"]:.3f}). '
         f'Cells with fewer than {min_n} households are not shown (–). There are no households without migrants.')
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
