"""Analyze remittance probability by Core vs. Kin network sizes with expanded bins
and specific configuration curves using MMP data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

# %% 1. Settings & Paths
ROOT = Path('C:/git-projects/covid-remittances')
DATA = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
# Separate corrected outputs from the earlier figures made with invalid counts.
OUT = ROOT / 'oct-26/outputs/extended-network-analysis-corrected'

START_YEAR, END_YEAR = 2010, 2019
KEY = ['country', 'commun', 'surveypl', 'hhnum']
MIN_CELL_SIZE = 15  # Lowered guardrail threshold
# Core = head/spouse/son/daughter. Kin = other relatives, excluding friends.
# Within each overlapping relationship, use max(PERS, HOUSE), matching the
# default synthetic-network script. This assumes overlap, not proven identity.
CORE_CODES = [1, 2, 3]
KIN_CODES = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]

PARENT_SLOTS = [('mousnw', 4), ('fausnw', 4)]
EXTENDED_COUNTS = {
    'uncinus': 7,    # Aunts/Uncles
    'coinus': 8,     # Cousins
    'neinus': 6,     # Nieces/Nephews
    'sbilinus': 15,  # Siblings-in-law
    'chilinus': 14,  # Children-in-law
    'pilinus': 16    # Parents-in-law
}


def build_network_counts(h, p):
    """Keep known lower counts and uncertainty; never count a status code as people.

    Complete counts refer to the selected recorded fields, not complete real
    family networks. HOUSE/PERS can overlap within relationship categories.
    """
    assert not h.duplicated(KEY).any(), 'Duplicate HOUSE keys'
    assert not p.duplicated(KEY + ['persnum']).any(), 'Duplicate PERS keys'
    p = p.merge(h[KEY + ['surveyyr']], on=KEY, how='inner',
                validate='many_to_one', suffixes=('', '_house'))
    assert p.surveyyr.eq(p.surveyyr_house).all(), 'HOUSE/PERS year mismatch'
    result = h.set_index(KEY).copy()
    alive = ~p.hhmemshp.eq(8888)
    current = p.uscurtrp.eq(1) & alive
    family = CORE_CODES + KIN_CODES
    counts = (p.loc[current & p.relhead.isin(family)]
              .groupby(KEY + ['relhead']).size().unstack('relhead', fill_value=0)
              .reindex(index=result.index, columns=family, fill_value=0).fillna(0))
    has_roster = p.groupby(KEY).size().reindex(result.index, fill_value=0).gt(0)
    unknown_status = ~p.uscurtrp.isin([1, 2, 8888]) & alive
    unknown_relation = ~p.relhead.isin(family + [19]) & (current | unknown_status)
    for name, codes in [('core', CORE_CODES), ('kin', KIN_CODES)]:
        uncertain = unknown_relation | (unknown_status & p.relhead.isin(codes))
        flags = p.assign(uncertain=uncertain).groupby(KEY).uncertain.any()
        result[name + '_unknown'] = flags.reindex(result.index, fill_value=False) | ~has_roster
    result['core_count_known'] = counts[CORE_CODES].sum(axis=1).astype(int)
    kin = counts[KIN_CODES].copy()
    result['house_unknown_fields'] = 0
    result['house_alive_status_conflicts'] = 0
    result['assumed_overlap'] = 0
    house_counts = {}
    for rel, status_alive_pairs in [
        (4, [('mousnw', 'molive'), ('fausnw', 'falive')]),
        (5, [(f'sbusnw{i}', f'sblive{i}') for i in range(1, 13)])]:
        values = []
        for status, living in status_alive_pairs:
            # 1=yes -> 1 person; 2=no and 8888=not applicable -> zero.
            # Missing/9999 remain unknown. Confirmed deceased people count zero.
            value = result[status].map({1: 1., 2: 0., 8888: 0.})
            dead = result[living].eq(2)
            conflict = result[status].eq(1) & dead
            result['house_alive_status_conflicts'] += conflict.astype(int)
            value = value.mask(dead, 0.)
            result['house_unknown_fields'] += value.isna().astype(int)
            result['kin_unknown'] |= value.isna() | conflict
            values.append(value)
        house_counts[rel] = pd.concat(values, axis=1).sum(axis=1)
    for field, rel in EXTENDED_COUNTS.items():
        value = result[field].replace({8888: np.nan, 9999: np.nan})
        assert (value.dropna().ge(0) & value.dropna().mod(1).eq(0)).all(), f'Invalid {field}'
        result['house_unknown_fields'] += value.isna().astype(int)
        result['kin_unknown'] |= value.isna()
        # Zero here is only a lower bound; the uncertainty flag prevents it
        # entering a zero-network analysis cell as a known count.
        house_counts[rel] = value.fillna(0)
    for rel, known_house_count in house_counts.items():
        result['assumed_overlap'] += np.minimum(kin[rel], known_house_count).astype(int)
        kin[rel] = np.maximum(kin[rel], known_house_count)
    result['kin_count_known'] = kin.sum(axis=1).astype(int)
    result['core_count'] = result.core_count_known.where(~result.core_unknown)
    result['kin_count'] = result.kin_count_known.where(~result.kin_unknown)
    result['network_counts_complete'] = ~result.core_unknown & ~result.kin_unknown
    return result.reset_index()


# ---------------------------------------------------------------------------
# Helpers for the added plots (unweighted, same convention as the heatmap)
# ---------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    """Naive binomial interval; does not account for community clustering."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def prob_ci(df, by):
    """Remittance probability (%) with Wilson 95% CI per group; drops cells with n < MIN_CELL_SIZE."""
    known = df.loc[df['receives_us'].notna()]
    s = known.groupby(by, observed=True)['receives_us'].agg(k='sum', n='count').reset_index()
    s = s[s['n'] >= MIN_CELL_SIZE].copy()
    lo, hi = wilson_ci(s['k'], s['n'])
    s['p'] = 100 * s['k'] / s['n']
    s['lo'] = 100 * lo
    s['hi'] = 100 * hi
    return s


def plot_interaction_lines(h):
    """Plot 1: P(remit) vs kin count, one line per core level, with Wilson CI bands."""
    d = h.assign(core_grp=pd.cut(h['core_count'], bins=[-1, 0, 1, 2, np.inf],
                                 labels=['0', '1', '2', '3+']))
    s = prob_ci(d, ['core_grp', 'kin_bin'])
    kin_levels = list(h['kin_bin'].cat.categories)
    pos = {lvl: i for i, lvl in enumerate(kin_levels)}
    colors = dict(zip(['0', '1', '2', '3+'], sns.color_palette('viridis', 4)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for lvl in ['0', '1', '2', '3+']:
        sub = s[s['core_grp'] == lvl].copy()
        if sub.empty:
            continue
        sub['x'] = sub['kin_bin'].map(pos).astype(float)
        sub = sub.sort_values('x')
        ax.plot(sub['x'], sub['p'], marker='o', lw=2, color=colors[lvl], label=f'Core = {lvl}')
        ax.fill_between(sub['x'], sub['lo'], sub['hi'], color=colors[lvl], alpha=0.15)
    ax.set_xticks(range(len(kin_levels)))
    ax.set_xticklabels(kin_levels)
    ax.set_xlabel('Kin Network Size Abroad')
    ax.set_ylabel('Remittance Probability (%)')
    ax.set_ylim(0, 100)
    ax.set_title(f'Remittance Probability by Kin Size, Separately by Core Size\n'
                 f'(95% Wilson CI; cells with n<{MIN_CELL_SIZE} omitted; {START_YEAR}-{END_YEAR})')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title='Core migrants')
    fig.tight_layout()
    fig.savefig(OUT / 'plot1_interaction_lines_kin_by_core.png', dpi=170)
    plt.close(fig)


def plot_core_share(h):
    """Plot 2: exact network sizes 1-4, with a separate pooled 5+ panel."""
    d = h.copy()
    d['total_net'] = d['core_count'] + d['kin_count']
    d = d.loc[d['total_net'] > 0].copy()
    d['core_share'] = d['core_count'] / d['total_net']
    total_labels = ['1', '2', '3', '4', '5+']
    d['total_bin'] = pd.cut(d['total_net'], bins=[0, 1, 2, 3, 4, np.inf], labels=total_labels)
    share_labels = ['0%', '1-25%', '26-50%', '51-75%', '76-99%', '100%']
    d['share_bin'] = pd.cut(d['core_share'], bins=[-0.001, 0, 0.25, 0.5, 0.75, 0.9999, 1.0],
                            labels=share_labels)

    s = prob_ci(d, ['total_bin', 'share_bin'])
    pos = {lvl: i for i, lvl in enumerate(share_labels)}

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharey=True)
    for ax, tb in zip(axes, total_labels):
        sub = s[s['total_bin'] == tb].copy()
        sub['x'] = sub['share_bin'].map(pos).astype(float)
        sub = sub.sort_values('x')
        if not sub.empty:
            ax.errorbar(sub['x'], sub['p'],
                        yerr=[sub['p'] - sub['lo'], sub['hi'] - sub['p']],
                        marker='o', lw=2, capsize=3, color='teal')
            for _, r in sub.iterrows():
                ax.annotate(f"n={int(r['n'])}", (r['x'], r['hi']), textcoords='offset points',
                            xytext=(0, 4), ha='center', fontsize=7)
        ax.set_xticks(range(len(share_labels)))
        ax.set_xticklabels(share_labels, rotation=45, ha='right')
        ax.set_title(f'Total network = {tb}')
        ax.set_xlabel('Core share of network')
        ax.set_xlim(-0.5, len(share_labels) - 0.5)
        ax.grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylabel('Remittance Probability (%)')
    axes[0].set_ylim(0, 105)
    fig.suptitle(f'Receipt by Core Share: Exact Network Sizes 1-4; Pooled 5+\n'
                 f'(95% Wilson CI; cells with n<{MIN_CELL_SIZE} omitted)', y=1.05)
    fig.tight_layout()
    fig.savefig(OUT / 'plot2_core_share_by_total_size.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_cell_sizes_with_marginals(h):
    """Plot 6: cell-size heatmap (n with known receipt) plus marginal histograms of core and kin."""
    core_levels = list(h['core_bin'].cat.categories)
    kin_levels = list(h['kin_bin'].cat.categories)

    known = h.loc[h['receives_us'].notna()]
    n_mat = (known.groupby(['core_bin', 'kin_bin'], observed=False).size()
             .unstack('kin_bin').reindex(index=core_levels, columns=kin_levels).fillna(0).astype(int))

    kin_marg = h['kin_bin'].value_counts().reindex(kin_levels).fillna(0)
    core_marg = h['core_bin'].value_counts().reindex(core_levels).fillna(0)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.2], height_ratios=[1.2, 5],
                          hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    sns.heatmap(n_mat, annot=True, fmt='d', cmap='Greys',
                norm=LogNorm(vmin=1, vmax=max(2, n_mat.values.max())),
                mask=(n_mat == 0), cbar=False, ax=ax_main, linewidths=0.5, linecolor='white')
    ax_main.set_xlabel('Kin Network Size Abroad')
    ax_main.set_ylabel('Core Network Size Abroad')

    xs = np.arange(len(kin_levels)) + 0.5
    ys = np.arange(len(core_levels)) + 0.5
    ax_top.bar(xs, kin_marg.values, width=0.8, color='teal')
    for x, v in zip(xs, kin_marg.values):
        ax_top.text(x, v, f'{int(v)}', ha='center', va='bottom', fontsize=8)
    ax_top.set_ylabel('Households')
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title(f'Where the Data Are: Households with Known Receipt per Cell (log shading)\n'
                     f'Complete-network households shown in marginals ({START_YEAR}-{END_YEAR})')

    ax_right.barh(ys, core_marg.values, height=0.8, color='darkorange')
    for y, v in zip(ys, core_marg.values):
        ax_right.text(v, y, f' {int(v)}', ha='left', va='center', fontsize=8)
    ax_right.set_xlabel('Households')
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlim(0, core_marg.max() * 1.25)

    fig.savefig(OUT / 'plot6_cell_sizes_and_marginals.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_year_small_multiples(h):
    """Plot 7: one panel per survey year; P(remit) vs kin size, lines for core = 0 vs core >= 1."""
    d = h.copy()
    d['kin_coarse'] = pd.cut(d['kin_count'], bins=[-1, 0, 1, 2, np.inf], labels=['0', '1', '2', '3+'])
    d['core_coarse'] = pd.cut(d['core_count'], bins=[-1, 0, np.inf], labels=['Core = 0', 'Core ≥ 1'])
    kin_levels = ['0', '1', '2', '3+']
    pos = {lvl: i for i, lvl in enumerate(kin_levels)}
    colors = {'Core = 0': 'teal', 'Core ≥ 1': 'darkorange'}

    years = list(range(START_YEAR, END_YEAR + 1))
    ncols = 5
    nrows = int(np.ceil(len(years) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, yr in zip(axes, years):
        s = prob_ci(d.loc[d['surveyyr'] == yr], ['core_coarse', 'kin_coarse'])
        for lvl, col in colors.items():
            sub = s[s['core_coarse'] == lvl].copy()
            if sub.empty:
                continue
            sub['x'] = sub['kin_coarse'].map(pos).astype(float)
            sub = sub.sort_values('x')
            ax.plot(sub['x'], sub['p'], marker='o', lw=2, color=col, label=lvl)
            ax.fill_between(sub['x'], sub['lo'], sub['hi'], color=col, alpha=0.15)
        n_year = int(d.loc[(d['surveyyr'] == yr) & d['receives_us'].notna()].shape[0])
        ax.set_title(f'{yr} (n={n_year})')
        ax.set_xticks(range(len(kin_levels)))
        ax.set_xticklabels(kin_levels)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
    for ax in axes[len(years):]:
        ax.set_visible(False)

    for ax in axes[::ncols]:
        ax.set_ylabel('Remittance Probability (%)')
    for ax in axes[-ncols:]:
        ax.set_xlabel('Kin migrants')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper right', ncol=2)
    fig.suptitle(f'Receipt by Survey Year (Different Community Samples, Not a Panel)\n'
                 f'(95% Wilson CI; cells with n<{MIN_CELL_SIZE} omitted)', y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'plot7_year_small_multiples.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def run_extended_analysis():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== 1. Loading MMP HOUSE & PERS Data ===")
    sibling_us_cols = [f'sbusnw{i}' for i in range(1, 13)]
    parent_us_cols = ['mousnw', 'fausnw']
    extended_us_cols = list(EXTENDED_COUNTS.keys())

    living_cols = ['molive', 'falive'] + [f'sblive{i}' for i in range(1, 13)]
    house_cols = KEY + ['surveyyr', 'weight', 'remus', 'members'] + parent_us_cols + sibling_us_cols + extended_us_cols + living_cols
    h = pd.read_stata(DATA / 'house174.dta', columns=house_cols, convert_categoricals=False)

    h = h.loc[h.country.eq(1) & h.surveypl.eq(1) & h.surveyyr.between(START_YEAR, END_YEAR)].copy()
    h['receives_us'] = h['remus'].map({1: 1.0, 2: 0.0})

    p = pd.read_stata(DATA / 'pers174.dta',
                      columns=KEY + ['surveyyr', 'persnum', 'hhmemshp', 'relhead', 'uscurtrp'],
                      convert_categoricals=False)
    h = build_network_counts(h, p)
    assert len(h) and h.weight.notna().all() and h.weight.gt(0).all()
    # Keep ALL selected households for audit; only complete counts enter plots.
    h.to_csv(OUT / 'household_network_counts.csv', index=False)
    coverage = h.groupby('network_counts_complete').agg(
        households=('hhnum', 'size'), known_receipt=('receives_us', 'count'),
        recipients=('receives_us', 'sum'))
    coverage.to_csv(OUT / 'network_count_coverage.csv')
    print(coverage.to_string())
    print('Known core/kin connection counts:', int(h.core_count_known.sum()), int(h.kin_count_known.sum()))
    h = h.loc[h.network_counts_complete].copy()
    assert len(h), 'No households with complete recorded network counts.'

    print("=== 2. Applying Expanded Bins (Core: 0-5+, Kin: 0-8+) ===")
    h['core_bin'] = pd.cut(h['core_count'], bins=[-1, 0, 1, 2, 3, 4, np.inf], labels=['0', '1', '2', '3', '4', '5+'])
    h['kin_bin'] = pd.cut(h['kin_count'], bins=[-1, 0, 1, 2, 3, 4, 5, 6, 7, np.inf], labels=['0', '1', '2', '3', '4', '5', '6', '7', '8+'])

    # Matrix Summary
    rows = []
    for name, g in h.groupby(['core_bin', 'kin_bin'], observed=True):
        known = g.loc[g.receives_us.notna()]
        n_known = len(known)
        prob_pct = 100 * known.receives_us.mean() if n_known >= MIN_CELL_SIZE else np.nan
        weighted_prob = 100 * np.average(known.receives_us, weights=known.weight) if n_known >= MIN_CELL_SIZE and known.weight.sum() > 0 else np.nan

        rows.append({
            'core_bin': name[0],
            'kin_bin': name[1],
            'households': len(g),
            'known_receipt': n_known,
            'remittance_probability_pct': prob_pct,
            'weighted_receipt_pct': weighted_prob,
            'small_cell': n_known < MIN_CELL_SIZE
        })

    summary = pd.DataFrame(rows)
    assert summary.households.sum() == len(h)
    assert summary.known_receipt.sum() == h.receives_us.notna().sum()
    summary.to_csv(OUT / 'remittance_probability_extended_matrix.csv', index=False)

    print("\n=== 3. Generating Expanded Heatmap ===")
    pivot_table = summary.pivot(index='core_bin', columns='kin_bin', values='remittance_probability_pct')

    annot_labels = np.empty(pivot_table.shape, dtype=object)
    for r_idx, row_val in enumerate(pivot_table.index):
        for c_idx, col_val in enumerate(pivot_table.columns):
            cell_data = summary.loc[(summary['core_bin'] == row_val) & (summary['kin_bin'] == col_val)]
            if len(cell_data) > 0:
                n = cell_data['known_receipt'].values[0]
                pct = cell_data['remittance_probability_pct'].values[0]
                if pd.isna(pct) or n < MIN_CELL_SIZE:
                    annot_labels[r_idx, c_idx] = f"n={n}\n(<{MIN_CELL_SIZE})"
                else:
                    annot_labels[r_idx, c_idx] = f"{pct:.1f}%\nn={n}"
            else:
                annot_labels[r_idx, c_idx] = "-"

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot_table, annot=annot_labels, fmt="", cmap="YlGnBu", cbar_kws={'label': 'Remittance Probability (%)'}, ax=ax, mask=pivot_table.isna())
    ax.set_title(f'Probability of Receiving US Remittances (%)\nExpanded Core (0-5+) vs. Kin (0-8+) ({START_YEAR}-{END_YEAR})')
    ax.set_xlabel('Kin Network Size Abroad')
    ax.set_ylabel('Core Network Size Abroad')
    fig.tight_layout()
    fig.savefig(OUT / 'expanded_core_vs_kin_heatmap.png', dpi=170)
    plt.close(fig)

    print("=== 4. Generating Configuration Curves ===")
    # Setup 1: Increasing kin migrants with core == 0
    setup1 = h.loc[h.core_count.eq(0)].groupby('kin_count')['receives_us'].agg(['mean', 'size', 'count']).reset_index()
    setup1.columns = ['kin_count', 'probability', 'total_obs', 'known_obs']
    setup1 = setup1[setup1['known_obs'] >= MIN_CELL_SIZE]
    setup1['probability'] *= 100

    # Setup 2: Increasing core migrants with kin == 0
    setup2 = h.loc[h.kin_count.eq(0)].groupby('core_count')['receives_us'].agg(['mean', 'size', 'count']).reset_index()
    setup2.columns = ['core_count', 'probability', 'total_obs', 'known_obs']
    setup2 = setup2[setup2['known_obs'] >= MIN_CELL_SIZE]
    setup2['probability'] *= 100

    # Setup 3: More kin than core migrants (kin_count > core_count)
    h['net_diff'] = h['kin_count'] - h['core_count']
    setup3 = h.loc[(h['net_diff'] > 0) & (h['core_count'] >= 1)].groupby('net_diff')['receives_us'].agg(['mean', 'size', 'count']).reset_index()
    setup3.columns = ['excess_kin', 'probability', 'total_obs', 'known_obs']
    setup3 = setup3[setup3['known_obs'] >= MIN_CELL_SIZE]
    setup3['probability'] *= 100
    for name, table in [('kin_only', setup1), ('core_only', setup2), ('excess_kin', setup3)]:
        table.to_csv(OUT / f'configuration_{name}.csv', index=False)

    # Plotting Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(setup1['kin_count'], setup1['probability'], marker='o', color='teal', lw=2)
    axes[0].set_title('Kin Migrants (Core = 0)')
    axes[0].set_xlabel('Number of Kin Migrants')
    axes[0].set_ylabel('Remittance Probability (%)')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(setup2['core_count'], setup2['probability'], marker='s', color='darkorange', lw=2)
    axes[1].set_title('Core Migrants (Kin = 0)')
    axes[1].set_xlabel('Number of Core Migrants')
    axes[1].set_ylabel('Remittance Probability (%)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(setup3['excess_kin'], setup3['probability'], marker='^', color='purple', lw=2)
    axes[2].set_title('Excess Kin (Kin > Core)')
    axes[2].set_xlabel('Excess Kin Migrants (Kin minus Core)')
    axes[2].set_ylabel('Remittance Probability (%)')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle('Remittance Probability Across Specific Network Configurations', fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / 'network_configuration_curves.png', dpi=170)
    plt.close(fig)

    print("=== 5. Generating Additional Plots (interaction lines, core share, cell sizes, by year) ===")
    plot_interaction_lines(h)
    plot_core_share(h)
    plot_cell_sizes_with_marginals(h)
    plot_year_small_multiples(h)

    (OUT / 'analysis_notes.txt').write_text(
        'Core=head/spouse/son/daughter; kin=other relatives, excluding friends.\n'
        'Parent/sibling status: 1=yes, 2=no, 8888=not applicable; unknown stays unknown.\n'
        'Recorded deceased people excluded; alive/current-status conflicts flagged.\n'
        'Within relationship, max(PERS,HOUSE) assumes maximum overlap.\n'
        'Complete counts mean complete selected fields, not a complete real family network.\n'
        'Incomplete counts are retained in the audit and excluded from all plotted cells.\n'
        'Plots are unweighted; matrix also reports MMP-weighted rates, not national estimates.\n'
        'Wilson intervals are naive binomial intervals: no community-cluster adjustment.\n'
        'Survey years sample different communities; comparisons are not household transitions.\n'
        'The 5+ composition panel pools different network sizes.\n'
        'Receipt is a household outcome, not an individual sending probability.\n', encoding='utf-8')

    print(f"\nAll expanded outputs and configuration curves successfully saved in: {OUT}")

if __name__ == '__main__':
    run_extended_analysis()
