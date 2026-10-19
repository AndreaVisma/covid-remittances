"""Minimal national ENIGH comparison. Run with Python; needs pandas/numpy/matplotlib.
Inputs stay unchanged. Definitions and price conversion: enigh-remittances-notes.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% 1. Settings and small helpers
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'oct-26/outputs/enigh-remittances'
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [2020, 2022]
COLORS = ['#247BA0', '#D97732']
# Approximate conversion inferred from INEGI's published 2020 mean at 2022 prices
# (57,370, rounded), divided by the original 2020 mean (50,309.31315023509).
# This is not a newly estimated CPI series. Nominal values are also exported.
PRICE = {2020: 57370 / 50309.31315023509, 2022: 1.0}

def mean(x, w):
    return np.average(x, weights=w)

def quantile(x, w, q):
    order = np.argsort(np.asarray(x), kind='stable')
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    return x[np.searchsorted(np.cumsum(w), q * w.sum(), side='left')]

def save(fig, filename):
    fig.text(.5, .01, 'Source: ENIGH household microdata; survey-weighted descriptive estimates.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)

# %% 2. Read, validate and classify. Keys/geography remain strings.
frames = []
for year in YEARS:
    d = pd.read_csv(ROOT / f'data/concentradohogar_{str(year)[2:]}.csv',
                    dtype={'folioviv': str, 'foliohog': str, 'ubica_geo': str})
    needed = ['remesas', 'factor', 'ing_cor', 'tot_integ', 'menores', 'p12_64',
              'p65mas', 'ocupados', 'edad_jefe', 'sexo_jefe', 'tam_loc']
    assert d[needed].notna().all().all()
    assert not d.duplicated(['folioviv', 'foliohog']).any()
    assert d.factor.gt(0).all() and d.remesas.ge(0).all()
    assert d.remesas.le(d.ing_cor + .01).all()
    assert (d.menores + d.p12_64 + d.p65mas).eq(d.tot_integ).all()
    assert d.sexo_jefe.isin([1, 2]).all() and d.tam_loc.isin([1, 2, 3, 4]).all()
    d['year'] = year
    d['state'] = d.ubica_geo.str.zfill(5).str[:2]
    d['recipient'] = d.remesas.gt(0)
    d['remit_real'] = d.remesas * PRICE[year]
    d['dependence'] = d.remesas.div(d.ing_cor.where(d.ing_cor.gt(0))) * 100
    d['Head sex'] = d.sexo_jefe.map({1: 'Male', 2: 'Female'})
    d['Head age'] = pd.cut(d.edad_jefe, [0, 39, 64, np.inf], labels=['Under 40', '40-64', '65+'])
    d['Household size'] = pd.cut(d.tot_integ, [0, 1, 2, 4, np.inf], labels=['1', '2', '3-4', '5+'])
    d['Employment'] = pd.cut(d.ocupados, [-1, 0, 1, np.inf], labels=['No employed member', '1 employed member', '2+ employed members'])
    d['Age composition'] = np.select([
        d.menores.gt(0) & d.p65mas.gt(0), d.menores.gt(0), d.p65mas.gt(0)],
        ['Under 12 and 65+', 'Under 12, no 65+', '65+, no under 12'], default='Only ages 12-64')
    d['Locality'] = np.where(d.tam_loc.eq(4), 'Rural (<2,500)', 'Urban (2,500+)')
    frames.append(d)
h = pd.concat(frames, ignore_index=True)

# %% 3. National results and group tables, with explicit denominators
national = []
groups = []
dimensions = ['Head sex', 'Head age', 'Household size', 'Employment', 'Age composition', 'Locality']
for year, d in h.groupby('year'):
    r = d.loc[d.recipient]
    national.append(dict(year=year, sample_n=len(d), recipient_sample_n=len(r),
        households=d.factor.sum(), recipient_households=r.factor.sum(),
        receipt_pct=100 * r.factor.sum() / d.factor.sum(),
        mean_recipient_nominal=mean(r.remesas, r.factor),
        mean_recipient_real=mean(r.remit_real, r.factor),
        median_recipient_real=quantile(r.remit_real, r.factor, .5),
        p25_recipient_real=quantile(r.remit_real, r.factor, .25),
        p75_recipient_real=quantile(r.remit_real, r.factor, .75),
        mean_all_households_real=mean(d.remit_real, d.factor),
        survey_total_real=(d.remit_real * d.factor).sum(),
        mean_dependence_pct=mean(r.dependence, r.factor),
        median_dependence_pct=quantile(r.dependence, r.factor, .5),
        at_least_half_income_pct=100 * mean(r.dependence.ge(50), r.factor)))
    for dimension in dimensions + ['state']:
        for category, g in d.groupby(dimension, observed=True, sort=False):
            recipients = g.loc[g.recipient]
            groups.append(dict(year=year, dimension=dimension, category=str(category),
                sample_n=len(g), recipient_sample_n=len(recipients),
                households=g.factor.sum(), recipient_households=recipients.factor.sum(),
                receipt_pct=100 * recipients.factor.sum() / g.factor.sum(),
                recipient_composition_pct=100 * recipients.factor.sum() / r.factor.sum(),
                all_household_composition_pct=100 * g.factor.sum() / d.factor.sum()))
national = pd.DataFrame(national).set_index('year')
groups = pd.DataFrame(groups)
assert national.households.to_dict() == {2020: 35749659, 2022: 37560123}
assert np.allclose(national.survey_total_real,
                   national.recipient_households * national.mean_recipient_real)
assert np.allclose(groups.groupby(['year', 'dimension']).recipient_composition_pct.sum(), 100)
assert np.allclose(groups.groupby(['year', 'dimension']).all_household_composition_pct.sum(), 100)
national.to_csv(OUT / 'national_summary.csv')
groups.loc[groups.dimension.ne('state')].to_csv(OUT / 'household_groups.csv', index=False)
groups.loc[groups.dimension.eq('state')].to_csv(OUT / 'states_for_later.csv', index=False)
group_changes = groups.loc[groups.dimension.ne('state')].pivot(
    index=['dimension', 'category'], columns='year',
    values=['receipt_pct', 'recipient_composition_pct'])
for metric in ['receipt_pct', 'recipient_composition_pct']:
    group_changes[(metric, 'change_pp')] = group_changes[(metric, 2022)] - group_changes[(metric, 2020)]
group_changes.columns = [f'{a}_{b}' for a, b in group_changes.columns]
group_changes.to_csv(OUT / 'group_changes.csv')
change = pd.DataFrame({'2020': national.loc[2020], '2022': national.loc[2022]})
change['difference'] = change['2022'] - change['2020']
change['relative_change_pct'] = 100 * (change['2022'] / change['2020'] - 1)
change.to_csv(OUT / 'national_changes.csv')

# %% 4. Four easily inspected figures
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
metrics = [('receipt_pct', 'Households receiving remittances', '% of all households', 1),
           ('recipient_households', 'Number of recipient households', 'Millions of households', 1e6),
           ('mean_recipient_real', 'Mean amount among recipients', 'Quarterly pesos (approx. 2022 prices)', 1)]
for ax, (col, title, unit, scale) in zip(axes, metrics):
    values = national[col] / scale
    bars = ax.bar([str(y) for y in YEARS], values, color=COLORS, width=.6)
    ax.bar_label(bars, labels=[f'{v:,.2f}' if scale == 1e6 else f'{v:,.1f}' for v in values], padding=4)
    ax.set_ylim(0, values.max() * 1.2)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(unit)
fig.suptitle('Remittances in Mexico: reach and amounts, 2020-2022')
save(fig, '01_national_overview.png')

for metric, title, filename, unit in [
    ('recipient_composition_pct', 'Who receives remittances?', '02_recipient_composition.png', '% of all recipient households'),
    ('receipt_pct', 'How common is receipt within each household group?', '03_receipt_by_group.png', '% of households in the indicated group')]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, dimension in zip(axes.flat, dimensions):
        t = groups.loc[groups.dimension.eq(dimension)].pivot(index='category', columns='year', values=metric)
        order = {'Head sex': ['Male', 'Female'], 'Head age': ['Under 40', '40-64', '65+'],
            'Household size': ['1', '2', '3-4', '5+'],
            'Employment': ['No employed member', '1 employed member', '2+ employed members'],
            'Age composition': ['Only ages 12-64', 'Under 12, no 65+', '65+, no under 12', 'Under 12 and 65+'],
            'Locality': ['Rural (<2,500)', 'Urban (2,500+)']}[dimension]
        t = t.reindex(order)
        y = np.arange(len(t))
        for i, year in enumerate(YEARS):
            bars = ax.barh(y + (i - .5) * .34, t[year], height=.32, color=COLORS[i], label=str(year))
            ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
        ax.set_yticks(y, t.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, t.to_numpy().max() * 1.22)
        ax.set_title(dimension)
        ax.set_xlabel(unit, fontsize=9)
    handles, labels_legend = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(.5, .925), ncol=2, frameon=False)
    fig.suptitle(title + '\nNational survey-weighted percentages')
    save(fig, filename)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
for year, color in zip(YEARS, COLORS):
    row = national.loc[year]
    axes[0].errorbar(str(year), row.median_recipient_real,
        yerr=[[row.median_recipient_real-row.p25_recipient_real], [row.p75_recipient_real-row.median_recipient_real]],
        fmt='o', capsize=8, color=color, markersize=8)
    axes[0].annotate(f'{row.median_recipient_real:,.0f}', (str(year), row.median_recipient_real), xytext=(12, 0), textcoords='offset points')
axes[0].set_xlim(-.5, 1.7)
axes[0].set_ylim(bottom=0)
axes[0].set_title('Median and middle 50% of recipient amounts')
axes[0].set_ylabel('Quarterly pesos (approx. 2022 prices)')
dep = []
labels = ['Up to 10%', '>10-25%', '>25-50%', '>50%']
for year, d in h.loc[h.recipient].groupby('year'):
    bins = pd.cut(d.dependence, [0, 10, 25, 50, 100.001], labels=labels, include_lowest=True)
    shares = d.groupby(bins, observed=False).factor.sum() / d.factor.sum() * 100
    assert np.isclose(shares.sum(), 100)
    dep.extend(dict(year=year, income_share=label, recipient_pct=value) for label, value in shares.items())
    left = 0
    for label, color in zip(labels, ['#B8D8E8', '#68A7C4', '#DDA15E', '#A44A21']):
        value = shares[label]
        axes[1].barh(str(year), value, left=left, color=color, label=label if year == 2020 else None)
        axes[1].text(left + value / 2, str(year), f'{value:.1f}%', ha='center', va='center', fontsize=9)
        left += value
axes[1].set_xlim(0, 100)
axes[1].set_xlabel('% of recipient households')
axes[1].set_title('Remittances as a share of total current income')
axes[1].legend(title='Household income share', loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2)
fig.suptitle('Amount distribution and reliance on remittances\nRecipients only; whiskers show quartiles, not confidence intervals')
save(fig, '04_amounts_and_dependence.png')
pd.DataFrame(dep).to_csv(OUT / 'dependence_distribution.csv', index=False)
report = 'ENIGH national remittance comparison\n\n' + change.round(3).to_string()
report += '\n\nWeighted descriptive estimates, not significance tests. Separate cross-sections.\n'
report += 'Amounts are quarterly; real amounts use approximate 2022 prices. All foreign origins, not US only.\n'
report += 'Survey totals are not reconciled to Banco de Mexico flows. See analysis notes for definitions.\n'
(OUT / 'REPORT.txt').write_text(report, encoding='utf-8')
print(report)
print(f'\nSaved four plots and summary tables to {OUT}')
