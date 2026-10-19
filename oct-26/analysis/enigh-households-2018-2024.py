"""Four ENIGH vintages and origin-household templates. pandas/numpy/matplotlib only.
Run: python oct-26/analysis/enigh-households-2018-2024.py
No inputs are modified. See enigh-households-2018-2024-notes.md for definitions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %% 1. Settings. Change grouping lists here to change the model representation.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'oct-26/outputs/enigh-households-2018-2024'
OUT.mkdir(parents=True, exist_ok=True)
YEARS = [2018, 2020, 2022, 2024]
SIMPLE = ['locality', 'age_composition', 'origin_work']  # 2 x 4 x 2 = 16
DETAILED = SIMPLE + ['employment', 'size_group', 'family_class']
BASE_YEAR = 2018
# Same approximate method as the previous analysis, now with common 2024 prices.
# Published 2024-price national mean / original-vintage national mean.
# INEGI 2024 presentation: national row of quarterly current income by decile.
NOMINAL_MEANS = {2018: 49851.005351966094, 2020: 50309.31315023509,
                 2022: 63695.45790004601, 2024: 77863.84301199632}
PUBLISHED_REAL = {2018: 67319, 2020: 63400, 2022: 70391, 2024: 77864}
PRICE = {y: PUBLISHED_REAL[y] / NOMINAL_MEANS[y] if y != 2024 else 1. for y in YEARS}
EXPECTED_HOUSEHOLDS = {2018: 34400515, 2020: 35749659, 2022: 37560123, 2024: 38830230}
ORDERS = {
    'head_sex': ['Male', 'Female'], 'head_age': ['Under 40', '40-64', '65+'],
    'size_group': ['1', '2', '3-4', '5+'],
    'employment': ['0 employed', '1 employed', '2+ employed'],
    'age_composition': ['Only ages 12-64', 'Under 12, no 65+', '65+, no under 12', 'Under 12 and 65+'],
    'locality': ['Rural', 'Urban']}

def wmean(x, w):
    return np.average(x, weights=w)

def wquantile(x, w, q):
    order = np.argsort(np.asarray(x), kind='stable')
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    return x[np.searchsorted(w.cumsum(), q * w.sum())]

def save(fig, name):
    fig.text(.5, .01, 'ENIGH 2018-2024; survey-weighted descriptive estimates, no significance tests.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .045, 1, .92))
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)

# %% 2. Comparable attributes and household templates
raw_cols = ['folioviv', 'foliohog', 'ubica_geo', 'tam_loc', 'est_dis', 'upm', 'factor',
    'clase_hog', 'sexo_jefe', 'edad_jefe', 'educa_jefe', 'tot_integ', 'hombres', 'mujeres',
    'menores', 'p12_64', 'p65mas', 'ocupados', 'percep_ing', 'perc_ocupa', 'ing_cor',
    'ingtrab', 'jubilacion', 'bene_gob', 'estim_alqu', 'remesas', 'gasto_mon']
frames, audit = [], []
for year in YEARS:
    d = pd.read_csv(ROOT / f'data/concentradohogar_{str(year)[2:]}.csv', usecols=raw_cols,
                    dtype={c: str for c in ['folioviv', 'foliohog', 'ubica_geo', 'est_dis', 'upm']})
    assert d.notna().all().all(), f'Missing inputs in {year}'
    assert not d.duplicated(['folioviv', 'foliohog']).any()
    assert d.factor.gt(0).all() and d.tot_integ.gt(0).all()
    assert d.remesas.ge(0).all() and d.remesas.le(d.ing_cor + .01).all()
    assert (d.menores + d.p12_64 + d.p65mas).eq(d.tot_integ).all()
    assert (d.hombres + d.mujeres).eq(d.tot_integ).all()
    assert d.tam_loc.isin([1, 2, 3, 4]).all() and d.clase_hog.isin([1, 2, 3, 4, 5]).all()
    assert d.sexo_jefe.isin([1, 2]).all()
    assert d.factor.sum() == EXPECTED_HOUSEHOLDS[year]
    assert np.isclose(wmean(d.ing_cor, d.factor), NOMINAL_MEANS[year], atol=.01)
    d['year'] = year
    d['household_id'] = str(year) + ':' + d.folioviv + ':' + d.foliohog
    d['state'] = d.ubica_geo.str.zfill(5).str[:2]
    d['draw_probability'] = d.factor / d.factor.sum()  # Within year, never pooled!
    d['recipient'] = d.remesas.gt(0)
    d['locality'] = np.where(d.tam_loc.eq(4), 'Rural', 'Urban')
    d['origin_work'] = np.where(d.ocupados.eq(0), 'No employed', 'Employed')
    d['employment'] = pd.cut(d.ocupados, [-1, 0, 1, np.inf], labels=ORDERS['employment']).astype(str)
    d['size_group'] = pd.cut(d.tot_integ, [0, 1, 2, 4, np.inf], labels=ORDERS['size_group']).astype(str)
    d['head_age'] = pd.cut(d.edad_jefe, [0, 39, 64, np.inf], labels=ORDERS['head_age']).astype(str)
    d['head_sex'] = d.sexo_jefe.map({1: 'Male', 2: 'Female'})
    d['age_composition'] = np.select([
        d.menores.gt(0) & d.p65mas.gt(0), d.menores.gt(0), d.p65mas.gt(0)],
        ['Under 12 and 65+', 'Under 12, no 65+', '65+, no under 12'], default='Only ages 12-64')
    d['family_class'] = d.clase_hog.map({1: 'One person', 2: 'Nuclear', 3: 'Extended',
                                       4: 'Composite', 5: 'Nonrelated co-residents'})
    d['simple_type'] = d[SIMPLE].agg(' | '.join, axis=1)
    d['detailed_type'] = d[DETAILED].agg(' | '.join, axis=1)
    for source, name in [('remesas', 'remittance'), ('ing_cor', 'income'), ('ingtrab', 'labor'),
                         ('jubilacion', 'pension'), ('bene_gob', 'government'),
                         ('estim_alqu', 'imputed_rent'), ('gasto_mon', 'monetary_spending')]:
        d[name + '_real'] = d[source] * PRICE[year]
    d['nonremittance_income_real'] = d.income_real - d.remittance_real
    d['other_nonremittance_income_real'] = (d.nonremittance_income_real - d.labor_real
        - d.pension_real - d.government_real - d.imputed_rent_real)
    d['nonremittance_per_capita_real'] = d.nonremittance_income_real / d.tot_integ
    d['dependence_pct'] = 100 * d.remesas / d.ing_cor.where(d.ing_cor.gt(0))
    audit.append(dict(year=year, sample_n=len(d), recipients_n=int(d.recipient.sum()),
        households=d.factor.sum(), price_multiplier=PRICE[year],
        original_income_mean=wmean(d.ing_cor, d.factor), zero_income_n=int(d.ing_cor.eq(0).sum())))
    frames.append(d)
h = pd.concat(frames, ignore_index=True)
assert h.household_id.is_unique
# Full weighted donor population, not fictional kinship or migrant links.
h.to_csv(OUT / 'origin_household_templates.csv', index=False)
h.loc[h.year.eq(BASE_YEAR)].to_csv(OUT / 'baseline_2018_templates.csv', index=False)
pd.DataFrame(audit).to_csv(OUT / 'input_checks_and_prices.csv', index=False)

# %% 3. Additive group statistics: sum totals, THEN recompute rates when collapsing.
z = h.copy()
z['sample_n'] = 1
z['recipient_sample_n'] = z.recipient.astype(int)
z['weight_squared'] = z.factor ** 2
z['recipient_weight'] = z.factor * z.recipient
for col in ['remittance_real', 'nonremittance_income_real', 'labor_real', 'pension_real',
            'government_real', 'imputed_rent_real', 'other_nonremittance_income_real',
            'monetary_spending_real', 'tot_integ', 'menores', 'p12_64', 'p65mas', 'ocupados']:
    z[col + '_weighted_sum'] = z.factor * z[col]
z['recipient_dependence_weighted_sum'] = z.factor * z.dependence_pct.where(z.recipient, 0)
SUMS = ['sample_n', 'recipient_sample_n', 'factor', 'weight_squared', 'recipient_weight'] + [c for c in z if c.endswith('_weighted_sum')]

def summarize(columns):
    t = z.groupby(['year'] + columns, observed=True)[SUMS].sum().reset_index()
    t['receipt_pct'] = 100 * t.recipient_weight / t.factor
    t['household_share_pct'] = 100 * t.factor / t.groupby('year').factor.transform('sum')
    t['recipient_share_pct'] = 100 * t.recipient_weight / t.groupby('year').recipient_weight.transform('sum')
    t['mean_recipient_real'] = t.remittance_real_weighted_sum / t.recipient_weight.replace(0, np.nan)
    t['mean_nonremittance_income_real'] = t.nonremittance_income_real_weighted_sum / t.factor
    t['mean_members'] = t.tot_integ_weighted_sum / t.factor
    t['weight_effective_n'] = t.factor ** 2 / t.weight_squared
    t['sparse_receipt_flag'] = (t.sample_n < 100) | (t.recipient_sample_n < 30)
    # Diagnostic only: effective_n omits survey clustering; flags are not CIs.
    return t

simple = summarize(SIMPLE)
detailed = summarize(DETAILED)
simple['simple_type'] = simple[SIMPLE].agg(' | '.join, axis=1)
detailed['simple_type'] = detailed[SIMPLE].agg(' | '.join, axis=1)
detailed['detailed_type'] = detailed[DETAILED].agg(' | '.join, axis=1)
for t in [simple, detailed]:
    assert np.allclose(t.groupby('year').factor.sum(), h.groupby('year').factor.sum())
    assert np.allclose(t.groupby('year').recipient_weight.sum(), h.loc[h.recipient].groupby('year').factor.sum())
    assert np.allclose(t.groupby('year').household_share_pct.sum(), 100)
# Verify detailed groups can be collapsed exactly to simple groups.
collapsed = detailed.groupby(['year'] + SIMPLE)[SUMS].sum().sort_index()
assert np.allclose(collapsed, simple.set_index(['year'] + SIMPLE)[SUMS].sort_index())
simple.to_csv(OUT / 'simple_type_targets.csv', index=False)
detailed.to_csv(OUT / 'detailed_type_targets.csv', index=False)
detailed[['detailed_type', 'simple_type'] + DETAILED].drop_duplicates().to_csv(OUT / 'type_mapping.csv', index=False)
groups = []
for dim in list(ORDERS) + ['family_class', 'state']:
    t = summarize([dim]).rename(columns={dim: 'category'})
    t['dimension'] = dim
    groups.append(t)
groups = pd.concat(groups, ignore_index=True)
groups.loc[groups.dimension.ne('state')].to_csv(OUT / 'household_groups.csv', index=False)
groups.loc[groups.dimension.eq('state')].to_csv(OUT / 'states_for_later.csv', index=False)
gc = groups.loc[groups.dimension.ne('state')].pivot(index=['dimension', 'category'], columns='year', values='receipt_pct')
for a, b in zip(YEARS[:-1], YEARS[1:]):
    gc[f'change_pp_{a}_{b}'] = gc[b] - gc[a]
gc.to_csv(OUT / 'group_receipt_changes.csv')

# %% 4. National results, conditional amounts and fixed-composition receipt
national, dependence = [], []
for year, d in h.groupby('year'):
    r = d.loc[d.recipient]
    national.append(dict(year=year, sample_n=len(d), recipient_sample_n=len(r), households=d.factor.sum(),
        recipient_households=r.factor.sum(), receipt_pct=100 * r.factor.sum() / d.factor.sum(),
        mean_recipient_nominal=wmean(r.remesas, r.factor), mean_recipient_real=wmean(r.remittance_real, r.factor),
        p25_recipient_real=wquantile(r.remittance_real, r.factor, .25),
        median_recipient_real=wquantile(r.remittance_real, r.factor, .5),
        p75_recipient_real=wquantile(r.remittance_real, r.factor, .75),
        mean_all_households_real=wmean(d.remittance_real, d.factor),
        survey_total_real=(d.remittance_real * d.factor).sum(),
        mean_dependence_pct=wmean(r.dependence_pct, r.factor),
        at_least_half_income_pct=100 * wmean(r.dependence_pct.ge(50), r.factor)))
    bands = pd.cut(r.dependence_pct, [0, 10, 25, 50, 100.001],
                   labels=['Up to 10%', '>10-25%', '>25-50%', '>50%'], include_lowest=True)
    shares = r.groupby(bands, observed=False).factor.sum() / r.factor.sum() * 100
    assert np.isclose(shares.sum(), 100)
    dependence.extend(dict(year=year, income_share=k, recipient_pct=v) for k, v in shares.items())
national = pd.DataFrame(national).set_index('year')
national.to_csv(OUT / 'national_summary.csv')
changes = []
for a, b in [(2018, 2020), (2020, 2022), (2022, 2024), (2018, 2024)]:
    for metric in national.columns:
        changes.append(dict(from_year=a, to_year=b, metric=metric,
            difference=national.loc[b, metric] - national.loc[a, metric],
            relative_change_pct=100 * (national.loc[b, metric] / national.loc[a, metric] - 1)))
pd.DataFrame(changes).to_csv(OUT / 'national_changes.csv', index=False)
pd.DataFrame(dependence).to_csv(OUT / 'dependence_distribution.csv', index=False)
rates = simple.pivot(index='simple_type', columns='year', values='receipt_pct')
shares = simple.pivot(index='simple_type', columns='year', values='household_share_pct') / 100
assert rates.notna().all().all() and shares.notna().all().all(), 'Missing support: do not silently renormalize.'
std = pd.DataFrame({'observed_receipt_pct': national.receipt_pct,
                   'fixed_2018_composition_pct': rates.mul(shares[BASE_YEAR], axis=0).sum()})
std['within_type_change_pp'] = std.fixed_2018_composition_pct - std.loc[BASE_YEAR, 'fixed_2018_composition_pct']
std['composition_remainder_pp'] = std.observed_receipt_pct - std.fixed_2018_composition_pct
assert np.allclose((rates * shares).sum(), national.receipt_pct)
std.index.name = 'year'
std.to_csv(OUT / 'fixed_composition_receipt.csv')

# %% 5. Plots: emphasize rates, with amounts and composition as context.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col, title, unit in zip(axes,
    ['receipt_pct', 'mean_recipient_real', 'at_least_half_income_pct'],
    ['Household receipt rate', 'Mean amount among recipients', 'At least half of income from remittances'],
    ['% of all households', 'Quarterly pesos (approx. 2024 prices)', '% of recipient households']):
    ax.plot(YEARS, national[col], 'o-', color='#247BA0')
    for year, value in national[col].items():
        label = f'{value:,.0f}' if col == 'mean_recipient_real' else f'{value:.2f}'
        ax.annotate(label, (year, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
    ax.set(title=title, ylabel=unit, xticks=YEARS)
    ax.set_xlim(2017.3, 2024.7)
    ax.set_ylim(0, national[col].max() * 1.22)
fig.suptitle('ENIGH remittances, 2018-2024')
save(fig, '01_national_overview.png')

for metric, title, filename, unit in [
    ('receipt_pct', 'Receipt rates within household groups', '02_receipt_by_group.png', '% of households in group'),
    ('recipient_share_pct', 'Composition of remittance recipients', '03_recipient_composition.png', '% of all recipient households')]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (dim, order) in zip(axes.flat, ORDERS.items()):
        t = groups.loc[groups.dimension.eq(dim)].pivot(index='category', columns='year', values=metric).reindex(order)
        for category, row in t.iterrows():
            ax.plot(YEARS, row, 'o-', label=category, linewidth=1.8)
        ax.set(title=dim.replace('_', ' ').title(), ylabel=unit, xticks=YEARS)
        ax.set_ylim(bottom=0, top=t.to_numpy().max() * 1.22)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    fig.suptitle(title + '\nSeparate cross-sections; connecting lines do not track the same households')
    save(fig, filename)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for year in YEARS:
    row = national.loc[year]
    axes[0].errorbar(year, row.median_recipient_real,
        yerr=[[row.median_recipient_real-row.p25_recipient_real], [row.p75_recipient_real-row.median_recipient_real]],
        fmt='o', capsize=7, color='#247BA0')
axes[0].set(title='Median and middle 50% of recipient amounts', xticks=YEARS,
            ylabel='Quarterly pesos (approx. 2024 prices)', ylim=(0, national.p75_recipient_real.max()*1.12))
dep = pd.DataFrame(dependence).pivot(index='year', columns='income_share', values='recipient_pct')
dep = dep[['Up to 10%', '>10-25%', '>25-50%', '>50%']]
dep.plot.barh(stacked=True, ax=axes[1], color=['#B8D8E8', '#68A7C4', '#DDA15E', '#A44A21'])
axes[1].set(title='Remittances as a share of household income', xlabel='% of recipients', ylabel='', xlim=(0,100))
axes[1].legend(loc='upper center', bbox_to_anchor=(.5,-.16), ncol=2, frameon=False)
fig.suptitle('Amounts and dependence among recipients\nWhiskers show quartiles, not confidence intervals')
save(fig, '04_amounts_and_dependence.png')

fig, ax = plt.subplots(figsize=(10, 8))
flags = simple.pivot(index='simple_type', columns='year', values='sparse_receipt_flag').reindex(rates.index)
im = ax.imshow(rates, aspect='auto', vmin=0, vmax=max(15, rates.to_numpy().max()), cmap='YlGnBu')
ax.set_xticks(range(len(YEARS)), YEARS)
ax.set_yticks(range(len(rates)), rates.index, fontsize=9)
for i in range(len(rates)):
    for j in range(len(YEARS)):
        value = rates.iloc[i, j]
        ax.text(j, i, f'{value:.1f}' + ('*' if flags.iloc[i,j] else ''), ha='center', va='center',
                color='white' if value > im.norm.vmax*.6 else 'black', fontsize=9)
fig.colorbar(im, ax=ax, label='Within-type receipt (%)')
fig.suptitle('16 origin-household types: calibration targets\n* Fewer than 100 sampled households or 30 sampled recipients')
save(fig, '05_simple_type_receipt.png')

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(YEARS, std.observed_receipt_pct, 'o-', label='Observed composition')
ax.plot(YEARS, std.fixed_2018_composition_pct, 's--', label='Fixed 2018 mix of 16 types')
ax.set(xticks=YEARS, ylabel='Households receiving remittances (%)', ylim=(0, max(std.max().iloc[:2])*1.2))
ax.legend(frameon=False)
fig.suptitle('Receipt change with and without population-composition shifts')
save(fig, '06_fixed_composition.png')
report = 'ENIGH 2018-2024, national weighted estimates\n\n' + national.round(3).to_string()
report += '\n\nFIXED COMPOSITION\n' + std.round(3).to_string()
report += '\n\nSparse simple cells: ' + str(int(simple.sparse_receipt_flag.sum())) + '/' + str(len(simple))
report += '\nHousehold templates preserve observed joint attributes. Migrant linkages are unobserved.\n'
report += 'Real amounts use approximate 2024 prices; see notes. No causal or significance claims.\n'
(OUT / 'REPORT.txt').write_text(report, encoding='utf-8')
print(report)
print(f'\nOutputs: {OUT}')
