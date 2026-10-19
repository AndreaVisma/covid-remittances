"""MMP174 household composition and US remittances. Run by section or as a script.

Requires pandas, numpy, matplotlib. Source files are never changed.
Outputs are descriptive of sampled communities, NOT nationally representative.
Codebooks: https://mmp.research.brown.edu/mmp/documentation
"""

# %% 1. Settings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save figures without opening windows.
import matplotlib.pyplot as plt

ROOT = Path('C:/git-projects/covid-remittances')
DATA = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
OUT = ROOT / 'oct-26/outputs/mmp-households'
OUT.mkdir(parents=True, exist_ok=True)
START_YEAR, END_YEAR = 2010, 2019  # Change these for a different pooled sample.
KEY = ['country', 'commun', 'surveypl', 'hhnum']

# %% 2. Read only necessary columns; export their original labels
sibling_cols = [f'sbusnw{i}' for i in range(1, 13)]
extended_cols = ['uncinus', 'coinus', 'neinus', 'sbilinus', 'chilinus', 'pilinus']
house_cols = KEY + ['surveyyr', 'weight', 'members', 'workers', 'nonwork',
    'hsex', 'hage', 'lifecycl', 'remus', 'remussiz', 'mousnw', 'fausnw', 'frinus'] + sibling_cols + extended_cols
person_cols = KEY + ['surveyyr', 'persnum', 'hhmemshp', 'relhead', 'age',
    'uscurtrp', 'usstatel', 'occ', 'usoccl', 'cncurtrp', 'docurtrp']
h = pd.read_stata(DATA / 'house174.dta', columns=house_cols, convert_categoricals=False)
p = pd.read_stata(DATA / 'pers174.dta', columns=person_cols, convert_categoricals=False)
assert not h.duplicated(KEY).any(), 'Duplicate household keys'
assert not p.duplicated(KEY + ['persnum']).any(), 'Duplicate person keys'
labels = []
for filename, columns in [('house174.dta', house_cols), ('pers174.dta', person_cols)]:
    with pd.io.stata.StataReader(DATA / filename) as reader:
        original_labels = reader.variable_labels()
    labels.extend({'file': filename, 'variable': c, 'label': original_labels[c]} for c in columns)
pd.DataFrame(labels).to_csv(OUT / 'variable_labels.csv', index=False)

# Coverage includes years before the remittance question existed. Do not code
# unasked questions as nonreceipt. REMUS introduced in community 72.
h['receives_us'] = h['remus'].map({1: 1.0, 2: 0.0})
coverage = h.groupby(['surveypl', 'surveyyr']).agg(
    households=('hhnum', 'size'), communities=('commun', 'nunique'),
    known_receipt=('receives_us', 'count'), recipients=('receives_us', 'sum'))
coverage.to_csv(OUT / 'coverage.csv')
all_years = (int(h.surveyyr.min()), int(h.surveyyr.max()))
h = h.loc[h.country.eq(1) & h.surveypl.eq(1) & h.surveyyr.between(START_YEAR, END_YEAR)].copy()
assert len(h), 'No households in the selected years'
assert h.weight.notna().all() and h.weight.gt(0).all()
p = p.merge(h[KEY + ['surveyyr']], on=KEY, how='inner', validate='many_to_one', suffixes=('', '_house'))
assert p.surveyyr.eq(p.surveyyr_house).all()
assert len(p[KEY].drop_duplicates()) == len(h), 'Household missing from roster'

# %% 3. Composition and recorded US connections
# HHMEMSHP=2 means member of survey household; 1 means OTHER household;
# 8888 means deceased. Membership is not a guarantee of physical presence.
# PERS includes nonresident children: counting all rows inflates household size.
p['age_clean'] = p.age.replace({8888: np.nan, 9999: np.nan})
p['us_now'] = p.uscurtrp.map({1: 1.0, 2: 0.0, 8888: 0.0})
p.loc[p.hhmemshp.eq(8888), 'us_now'] = 0.0
p['child'] = p.relhead.eq(3)  # Son/daughter; step/foster children not included.
p['child_us'] = p.us_now.where(p.child, 0.0)
p['head_spouse_us'] = p.us_now.where(p.relhead.isin([1, 2]), 0.0)
p['member'] = p.hhmemshp.eq(2)
p['member_under18'] = (p.age_clean < 18).astype(float).where(p.age_clean.notna()).where(p.member, 0.0)
p['member_65plus'] = (p.age_clean >= 65).astype(float).where(p.age_clean.notna()).where(p.member, 0.0)

# A count is unknown if any relevant person's status is unknown.
count_cols = ['child_us', 'head_spouse_us', 'us_now', 'member_under18', 'member_65plus']
grouped = p.groupby(KEY)
counts = grouped[count_cols].sum()
counts = counts.where(grouped[count_cols].count().eq(grouped.size(), axis=0))
counts['roster_members'] = grouped.member.sum()
counts['membership_unknown'] = grouped.hhmemshp.apply(lambda s: (~s.isin([1, 2, 8888])).sum())
counts.loc[counts.membership_unknown.gt(0), ['member_under18', 'member_65plus']] = np.nan

# Geographic dispersion only among explicitly recorded current US migrants.
# Last-trip state is NOT used as a current location for returned migrants.
current = p.loc[p.us_now.eq(1)].copy()
current['known_state'] = current.usstatel.where(current.usstatel.between(101, 151))
states = current.groupby(KEY).agg(recorded_us_people=('persnum', 'size'),
    known_us_states=('known_state', 'count'), distinct_us_states=('known_state', 'nunique'))
states['state_coverage_complete'] = states.recorded_us_people.eq(states.known_us_states)
h = h.merge(counts.reset_index(), on=KEY, validate='one_to_one')
h = h.merge(states.reset_index(), on=KEY, how='left', validate='one_to_one')
h['roster_size_matches'] = h.roster_members.eq(h.members)
# Do not assign an age-composition type when the two files disagree on size.
h.loc[~h.roster_size_matches, ['member_under18', 'member_65plus']] = np.nan

# HOUSE sibling/parent codes: 8888=no sibling/never migrated, hence zero
# *recorded* connections; 9999 and system missing remain unknown.
sb = h[sibling_cols].replace({1: 1.0, 2: 0.0, 8888: 0.0, 9999: np.nan})
h['siblings_us'] = sb.sum(axis=1, min_count=len(sibling_cols))
par = h[['mousnw', 'fausnw']].replace({1: 1.0, 2: 0.0, 8888: 0.0, 9999: np.nan})
h['parents_us'] = par.sum(axis=1, min_count=2)
ext = h[extended_cols].replace({8888: np.nan, 9999: np.nan})
h['extended_kin_us'] = ext.sum(axis=1, min_count=len(extended_cols))
h['friends_us'] = h.frinus.replace({8888: np.nan, 9999: np.nan})
# Do NOT add HOUSE relatives to PERS counts: some relatives may overlap.
h['other_kin_us'] = h[['siblings_us', 'parents_us', 'extended_kin_us']].sum(axis=1, min_count=3)
h['remittance_size'] = h.remussiz.where(h.receives_us.eq(1)).map({1: 'Small', 2: 'Intermediate', 3: 'Substantial'})
h['head_sex'] = h.hsex.map({1: 'Male', 2: 'Female'})
h['head_age_group'] = pd.cut(h.hage.replace(9999, np.nan), [0, 39, 64, np.inf], labels=['Under 40', '40-64', '65+'])
h['worker_group'] = pd.cut(h.workers.replace(9999, np.nan), [-1, 0, 1, np.inf], labels=['0', '1', '2+'])
h['life_cycle'] = h.lifecycl.map({1: 'No children', 2: 'All children under 13',
    3: 'Some children teenagers', 4: 'All children teenagers', 5: 'All children adults'})
for column in ['child_us', 'siblings_us', 'extended_kin_us']:
    h[column + '_group'] = pd.cut(h[column], [-1, 0, 1, np.inf], labels=['0', '1', '2+'])
h['composition'] = np.select([
    h.member_under18.isna() | h.member_65plus.isna(),
    h.member_under18.gt(0) & h.member_65plus.gt(0),
    h.member_under18.gt(0), h.member_65plus.gt(0)],
    ['Unknown', 'Under-18 and 65+ members', 'Under-18, no 65+ members', '65+, no under-18 members'],
    default='Only 18-64 members')
# Mutually exclusive, explicitly partial typology. Relatives are potential
# connections, not identified senders, earners, or independent US households.
h['network_type'] = np.select([
    h.head_spouse_us.gt(0), h.child_us.ge(2), h.child_us.eq(1),
    h.head_spouse_us.isna() | h.child_us.isna(), h.other_kin_us.gt(0),
    h.other_kin_us.isna()],
    ['Head/spouse on US trip', '2+ children on US trips', '1 child on US trip',
     'Unknown core connection', 'Other US kin, no core connection', 'Unknown other kin'],
    default='No recorded core/other US kin')
# Household profiles are saved again after the additional compact typology.

# %% 4. One small helper for all descriptive tables
# Weights describe the pooled sampled frames, NOT the Mexican population.
# Both weighted and unweighted rates and actual denominators are retained.
def summarize(frame, column):
    rows = []
    for name, g in frame.groupby(column, observed=True, dropna=False):
        known = g.loc[g.receives_us.notna()]
        recipients = g.loc[g.receives_us.eq(1)]
        sized = recipients.loc[recipients.remittance_size.notna()]
        row = {'group': str(name), 'n': len(g), 'communities': g.commun.nunique(),
            'sample_share_pct': 100 * len(g) / len(frame),
            'weighted_share_pct': 100 * g.weight.sum() / frame.weight.sum(),
            'known_receipt_n': len(known), 'recipient_n': len(recipients),
            'receipt_pct': 100 * known.receives_us.mean(),
            'weighted_receipt_pct': 100 * np.average(known.receives_us, weights=known.weight) if len(known) else np.nan,
            'known_size_n': len(sized),
            'substantial_among_known_recipients_pct': 100 * sized.remittance_size.eq('Substantial').mean(),
            'weighted_substantial_pct': 100 * np.average(sized.remittance_size.eq('Substantial'), weights=sized.weight) if len(sized) else np.nan,
            'mean_members': g.members.mean(), 'mean_workers': g.workers.mean()}
        rows.append(row)
    return pd.DataFrame(rows)

tables = {}
for column in ['network_type', 'child_us_group', 'siblings_us_group', 'extended_kin_us_group',
               'head_sex', 'head_age_group', 'worker_group', 'life_cycle', 'composition', 'surveyyr', 'commun']:
    tables[column] = summarize(h, column)
    tables[column].to_csv(OUT / f'by_{column}.csv', index=False)
# Basic accounting checks, including the missing-category rows.
for column in ['network_type', 'child_us_group', 'worker_group', 'composition']:
    assert tables[column]['n'].sum() == len(h)
    assert tables[column].known_receipt_n.sum() == h.receives_us.notna().sum()
    assert tables[column].recipient_n.sum() == h.receives_us.eq(1).sum()
for column in ['child_us', 'head_spouse_us']:
    known = h[[column, 'us_now']].dropna()
    assert known[column].le(known.us_now).all()
assert h.loc[h.receives_us.ne(1), 'remittance_size'].isna().all()
# Same tables for the most recent five years: changing communities, NOT a trend.
recent = h.loc[h.surveyyr.ge(max(START_YEAR, END_YEAR - 4))]
if len(recent):
    summarize(recent, 'network_type').to_csv(OUT / 'recent_network_types.csv', index=False)
# A joint distribution preserves associations needed for synthetic households.
h.groupby(['network_type', 'composition', 'worker_group'], observed=True, dropna=False).agg(
    households=('hhnum', 'size'), weight_sum=('weight', 'sum'),
    known_receipt=('receives_us', 'count'), recipients=('receives_us', 'sum')
).to_csv(OUT / 'joint_household_types.csv')
dispersion = h.loc[h.us_now.ge(2) & h.state_coverage_complete.eq(True)].copy()
dispersion['dispersion'] = np.where(dispersion.distinct_us_states.ge(2), 'Several US states', 'One US state')
summarize(dispersion, 'dispersion').to_csv(OUT / 'by_us_state_dispersion.csv', index=False)

# %% 5. A compact figure and a text report
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, column, title in zip(axes, ['child_us_group', 'worker_group'],
                           ['Children currently on US trips', 'Household workers']):
    t = tables[column].loc[lambda x: x.group.ne('nan')]
    x = np.arange(len(t))
    ax.bar(x - .18, t.receipt_pct, width=.36, label='Unweighted')
    ax.bar(x + .18, t.weighted_receipt_pct, width=.36, label='MMP weights')
    ax.set_xticks(x, [f'{a}\n(n={b:,})' for a, b in zip(t.group, t.known_receipt_n)])
    ax.set_title(title)
    ax.set_ylabel('Households receiving US remittances (%)')
    ax.legend()
fig.suptitle(f'MMP174: Mexico survey sample, {START_YEAR}-{END_YEAR}; descriptive associations')
fig.tight_layout()
fig.savefig(OUT / 'receipt_comparisons.png', dpi=170)
plt.close(fig)

report = [f'MMP174 household analysis: {START_YEAR}-{END_YEAR}',
    f'Source coverage: {all_years[0]}-{all_years[1]}; no COVID-period observations.',
    f'Selected: {len(h):,} households, {h.commun.nunique()} communities; {len(p):,} roster people.',
    f'Receipt known: {h.receives_us.notna().sum():,}; recipients: {h.receives_us.eq(1).sum():,}.',
    f'Roster membership differs from HOUSE members for {(~h.roster_size_matches).sum():,} households.',
    'Use HOUSE members for size; roster age composition is flagged separately.',
    '', 'NETWORK TYPES', tables['network_type'].round(2).to_string(index=False),
    '', 'CHILDREN ON US TRIPS', tables['child_us_group'].round(2).to_string(index=False),
    '', 'WORKERS', tables['worker_group'].round(2).to_string(index=False),
    '', 'INTERPRETATION LIMITS',
    '- Pooled sampled communities, not nationally representative, even with weights.',
    '- Year comparisons mix different communities and are not a household panel.',
    '- Missing/unasked receipt is excluded from rate denominators, never coded no.',
    '- US connections are not confirmed senders or independent earning units.',
    '- Original network_type separates head/spouse from children; compact_type uses head/children as core.',
    '- No-recorded-kin type is not proof of no US ties; friends and some relatives are omitted.',
    '- Current US trip is a migration proxy; it does not enumerate all US-born relatives.',
    '- Size is ordinal relative to monthly salary, not dollars or a numeric income share.',
    '- No monetary Gini or pandemic inequality change can be estimated from these outputs.',
    '- MIG last-trip dollar amounts are deliberately not treated as current household receipts.',
    '- State dispersion excludes unknown states and is not occupational diversification.',
    '', 'Source: MMP174, Mesoamerican Migration Project, Brown University / El Colegio de Mexico.',
    'https://mmp.research.brown.edu',
    'Codebooks: https://mmp.research.brown.edu/mmp/documentation']
(OUT / 'REPORT.txt').write_text('\n'.join(report), encoding='utf-8')
print('\n'.join(report))
print(f'\nOutputs saved in: {OUT}')

# %% 6. Compact typology: who abroad / origin work / core occupation abroad
# User definition: CORE = head or son/daughter (RELHEAD 1,3).
# Spouse is OTHER KIN here; the earlier head/spouse typology stays unchanged.
# OCC is principal occupation at survey. USOCCL is last-trip occupation and
# is retained for auditing, NOT substituted for missing current occupation.
o = p.occ
p['occupation_group'] = np.select([
    o.isin([43, 54]), o.between(10, 99), o.between(410, 419),
    o.between(510, 559), o.between(710, 839),
    o.between(110, 149) | o.between(210, 219) | o.between(610, 629)],
    ['Working student/pensioner', 'Not working', 'Agriculture',
     'Production/construction/transport', 'Sales/personal services',
     'Professional/administrative'], default='Unknown')
p['working'] = np.where(p.occupation_group.eq('Unknown'), np.nan,
                        (~p.occupation_group.eq('Not working')).astype(float))
p['core'] = p.relhead.isin([1, 3])
p['core_abroad'] = p.us_now.where(p.core, 0.0)
# Include spouse, other relatives and step/foster children in the kin category.
p['kin_abroad'] = p.us_now.where(p.relhead.between(2, 20) & ~p.core & p.relhead.ne(19), 0.0)
p['unknown_relationship_abroad'] = p.us_now.where(p.relhead.eq(9999), 0.0)
g = p.groupby(KEY)
for col in ['core_abroad', 'kin_abroad', 'unknown_relationship_abroad']:
    # Presence can be established even if another relative has unknown status.
    present = g[col].max().where(g[col].count().eq(g.size()) | g[col].max().eq(1))
    h = h.merge(present.rename(col).reset_index(), on=KEY, validate='one_to_one')
kin_fields = pd.concat([sb, par, ext.gt(0).astype(float).where(ext.notna())], axis=1)
kin_presence = kin_fields.max(axis=1).where(kin_fields.notna().all(axis=1) | kin_fields.max(axis=1).eq(1))
h['kin_abroad'] = pd.concat([h.kin_abroad, kin_presence.rename('house_kin')], axis=1).max(axis=1).where(
    (h.kin_abroad.notna() & kin_presence.notna()) | h.kin_abroad.eq(1) | kin_presence.eq(1))
h['who_abroad'] = np.select([
    h.core_abroad.eq(1) & h.kin_abroad.eq(1),
    h.core_abroad.eq(1) & h.kin_abroad.eq(0),
    h.core_abroad.eq(0) & h.kin_abroad.eq(1),
    h.core_abroad.eq(0) & h.kin_abroad.eq(0) & h.unknown_relationship_abroad.eq(0)],
    ['Core + kin', 'Core only', 'Kin only', 'None recorded'], default='Uncertain')

# Origin-work proxy: survey members who are not on an ongoing US, Canadian,
# or domestic migration. No location information is inferred from last-trip state.
away = pd.DataFrame({c: p[c].map({1: 1., 2: 0., 8888: 0.})
                    for c in ['uscurtrp', 'cncurtrp', 'docurtrp']})
away.loc[p.hhmemshp.eq(8888), :] = 0
known_home = away.eq(0).all(axis=1)
known_away = away.eq(1).any(axis=1)
origin_contribution = p.working.where(known_home)
origin_contribution = origin_contribution.mask(known_away | ~p.member, 0.)
origin_contribution = origin_contribution.mask(~p.hhmemshp.isin([1, 2, 8888]))
p['origin_worker'] = origin_contribution
g = p.groupby(KEY)
origin_count = g.origin_worker.sum().where(g.origin_worker.count().eq(g.size()))
h = h.merge(origin_count.rename('origin_workers').reset_index(), on=KEY, validate='one_to_one')
h.loc[~h.roster_size_matches, 'origin_workers'] = np.nan
h['origin_work'] = np.select([h.origin_workers.eq(0), h.origin_workers.gt(0)],
                             ['No origin worker', 'Origin worker(s)'], default='Origin work unknown')

# Only core migrants are sufficiently enumerated to summarize their occupations.
# Even this measures occupations, not employment contracts, hours or earnings.
abroad = p.loc[p.core & p.us_now.eq(1)].copy()
job_rows = []
for key, group in abroad.groupby(KEY):
    jobs = group.occupation_group
    label = 'Unknown/partial' if jobs.eq('Unknown').any() else (
        jobs.iloc[0] if jobs.nunique() == 1 else 'Mixed occupations/statuses')
    job_rows.append(dict(zip(KEY, key), core_job=label, core_people=len(group),
                         core_occupation_known=int(jobs.ne('Unknown').sum())))
jobs = pd.DataFrame(job_rows, columns=KEY + ['core_job', 'core_people', 'core_occupation_known'])
h = h.merge(jobs, on=KEY, how='left', validate='one_to_one')
h.loc[h.core_abroad.eq(0), 'core_job'] = 'No recorded core abroad'
h['core_job'] = h.core_job.fillna('Unknown/partial')
h.loc[h.core_abroad.isna(), 'core_job'] = 'Unknown/partial'
h['compact_type'] = h.who_abroad + ' | ' + h.origin_work
h['employment_type'] = h.compact_type + ' | ' + h.core_job
h.to_csv(OUT / 'household_profiles.csv', index=False)
abroad[KEY + ['persnum', 'relhead', 'occ', 'usoccl', 'occupation_group']].to_csv(
    OUT / 'core_abroad_occupations.csv', index=False)
for col in ['who_abroad', 'origin_work', 'core_job', 'compact_type', 'employment_type']:
    tables[col] = summarize(h, col)
    assert tables[col]['n'].sum() == len(h)
    assert tables[col].recipient_n.sum() == h.receives_us.eq(1).sum()
    tables[col].to_csv(OUT / f'by_{col}.csv', index=False)

# %% 7. Additional plots (all rates unweighted unless explicitly labelled)
stamp = f'MMP174, {START_YEAR}-{END_YEAR}; sampled communities, descriptive only'
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
t = tables['compact_type'].sort_values('n')
axes[0].barh(t.group, t.sample_share_pct, color='#277DA1')
axes[0].set_xlabel('Share of sampled households (%)')
axes[1].barh(t.group, t.receipt_pct, color='#D97706')
axes[1].set_xlabel('Receiving US remittances (%)')
for i, row in enumerate(t.itertuples()):
    axes[1].text(row.receipt_pct + 1, i, f'n={row.known_receipt_n}', va='center', fontsize=8)
axes[1].set_xlim(0, 110)
fig.suptitle('Compact household types\n' + stamp)
fig.tight_layout()
fig.savefig(OUT / '02_compact_types.png', dpi=170)
plt.close(fig)

# Heatmap with denominators; suppress rates for cells with fewer than 20 answers.
cells = []
for (who, work), group in h.groupby(['who_abroad', 'origin_work']):
    known = group.receives_us.dropna()
    cells.append({'who': who, 'work': work, 'n': len(known), 'rate': known.mean() * 100})
cells = pd.DataFrame(cells)
cells.to_csv(OUT / 'compact_heatmap_cells.csv', index=False)
rates = cells.pivot(index='who', columns='work', values='rate')
ns = cells.pivot(index='who', columns='work', values='n').reindex_like(rates)
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(rates.where(ns.ge(20)), vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
ax.set_xticks(range(len(rates.columns)), rates.columns)
ax.set_yticks(range(len(rates)), rates.index)
for i in range(len(rates)):
    for j in range(len(rates.columns)):
        n, rate = ns.iloc[i, j], rates.iloc[i, j]
        text = 'No observations' if pd.isna(n) else (f'{rate:.1f}%\nn={n:.0f}' if n >= 20 else f'n={n:.0f}\nsmall cell')
        ax.text(j, i, text, ha='center', va='center', color='white' if n >= 20 and rate > 55 else 'black')
fig.colorbar(im, ax=ax, label='Receiving US remittances (%)')
ax.set_title('Who is abroad and work at origin\n' + stamp)
fig.tight_layout()
fig.savefig(OUT / '03_receipt_heatmap.png', dpi=170)
plt.close(fig)

# Remittance importance: denominator is recipients with a known ordinal answer.
sizes = pd.crosstab(h.who_abroad, h.remittance_size).reindex(columns=['Small', 'Intermediate', 'Substantial'], fill_value=0)
sizes.to_csv(OUT / 'size_by_who_abroad.csv')
fig, ax = plt.subplots(figsize=(10, 5))
sizes.div(sizes.sum(axis=1), axis=0).mul(100).plot.barh(stacked=True, ax=ax, color=['#90BE6D', '#F9C74F', '#F9844A'])
ax.set_yticklabels([f'{name} (n={n})' for name, n in sizes.sum(axis=1).items()])
ax.set_xlabel('Recipients with known size (%)')
ax.set_ylabel('')
ax.set_title('Reported remittance importance relative to salary\n' + stamp)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3)
fig.tight_layout()
fig.savefig(OUT / '04_remittance_importance.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
counts = abroad.occupation_group.value_counts().sort_values()
axes[0].barh(counts.index, counts.values, color='#277DA1')
axes[0].set_xlabel('Core people currently on US trips')
axes[0].set_title('Current principal occupation (person counts)')
t = tables['core_job'].loc[lambda x: x.group.ne('No recorded core abroad')].sort_values('receipt_pct')
axes[1].barh(t.group, t.receipt_pct, color='#D97706')
axes[1].set_xlabel('Households receiving remittances (%)')
axes[1].set_title('Household core-occupation profile')
axes[1].set_yticks(range(len(t)), [f'{r.group} (n={r.known_receipt_n})' for r in t.itertuples()])
fig.suptitle(stamp + '\nExtended-kin employment is unavailable; mixed profiles are retained')
fig.tight_layout()
fig.savefig(OUT / '05_abroad_occupations.png', dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ['composition', 'head_age_group'], ['Member age composition', 'Age of household head']):
    t = tables[col].sort_values('receipt_pct')
    ax.barh(t.group, t.receipt_pct, color='#577590')
    ax.set_yticks(range(len(t)), [f'{r.group} (n={r.known_receipt_n})' for r in t.itertuples()])
    ax.set_title(title)
    ax.set_xlabel('Receiving US remittances (%)')
fig.suptitle(stamp)
fig.tight_layout()
fig.savefig(OUT / '06_composition.png', dpi=170)
plt.close(fig)

with (OUT / 'REPORT.txt').open('a', encoding='utf-8') as f:
    f.write('\n\nCOMPACT TYPOLOGY (core=head or son/daughter; spouse=kin)\n')
    f.write(tables['compact_type'].round(2).to_string(index=False))
    f.write('\n\nCORE OCCUPATION PROFILES\n' + tables['core_job'].round(2).to_string(index=False))
    f.write('\nOrigin work uses roster occupations, excluding members on recorded ongoing migrations.\n')
    f.write('Unknown occupations/location are retained as unknown; this is a location proxy.\n')
    f.write('Current OCC is used; last-trip USOCCL is exported only for audit. Kin jobs are unavailable.\n')
print('\nCompact types:\n', tables['compact_type'][['group', 'n', 'receipt_pct']].round(2).to_string(index=False))
