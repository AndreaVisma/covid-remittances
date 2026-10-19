"""Build an MMP-like migrant population and optionally match ENIGH economic donors.

Run from any directory: python oct-26/simulations/synthetic-migrants.py
Requires pandas, numpy and pyarrow. No source data are modified.
Numbered cells can also be run interactively (set ROOT yourself in that case).
This is empirical resampling, not a national population estimate or a forecast.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# %% 1. Settings: population size, source window, reproducibility and outputs.
N_AGENTS = 10_000                 # Number of migrant people, NOT households.
SEED = 42
START_YEAR, END_YEAR = 2010, 2019 # Pooled interview dates; not a 2019 snapshot.
USE_MMP_WEIGHTS = False          # False reproduces the observed sample mix.
MATCH_ENIGH = True               # False creates only MMP migrants/households.
ENIGH_YEAR = 2018
SAVE_PARQUET = True

ROOT = Path('C:/git-projects/covid-remittances')
MMP = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
ENIGH = ROOT / 'oct-26/outputs/enigh-households-2018-2024/origin_household_templates.csv'
OUT = ROOT / 'oct-26/outputs/synthetic-migrants'
KEY = ['country', 'commun', 'surveypl', 'hhnum']
UNKNOWN = 'Unknown'

# %% 2. Editable category dictionaries.
# Numeric intervals INCLUDE both endpoints. Gaps become Unknown; overlaps fail.
# These groups label existing people: changing bins does not redraw their ages.
AGE_GROUPS = {'Under 18': (0, 17), '18-29': (18, 29), '30-44': (30, 44),
              '45-64': (45, 64), '65+': (65, 120)}
EDUCATION_GROUPS = {'0 years': (0, 0), '1-6 years': (1, 6),
                    '7-9 years': (7, 9), '10-12 years': (10, 12), '13+ years': (13, 40)}
SIZE_GROUPS = {'0': (0, 0), '1': (1, 1), '2': (2, 2), '3-4': (3, 4), '5+': (5, 100)}
WORKER_GROUPS = {'0': (0, 0), '1': (1, 1), '2+': (2, 100)}
SEX = {1: 'Male', 2: 'Female'}
MARITAL_STATUS = {1: 'Never married', 2: 'Married', 3: 'Consensual union',
                  4: 'Widowed', 5: 'Divorced', 6: 'Separated'}
RELATIONSHIP = {
    1: 'Head', 2: 'Spouse', 3: 'Son/daughter', 4: 'Father/mother',
    5: 'Brother/sister', 6: 'Niece/nephew', 7: 'Uncle/aunt', 8: 'Cousin',
    9: 'Grandparent', 10: 'Grandchild', 11: 'Stepchild', 12: 'Stepsibling',
    13: 'Stepparent', 14: 'Son/daughter-in-law', 15: 'Brother/sister-in-law',
    16: 'Father/mother-in-law', 17: 'Other blood relative',
    18: 'Other relative by marriage', 19: 'Non-relative', 20: 'Adopted/foster child'}
# Explicit code sets can be split, combined or renamed. range(a,b) excludes b.
# Occupation is NOT industry or an estimate of pandemic exposure.
JOB_GROUPS = {
    'Working student/pensioner': [43, 54],
    'Not working': [i for i in range(10, 100) if i not in [43, 54]],
    'Agriculture': list(range(410, 420)),
    'Production/construction/transport': list(range(510, 560)),
    'Sales/personal services': list(range(710, 730)) + list(range(810, 840)),
    'Professional/administrative': list(range(110, 150)) + list(range(210, 220))
                                 + list(range(610, 630))}
# Employment proxy uses original codes, independently of the chosen job labels.
WORKING_CODES = [43, 54] + list(range(110, 150)) + list(range(210, 220)) + list(range(410, 420)) + list(range(510, 560)) + list(range(610, 630)) + list(range(710, 730)) + list(range(810, 840))
NONWORKING_CODES = [i for i in range(10, 100) if i not in [43, 54]]
CHILD_AGE_LIMIT = 12             # Must stay 12 for supplied ENIGH summary data.
OLDER_AGE_LIMIT = 65             # Must stay 65 unless ENIGH is rebuilt from roster.
EMPLOYMENT_MIN_AGE = 14          # Matches ENIGH's supplied occupied-member count.

# State is always an exact matching constraint. Other features incur the chosen
# penalty when categories differ. Missing MMP features do not count as matches.
MATCH_WEIGHTS = {'size_group': 2.0, 'children_present': 1.0,
                 'older_present': 1.0, 'worker_group': 1.0}

# Fixed SOURCE code dictionaries below: change analytical bins above freely,
# but change these mappings only against the MMP codebooks.
# Appendix A: community IDs -> origin state (not migrant's state of birth).
COMMUNITIES_BY_STATE = {
    1: [69, 71], 2: [63, 64, 65, 66], 6: [33, 67, 68],
    8: list(range(78, 86)), 10: [73, 74, 75, 76],
    11: [1, 2, 4, 5, 13, 15, 16, 26, 27, 55, 56, 72, 87, 88, 89, 131],
    12: [31, 41, 42, 43], 13: [92, 93],
    14: [3, 6, 7, 17, 20, 21, 23, 24, 25, 28, 57, 58, 91, 97, 98, 99,
         107, 120, 122, 123, 124, 144, 145, 146, 147, 151, 152, 154],
    15: [110, 111, 112, 113, 114, 155, 156, 157],
    16: [8, 9, 10, 14, 19, 22, 29, 108, 109, 121],
    17: [115, 116, 117, 118, 162, 163, 164, 165, 166], 18: [11, 12], 19: [77, 86],
    20: [49, 50, 51, 52, 171, 172, 173, 174],
    21: [54, 59, 60, 61, 132, 133, 134, 148, 149, 150, 158, 159, 161],
    22: [135, 136, 137, 138, 139, 160, 169, 170],
    24: [32, 36, 37, 38, 39, 44, 45, 47, 48, 90, 129, 130],
    25: [53, 62, 70], 27: [140, 141, 142, 143], 29: [94, 95, 96],
    30: [100, 101, 102, 103, 104, 105, 106, 119],
    31: [125, 126, 127, 128, 167, 168], 32: [18, 30, 34, 35, 40, 46, 153]}
MEXICO_STATES = {1: 'Aguascalientes', 2: 'Baja California', 6: 'Colima', 8: 'Chihuahua',
    10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero', 13: 'Hidalgo', 14: 'Jalisco',
    15: 'Estado de Mexico', 16: 'Michoacan', 17: 'Morelos', 18: 'Nayarit',
    19: 'Nuevo Leon', 20: 'Oaxaca', 21: 'Puebla', 22: 'Queretaro',
    24: 'San Luis Potosi', 25: 'Sinaloa', 27: 'Tabasco', 29: 'Tlaxcala',
    30: 'Veracruz', 31: 'Yucatan', 32: 'Zacatecas'}
US_STATES = {100: 'Alabama', 102: 'Alaska', 103: 'Arizona', 104: 'Arkansas',
    105: 'California', 106: 'Colorado', 107: 'Connecticut', 108: 'Delaware',
    109: 'District of Columbia', 110: 'Florida', 111: 'Georgia', 112: 'Hawaii',
    113: 'Idaho', 114: 'Illinois', 115: 'Indiana', 116: 'Iowa', 117: 'Kansas',
    118: 'Kentucky', 119: 'Louisiana', 120: 'Maine', 121: 'Maryland',
    122: 'Massachusetts', 123: 'Michigan', 124: 'Minnesota', 125: 'Mississippi',
    126: 'Missouri', 127: 'Montana', 128: 'Nebraska', 129: 'Nevada',
    130: 'New Hampshire', 131: 'New Jersey', 132: 'New Mexico', 133: 'New York',
    134: 'North Carolina', 135: 'North Dakota', 136: 'Ohio', 137: 'Oklahoma',
    138: 'Oregon', 139: 'Pennsylvania', 140: 'Rhode Island', 141: 'South Carolina',
    142: 'South Dakota', 143: 'Tennessee', 144: 'Texas', 145: 'Utah',
    146: 'Vermont', 147: 'Virginia', 148: 'Washington', 149: 'West Virginia',
    150: 'Wisconsin', 151: 'Wyoming', 152: 'Various states', 153: 'Puerto Rico',
    199: 'US state unknown'}


def grouped(values, intervals):
    """Apply editable inclusive intervals, leaving missing/uncovered values Unknown."""
    result = pd.Series(UNKNOWN, index=values.index)
    used = pd.Series(False, index=values.index)
    for label, (low, high) in intervals.items():
        assert low <= high, f'Invalid interval: {label}'
        mask = values.between(low, high)
        assert not (used & mask).any(), f'Overlapping intervals: {label}'
        result.loc[mask] = label
        used |= mask
    return result


# %% 3. Load MMP households and people, keeping provenance and unknown values.
assert isinstance(N_AGENTS, int) and N_AGENTS > 0
assert START_YEAR <= END_YEAR
assert CHILD_AGE_LIMIT == 12 and OLDER_AGE_LIMIT == 65, 'Rebuild ENIGH roster groups before changing age cutoffs.'
rng_network, rng_match = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
hcols = KEY + ['surveyyr', 'weight', 'members', 'workers', 'hage', 'hsex', 'remus', 'remussiz']
pcols = KEY + ['surveyyr', 'persnum', 'hhmemshp', 'age', 'sex', 'relhead', 'marstat',
               'edyrs', 'occ', 'usoccl', 'uscurtrp', 'usstatel', 'usplacel', 'usyr1',
               'usyrl', 'ustrips', 'usexp', 'cncurtrp', 'docurtrp']
h = pd.read_stata(MMP / 'house174.dta', columns=hcols, convert_categoricals=False)
p = pd.read_stata(MMP / 'pers174.dta', columns=pcols, convert_categoricals=False)
assert not h.duplicated(KEY).any() and not p.duplicated(KEY + ['persnum']).any()
h = h.loc[h.country.eq(1) & h.surveypl.eq(1) & h.surveyyr.between(START_YEAR, END_YEAR)].copy()
assert len(h), 'No Mexico households in the selected interview window.'
h['mmp_household_id'] = h[KEY].astype(int).astype(str).agg(':'.join, axis=1)
p = p.merge(h[KEY + ['mmp_household_id', 'surveyyr']], on=KEY, how='inner',
            validate='many_to_one', suffixes=('', '_house'))
assert p.surveyyr.eq(p.surveyyr_house).all(), 'HOUSE/PERS interview year mismatch.'
assert p.mmp_household_id.nunique() == len(h), 'A selected household lacks a PERS roster.'
community_state = {c: s for s, communities in COMMUNITIES_BY_STATE.items() for c in communities}
assert len(community_state) == sum(map(len, COMMUNITIES_BY_STATE.values())) == 174
h['origin_state_code'] = h.commun.map(community_state)
assert h.origin_state_code.notna().all(), 'Community missing from state dictionary.'
h['origin_state'] = h.origin_state_code.map(MEXICO_STATES)

# Retain original numeric codes alongside readable categories. Never replace
# unknown occupation with a last-trip job; those refer to different moments.
p['age_years'] = p.age.replace({8888: np.nan, 9999: np.nan})
p['education_years'] = p.edyrs.replace({8888: np.nan, 9999: np.nan})
p['age_group'] = grouped(p.age_years, AGE_GROUPS)
p['education_group'] = grouped(p.education_years, EDUCATION_GROUPS)
for raw, name, mapping in [('sex', 'sex_label', SEX), ('relhead', 'relationship', RELATIONSHIP),
                            ('marstat', 'marital_status', MARITAL_STATUS), ('usstatel', 'us_state', US_STATES)]:
    p[name] = p[raw].map(mapping).fillna(UNKNOWN)
job_codes = [code for codes in JOB_GROUPS.values() for code in codes]
assert len(job_codes) == len(set(job_codes)), 'Occupation codes belong to multiple groups.'
job_map = {code: group for group, codes in JOB_GROUPS.items() for code in codes}
p['job_group'] = p.occ.map(job_map).fillna(UNKNOWN)

# %% 4. Build origin-household matching features from the full PERS roster.
away = p[['uscurtrp', 'cncurtrp', 'docurtrp']].apply(lambda x: x.map({1: 1., 2: 0., 8888: 0.}))
known_away = away.eq(1).any(axis=1)
known_home = away.eq(0).all(axis=1)
member = p.hhmemshp.eq(2)
excluded = p.hhmemshp.isin([1, 8888]) | known_away
p['resident'] = pd.Series(np.nan, index=p.index).mask(member & known_home, 1.).mask(excluded, 0.)
p['resident_child'] = p.age_years.lt(CHILD_AGE_LIMIT).astype(float).where(p.age_years.notna()).where(p.resident.eq(1))
p['resident_older'] = p.age_years.ge(OLDER_AGE_LIMIT).astype(float).where(p.age_years.notna()).where(p.resident.eq(1))
working = p.occ.map({**dict.fromkeys(WORKING_CODES, 1.), **dict.fromkeys(NONWORKING_CODES, 0.)})
working = working.where(p.age_years.ge(EMPLOYMENT_MIN_AGE), 0.).where(p.age_years.notna())
p['resident_worker'] = working.where(p.resident.eq(1))
features = ['resident', 'resident_child', 'resident_older', 'resident_worker']
for col in features[1:]:
    p.loc[p.resident.eq(0), col] = 0.
g = p.groupby('mmp_household_id')
counts = g[features].sum().where(g[features].count().eq(g.size(), axis=0))
h = h.merge(counts, on='mmp_household_id', validate='one_to_one')
roster_members = member.groupby(p.mmp_household_id).sum()
membership_known = p.hhmemshp.isin([1, 2, 8888]).groupby(p.mmp_household_id).all()
h['roster_complete'] = h.mmp_household_id.map(roster_members).eq(h.members) & h.mmp_household_id.map(membership_known)
h.loc[~h.roster_complete, features] = np.nan
h['size_group'] = grouped(h.resident, SIZE_GROUPS)
h['worker_group'] = grouped(h.resident_worker, WORKER_GROUPS)
for source, target in [('resident_child', 'children_present'), ('resident_older', 'older_present')]:
    h[target] = h[source].gt(0).map({True: 'Yes', False: 'No'}).where(h[source].notna(), UNKNOWN)

m = p.loc[p.uscurtrp.eq(1) & ~p.hhmemshp.eq(8888)].copy()
assert len(m), 'No recorded current US migrants in the selected sample.'
m['mmp_person_id'] = m.mmp_household_id + ':' + m.persnum.astype(int).astype(str)
m = m.merge(h[['mmp_household_id', 'origin_state_code', 'origin_state']], on='mmp_household_id', validate='many_to_one')
h['network_size'] = h.mmp_household_id.map(m.groupby('mmp_household_id').size())
h = h.loc[h.network_size.notna()].copy().reset_index(drop=True)
h['network_size'] = h.network_size.astype(int)
assert h.weight.gt(0).all() and h.weight.notna().all()

# %% 5. Draw whole networks until exactly N_AGENTS migrants have been generated.
assert h.network_size.eq(1).any(), 'Exact-size sampling requires a one-migrant household donor.'
sizes = h.network_size.to_numpy()
weights = h.weight.to_numpy() if USE_MMP_WEIGHTS else np.ones(len(h))
draws, remaining, constrained_draws = [], N_AGENTS, 0
while remaining:
    eligible = np.flatnonzero(sizes <= remaining)
    constrained_draws += int(len(eligible) < len(h))
    probability = weights[eligible] / weights[eligible].sum()
    chosen = rng_network.choice(eligible, p=probability)
    draws.append(chosen)
    remaining -= int(sizes[chosen])
households = h.iloc[draws].copy().reset_index(drop=True)
households.insert(0, 'synthetic_household_id', np.arange(1, len(households) + 1))
migrants = households[['synthetic_household_id', 'mmp_household_id']].merge(m, on='mmp_household_id', how='left')
migrants.insert(0, 'synthetic_migrant_id', np.arange(1, len(migrants) + 1))

# %% 6. Match each synthetic origin household to an ENIGH economic donor.
if MATCH_ENIGH:
    assert ENIGH.exists(), 'First run oct-26/analysis/enigh-households-2018-2024.py.'
    e = pd.read_csv(ENIGH, dtype={'household_id': str, 'folioviv': str, 'foliohog': str,
                                'ubica_geo': str, 'est_dis': str, 'upm': str})
    e = e.loc[e.year.eq(ENIGH_YEAR)].copy().reset_index(drop=True)
    assert len(e) and e.household_id.is_unique and e.factor.gt(0).all()
    e['size_group'] = grouped(e.tot_integ, SIZE_GROUPS)
    e['worker_group'] = grouped(e.ocupados, WORKER_GROUPS)
    e['children_present'] = e.menores.gt(0).map({True: 'Yes', False: 'No'}).where(e.menores.notna(), UNKNOWN)
    e['older_present'] = e.p65mas.gt(0).map({True: 'Yes', False: 'No'}).where(e.p65mas.notna(), UNKNOWN)
    assert MATCH_WEIGHTS and all(np.isfinite(w) and w > 0 for w in MATCH_WEIGHTS.values())
    assert set(MATCH_WEIGHTS) <= {'size_group', 'worker_group', 'children_present', 'older_present'}
    cache, matches = {}, []
    for row in households.to_dict('records'):
        profile = (row['origin_state_code'],) + tuple(row[c] for c in MATCH_WEIGHTS)
        if profile not in cache:
            candidates = e.loc[e.state.eq(row['origin_state_code'])]
            assert len(candidates), f'No ENIGH donor in state {row["origin_state_code"]}.'
            score = pd.Series(0., index=candidates.index)
            usable = [c for c in MATCH_WEIGHTS if row[c] != UNKNOWN]
            for c in usable:
                score += candidates[c].ne(row[c]) * MATCH_WEIGHTS[c]
            best = candidates.loc[score.eq(score.min())]
            cache[profile] = (best.index.to_numpy(), best.factor.to_numpy() / best.factor.sum(),
                              float(score.min()), len(usable))
        indices, probabilities, distance, n_features = cache[profile]
        donor = int(rng_match.choice(indices, p=probabilities))
        matches.append((donor, distance, n_features, len(indices)))
    match = pd.DataFrame(matches, columns=['enigh_row', 'match_distance', 'match_features_used', 'match_candidate_count'])
    households = pd.concat([households, match.drop(columns='enigh_row'),
                            e.iloc[match.enigh_row].reset_index(drop=True).add_prefix('enigh_')], axis=1)
    households['match_complete_exact'] = households.match_distance.eq(0) & households.match_features_used.eq(len(MATCH_WEIGHTS))
    assert households.origin_state_code.eq(households.enigh_state).all()
    migrants = migrants.merge(households[['synthetic_household_id', 'enigh_household_id']],
                              on='synthetic_household_id', validate='many_to_one')

# %% 7. Validate links and compare source/synthetic distributions, then save.
assert len(migrants) == N_AGENTS and migrants.synthetic_migrant_id.is_unique
assert households.synthetic_household_id.is_unique
assert migrants.synthetic_household_id.isin(households.synthetic_household_id).all()
actual_sizes = migrants.groupby('synthetic_household_id').size()
assert households.synthetic_household_id.map(actual_sizes).eq(households.network_size).all()
assert not migrants.duplicated(['synthetic_household_id', 'mmp_person_id']).any()

m['age_job_state'] = m[['age_group', 'job_group', 'us_state']].agg(' | '.join, axis=1)
migrants['age_job_state'] = migrants[['age_group', 'job_group', 'us_state']].agg(' | '.join, axis=1)
checks = []
for source, synthetic, columns, unit in [
    (m, migrants, ['age_group', 'job_group', 'relationship', 'us_state', 'origin_state', 'age_job_state'], 'migrant'),
    (h, households, ['network_size'], 'household')]:
    source_weights = source.mmp_household_id.map(h.set_index('mmp_household_id').weight) if USE_MMP_WEIGHTS else pd.Series(1., index=source.index)
    for col in columns:
        observed = source_weights.groupby(source[col]).sum() / source_weights.sum()
        simulated = synthetic[col].value_counts(normalize=True)
        table = pd.concat([observed.rename('source_share'), simulated.rename('synthetic_share')], axis=1).fillna(0)
        table['difference_pp'] = 100 * (table.synthetic_share - table.source_share)
        table.index.name = 'category'
        checks.append(table.reset_index().assign(characteristic=col, unit=unit))
diagnostics = pd.concat(checks, ignore_index=True)
summary = {'requested_migrants': N_AGENTS, 'synthetic_migrants': len(migrants),
    'synthetic_households': len(households), 'source_migrants': len(m), 'source_linked_households': len(h),
    'seed': SEED, 'mmp_years': [START_YEAR, END_YEAR], 'use_mmp_weights': USE_MMP_WEIGHTS,
    'match_enigh': MATCH_ENIGH, 'enigh_year': ENIGH_YEAR if MATCH_ENIGH else None,
    'end_constrained_draws': constrained_draws,
    'match_complete_exact_pct': float(100 * households.match_complete_exact.mean())
    if MATCH_ENIGH else None}

if SAVE_PARQUET:
    OUT.mkdir(parents=True, exist_ok=True)
    # Save standard Parquet files
    migrants.to_parquet(OUT / 'migrants.parquet', index=False)
    households.to_parquet(OUT / 'households.parquet', index=False)
    diagnostics.to_csv(OUT / 'distribution_checks.csv', index=False)

    # Save JSON files for browser dashboard
    migrants.to_json(OUT / 'migrants.json', orient='records', indent=2)
    households.to_json(OUT / 'households.json', orient='records', indent=2)

    settings = {**summary, 'age_groups': AGE_GROUPS, 'education_groups': EDUCATION_GROUPS,
                'job_groups': JOB_GROUPS, 'size_groups': SIZE_GROUPS, 'worker_groups': WORKER_GROUPS,
                'match_weights': MATCH_WEIGHTS, 'sex': SEX, 'marital_status': MARITAL_STATUS,
                'relationship': RELATIONSHIP, 'employment_min_age': EMPLOYMENT_MIN_AGE,
                'child_age_limit': CHILD_AGE_LIMIT, 'older_age_limit': OLDER_AGE_LIMIT,
                'working_codes': WORKING_CODES, 'nonworking_codes': NONWORKING_CODES,
                'mmp_path': str(MMP), 'enigh_path': str(ENIGH)}
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')

print(json.dumps(summary, indent=2))
if SAVE_PARQUET:
    print(f'Saved parquet, json, and diagnostics in {OUT}')