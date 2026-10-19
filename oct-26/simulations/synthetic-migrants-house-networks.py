"""Build household-centred core/kin networks using HOUSE counts and PERS profiles.

Run from any directory: python oct-26/simulations/synthetic-migrants-house-networks.py
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
SAVE_JSON = True                  # Independent browser-dashboard export.
OVERLAP_RULE = 'maximum_overlap'  # Or 'no_overlap' as a sensitivity scenario.
MIN_PROFILE_DONORS = 20           # Broaden sparse donor pools before drawing.
# Unknown HOUSE counts are NOT zero. Generate known connections only, flag
# incomplete networks, and report excluded households with no known connection.

ROOT = Path('C:/git-projects/covid-remittances')
MMP = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
ENIGH = ROOT / 'oct-26/outputs/enigh-households-2018-2024/origin_household_templates.csv'
OUT = ROOT / 'oct-26/outputs/synthetic-migrants-house-networks'
KEY = ['country', 'commun', 'surveypl', 'hhnum']
UNKNOWN = 'Unknown'

# %% 2. Editable category dictionaries.
# Numeric intervals INCLUDE both endpoints. Gaps become Unknown; overlaps fail.
# These groups label existing people: changing bins does not redraw their ages.
CONNECTION_GROUPS = {'Core': [1, 2, 3], 'Kin': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]}
# Nonrelatives/friends (19) are deliberately outside this family-only model.
AGE_GROUPS = {'Minor': (0, 17), 'Working age': (18, 64), 'Older': (65, 120)}
EDUCATION_GROUPS = {'0-6 years': (0, 6), '7-12 years': (7, 12), '13+ years': (13, 40)}
STAY_GROUPS = {'Under 5 years': (0, 59), '5-9 years': (60, 119), '10+ years': (120, 1800)}
# Stay is reported USDURL in MONTHS for an ongoing trip, not lifetime USEXP.
# Minimal age assumptions for IMPUTED kin; observed PERS ages are untouched.
# (minimum age, minimum age minus head's age). None means no head-age bound.
KIN_AGE_RULES = {4: (18, 12), 14: (18, None), 15: (18, None), 16: (18, None)}
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


# %% 3. Load MMP and keep original identifiers, including HOUSE-only networks.
# HOUSE's parent/sibling slots and extended-family counts describe relatives of
# the head. They are not disjoint from PERS, and not identified across households.
PARENT_SLOTS = [('mother', 'mousnw', 'molive', 'mousst', 2),
                ('father', 'fausnw', 'falive', 'fausst', 1)]
EXTENDED_COUNTS = {7: 'uncinus', 8: 'coinus', 6: 'neinus',
                   15: 'sbilinus', 14: 'chilinus', 16: 'pilinus'}
sibling_cols = [f'{prefix}{i}' for i in range(1, 13)
                for prefix in ['sbusnw', 'sblive', 'sbusst', 'sbsex']]
house_kin_cols = [c for _, status, alive, state, _ in PARENT_SLOTS for c in [status, alive, state]]
house_kin_cols += sibling_cols + list(EXTENDED_COUNTS.values())
assert isinstance(N_AGENTS, int) and N_AGENTS > 0
assert MIN_PROFILE_DONORS > 0 and START_YEAR <= END_YEAR
assert OVERLAP_RULE in ['maximum_overlap', 'no_overlap']
assert CHILD_AGE_LIMIT == 12 and OLDER_AGE_LIMIT == 65, 'ENIGH summaries have fixed cutoffs.'
assert set(CONNECTION_GROUPS) == {'Core', 'Kin'}
assert set(CONNECTION_GROUPS['Core']).isdisjoint(CONNECTION_GROUPS['Kin'])
assert set(sum(CONNECTION_GROUPS.values(), [])) == set(range(1, 21)) - {19}
rng_network, rng_profile, rng_match = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(3)]
hcols = KEY + ['surveyyr', 'weight', 'members', 'workers', 'hage', 'hsex', 'remus', 'remussiz'] + house_kin_cols
pcols = KEY + ['surveyyr', 'persnum', 'hhmemshp', 'age', 'sex', 'relhead', 'marstat',
               'edyrs', 'occ', 'usoccl', 'uscurtrp', 'usstatel', 'usdurl', 'cncurtrp', 'docurtrp']
h = pd.read_stata(MMP / 'house174.dta', columns=hcols, convert_categoricals=False)
p = pd.read_stata(MMP / 'pers174.dta', columns=pcols, convert_categoricals=False)
assert not h.duplicated(KEY).any() and not p.duplicated(KEY + ['persnum']).any()
h = h.loc[h.country.eq(1) & h.surveypl.eq(1) & h.surveyyr.between(START_YEAR, END_YEAR)].copy()
assert len(h), 'No Mexico households in the selected window.'
h['mmp_household_id'] = h[KEY].astype(int).astype(str).agg(':'.join, axis=1)
p = p.merge(h[KEY + ['mmp_household_id', 'surveyyr']], on=KEY, validate='many_to_one', suffixes=('', '_house'))
assert p.surveyyr.eq(p.surveyyr_house).all()
assert p.mmp_household_id.nunique() == len(h), 'Missing PERS roster for selected household.'
community_state = {c: s for s, communities in COMMUNITIES_BY_STATE.items() for c in communities}
assert len(community_state) == sum(map(len, COMMUNITIES_BY_STATE.values())) == 174
h['origin_state_code'] = h.commun.map(community_state)
assert h.origin_state_code.notna().all()
h['origin_state'] = h.origin_state_code.map(MEXICO_STATES)
p['age_years'] = p.age.where(p.age.between(0, 120))
p['education_years'] = p.edyrs.where(p.edyrs.between(0, 40))
p['stay_months'] = p.usdurl.where(p.usdurl.between(0, 1800))
job_codes = [code for codes in JOB_GROUPS.values() for code in codes]
assert len(job_codes) == len(set(job_codes)), 'Overlapping job categories.'
job_map = {code: group for group, codes in JOB_GROUPS.items() for code in codes}
connection_map = {code: group for group, codes in CONNECTION_GROUPS.items() for code in codes}




# %% 4. Build origin-household matching features from the full PERS roster.
# HHMEMSHP=2 includes absent members. Resident is a proxy: household member
# without a recorded ongoing US, Canadian or domestic migration. Unknown status
# remains unknown; one positive migration flag is enough to establish absence.
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


# %% 5. Reconcile PERS and HOUSE, category by category, into connection slots.
# Observed PERS people are retained. For each overlapping relationship category,
# default total = max(PERS count, known HOUSE count). Only the positive HOUSE
# excess becomes added slots. This assumes maximum overlap, NOT proven identity.
# no_overlap instead adds both sources, providing an upper-count sensitivity.
# Relationships reported in different HOUSE categories are assumed distinct.
# Unknown counts add no invented people, but remain flagged as unknown below.
current = p.uscurtrp.eq(1) & ~p.hhmemshp.eq(8888)
observed = p.loc[current & p.relhead.isin(connection_map)].copy()
observed['mmp_person_id'] = observed.mmp_household_id + ':' + observed.persnum.astype(int).astype(str)
observed = observed.merge(h[['mmp_household_id', 'origin_state_code']], on='mmp_household_id', validate='many_to_one')
observed['connection_group'] = observed.relhead.map(connection_map)
observed = observed.reset_index(drop=True)
assert len(observed), 'No PERS family migrant profiles available for this window.'
pers_groups = {key: group.to_dict('records') for key, group in observed.groupby('mmp_household_id')}
uncertain = (~p.uscurtrp.isin([1, 2, 8888]) | (current & ~p.relhead.isin(list(connection_map) + [19]))) & ~p.hhmemshp.eq(8888)
h['pers_network_uncertain'] = h.mmp_household_id.map(uncertain.groupby(p.mmp_household_id).any())
slot_rows, audit_rows, household_flags = [], [], []
for row in h.to_dict('records'):
    hid = row['mmp_household_id']
    people = pers_groups.get(hid, [])
    # A slot has a family relationship and possibly a HOUSE-observed sex/state.
    # It has no fabricated person identity. The stable slot ID is assigned later.
    house_slots = {rel: [] for rel in [4, 5] + list(EXTENDED_COUNTS)}
    unknown_by_rel = dict.fromkeys(house_slots, 0)
    conflicts = 0
    individual_slots = [(4, name, status, alive, state, sex)
                        for name, status, alive, state, sex in PARENT_SLOTS]
    individual_slots += [(5, f'sibling_{i}', f'sbusnw{i}', f'sblive{i}', f'sbusst{i}', row[f'sbsex{i}'])
                         for i in range(1, 13)]
    for rel, name, status, alive, state, sex in individual_slots:
        value = row[status]
        conflicts += int(value == 1 and row[alive] == 2)
        if value == 1 and row[alive] != 2:
            house_slots[rel].append({'house_field': name, 'house_sex': sex if sex in SEX else np.nan,
                                    'house_us_state': row[state] if row[state] in US_STATES and row[state] != 199 else np.nan})
        elif value not in [1, 2, 8888] and row[alive] != 2:
            unknown_by_rel[rel] += 1
    for rel, field in EXTENDED_COUNTS.items():
        value = row[field]
        if pd.isna(value) or value in [8888, 9999]:
            unknown_by_rel[rel] += 1
        else:
            assert value >= 0 and value == int(value), f'Invalid count in {field}: {value}'
            house_slots[rel] = [{'house_field': field, 'house_sex': np.nan, 'house_us_state': np.nan}
                                for _ in range(int(value))]
    # PERS is used directly where personal profiles are observed.
    for person in people:
        slot_rows.append({'mmp_household_id': hid, 'relhead': int(person['relhead']),
                          'observed_person_id': person['mmp_person_id'], 'connection_source': 'PERS',
                          'house_field': None, 'house_sex': np.nan, 'house_us_state': np.nan})
    for rel, slots in house_slots.items():
        same_relatives = [person for person in people if person['relhead'] == rel]
        n_pers = len(same_relatives)
        overlap = min(n_pers, len(slots)) if OVERLAP_RULE == 'maximum_overlap' else 0
        # With no cross-file IDs, which HOUSE slots overlap cannot be established.
        # Prefer equal known sex/state to suppress plausible duplicates first;
        # remaining count-based overlap is explicitly an assumption in the audit.
        remaining_slots = slots.copy()
        for person in same_relatives[:overlap]:
            penalties = [int(pd.notna(s['house_sex']) and s['house_sex'] != person['sex'])
                         + int(pd.notna(s['house_us_state']) and s['house_us_state'] != person['usstatel'])
                         for s in remaining_slots]
            remaining_slots.pop(int(np.argmin(penalties)))
        for slot in remaining_slots:
            slot_rows.append({'mmp_household_id': hid, 'relhead': rel, 'observed_person_id': None,
                              'connection_source': 'HOUSE', **slot})
        audit_rows.append({'mmp_household_id': hid, 'relationship_code': rel,
                           'pers_count': n_pers, 'house_known_count': len(slots),
                           'house_unknown_fields': unknown_by_rel[rel], 'assumed_overlap': overlap,
                           'added_connections': len(remaining_slots),
                           'reconciled_count': n_pers + len(remaining_slots)})
    household_flags.append({'mmp_household_id': hid, 'house_unknown_fields': sum(unknown_by_rel.values()),
                            'house_alive_status_conflicts': conflicts})
slots = pd.DataFrame(slot_rows)
slots['source_slot_id'] = slots.mmp_household_id + ':slot:' + (slots.groupby('mmp_household_id').cumcount() + 1).astype(str)
slots['connection_group'] = slots.relhead.map(connection_map)
audit = pd.DataFrame(audit_rows)
h = h.merge(pd.DataFrame(household_flags), on='mmp_household_id', validate='one_to_one')
h['recorded_fields_complete'] = ~h.pers_network_uncertain & h.house_unknown_fields.eq(0) & h.house_alive_status_conflicts.eq(0)
for group in CONNECTION_GROUPS:
    h[group.lower() + '_count'] = h.mmp_household_id.map(slots.loc[slots.connection_group.eq(group)].groupby('mmp_household_id').size()).fillna(0).astype(int)
h['network_size'] = h.core_count + h.kin_count
h['network_type'] = np.select([h.core_count.gt(0) & h.kin_count.gt(0), h.core_count.gt(0), h.kin_count.gt(0)],
                              ['Core + kin', 'Core only', 'Kin only'], default='No recorded connection')
# These are known-count targets, not claims that unreported links do not exist.
source_households = h.copy()
h = h.loc[h.network_size.gt(0)].reset_index(drop=True)
assert len(h) and h.weight.gt(0).all()
assert audit.reconciled_count.eq(audit.pers_count + audit.house_known_count - audit.assumed_overlap).all()


# %% 6. Sample whole reconciled networks to reach exactly N_AGENTS.
# One agent represents one migrant connection to one origin household. Relatives
# shared between different surveyed households cannot be deduplicated from these
# data. Do not read the total as a count of unique real-world Mexican emigrants.
# Close to the target only networks fitting the remainder can be drawn. This
# preserves networks but introduces boundary bias, especially in tiny runs.
assert h.network_size.eq(1).any(), 'Exact target requires at least one singleton network.'
sizes = h.network_size.to_numpy()
weights = h.weight.to_numpy() if USE_MMP_WEIGHTS else np.ones(len(h))
draws, remaining, constrained_draws = [], N_AGENTS, 0
while remaining:
    eligible = np.flatnonzero(sizes <= remaining)
    constrained_draws += int(len(eligible) < len(h))
    chosen = rng_network.choice(eligible, p=weights[eligible] / weights[eligible].sum())
    draws.append(chosen)
    remaining -= int(sizes[chosen])
households = h.iloc[draws].reset_index(drop=True)
households.insert(0, 'synthetic_household_id', np.arange(1, len(households) + 1))
migrants = households[['synthetic_household_id', 'mmp_household_id', 'origin_state_code', 'origin_state', 'hage']].merge(slots, on='mmp_household_id')
migrants.insert(0, 'synthetic_migrant_id', np.arange(1, len(migrants) + 1))


# %% 7. Fill personal characteristics, retaining observed versus imputed status.
# Copy a WHOLE donor profile: age/education/job/sex/stay/state stay together.
# Known HOUSE sex/state take priority. Donor pools first condition on these
# attributes and origin/relationship; sparse pools broaden transparently.
# The recent PERS pool has very few non-core relatives. MIN_PROFILE_DONORS avoids
# letting a handful determine all kin jobs. "all_*" pools borrow from core too:
# this is an assumption about kin, never an empirical kin distribution.
# Age constraints and known sex are never relaxed. If no donor satisfies them,
# keep personal fields unknown. Do not invent implausibly young parents.
profile_cols = ['age_years', 'education_years', 'occ', 'sex', 'marstat', 'stay_months', 'usstatel']
lookup = observed.set_index('mmp_person_id', drop=False)
profile_cache, profile_rows = {}, []
for row in migrants.to_dict('records'):
    is_imputed = row['connection_source'] == 'HOUSE'
    if not is_imputed:
        donor = lookup.loc[row['observed_person_id']]
        pool_name, pool_size = 'Observed PERS', 1
        min_age = np.nan
    else:
        minimum, gap = KIN_AGE_RULES.get(row['relhead'], (0, None))
        head_age = row['hage'] if 0 <= row['hage'] <= 120 else None
        min_age = max(minimum, head_age + gap) if gap is not None and head_age is not None else minimum
        known_sex = int(row['house_sex']) if pd.notna(row['house_sex']) else None
        known_state = int(row['house_us_state']) if pd.notna(row['house_us_state']) else None
        key = (row['relhead'], row['origin_state_code'], min_age, known_sex, known_state)
        if key not in profile_cache:
            # Donors with unknown age cannot establish compatibility with a
            # positive age bound; otherwise their missing age may be retained.
            age_ok = observed.age_years.ge(min_age) | (observed.age_years.isna() & (min_age == 0))
            sex_ok = observed.sex.eq(known_sex) if known_sex is not None else pd.Series(True, index=observed.index)
            base = observed.loc[age_ok & sex_ok]
            same_origin = base.origin_state_code.eq(row['origin_state_code'])
            same_destination = base.usstatel.eq(known_state) if known_state is not None else pd.Series(True, index=base.index)
            same_relation = base.relhead.eq(row['relhead'])
            kin = base.connection_group.eq('Kin')
            masks = {'relationship_origin_destination': same_relation & same_origin & same_destination,
                     'relationship_anywhere': same_relation, 'kin_origin_destination': kin & same_origin & same_destination,
                     'kin_anywhere': kin, 'all_origin_destination': same_origin & same_destination,
                     'all_origin': same_origin, 'all_anywhere': pd.Series(True, index=base.index)}
            candidates, pool_name = base.iloc[:0], 'No compatible donor'
            for name, mask in masks.items():
                if mask.sum() >= MIN_PROFILE_DONORS:
                    candidates, pool_name = base.loc[mask], name
                    break
            if candidates.empty and len(base):
                # None meet the preferred minimum; choose the first nonempty
                # pool, retaining the small size as a diagnostic.
                for name, mask in masks.items():
                    if mask.any():
                        candidates, pool_name = base.loc[mask], 'Sparse: ' + name
                        break
            probabilities = np.ones(len(candidates))
            if USE_MMP_WEIGHTS:
                probabilities = candidates.mmp_household_id.map(h.set_index('mmp_household_id').weight).to_numpy()
            if len(candidates):
                probabilities = probabilities / probabilities.sum()
            profile_cache[key] = (candidates.index.to_numpy(), probabilities, pool_name)
        indices, probabilities, pool_name = profile_cache[key]
        pool_size = len(indices)
        donor = observed.loc[rng_profile.choice(indices, p=probabilities)] if pool_size else None
    values = {c: donor[c] if donor is not None else np.nan for c in profile_cols}
    # HOUSE location and sex are observed at the link level, even when the
    # remaining profile is imputed. State overrides are flagged for diagnostics.
    state_override = is_imputed and pd.notna(row['house_us_state'])
    sex_override = is_imputed and pd.notna(row['house_sex'])
    donor_state = values['usstatel']
    if state_override:
        values['usstatel'] = row['house_us_state']
    if sex_override:
        values['sex'] = row['house_sex']
    profile_rows.append({**values, 'profile_imputed': is_imputed, 'profile_pool': pool_name,
                         'profile_pool_size': pool_size, 'imputation_min_age': min_age,
                         'profile_donor_person_id': donor['mmp_person_id'] if donor is not None else None,
                         'profile_donor_relationship': int(donor['relhead']) if donor is not None else None,
                         'profile_donor_survey_year': int(donor['surveyyr']) if donor is not None else None,
                         'profile_donor_origin_state': int(donor['origin_state_code']) if donor is not None else None,
                         'us_state_source': 'HOUSE' if state_override else ('Imputed PERS' if is_imputed else 'PERS'),
                         'sex_source': 'HOUSE' if sex_override else ('Imputed PERS' if is_imputed else 'PERS'),
                         'destination_differs_from_donor': bool(state_override and donor_state != row['house_us_state'])})
migrants = pd.concat([migrants, pd.DataFrame(profile_rows)], axis=1)
migrants['relationship'] = migrants.relhead.map(RELATIONSHIP)
migrants['job_group'] = migrants.occ.map(job_map).fillna(UNKNOWN)
migrants['age_group'] = grouped(migrants.age_years, AGE_GROUPS)
migrants['education_group'] = grouped(migrants.education_years, EDUCATION_GROUPS)
migrants['stay_group'] = grouped(migrants.stay_months, STAY_GROUPS)
migrants['us_state'] = migrants.usstatel.map(US_STATES).fillna(UNKNOWN)
migrants['sex_label'] = migrants.sex.map(SEX).fillna(UNKNOWN)
migrants['marital_status'] = migrants.marstat.map(MARITAL_STATUS).fillna(UNKNOWN)
# These links are potential contributors. No sender activation or transfers are
# assigned here; household REMUS is a receipt observation, not a sender label.



# %% 8. Match each synthetic origin household to an ENIGH economic donor.
# Match on grouped origin characteristics within state, using survey weights
# among tied closest donors. Do not match on remittance receipt or amount.
# ENIGH supplies an economic profile, not the actual family of these migrants.
# Prefixed ENIGH amounts remain donor observations (quarterly pesos), not
# generated transfers. The MMP structure and its uncertainty flags are retained.
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


# %% 9. Validate counts/provenance and save both Parquet and dashboard JSON.
# Compare network structure with reconciled HOUSE/PERS targets, NOT with the
# child-heavy original PERS marginal. Imputed personal profiles have no observed
# HOUSE counterpart; profile_distributions.csv describes them without claiming
# that their job/age distributions are measured kin-population targets.
assert len(migrants) == N_AGENTS and migrants.synthetic_migrant_id.is_unique
assert households.synthetic_household_id.is_unique
assert migrants.synthetic_household_id.isin(households.synthetic_household_id).all()
assert not migrants.duplicated(['synthetic_household_id', 'source_slot_id']).any()
actual_sizes = migrants.groupby('synthetic_household_id').size()
assert households.synthetic_household_id.map(actual_sizes).eq(households.network_size).all()
for group in CONNECTION_GROUPS:
    actual = migrants.loc[migrants.connection_group.eq(group)].groupby('synthetic_household_id').size()
    assert households.synthetic_household_id.map(actual).fillna(0).eq(households[group.lower() + '_count']).all()
assert migrants.profile_imputed.eq(migrants.connection_source.eq('HOUSE')).all()
assert migrants.loc[~migrants.profile_imputed, 'observed_person_id'].notna().all()
assert migrants.loc[migrants.profile_imputed, 'observed_person_id'].isna().all()
assert (migrants.age_years.ge(migrants.imputation_min_age) | migrants.age_years.isna() | ~migrants.profile_imputed).all()
checks = []
slot_targets = slots.merge(h[['mmp_household_id', 'origin_state']], on='mmp_household_id', validate='many_to_one')
for source, synthetic, columns, unit in [
    (slot_targets, migrants, ['connection_group', 'origin_state'], 'connection'),
    (h, households, ['network_type', 'network_size', 'recorded_fields_complete'], 'household')]:
    source_weights = source.mmp_household_id.map(h.set_index('mmp_household_id').weight) if USE_MMP_WEIGHTS else pd.Series(1., index=source.index)
    for col in columns:
        target = source_weights.groupby(source[col]).sum() / source_weights.sum()
        actual = synthetic[col].value_counts(normalize=True)
        table = pd.concat([target.rename('target_share'), actual.rename('synthetic_share')], axis=1).fillna(0)
        table['difference_pp'] = 100 * (table.synthetic_share - table.target_share)
        table.index.name = 'category'
        checks.append(table.reset_index().assign(characteristic=col, unit=unit))
diagnostics = pd.concat(checks, ignore_index=True)
profile_tables = []
for col in ['age_group', 'job_group', 'education_group', 'stay_group', 'us_state', 'profile_pool']:
    table = migrants.groupby(['connection_group', 'profile_imputed', col]).size().rename('n').reset_index()
    table = table.rename(columns={col: 'category'})
    table['characteristic'] = col
    profile_tables.append(table)
profile_distributions = pd.concat(profile_tables, ignore_index=True)
summary = {
    'requested_agents': N_AGENTS, 'synthetic_agents': len(migrants), 'synthetic_households': len(households),
    'agent_unit': 'One migrant connection to one origin household; not cross-household deduplicated',
    'source_households_all': len(source_households), 'source_households_with_connections': len(h),
    'source_pers_family_migrants': len(observed), 'source_reconciled_connections': len(slots),
    'source_added_house_connections': int(slots.connection_source.eq('HOUSE').sum()),
    'source_assumed_overlap': int(audit.assumed_overlap.sum()),
    'source_core_connections': int(slots.connection_group.eq('Core').sum()),
    'source_kin_connections': int(slots.connection_group.eq('Kin').sum()),
    'source_incomplete_recorded_networks': int((~source_households.recorded_fields_complete).sum()),
    'source_no_known_connection_but_incomplete': int((source_households.network_size.eq(0) & ~source_households.recorded_fields_complete).sum()),
    'synthetic_core_agents': int(migrants.connection_group.eq('Core').sum()),
    'synthetic_kin_agents': int(migrants.connection_group.eq('Kin').sum()),
    'imputed_profiles': int(migrants.profile_imputed.sum()),
    'imputed_without_compatible_donor': int((migrants.profile_imputed & migrants.profile_donor_person_id.isna()).sum()),
    'imputed_borrowing_core_profile': int((migrants.profile_imputed & migrants.profile_donor_relationship.isin(CONNECTION_GROUPS['Core'])).sum()),
    'match_complete_exact_pct': float(100 * households.match_complete_exact.mean()) if MATCH_ENIGH else None,
    'end_constrained_draws': constrained_draws}
settings = {**summary, 'seed': SEED, 'mmp_years': [START_YEAR, END_YEAR], 'use_mmp_weights': USE_MMP_WEIGHTS,
    'overlap_rule': OVERLAP_RULE, 'minimum_profile_donors': MIN_PROFILE_DONORS,
    'connection_groups': CONNECTION_GROUPS, 'age_groups': AGE_GROUPS, 'job_groups': JOB_GROUPS,
    'education_groups': EDUCATION_GROUPS, 'stay_groups_months': STAY_GROUPS, 'kin_age_rules': KIN_AGE_RULES,
    'size_groups': SIZE_GROUPS, 'worker_groups': WORKER_GROUPS, 'match_weights': MATCH_WEIGHTS,
    'match_enigh': MATCH_ENIGH, 'enigh_year': ENIGH_YEAR if MATCH_ENIGH else None,
    'save_parquet': SAVE_PARQUET, 'save_json': SAVE_JSON,
    'mmp_path': str(MMP), 'enigh_path': str(ENIGH),
    'unknown_count_policy': 'Generate known counts only; preserve incompleteness flags',
    'population_scope': 'Selected Mexico-interviewed households with at least one known family connection abroad'}
if SAVE_PARQUET or SAVE_JSON:
    OUT.mkdir(parents=True, exist_ok=True)
    if SAVE_PARQUET:
        migrants.to_parquet(OUT / 'migrants.parquet', index=False)
        households.to_parquet(OUT / 'households.parquet', index=False)
    if SAVE_JSON:
        migrants.to_json(OUT / 'migrants.json', orient='records', indent=2)
        households.to_json(OUT / 'households.json', orient='records', indent=2)
    diagnostics.to_csv(OUT / 'distribution_checks.csv', index=False)
    profile_distributions.to_csv(OUT / 'profile_distributions.csv', index=False)
    # Audit includes zero and unknown categories: missing must not become zero.
    audit.to_csv(OUT / 'overlap_audit.csv', index=False)
    source_households.to_csv(OUT / 'household_network_targets.csv', index=False)
    (OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
if SAVE_PARQUET or SAVE_JSON:
    print(f'Saved alternative population and diagnostics in {OUT}')

