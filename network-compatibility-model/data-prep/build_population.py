"""Build the household-migrant network population from MMP (replaces the
oct-26 synthetic-migrants-house-networks population for this model).

Run: python network-compatibility-model/data-prep/build_population.py
Dependencies: numpy, pandas, pyarrow. Source data are read only.

Changes from oct-26/simulations/synthetic-migrants-house-networks.py
(population audit, 19 Oct 2026):
  1. Every connected MMP household is used exactly once. No resampling, no
     10,000-agent target, so calibration uses the full MMP sample.
  2. Links:
       Core       roster head/spouse/child currently on a US trip (PERS)
       Close kin  head's parents and siblings currently in the US (HOUSE
                  slots), plus any other roster member currently in the US
       Distant    HOUSE counts of uncles/aunts, cousins, nieces/nephews and
                  in-laws in the US, collapsed into ONE link per household
     Distant relatives are not independent potential senders: in MMP, receipt
     rises with close kin but not with the number of distant kin.
  3. Imputed profiles are labelled as such. Kin occupations come from roster
     donors, so occupational differences across households are partly
     imputation noise: treat rho / occupational concentration as sensitivity.
Unchanged: Mexico interviews 2010-19, household composition features, the
maximum-overlap rule between roster and questionnaire relatives, and the
recorded_fields_complete flag.

Outputs (MMP-derived microdata, git-ignored): population/households.parquet,
population/migrants.parquet, population/run_settings.json.
Aggregate summary (committable): outputs/population_summary.csv.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

# %% Settings
ROOT = Path('C:/git-projects/covid-remittances')
MMP = ROOT / 'data/Mesoamerican Migration Core Data SAV and DAT Files/DTA Files-20261004T141113Z-1-001/DTA Files'
HERE = Path(__file__).resolve().parent
POP_OUT = HERE / 'population'
SUMMARY_OUT = HERE / 'outputs'
SEED = 42
START_YEAR, END_YEAR = 2010, 2019
MIN_PROFILE_DONORS = 20
KEY = ['country', 'commun', 'surveypl', 'hhnum']
CORE_REL = [1, 2, 3]
NONRELATIVE = 19
PARENT_SLOTS = [('mousnw', 'molive', 'mousst', 2, 4), ('fausnw', 'falive', 'fausst', 1, 4)]
SIBLING_SLOTS = [(f'sbusnw{i}', f'sblive{i}', f'sbusst{i}', f'sbsex{i}', 5) for i in range(1, 13)]
DISTANT_COUNTS = {7: 'uncinus', 8: 'coinus', 6: 'neinus', 15: 'sbilinus', 14: 'chilinus', 16: 'pilinus'}
UNKNOWN_CODES = [8888, 9999]
# Parents must be adults and at least 12 years older than the head.
PARENT_MIN_AGE, PARENT_HEAD_GAP = 18, 12
CHILD_AGE_LIMIT, OLDER_AGE_LIMIT, EMPLOYMENT_MIN_AGE = 12, 65, 14
JOB_GROUPS = {
    'Working student/pensioner': [43, 54],
    'Not working': [i for i in range(10, 100) if i not in [43, 54]],
    'Agriculture': list(range(410, 420)),
    'Production/construction/transport': list(range(510, 560)),
    'Sales/personal services': list(range(710, 730)) + list(range(810, 840)),
    'Professional/administrative': list(range(110, 150)) + list(range(210, 220)) + list(range(610, 630)),
}
WORKING_CODES = [c for g, codes in JOB_GROUPS.items() if g != 'Not working' for c in codes]
NONWORKING_CODES = JOB_GROUPS['Not working']
# Appendix A of the MMP codebook: community -> origin state (copied from oct-26).
COMMUNITIES_BY_STATE = {
    1: [69, 71], 2: [63, 64, 65, 66], 6: [33, 67, 68], 8: list(range(78, 86)), 10: [73, 74, 75, 76],
    11: [1, 2, 4, 5, 13, 15, 16, 26, 27, 55, 56, 72, 87, 88, 89, 131], 12: [31, 41, 42, 43], 13: [92, 93],
    14: [3, 6, 7, 17, 20, 21, 23, 24, 25, 28, 57, 58, 91, 97, 98, 99, 107, 120, 122, 123, 124, 144, 145,
         146, 147, 151, 152, 154],
    15: [110, 111, 112, 113, 114, 155, 156, 157], 16: [8, 9, 10, 14, 19, 22, 29, 108, 109, 121],
    17: [115, 116, 117, 118, 162, 163, 164, 165, 166], 18: [11, 12], 19: [77, 86],
    20: [49, 50, 51, 52, 171, 172, 173, 174], 21: [54, 59, 60, 61, 132, 133, 134, 148, 149, 150, 158, 159, 161],
    22: [135, 136, 137, 138, 139, 160, 169, 170], 24: [32, 36, 37, 38, 39, 44, 45, 47, 48, 90, 129, 130],
    25: [53, 62, 70], 27: [140, 141, 142, 143], 29: [94, 95, 96], 30: [100, 101, 102, 103, 104, 105, 106, 119],
    31: [125, 126, 127, 128, 167, 168], 32: [18, 30, 34, 35, 40, 46, 153]}


# %% Load
def load():
    hcols = (KEY + ['surveyyr', 'weight', 'members', 'hage', 'remus']
             + [c for slot in PARENT_SLOTS for c in slot[:3]]
             + [c for slot in SIBLING_SLOTS for c in slot[:4]] + list(DISTANT_COUNTS.values()))
    pcols = KEY + ['surveyyr', 'persnum', 'hhmemshp', 'age', 'sex', 'relhead', 'occ', 'usstatel',
                   'uscurtrp', 'cncurtrp', 'docurtrp']
    h = pd.read_stata(MMP / 'house174.dta', columns=hcols, convert_categoricals=False)
    p = pd.read_stata(MMP / 'pers174.dta', columns=pcols, convert_categoricals=False)
    assert not h.duplicated(KEY).any() and not p.duplicated(KEY + ['persnum']).any()
    h = h.loc[h.country.eq(1) & h.surveypl.eq(1) & h.surveyyr.between(START_YEAR, END_YEAR)].copy()
    h['mmp_household_id'] = h[KEY].astype(int).astype(str).agg(':'.join, axis=1)
    p = p.merge(h[KEY + ['mmp_household_id']], on=KEY, validate='many_to_one')
    state = {c: s for s, cs in COMMUNITIES_BY_STATE.items() for c in cs}
    h['origin_state_code'] = h.commun.map(state)
    assert h.origin_state_code.notna().all()
    p['age_years'] = p.age.where(p.age.between(0, 120))
    return h.reset_index(drop=True), p


# %% Household composition (same definitions as oct-26)
def composition(h, p):
    away = p[['uscurtrp', 'cncurtrp', 'docurtrp']].apply(lambda x: x.map({1: 1., 2: 0., 8888: 0.}))
    member = p.hhmemshp.eq(2)
    excluded = p.hhmemshp.isin([1, 8888]) | away.eq(1).any(axis=1)
    p['resident'] = pd.Series(np.nan, index=p.index).mask(member & away.eq(0).all(axis=1), 1.).mask(excluded, 0.)
    p['resident_child'] = p.age_years.lt(CHILD_AGE_LIMIT).astype(float).where(p.age_years.notna()).where(p.resident.eq(1))
    p['resident_older'] = p.age_years.ge(OLDER_AGE_LIMIT).astype(float).where(p.age_years.notna()).where(p.resident.eq(1))
    working = p.occ.map({**dict.fromkeys(WORKING_CODES, 1.), **dict.fromkeys(NONWORKING_CODES, 0.)})
    working = working.where(p.age_years.ge(EMPLOYMENT_MIN_AGE), 0.).where(p.age_years.notna())
    p['resident_worker'] = working.where(p.resident.eq(1))
    feats = ['resident', 'resident_child', 'resident_older', 'resident_worker']
    for c in feats[1:]:
        p.loc[p.resident.eq(0), c] = 0.
    g = p.groupby('mmp_household_id')
    counts = g[feats].sum().where(g[feats].count().eq(g.size(), axis=0))
    h = h.merge(counts, on='mmp_household_id', how='left', validate='one_to_one')
    roster = member.groupby(p.mmp_household_id).sum()
    known = p.hhmemshp.isin([1, 2, 8888]).groupby(p.mmp_household_id).all()
    complete = h.mmp_household_id.map(roster).eq(h.members) & h.mmp_household_id.map(known).fillna(False)
    h.loc[~complete, feats] = np.nan
    return h


# %% Links
def links(h, p):
    current = p.uscurtrp.eq(1) & ~p.hhmemshp.eq(8888)
    roster = p.loc[current & p.relhead.between(1, 20) & p.relhead.ne(NONRELATIVE)].copy()
    roster['person_id'] = roster.mmp_household_id + ':' + roster.persnum.astype(int).astype(str)
    by_hh = {k: g for k, g in roster.groupby('mmp_household_id')}
    uncertain = ((~p.uscurtrp.isin([1, 2, 8888]) | (current & ~p.relhead.between(1, 20)))
                 & ~p.hhmemshp.eq(8888)).groupby(p.mmp_household_id).any()
    rows, flags = [], []
    for row in h.to_dict('records'):
        hid = row['mmp_household_id']
        people = by_hh.get(hid, roster.iloc[:0])
        unknown, conflicts = 0, 0
        # Individual HOUSE slots: parents (relhead 4) and siblings (5).
        house = {4: [], 5: []}
        for status, alive, st, sex, rel in PARENT_SLOTS + SIBLING_SLOTS:
            sex_value = sex if isinstance(sex, (int, float)) and rel == 4 else row.get(sex)
            v = row[status]
            conflicts += int(v == 1 and row[alive] == 2)
            if v == 1 and row[alive] != 2:
                house[rel].append(dict(house_sex=sex_value if sex_value in (1, 2) else np.nan))
            elif v not in [1, 2, 8888] and row[alive] != 2:
                unknown += 1
        distant_total = 0
        for rel, field in DISTANT_COUNTS.items():
            v = row[field]
            if pd.isna(v) or v in UNKNOWN_CODES:
                unknown += 1
                continue
            # Maximum overlap with roster members of the same relationship.
            distant_total += max(int(v) - int(people.relhead.eq(rel).sum()), 0)
        for person in people.to_dict('records'):
            rel = int(person['relhead'])
            rows.append(dict(mmp_household_id=hid, relhead=rel, source='PERS', person_id=person['person_id'],
                             link_type='Core' if rel in CORE_REL else 'Close kin', house_sex=np.nan))
        for rel, slots in house.items():
            n_pers = int(people.relhead.eq(rel).sum())
            for slot in slots[n_pers:]:                       # maximum overlap
                rows.append(dict(mmp_household_id=hid, relhead=rel, source='HOUSE', person_id=None,
                                 link_type='Close kin', **slot))
        if distant_total > 0:
            rows.append(dict(mmp_household_id=hid, relhead=-1, source='HOUSE', person_id=None,
                             link_type='Distant kin (aggregated)', house_sex=np.nan,
                             distant_relatives=distant_total))
        flags.append(dict(mmp_household_id=hid, house_unknown_fields=unknown, house_alive_conflicts=conflicts,
                          distant_relatives=distant_total))
    links = pd.DataFrame(rows)
    flags = pd.DataFrame(flags)
    flags['pers_network_uncertain'] = flags.mmp_household_id.map(uncertain).fillna(False).astype(bool)
    flags['recorded_fields_complete'] = (~flags.pers_network_uncertain & flags.house_unknown_fields.eq(0)
                                         & flags.house_alive_conflicts.eq(0))
    return links, flags, roster


# %% Profiles
def profiles(links, roster, h, rng):
    """Observed roster profiles are kept. HOUSE links copy a whole donor profile
    from roster migrants: same relationship if >= MIN_PROFILE_DONORS donors,
    else same origin state, else all; known sex and parent age bounds hold."""
    roster = roster.merge(h[['mmp_household_id', 'origin_state_code']], on='mmp_household_id')
    lookup = roster.set_index('person_id')
    head_age = h.set_index('mmp_household_id').hage
    out = []
    for row in links.to_dict('records'):
        if row['source'] == 'PERS':
            d = lookup.loc[row['person_id']]
            out.append(dict(age_years=d.age_years, sex=d.sex, occ=d.occ, profile_imputed=False,
                            profile_pool='Observed'))
            continue
        base = roster
        if row['relhead'] == 4:
            ha = head_age.get(row['mmp_household_id'])
            min_age = max(PARENT_MIN_AGE, ha + PARENT_HEAD_GAP) if pd.notna(ha) and 0 <= ha <= 120 else PARENT_MIN_AGE
            base = base.loc[base.age_years.ge(min_age)]
        else:
            base = base.loc[base.age_years.ge(18)]
        if pd.notna(row['house_sex']):
            base = base.loc[base.sex.eq(row['house_sex'])]
        origin = h.loc[h.mmp_household_id.eq(row['mmp_household_id']), 'origin_state_code'].iloc[0]
        pools = [('relationship', base.relhead.eq(row['relhead'])),
                 ('non-core kin', ~base.relhead.isin(CORE_REL)),
                 ('origin state', base.origin_state_code.eq(origin)),
                 ('all', pd.Series(True, index=base.index))]
        name, cand = next(((n, base.loc[m]) for n, m in pools if m.sum() >= MIN_PROFILE_DONORS), ('all', base))
        if cand.empty:
            # e.g. parents of an old head: keep the profile unknown, never invent one.
            out.append(dict(age_years=np.nan, sex=row['house_sex'], occ=np.nan, profile_imputed=True,
                            profile_pool='No compatible donor'))
            continue
        d = cand.iloc[rng.integers(len(cand))]
        out.append(dict(age_years=d.age_years, sex=row['house_sex'] if pd.notna(row['house_sex']) else d.sex,
                        occ=d.occ, profile_imputed=True, profile_pool=name))
    return pd.concat([links.reset_index(drop=True), pd.DataFrame(out)], axis=1)


# %% Main
def main():
    POP_OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    h, p = load()
    h = composition(h, p)
    lk, flags, roster = links(h, p)
    h = h.merge(flags, on='mmp_household_id', validate='one_to_one')
    count = lambda t: h.mmp_household_id.map(lk.loc[lk.link_type.eq(t)].groupby('mmp_household_id').size()).fillna(0).astype(int)
    h['core_count'] = count('Core')
    h['close_kin_count'] = count('Close kin')
    h['distant_link'] = count('Distant kin (aggregated)')
    h['kin_count'] = h.close_kin_count + h.distant_link
    h['network_size'] = h.core_count + h.kin_count
    all_households = h.copy()
    h = h.loc[h.network_size.gt(0)].reset_index(drop=True)
    h.insert(0, 'synthetic_household_id', np.arange(1, len(h) + 1))

    m = lk.merge(h[['mmp_household_id', 'synthetic_household_id']], on='mmp_household_id', validate='many_to_one')
    m = profiles(m, roster, h, rng)
    job_map = {c: g for g, codes in JOB_GROUPS.items() for c in codes}
    m['job_group'] = m.occ.map(job_map).fillna('Unknown')
    m['connection_group'] = np.where(m.link_type.eq('Core'), 'Core', 'Kin')
    m = m.sort_values(['synthetic_household_id', 'link_type']).reset_index(drop=True)
    m.insert(0, 'synthetic_migrant_id', np.arange(1, len(m) + 1))
    assert np.array_equal(m.groupby('synthetic_household_id').size().to_numpy(), h.network_size.to_numpy())

    h.to_parquet(POP_OUT / 'households.parquet', index=False)
    m.to_parquet(POP_OUT / 'migrants.parquet', index=False)

    # Aggregate summary: receipt by close-kin count and distant relatives (audit check).
    known = all_households.remus.isin([1, 2])
    a = all_households.loc[known].assign(received=lambda d: d.remus.eq(1))
    a['type'] = np.select([a.core_count.gt(0), a.close_kin_count.gt(0), a.distant_link.gt(0)],
                          ['Has core', 'Close kin, no core', 'Distant only'], 'No link')
    a['close_bin'] = pd.cut(a.close_kin_count, [-1, 0, 1, 2, 4, 100], labels=['0', '1', '2', '3-4', '5+'])
    a['distant_bin'] = pd.cut(a.distant_relatives, [-1, 0, 4, 9, 19, 10_000], labels=['0', '1-4', '5-9', '10-19', '20+'])
    summary = pd.concat([
        a.groupby('type').received.agg(n='count', receipt='mean').reset_index().rename(columns={'type': 'group'}).assign(dimension='link type'),
        a.loc[a.core_count.eq(0)].groupby('close_bin', observed=True).received.agg(n='count', receipt='mean').reset_index()
         .rename(columns={'close_bin': 'group'}).assign(dimension='close kin (no core)'),
        a.loc[a.core_count.eq(0) & a.close_kin_count.eq(0)].groupby('distant_bin', observed=True).received
         .agg(n='count', receipt='mean').reset_index().rename(columns={'distant_bin': 'group'}).assign(dimension='distant relatives (distant only)'),
        h.network_size.clip(upper=10).value_counts().sort_index().rename('n').reset_index()
         .rename(columns={'network_size': 'group'}).assign(dimension='network size (10 = 10+)'),
        m.groupby('link_type').size().rename('n').reset_index().rename(columns={'link_type': 'group'}).assign(dimension='links'),
        m.groupby('job_group').size().rename('n').reset_index().rename(columns={'job_group': 'group'}).assign(dimension='occupation'),
    ], ignore_index=True)[['dimension', 'group', 'n', 'receipt']]
    summary.to_csv(SUMMARY_OUT / 'population_summary.csv', index=False)

    settings = dict(seed=SEED, survey_years=[START_YEAR, END_YEAR], households_all=len(all_households),
                    households_connected=len(h), migrants=len(m),
                    links=m.link_type.value_counts().to_dict(),
                    imputed_profiles=int(m.profile_imputed.sum()),
                    recorded_fields_complete=int(h.recorded_fields_complete.sum()),
                    max_network_size=int(h.network_size.max()),
                    distant_rule='HOUSE counts of relhead 6,7,8,14,15,16 (minus roster overlap) collapsed into one link',
                    close_rule='PERS roster relatives currently in US + HOUSE parent/sibling slots (maximum overlap)',
                    connection_groups=dict(Core=CORE_REL), use_mmp_weights=False)
    (POP_OUT / 'run_settings.json').write_text(json.dumps(settings, indent=2, default=int))
    pd.set_option('display.width', 200)
    print(json.dumps(settings, indent=1, default=int))
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
