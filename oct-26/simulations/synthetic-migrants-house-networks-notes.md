# Alternative population: household-centred core/kin networks

Run `python oct-26/simulations/synthetic-migrants-house-networks.py`.
Requires pandas, numpy and pyarrow. This is a standalone script with a manually
set `ROOT`, numbered cells, editable dictionaries and no dependency on executing
the older script. Its outputs go to `oct-26/outputs/synthetic-migrants-house-networks/`.
The original population, script and user edits are preserved.

## Settings and classification

Set `N_AGENTS`, `SEED`, MMP interview years and optional ENIGH year in section 1.
`SAVE_PARQUET` and `SAVE_JSON` independently control the two export formats.
Turn both off to leave results in memory. Saved runs replace this alternative's
generated outputs. `USE_MMP_WEIGHTS=False` reproduces the sample's household mix;
True uses the community/sample expansion weights. Neither is nationally representative.

Section 2 contains editable analytical dictionaries:

| Characteristic | Default |
| --- | --- |
| Core | Head, spouse, son/daughter: RELHEAD 1, 2, 3 |
| Kin | All other relatives; nonrelatives/friends excluded |
| Age | Minor 0–17, working age 18–64, older 65+ |
| Education | 0–6, 7–12, 13+ completed school years |
| Ongoing US stay | Under 5, 5–9, 10+ years, using duration in months |
| Job | Agriculture; production/construction/transport; sales/services; professional/administrative; not working; working student/pensioner |

Unknowns remain explicit. Working age does not mean employed; job is occupation,
not industry. Schooling groups describe years, not inferred qualifications.
Stay comes from PERS `USDURL` for an ongoing trip, not cumulative US experience,
first departure or inferred continuous residence. For added kin it is imputed.
All profiles describe interview-era conditions, not a reconstructed 2019 stock.

## Network construction and overlap

Use PERS to keep individually recorded relatives on current US trips, excluding
recorded deceased people. HOUSE contributes parents/siblings currently in the US
and counts of uncles/aunts, cousins, nieces/nephews, siblings-in-law,
children-in-law and parents-in-law. Parent/sibling sex and state are retained
when reported. Confirmed deceased parent/sibling slots are excluded, with
contradictory current-residence reports counted as conflicts.

Within each overlapping relationship category, the default is:

`reconciled count = max(PERS count, known HOUSE count)`

This assumes maximum overlap between those sources, not verified identities.
The observed PERS profiles survive and only the HOUSE excess becomes added
connections. For parent/sibling slots, suppression of assumed duplicates first
prefers compatible sex/state. `OVERLAP_RULE='no_overlap'` instead adds the two
counts as a sensitivity scenario. It is not a claim that those people differ.
Distinct relationship categories are treated as distinct connections; possible
overlap through multiple family roles cannot be resolved here.

Unknown HOUSE counts do not become observed zeros. The script generates known
connections only and retains `house_unknown_fields`, `pers_network_uncertain`,
`house_alive_status_conflicts` and `recorded_fields_complete`. The latter means
the selected recorded fields are complete, not that all actual relatives have
been enumerated. Unknown-only households without any known connection are
excluded from sampling and counted in the run summary. Full selected-household
targets, including households not sampled, are saved for audit.

Whole reconciled household networks are resampled. Near the target only networks
that fit the remaining places are eligible, preserving exact `N_AGENTS` without
cutting off relatives. This causes a boundary bias toward small networks,
potentially material for small populations. A singleton donor is required.

**An agent represents a migrant connection to one origin household.** The same
real cousin may have been reported by several surveyed households; the data do
not establish a universal migrant identity. This version creates separate agents
for those links. Its count is not the number of unique actual emigrants, and
budgets shared across multiple recipient households are not modelled yet.
Large reported extended-family counts are preserved, not silently capped.

## Personal-profile imputation

PERS-linked people keep their observed age, sex, schooling, occupation, marital
status, ongoing-trip duration and state. Added HOUSE connections draw these
attributes together from one current PERS migrant, preserving a joint donor
profile. Every added profile is marked `profile_imputed=True`.

Donor pools broaden in this order until at least `MIN_PROFILE_DONORS` are found:

1. Same relationship, origin state and known destination.
2. Same relationship anywhere in the selected Mexico sample.
3. Kin with the same origin state and known destination.
4. Kin anywhere in that sample.
5. All migrants with the same origin state and known destination.
6. All migrants with the same origin state.
7. All migrants in that sample.

The default minimum is 20. There are only 18 individually observed kin under the
new definition in the 2010–2019 PERS sample, so broad borrowing is expected.
This does NOT identify the true job/age/stay distribution of extended kin.
`profile_pool`, `profile_pool_size`, `profile_donor_relationship` and donor IDs
make that assumption inspectable. If all pools are smaller than the minimum,
the first nonempty pool is used and marked `Sparse:`.

Known HOUSE sex remains a hard donor constraint. Editable `KIN_AGE_RULES` require
imputed parents to be at least 12 years older than the head when head age is
known; parents and in-law categories have a minimum age of 18. These are simple
model assumptions, not additional measured ages. The script does not assert
parent-in-law age gaps without the spouse's age. No compatible donor means
unknown personal attributes rather than relaxation of those constraints.

HOUSE-observed state/sex override donor values. Destination matching can be
relaxed when pools are sparse; `destination_differs_from_donor` marks actual
state overrides. This can break the observed job/state association and should
be considered in employment-shock sensitivity work. Missing values in donor
profiles remain missing. No separate field-by-field imputation is applied.

Independent random streams control network selection, profile imputation and
ENIGH matching. Repeated copies of an origin household can draw different kin
profiles and ENIGH economic donors while retaining their network counts.
Added relatives draw profiles independently conditional on the donor rules;
their shared destinations, jobs and shock correlations are not separately
estimated from HOUSE. Those correlations require sensitivity analysis.

## ENIGH matching and remittances

Optional ENIGH matching stays within origin state and minimizes category
differences in resident-size group, presence of under-12s, presence of 65+ people,
and employed-member group. MMP resident size and employment use the same proxies
as the first script. Uncertain features are omitted and counted in the match
diagnostics. Rural/urban locality is not harmonized in this version.

All ENIGH fields have the `enigh_` prefix. They supply an economic donor, not an
identified real family or a replacement MMP household roster. Raw monetary fields
are quarterly nominal pesos; `_real` fields follow the existing ENIGH template
analysis's approximate quarterly 2024-price amounts, including noncash income.
ENIGH receipt and amounts do not enter matching and are not simulated transfers.
MMP `remus` and `remussiz` are original household observations; the latter is
ordinal, not a monetary amount. No potential connection is marked as an active
sender. Activation and COVID-period behavioural changes require a later model.

## Files and joins

- `migrants.parquet` / `migrants.json`: one row per synthetic linked agent.
- `households.parquet` / `households.json`: one row per synthetic origin household.
- `household_network_targets.csv`: all selected MMP households, raw selected HOUSE
  kin variables, reconciled counts, core/kin type and missingness flags.
- `overlap_audit.csv`: PERS count, known HOUSE count, assumed overlap, added slots
  and reconciled count for each overlapping relationship category/household.
- `distribution_checks.csv`: network shares against reconciled targets, with the
  selected weighting convention. Personal traits are not validated against an
  unobserved HOUSE distribution.
- `profile_distributions.csv`: age/job/education/stay/state/pool counts separated
  by core/kin and observed/imputed profile status.
- `run_settings.json`: settings, category dictionaries and count diagnostics.

Join migrant and household files on `synthetic_household_id`.
`synthetic_migrant_id` is unique per run. `source_slot_id` identifies a source
connection template and repeats across synthetic household copies.
`observed_person_id` exists only for actual PERS-linked connections.
`profile_donor_person_id` identifies the profile supplier, NOT the true identity
of an added HOUSE relative. `mmp_household_id` always names the network's origin
household, not necessarily the profile donor's household. `enigh_household_id`
names the matched economic donor and may repeat. Each synthetic row counts once;
do not multiply source MMP/ENIGH weights again after weighted draws.

Coding sources: MMP174 [HOUSE](https://mmp.research.brown.edu/images/house-codebook-mmp174-.pdf),
[PERS](https://mmp.research.brown.edu/images/pers-codebook-mmp174-.pdf),
[community states](https://mmp.research.brown.edu/images/appendix-a-sample-information-mmp174-.pdf),
[state codes](https://mmp.research.brown.edu/images/appendix-c-states-mmp174-.pdf),
[occupations](https://mmp.research.brown.edu/images/appendix-d-occupations-mmp174-.pdf).
