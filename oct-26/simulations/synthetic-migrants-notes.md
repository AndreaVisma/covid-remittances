# Synthetic migrants and linked origin households

Run `python oct-26/simulations/synthetic-migrants.py` from the project root.
Dependencies: pandas, numpy and pyarrow. The script also works from other
directories, and its numbered `# %%` cells can be run in an editor.

## What to edit

- Section 1: `N_AGENTS` is the exact number of migrant people; `SEED` makes a
  run reproducible. Change the MMP interview window here as well.
- Section 2: age, schooling, job, household-size and worker-count dictionaries.
  Numeric intervals include both endpoints; job dictionaries contain source codes.
  Changing these labels does not change the underlying observed characteristics.
- `USE_MMP_WEIGHTS=False` reproduces the unweighted sample mix. Set True to
  sample networks using MMP household weights. Neither option is national.
- `MATCH_ENIGH=False` skips economic-donor matching. With matching on, `ENIGH_YEAR`
  chooses one vintage from the existing ENIGH template file. If it is missing,
  first run `oct-26/analysis/enigh-households-2018-2024.py`.
- `MATCH_WEIGHTS` sets penalties for differences in shared characteristics.
  State remains an exact constraint. All minimum-distance donors are eligible,
  and one is drawn using ENIGH survey weights within that candidate pool.
- `SAVE_PARQUET=False` keeps results in memory without writing outputs.

The default is 10,000 migrants, seed 42, Mexico interviews in 2010–2019 and
ENIGH economic donors from 2018. Existing data and analysis scripts are unchanged.

## What the script generates

The donor pool contains living-status-compatible PERS records on ongoing US
trips, linked to a HOUSE record. Each draw copies a complete observed migrant
network and creates a new synthetic origin household. Repeated draws of the
same observed household are independent synthetic households with new IDs.
There is no independent drawing of age, occupation, destination or relationship.

At the end of sampling, only networks fitting the remaining number of agents
are eligible. This preserves complete networks and reaches the exact target,
but introduces a small boundary bias, potentially large for tiny populations.
The script requires a singleton donor so every positive target can be reached.

The output contains only households linked to at least one enumerated current
US migrant. It is not an all-household Mexican population. Extended relatives
reported only in HOUSE are not generated as if their personal attributes were
known. One synthetic migrant has one observed origin-household link; additional
links across origin households are not identified or invented.

## Matching and interpretation

MMP supplies the origin-household structure and migrant relationships. ENIGH
supplies a plausible economic donor. This is statistical matching, not linkage
of real people across surveys. MMP relationships remain relative to the MMP
household head; they are not assigned to a particular ENIGH roster person.

Matching uses same state, resident-size group, presence of under-12s, presence
of people aged 65+, and employed-member group. MMP residence is a proxy based
on household membership and ongoing migration flags. Employment uses occupation
codes among proxy residents aged 14+. It is not an identical employment survey
measure. Unknown ages/statuses and incomplete rosters leave affected features
unknown; those features are omitted from distance, never counted as exact matches.
`match_features_used` records how many features could be compared.

`match_distance=0` means agreement on the available grouped features, not exact
agreement on raw counts or real family structure. `match_complete_exact` also
requires all configured features to be known. Positive distances explicitly mark
relaxed matches. `match_candidate_count` is the number of tied best donors.
Rural/urban matching is not included until comparable MMP locality information
is harmonized. This limits the economic comparability of the current matches.

All ENIGH columns have the `enigh_` prefix. The MMP structure is not overwritten
by ENIGH demographics. Income/expenditure/remittance values are observed donor
attributes, not simulated outcomes, causal counterfactuals or household budgets
already reconciled to the MMP roster. Original amounts are quarterly nominal
pesos of the ENIGH vintage; `_real` fields are approximately 2024-price quarterly
pesos, following the existing ENIGH template analysis. Current income includes
noncash components. ENIGH remittances cover foreign countries generally.
Remittance receipt and amounts do not enter matching.

Migrant occupation uses current `occ`; `usoccl` remains a separate last-trip
code. US location uses the ongoing trip's recorded state/city. Ages and jobs
remain as reported at interview: the pooled sample is not aged forward to 2019.
Unknown numeric source codes are retained alongside readable categories.

## Saved files

Outputs are in `oct-26/outputs/synthetic-migrants/` and replaced on each saved run.

| File | Contents |
| --- | --- |
| `migrants.parquet` | One row per migrant; personal attributes, synthetic household link, original MMP IDs and optional ENIGH donor ID |
| `households.parquet` | One row per synthetic origin household; MMP structure, network size, matching diagnostics and prefixed ENIGH donor attributes |
| `distribution_checks.csv` | Source versus synthetic shares, including joint age/job/state profiles and household network sizes |
| `run_settings.json` | Run summary and editable category/matching definitions used |

Join the Parquet files on `synthetic_household_id`. `synthetic_migrant_id` is
unique per agent. `mmp_person_id` and `mmp_household_id` identify the source
records and can repeat. `enigh_household_id` identifies an economic donor and
can also repeat. Two synthetic households using one ENIGH donor remain distinct.
IDs restart each run; preserve the seed/settings with each saved population.

```python
import pandas as pd
from pathlib import Path

out = Path('oct-26/outputs/synthetic-migrants')
migrants = pd.read_parquet(out / 'migrants.parquet')
households = pd.read_parquet(out / 'households.parquet')
one_network = migrants.loc[migrants.synthetic_household_id.eq(1)]
```

Each synthetic row counts once. MMP `weight`, ENIGH `factor`, and donor draw
probabilities are provenance metadata; do not apply them again to the synthetic
population. Marginal/joint distributions are matched approximately by resampling,
not exactly. Inspect the diagnostics before using small populations or sparse groups.

Source coding: MMP174 PERS codebook and Appendices A, C and D at
https://mmp.research.brown.edu/mmp/documentation (Appendix A:
https://mmp.research.brown.edu/images/appendix-a-sample-information-mmp174-.pdf).
ENIGH definitions and price treatment follow the existing
`oct-26/analysis/enigh-households-2018-2024-notes.md`.
