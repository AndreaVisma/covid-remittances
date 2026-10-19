# MMP174 household analysis

Run `python oct-26/analysis/mmp-households.py` from the project directory.
The script needs pandas, numpy and matplotlib. It has numbered `# %%` sections,
one summary function, and no classes or command-line framework. Change
`START_YEAR` and `END_YEAR` in section 1 to select another pooled survey sample.
When pasting sections into an interactive console, set ROOT manually to the
project directory instead of using `__file__`.

Outputs are in `oct-26/outputs/mmp-households/`. Re-running replaces these
analysis outputs, not the source DTA files.

## Scope and coding

The downloaded files are MMP174, with household interviews from 1982 to 2019.
The default uses Mexico interviews in 2010-2019: 7,154 households in 46
communities. Receipt is known for 7,153 households; 822 report US remittances.
This is a pre-pandemic structural baseline, not a COVID outcome dataset.

The analysis joins HOUSE to aggregated PERS using country, community, survey
place and household number, and checks interview years and unique keys. PERS
includes children in other households, so its full row count is NOT household
size. HHMEMSHP=2 identifies survey household members, but does not guarantee
physical presence in Mexico. Four selected households have differing HOUSE
and PERS membership totals; their age-composition classification is unknown.

REMUS: 1=yes, 2=no; other values remain missing. The question starts in
community 72, so earlier unasked questions cannot be interpreted as nonreceipt.
REMUSSIZ: 1=small, 2=intermediate, 3=substantial, relative to monthly salary.
This is not a monetary amount or a numerical share. Size percentages condition
on recipients with a known size answer.

USCURTRP=1 identifies people still on their last US migration; 2 identifies
returnees; 8888 is not applicable (never migrated). This proxy does not enumerate
all US-born relatives. Deceased people are excluded from current connections.
Unknown relevant statuses make derived counts unknown, rather than zero.
Children here means PERS RELHEAD=3 (son/daughter), not step/foster children.

HOUSE provides the head's sibling, parent, and extended-family connections.
For parent/sibling current-residence questions, 8888 means no sibling/never
migrated and contributes zero recorded links; 9999 and system missing remain
unknown. Counts from HOUSE and PERS are not added because relatives can overlap.
Network types prioritize head/spouse abroad, then multiple children, then one
child, then other kin. These are potential support structures, not actual
sender-recipient links. A no-recorded-kin classification is not evidence that
all possible foreign connections are absent.

Both unweighted and MMP-weighted results are saved. MMP weights expand sampled
frames, not Mexico nationally. Changes across survey years also change the
communities sampled. The recent-five-year output is a sample sensitivity check,
not evidence of a time trend. Tables are descriptive, without causal adjustment
or statistical-significance claims. Small cells should not become fixed model
parameters.

## Main descriptive findings (default window, unweighted)

Among households without a head/spouse on a US trip, receipt is 53.3% with one
child on a US trip and 68.3% with two or more. Receipt is 72.5% in the head/spouse
US-trip type. These patterns remain similar with MMP weights and in 2015-2019.

Among recipients with known size, substantial transfers are reported by 60.9%
of the head/spouse type, versus 13.7% of the one-child and 22.7% of the
multiple-child types. This suggests distinguishing support for a separated
partner/household from support by children, but is not a comparison of dollars.

Receipt is 27.6% among zero-worker households, versus 10.4% with one worker and
10.6% with two or more. It is 24.1% for heads aged 65+, versus 4.8% under 40.
These associations motivate recipient need and lifecycle heterogeneity, but do
not identify whether remittances cause lower labor supply or compensate for it.

Of 446 households with at least two recorded current US migrants and complete
state information, 76 (17.0%) span multiple US states. That is a conditional
network-dispersion statistic, not a national share or a measure of independent
employment risks. Missing destinations are excluded, not coded as one state.

## Outputs for simulation

- `household_profiles.csv`: one row per selected household, with observed
  attributes, derived connection counts, classifications and quality flags.
- `joint_household_types.csv`: joint frequencies of network type, age
  composition and worker count. Prefer this to independent draws of attributes.
- `by_*.csv`: sample sizes, valid denominators, receipt rates and ordinal size
  summaries; both weighted and unweighted where relevant.
- `recent_network_types.csv`: the most recent five selected years.
- `coverage.csv`: all survey years and places, including unasked receipt.
- `variable_labels.csv`: original DTA descriptions of the selected variables.
- `receipt_comparisons.png` and `REPORT.txt`: compact results.

Use these profiles to construct baseline household types, then impose alternative
US income-shock correlation and contributor-activation assumptions. Do not equate
two children abroad with two earners or two remitters. This analysis does not
estimate pandemic effects, contribution substitution, or remittance inequality.
MIG amounts describe the migrant's last trip and are intentionally not merged as
current household receipt amounts.

## Added compact typology and plots

The original classifications remain available. The new `compact_type` crosses:

1. **Who is in the US:** core only, kin only, core + kin, none recorded.
   Core means the head or sons/daughters (including adult children), exactly
   RELHEAD 1 or 3. Spouses and step/foster children are kin in this new definition.
   Kin combines PERS relatives and HOUSE family connections as presence flags,
   without adding overlapping counts. Unknown connections remain uncertain.
2. **Work at origin:** no worker versus at least one worker, with a separate
   unknown category. This uses current occupations of survey-household members,
   excluding those on recorded ongoing US, Canadian or domestic migrations.
   It is a location proxy; no worker does not mean no income. Incomplete roster
   counts, occupations or migration status can leave this classification unknown.

This gives eight substantive types, plus explicit uncertainty categories.
`employment_type` adds the current occupation profile of recorded core members
on US trips. Broad groups are agriculture; production/construction/transport;
sales/personal services; professional/administrative; not working; and working
student/pensioner. Multiple groups become mixed; any missing occupation becomes
unknown/partial. These are custom occupation bins, not industry, contract type,
formality, or a pandemic exposure ranking. Sales/services includes security.
Extended-kin jobs are not systematically observed, so kin-only households are
not assigned an inferred migrant occupation.

Current PERS `OCC` is used. Last-trip `USOCCL` is exported alongside it only for
audit in `core_abroad_occupations.csv`; it does not fill missing current jobs.
Neither occupation nor a US connection establishes that a person sends money.

New outputs:

- `by_compact_type.csv`, `by_employment_type.csv` and their component summaries:
  joint type frequencies and remittance receipt, including weighted estimates.
- `02_compact_types.png`: prevalence and receipt by compact type.
- `03_receipt_heatmap.png`: receipt by who abroad and origin work; rates hidden
  below 20 valid answers. Underlying cells are in `compact_heatmap_cells.csv`.
- `04_remittance_importance.png`: ordinal remittance size among recipients.
- `05_abroad_occupations.png`: core-person occupations and household receipt.
- `06_composition.png`: receipt by member ages and head age.

In the default sample, receipt for core-only households is 68.6% without an
origin worker and 59.4% with one or more. Corresponding core-plus-kin rates are
67.1% and 63.8%; kin-only rates are 10.6% and 5.9%. This supports separating core
connections from wider kin and keeping origin work as a dependence-related
attribute. It does not establish a causal labor-supply response or a benefit
from adding kin. For microsimulation, draw the joint compact types, then draw
occupation profiles conditionally where sample sizes permit. Preserve the
original head/spouse-versus-child distinction as an optional refinement, since
the compact core definition pools different family support relationships.

## Sources

MMP174, Mesoamerican Migration Project, Brown University and El Colegio de Mexico:
https://mmp.research.brown.edu

Coding verified against the HOUSE and PERS codebooks and weights appendix:

- https://mmp.research.brown.edu/images/house-codebook-mmp174-.pdf
- https://mmp.research.brown.edu/images/pers-codebook-mmp174-.pdf
- https://mmp.research.brown.edu/images/appendix-d-occupations-mmp174-.pdf
- https://mmp.research.brown.edu/images/appendix-h-weights-mmp-and-lamp-weights-2012-.pdf
