# ENIGH 2018-2024: remittances and origin-household templates

Run `python oct-26/analysis/enigh-households-2018-2024.py`.
Dependencies: pandas, numpy and matplotlib. Outputs are in
`oct-26/outputs/enigh-households-2018-2024/`. The earlier two-year script and
outputs are preserved. Four household summary CSVs are the only inputs.

## How to read the results

The primary outcome is receipt within a group: sum of survey weights of
households with `remesas > 0`, divided by sum of weights of all households in
the group. Composition among recipients has a different denominator: all
recipient households nationally. Both are exported, but rate comparisons are
the primary calibration targets. Percentage-point changes are in
`group_receipt_changes.csv`; national changes include adjacent vintages and
2018 to 2024.

The six plots show national rates/amounts, within-group receipt, recipient
composition, conditional amount distributions/dependence, the 16 simple types,
and receipt standardized to the 2018 mix of types. Lines connect independent
cross-sections; they do not trace household transitions or annual trajectories.
The medians and quartiles use the inverse weighted empirical distribution.

## A representation that can be simplified

Key results from the supplied vintages:

| Measure | 2018 | 2020 | 2022 | 2024 |
| --- | ---: | ---: | ---: | ---: |
| Receipt, all households (%) | 4.73 | 4.38 | 4.57 | 3.95 |
| Receipt, rural households (%) | 9.63 | 9.07 | 10.26 | 9.92 |
| Receipt, urban households (%) | 3.27 | 3.09 | 2.87 | 2.41 |
| Receipt with fixed 2018 type shares (%) | 4.73 | 4.38 | 4.50 | 4.10 |
| Mean recipient amount, quarterly approximate 2024 pesos | 11,250 | 11,981 | 15,975 | 15,149 |

Rural receipt is roughly maintained over the whole window, whereas urban
receipt falls in every observed vintage. These patterns are useful separate
calibration targets, rather than requiring the model to reproduce only the
national average. The national decline from 2018 to 2024 is about 0.78
percentage points: 0.63 points remain with the 2018 type mix fixed and 0.15
points are the composition remainder. These are descriptive decompositions,
not causal contributions. Seven of the 64 simple year-type cells are flagged
as sparse. Detailed tables have 205/216/204/209 observed types by vintage and
are primarily useful for pooling and template design, not fitting a separate
parameter for each cell.

`origin_household_templates.csv` has one row per original survey household per
year. `baseline_2018_templates.csv` contains only the pre-pandemic vintage.
These are weighted empirical donor populations, not newly generated families.
Sampling whole rows preserves their observed joint attributes. Household IDs
include the year and must not be used to link households longitudinally.

The file keeps original source variables plus clearly named derived fields:

| Fields | Meaning/use |
| --- | --- |
| `year`, `household_id`, `folioviv`, `foliohog` | Vintage and unique record identity; keep IDs as strings |
| `factor`, `draw_probability` | Survey expansion weight and sampling probability normalized within year |
| `state`, `ubica_geo`, `tam_loc`, `locality` | State/municipality identifiers and locality size; retain geography for later |
| `est_dis`, `upm` | Survey strata and primary sampling units; use year-specific design identifiers |
| `tot_integ`, `hombres`, `mujeres` | Household member counts |
| `menores`, `p12_64`, `p65mas` | Counts aged under 12, 12-64, 65+; these sum to household size |
| `ocupados`, `percep_ing`, `perc_ocupa` | Employed people aged 14+; income recipients; employed income recipients |
| `clase_hog`, `family_class` | Official one-person, nuclear, extended, composite or nonrelated-co-resident class |
| `edad_jefe`, `sexo_jefe`, `educa_jefe` | Head characteristics, retained as supplementary survey attributes |
| `head_age`, `head_sex`, `size_group`, `employment` | Coarsened versions for descriptive comparisons |
| `age_composition`, `origin_work` | Age-presence pattern and whether anyone is employed |
| `simple_type`, `detailed_type` | Readable joint classifications; separate component columns are also retained |
| `remesas`, `recipient`, `dependence_pct` | Original quarterly remittances, positive receipt, remittances/current income (%) |
| `ing_cor`, `ingtrab`, `jubilacion`, `bene_gob`, `estim_alqu`, `gasto_mon` | Original nominal quarterly income/spending components |
| `remittance_real`, `income_real`, `labor_real`, `pension_real`, `government_real`, `imputed_rent_real`, `monetary_spending_real` | Corresponding approximately 2024-price amounts |
| `nonremittance_income_real` | Current income minus remittances, suitable as a baseline resource attribute |
| `other_nonremittance_income_real` | Residual after labor, pension, government benefits and imputed rent; not necessarily cash |
| `nonremittance_per_capita_real` | Non-remittance current income divided by member count |

The **16 simple types** cross rural/urban locality, four age-composition groups,
and no employed/at least one employed member. The **detailed types** subdivide
these by 0/1/2+ employed people, size (1/2/3-4/5+) and official family class.
Only observed combinations are exported; structurally impossible combinations
are not invented. `type_mapping.csv` maps detailed types to simple ones.
Neither receipt nor transfer amount defines a household type, avoiding circular
definitions when calibrating behaviour.

Headship does not determine the simple types because MMP and ENIGH use different
head-selection conventions. Age presence and household class still do not reveal
exact kinship: two adults and a child cannot automatically be called a couple
with their child. POBLACION would be needed for relationship/individual-age detail.
There is no information here on migrants' identities, earnings or potential kin
abroad. Absence of migrant variables must never be encoded as zero migrant links.

## Using the templates in a model

First choose one vintage: use 2018 for a pre-pandemic baseline; use the others
as separate calibration/validation populations. Do not concatenate the four
years as though they were one population, and do not multiply donor weights
again after sampling with probabilities proportional to those weights.

For example, an optional 10,000-household synthetic population can be generated
from the baseline file without independently drawing household attributes:

```python
base = pd.read_csv('baseline_2018_templates.csv',
                  dtype={'folioviv': str, 'foliohog': str, 'household_id': str})
synthetic = base.sample(n=10000, replace=True, weights='draw_probability', random_state=42).copy()
synthetic = synthetic.rename(columns={'household_id': 'donor_household_id'})
synthetic['synthetic_household_id'] = range(len(synthetic))
synthetic['simulation_weight'] = base.factor.sum() / len(synthetic)
```

This optional sampling introduces Monte Carlo error. Keeping all weighted donors
avoids that error. Draw migrant networks conditional on origin characteristics
in a later step; keep the original observed recipient status and remittances as
validation variables, separate from simulated outcomes. Use non-remittance
resources when determining need. Spending is observed, partly enabled by
remittances, and is not an exogenous needs budget or a poverty threshold. These
files do not measure liquid savings or identify a causal income-without-migration
counterfactual.

## Aggregation without losing the denominators

`simple_type_targets.csv` and `detailed_type_targets.csv` contain additive sample
counts, weights, recipient weights and weighted sums of resources/composition.
They also include convenient derived rates and means. To collapse categories:

1. Sum `factor`, `recipient_weight`, sample counts and the relevant weighted sums.
2. Recompute receipt as `100 * recipient_weight / factor`.
3. Recompute conditional mean remittances as
   `remittance_real_weighted_sum / recipient_weight`; leave undefined if zero.
4. Recompute mean non-remittance income as
   `nonremittance_income_real_weighted_sum / factor`.

Do not average subgroup percentages or medians. Quantiles must be recalculated
from the household templates. `weight_effective_n = factor**2 / weight_squared`
uses sums within a cell and measures unequal-weight concentration only, not
survey-design precision. The sparse flag marks cells with fewer than 100 sampled
households OR 30 sampled recipients. It is a pooling warning, not a confidence
interval or evidence that an unflagged cell is statistically reliable.

The grouping lists `SIMPLE` and `DETAILED` are at the top of the script. For
example, removing locality from both produces an 8-type age/employment scheme.
Derived results and standardization are recomputed from the underlying records.

## Fixed-composition comparison

For each year t the standardized rate is
`sum_g(2018 household share_g * receipt_rate_g,t)` over the simple types.
All baseline types must be observed in every vintage; the script fails rather
than silently renormalizing missing types. The difference from the baseline
standardized rate is the within-type change component. Observed minus
standardized is the composition remainder (using current-year rates).
Together these exactly reconcile to the observed change from 2018.

This holds only the distribution of selected types fixed. Changes in income,
age, network prevalence or other attributes within a type can still contribute.
It is not a causal estimate or a literal panel with fixed households. Employment
is itself potentially affected by the pandemic and transfers; standardizing it
can remove a substantively relevant pathway. Compare alternative groupings when
using this exercise to assess mechanisms.

## Prices, comparability and validation

All monetary quantities are quarterly, not annual. Remittances cover foreign
origins generally (P041), not the US alone. Monetary results are approximately
at 2024 prices: the multiplier is INEGI's published 2024-price national mean
current income divided by the original-vintage mean. Published means are rounded
(2018: 67,319; 2020: 63,400; 2022: 70,391). The 2024 multiplier is 1. This follows
the previous script's approximation, not a reconstruction of month-specific
deflation. Original nominal fields and multipliers are retained. If revising
survey weights or file versions, recheck these benchmarks before interpreting
the ratios as price adjustments.

Dependence uses total current income, including imputed rent and noncash income;
it is not a cash-income share. No employed person does not mean no income.
The no-children category means no under-12 members, not no under-18s. The age
12-64 group is not an employed or legally adult population.

Checks cover unique keys, missing values, valid categories, positive weights,
age/sex-count reconciliation, national weights and income benchmarks, exact
recovery of simple-cell totals from detailed cells, and national rate recovery.
No survey-design standard errors or significance tests are supplied. State
tables are retained for future work, without state plots. The 2024 sampling
frame changed; preserve each vintage's weights and design variables.

Sources:

- 2018 definitions: https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/702825188061.pdf
- 2020 definitions: https://www.inegi.org.mx/contenidos/programas/enigh/nc/2020/doc/enigh2020_ns_descriptor_archivos_fd.pdf
- 2022 dictionary: https://www.inegi.org.mx/rnm/index.php/catalog/901/data-dictionary/F64
- 2024 definitions: https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/microdatos/889463924494.pdf
- Common-price comparison: https://www.inegi.org.mx/contenidos/programas/enigh/nc/2024/doc/enigh2024_ns_presentacion_resultados.pdf
