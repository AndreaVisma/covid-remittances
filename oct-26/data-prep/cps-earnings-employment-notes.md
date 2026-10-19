# CPS earnings and employment inputs

Run from the repository root with Python and `numpy`, `pandas`, `openpyxl` installed:

```powershell
python oct-26/data-prep/cps-earnings-employment.py
```

The script reads `data/cps_00001.dat` and the supplied `cps_00001_codebook.txt` beside the script. It writes `outputs/cps_earnings_employment.xlsx`, equivalent summary CSV files and `cps_run_settings.json`. The source microdata and existing ABM are unchanged. The workbook contains materialized analysis results, so it can be read directly with `pandas.read_excel` without Excel recalculation. Edit the Python settings and rerun to change definitions.

## Population and period

The default analysis uses Basic Monthly observations from 2010 through the latest available month in 2026, with 2019 as the calibration baseline. The extract ends in September 2026 and lacks October 2025. Missing survey months are not created or assigned zero earnings/unemployment. The script verifies that all twelve baseline months are present.

The analysis covers Hispanic/Latino adults aged 18 or older of any race. Age 18 aligns with the ABM's adult sending eligibility; these are not the standard BLS age-16-plus estimates. `HISPAN` codes 901 and 902 (unknown/nonresponse) are not classified as Latino. Mexican-origin results use codes 100, 102, 103, 104, 108 and 109, and are an overlapping subset. Do not add the two populations. Mexican origin does not establish Mexican birth, migration status, citizenship or undocumented status; those fields are absent.

All rows are person-observations at a survey month. Repeated respondents in different months are appropriate for cross-sectional time series and pooled average-month estimates; the counts are not unique people. `CPSIDV` remains an exact integer when linking records.

## Earnings

Earnings statistics use employed wage/salary workers in outgoing rotations (MISH 4 and 8), with positive `EARNWT` and valid `EARNWEEK2`. Part-time workers and workers temporarily absent from their jobs are retained. Self-employment, unpaid family work and military worker classes are excluded from this earnings universe.

The codebook gives fixed-width field positions and implied decimal places. The parser applies those scales once: earnings /100; person and earnings weights /10000. The NIU value 999999.99 is missing, not an amount and not zero. Valid reported zero earnings are retained.

`monthly_mean_usd = weighted_mean(EARNWEEK2, EARNWT) * 52/12`. The weighted empirical 25th percentile, median and 75th percentile are converted the same way. A percentile is the smallest observed value at which cumulative weight reaches its target. There is no interpolation or smoothing of the earnings distribution.

Amounts are nominal US dollars of the observation period, before deductions, at the respondent's current/main job. They are usual monthly-equivalent earnings, not measured calendar-month receipts, after-tax income or disposable remittance budgets. Earnings while temporarily absent may describe usual pay without guaranteeing pay was received that month. No inflation adjustment, tax deduction, benefits imputation or invented outlier trimming is applied.

Public-use earnings topcodes are retained. The diagnostic `fixed_topcode_share` measures the earnings-weighted share at or above 2884.61 weekly through March 2023. It is blank afterwards because phased/dynamic topcoding needs different treatment. Large later dynamic topcodes are not deleted using an arbitrary historical ceiling. Means are not corrected for topcoding. Missingness/allocation flags beyond those in the extract cannot be recovered; no claim is made to exclude all allocated earnings.

## 2019 baseline

The baseline pools all twelve 2019 months and calculates earnings with `EARNWT`. This represents an average worker-month, weighting months according to their estimated worker populations, rather than giving each monthly mean identical weight. Baseline means and quantiles are computed directly on the pooled microdata. Weight totals are divided by 12 to represent average monthly populations. Dividing all weights by 12 does not change means, rates, quantiles or Kish effective sample sizes.

Unemployment uses `WTFINL`, independently of earnings eligibility. Earnings and labour-force estimates therefore have different universes. Self-employed workers remain part of employment and unemployment calculations even though their current earnings are outside this wage/salary analysis.

## Occupational crosswalk

The script uses harmonized `OCC2010`, not raw occupation codes or `IND`. The workbook exports every codebook occupation label with its assigned group. The editable dictionaries define:

- Agriculture: farming, fishing and forestry occupations.
- Production/construction/transport: construction/extraction, installation/maintenance/repair, production, transportation and material moving.
- Sales/personal services: sales plus service occupations, including health-care support, protective service, food preparation, cleaning and personal care.
- Professional/administrative: management, business, professional and technical occupations, plus office/administrative support.
- Unknown: unassigned and NIU codes, kept visible rather than silently mapped to an earnings proxy.

These broad groups approximate the existing MMP categories. They are not a claim that the detailed CPS and MMP classifications coincide perfectly. The code-level crosswalk allows review and revision.

“Not working” is a current status, not an occupation. It should be handled by the ABM's employment/capacity rule, not an average wage inferred from missing earnings. “Working student/pensioner” is not separately recoverable without enrolment/pension information. Unknown earnings are not automatically assigned the overall mean. These decisions remain explicit for the next ABM version.

## Employment and unemployment

`unemployment_rate = sum(WTFINL for unemployed) / sum(WTFINL for employed or unemployed)`.

Employment is EMPSTAT 10/12; unemployment is 20/21/22. Civilian nonparticipation is 30–36 as listed in the codebook. Occupation refers to the current job of employed people and most recently reported occupation of unemployed people. New entrants and other people without assignable jobs contribute to the Unknown group and the overall total.

Overall employment/population and participation rates use all adult civilians. These ratios are deliberately blank for occupational groups because past occupation is not observed consistently among nonparticipants. The occupational unemployment rates should not be interpreted as unemployment risks of a fixed 2019 occupational cohort. People can change occupations, and a rate is a stock rather than a monthly job-loss transition probability.

The monthly tables are not seasonally adjusted. A 2019 annual baseline is useful for calibration, but same-calendar-month comparisons can be useful when studying seasonal agriculture. The script does not silently smooth sparse estimates.

The Unknown group has no employed observations in the default baseline and its unemployment rate is 100% by construction among those in its labour force. This is a classification diagnostic, not a usable job-loss input. `baseline_2019.use_for_abm` requires a known occupational group and both earnings and unemployment reliability flags; aggregate and Unknown rows are excluded by that flag.

Early-pandemic employment-status misclassification is retained as reported. Reclassification would require more information and assumptions; the extract does not include detailed absence reasons/hours. COVID job loss that appears as labour-force exit is not unemployment, which is why overall nonparticipation measures are retained.

## Workforce-composition adjustment

Raw earnings can rise when low-paid workers lose employment. Alongside the observed mean, the script directly standardizes to 2019 joint shares of age (18–34, 35–54, 55+), sex and education (high school or less versus some college or more), separately within each population and occupational group.

For cell c with 2019 weight share s and monthly earnings mean m, the adjusted mean is `sum(s_c * m_ct) / sum(s_c)` over supported cells. A monthly cell needs at least three earnings observations. At least 90% of the baseline cell mass must be supported; otherwise the adjusted estimate is blank. Supported shares are renormalized and coverage is exported. Baseline records with unknown cell characteristics are excluded from this reference and their count is documented.

This is partial adjustment, not the counterfactual pay of workers who became unemployed. It does not hold detailed occupation or unobserved skill constant, remove all selection, or account for changes in hours within cells. Medians alone would not solve those problems. Changing the cell definitions or coverage threshold should be treated as a sensitivity choice.

## Unemployment by previously observed earnings

`linked_earnings_risk` is exploratory. It links valid 2019 MISH4 earners to MISH8 in the same calendar month of 2020 using `CPSIDV`, retaining baseline occupation and earnings quartile. Cutoffs are the population-specific, pooled 2019 earnings quartiles across all occupations; ties at a boundary enter the lower category, so shares need not be exactly 25%.

The table reports linkage counts/coverage, unemployment among matched labour-force participants, and nonemployment among matched civilians. It uses origin `EARNWT` only. There are no longitudinal/attrition weights in this extract, so these are weighted matched-sample descriptions, not representative monthly unemployment rates by former income. All rows carry `use_for_abm=False`. The 2019–2020 links cannot supply a complete panel through the pandemic recovery. The file retains only Latino adults, so changes in ethnicity reporting also affect linkage coverage.

## Quality flags and uncertainty

Earnings reliability requires at least 30 observations and Kish effective n at least 20. Occupational unemployment reliability requires at least 100 labour-force observations and five unemployed observations. Raw estimates remain available alongside flags; insufficient evidence is not disguised as a zero or automatically filled. Kish n measures unequal-weight concentration and is not a survey-design-based sample size. No confidence intervals or formal significance claims are produced. Matched risk has its own sample flag, but a sufficient count does not overcome its selection/weighting limitation.

The script checks occupation totals against overall employment/unemployment, weekly-to-monthly conversion, unique output keys, valid rates, a baseline mean recomputed from persons, and the workbook after saving. The source file's SHA256 and run settings are saved for reproducibility.

## Findings from the supplied extract

The completed run reads 23,579,903 source rows and retains 2,319,395 Latino adult observations over 200 survey months. All four occupational groups in both populations meet the baseline reliability thresholds.

| 2019 occupational group, all Latinos age 18+ | Monthly-equivalent gross mean USD | Unemployment rate |
|---|---:|---:|
| Agriculture | 2,532.87 | 12.81% |
| Production/construction/transport | 3,404.57 | 3.76% |
| Sales/personal services | 2,589.19 | 4.09% |
| Professional/administrative | 4,416.36 | 2.87% |

For 2019–2022, the composition adjustment is available in all 48 months for each of the three larger occupational groups, in both populations. Agriculture meets the adjustment coverage requirement in only one of the 48 months in each population. The observed agriculture mean remains available separately; a quarterly or coarser-cell adjustment would be a distinct future sensitivity analysis.

Among eligible 2019 MISH4 Latino earners, 6,916 of 10,866 link to the relevant 2020 observation. For Mexican-origin earners, 4,250 of 6,576 link. Only 37 of 378 population/month/occupation/quartile cells meet the linked-table count thresholds. The exploratory table should not drive the baseline ABM. Later monthly topcodes can legitimately exceed $10,000 weekly; the script retains documented in-universe values rather than deleting them.

The focused fixture checks can be rerun with `python oct-26/data-prep/check-cps-earnings-employment.py`. They exercise decimal scaling, exact identifiers, NIU versus zero earnings, age/ethnicity/ASEC filtering, weighted means/quantiles, unemployment denominators, missing composition cells and occupation mapping.

## Reading the workbook in the future ABM

```python
from pathlib import Path
import pandas as pd

inputs = Path('C:/git-projects/covid-remittances/oct-26/data-prep/outputs/cps_earnings_employment.xlsx')
baseline = pd.read_excel(inputs, sheet_name='baseline_2019')
baseline = baseline.query("population == 'Latino'")  # Or 'Mexican origin'.
earnings = pd.read_excel(inputs, sheet_name='monthly_earnings')
employment = pd.read_excel(inputs, sheet_name='monthly_employment')
```

Use `population`, `year`, `month`, `job_group` as monthly keys. Baseline keys omit month. “All occupations” is an aggregate, not another job group. Check quality flags before selecting estimates. Do not turn blank adjusted estimates into zeros. If unemployment is simulated at agent level, do not also multiply employed earnings by (1 − unemployment rate).

## Sources

- [IPUMS CPS citation instructions](https://cps.ipums.org/cps/citation.shtml) and the supplied extract codebook (2026-10-13).
- [EARNWEEK2](https://cps.ipums.org/cps-action/variables/EARNWEEK2) and [outgoing rotation weights](https://cps.ipums.org/cps/outgoing_rotation_notes.shtml).
- [WTFINL](https://cps.ipums.org/cps-action/variables/WTFINL), [EMPSTAT](https://cps.ipums.org/cps-action/variables/EMPSTAT), [OCC2010](https://cps.ipums.org/cps-action/variables/OCC2010).
- [Validated linking identifier](https://cps.ipums.org/cps-action/variables/CPSIDV).
- [BLS explanation of COVID misclassification](https://www.bls.gov/blog/2020/update-on-the-misclassification-that-affected-the-unemployment-rate.htm).
