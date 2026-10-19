# Current implementation and saved results — audited 16 October 2026

This document describes `remittance-minimal-cps-abm.py` and the saved outputs in
`oct-26/outputs/remittance-minimal-cps-abm/`. It replaces earlier gross-earnings-budget
results. The audit reconciled the source, `run_settings.json`, validation CSVs,
and exported panels; it did not rerun calibration or change model behaviour.

For the research questions, literature assessment, and next experiments, see
[research questions and literature review](../../research-questions-literature-review.md).
The [three-month plan](../../research-plan-three-months.md) is an earlier intended
scope, not a description of modules already implemented.

## Current earnings budget

The current default is Mexican-origin CPS inputs. Transfer capacity is
`0.20 * max(individual_gross_earnings - 20000/12, 0)`.
The annual allowance is **$20,000**, or **$1,666.67 per month**, fixed in nominal
dollars during 2019–2022. This is a configured scenario assumption, not an
estimated historical living-cost requirement. The code/report retain a reference
to a user-supplied MIT table, but that label does not establish a documented MIT
source for the current $20,000 value. Geography, price year, and the derivation
of the current amount remain unresolved. The earlier $52,909/$4,409.08 figures
are not the current setting.

The 0.20 share applies to **surplus**, not gross earnings. Neither that share nor
the living allowance is fitted. No claim about an approximately 20% gross-income
remittance norm is used as an empirical calibration target here.

Persistent individual earnings multipliers follow a lognormal assumption inferred
from baseline occupational mean/median ratios. Within eligible occupations they
are normalized to mean one. Historical occupational means scale the same draws;
individual ranks and dispersion are held fixed. Set `EARNINGS_HETEROGENEITY=False`
to recover occupational mean assignment. These are assumed individual profiles,
not observed earnings histories.

Low-paid employed migrants can have no surplus and no capacity. Panels export
gross earnings and budgets; summaries distinguish employed agents with no surplus
and report remittance/gross-earnings ratios. There are no savings, benefits,
borrowing, or nonlabour resources.

# Minimal ABM with prepared CPS earnings and unemployment

Run from the project root:

```powershell
python oct-26/simulations/abms/remittance-minimal-cps-abm.py
python oct-26/simulations/abms/check-remittance-minimal-cps-abm.py
```

Dependencies: numpy, pandas, scipy, matplotlib, openpyxl and pyarrow. Paths and settings are at the top of the script. The existing ABM and synthetic population files are unchanged. **The only CPS input is `oct-26/data-prep/outputs/cps_earnings_employment.xlsx`. The script never loads the raw CPS extract or runs the CPS preparation script.**

## Exact behavioural specification

For migrant link j connected to household i, define a shared need index:

```
L_i = beta_worker * household_has_no_worker
    + beta_child * child_share
    + beta_old * old_share

theta_jit = alpha_core_or_kin + L_i + rho * sent_last_month
P(willing_jit) = logistic(theta_jit)
N_i = exp(log_need_scale_usd + L_i)
```

There are **seven fitted baseline coefficients**: two relationship intercepts, three household need coefficients, persistence, and the logarithm of the monetary scale. No need, persistence, occupation, capacity or network-size coefficient is fixed in theta. Budget and other migrants' contributions do not enter willingness. The old model's network dilution, residual-need penalty, fixed demographic coefficients and core/kin contribution fractions are removed.

Child share uses the MMP resident proxy's under-12 population; old share uses 65+. Both divide by the resident proxy count. Incomplete or internally inconsistent demographics receive explicit simulation fallbacks and are excluded from calibration. Source and synthetic eligibility rules are aligned. No actual household receipt flag determines the simulated sender state.

The same need slopes appear in willingness and perceived monetary support. This is a deliberate, strong restriction: one Banxico mean cannot separately identify a whole second set of amount slopes. Negative need associations are permitted by editable bounds. Consequently the coefficients describe calibrated associations, not causal effects or a validated poverty/essential-consumption budget. In particular, an estimated negative old-share coefficient does not establish that older people need less support.

All seven coefficients are calibrated, but the model still has structural assumptions: logistic/exponential functional forms, shared need slopes, budget ceilings, employment sampling, ordering and the meaning of a transaction. Search bounds, seeds, tolerances and objective weights are numerical choices, not empirical estimates of those assumptions.

## The two mechanisms

**Uncoordinated:** each willing migrant sends `min(current_budget, N_i)`. Contributions by other migrants are ignored. Several positive transfers can exceed the household's perceived need.

**Coordinated:** previous senders act first, with random order within incumbent/nonincumbent groups. Each willing migrant sends the minimum of their budget and remaining household need. A willing migrant whose household is already covered makes no transfer. Total receipts cannot exceed perceived need.

`sent_last_month` means an actual positive transfer. Willingness with zero capacity or full crowding out does not make someone an incumbent next month. Each link has at most one positive transfer per month. The script evaluates each mechanism with its own calibration, and also switches allocation rules while holding fitted parameters and random numbers fixed. The latter comparison isolates the allocation rule from recalibration.

## CPS capacity and employment

The default CPS population is **Mexican-origin adults age 18+**. The prepared workbook also contains Latino results. Mexican origin does not identify Mexican birth, migrant status, or legal status. Earnings are conditional gross monthly-equivalent means, and unemployment rates are the prepared occupational stock estimates.

- Baseline: fixed pooled 2019 earnings and unemployment by occupation.
- History: observed monthly values for January 2020 through December 2022, following 36 baseline burn-in months.
- Counterfactual: the same dates and random numbers with constant 2019 resources.
- No raw CPS microdata are read, and no survey weights are applied again to the aggregate workbook results.

An eligible agent is employed in a model month if a uniform draw exceeds the occupational unemployment rate. Thus rates are reproduced in expectation, with finite-agent sampling variation. These are **independent monthly employment-status draws, not unemployment rates misused as job-loss hazards**. The model does not estimate joblessness durations or employment transitions. It retains sending persistence separately. Realistic employment spells are a future extension needing transition estimates.

An employed agent's gross earnings are the occupational mean times their fixed individual earnings multiplier. The ceiling is 20% of earnings above the fixed $20,000 annual allowance. An unemployed agent has zero labour capacity. The allowance and share are assumptions; there is no individual tax model, savings, benefits, or nonlabour income. Banxico's mean does not identify these capacity assumptions.

Synthetic `Not working` links have no labour capacity and do not enter occupational employment draws. Unknown occupations use the **explicitly configured overall CPS proxy**, including its overall unemployment rate; this does not use the diagnostic Unknown-occupation unemployment rate. The same proxy is configured for working student/pensioner if present. Proxy flags are saved on migrant records, and `JOB_PROXY` can instead map these categories to None. Minors and unknown ages cannot transfer in this version.

The historical exercise uses observed mean earnings, preserving the prepared quality flags. It does not silently substitute missing composition-adjusted agriculture estimates or multiply every employed agent's pay by (1 − unemployment), which would double-count employment loss. The prepared CPS earnings can retain workforce-selection effects. Every historical job/month input and its quality flag is saved.

## Calibration

MMP targets are unweighted receipt rates from unique original connected-household records with known receipt, complete recorded network fields and usable resident demographics. They use three overlapping panels: network type crossed with no-worker status, presence of children, and presence of older residents. Each panel gets equal total weight, and cells within a panel are weighted by original sample size. Source and synthetic minimum cell counts are configurable. These 18 moments are not 18 independent samples.

Receipt fit uses the mean exact probability of any receipt conditional on the simulated preceding sender state and current employment. Along the path where everyone declines, the household is still uncovered, so `q = 1 − product(1 − p_j)` over agents with positive capacity. This is valid in both mechanisms and does not make final coordinated transfers independent. States and positive transfers are still simulated agent by agent over months.

The amount target is **$326.0555828675 per transaction**, the ratio of total Banxico dollars to total transactions across 2019. It is compared with total simulated dollars divided by positive link-month transfers after calibration burn-in. It is not compared with mean household receipts. Banxico covers Mexican inbound remittances broadly, while this population is MMP-like US-connected networks; using its mean as an external scale requires that approximation. The MMP `REMUSSIZ` categories are not interpreted as dollar measurements.

Calibration fits the seven coefficients jointly using simulated moments and common random numbers. Two starting persistence values reduce dependence on initialization. Finite differences have explicit absolute steps because discrete sender histories make the objective locally rough. The best evaluated parameter vector is retained; numerical optimizer success is reported separately from fit diagnostics. There is no prior/ridge penalty pulling behavioural coefficients toward the initial values.

Persistence sensitivity fixes rho at 0, 1.5 and 3 and refits every other coefficient. These finite-budget numerical profiles diagnose weak identification; they are not likelihood profiles, confidence intervals or proof of global optimization. Population receipt cross-sections do not directly observe monthly persistence. Different validation seeds check simulation reproducibility, not held-out survey prediction.

## Current saved results and what drives them

The population has 10,000 migrant-household links and 1,019 connected households.
Baseline validation runs 120 months, discards the first 36, and uses three seeds.
The fit uses a shorter bank and a separate seed. All 18 receipt cells are used;
the three overlapping panels are not independent samples.

| Fitted coefficient | Uncoordinated | Coordinated |
|---|---:|---:|
| core_intercept | 0.3896 | 0.7297 |
| kin_intercept | -4.5673 | -4.5753 |
| no_worker | 0.8310 | 0.8724 |
| child_share | -0.6122 | -0.5888 |
| old_share | -0.5859 | -0.5144 |
| sent_last_month | 0.7936 | 0.1785 |
| Need scale, USD | 1168.57 | 1265.93 |

Both optimizers report success, no retained coefficient is near its search bound,
and both mechanisms pass the declared practical stability checks on all seeds.
Changing initial previous-sender flags from empty to 50% produces a zero tested
post-burn-in receipt gap. These are numerical diagnostics, not external validation.

**Receipt fit remains incomplete.** Maximum absolute receipt errors are 9.11 and
9.00 percentage points, exceeding the declared 8-point tolerance. For kin-only
households with older residents, MMP receipt is 13.50%, versus roughly 4.4–4.6%
simulated. Both amount fits pass the 5% tolerance; maximum seed errors are 2.04%
and 2.22%. Positive-transfer means across seeds are $329.10–$333.30, compared
with the $326.06 target.

Averages of post-burn-in monthly summaries across three seeds:

| Diagnostic | Uncoordinated | Coordinated |
|---|---:|---:|
| Household receipt | 23.14% | 23.14% |
| Positive migrant-link transfers per month | 321.67 | 315.35 |
| Total simulated USD per month | 106,651.07 | 104,614.93 |
| Budget-bound positive transfers | 94.62% | 94.36% |
| Unmet perceived need / total perceived need | 91.17% | 91.88% |

Budget-binding shares are pooled budget-bound counts divided by pooled positive
transfer counts. Unmet perceived need includes nonreceiving connected households;
it is not a poverty rate or a measured consumption gap. The fitted need scale is
an intercept, not the mean household need. **Capacity constraints bind frequently**
under the current surplus budget; the earlier conclusion that budgets never bind
applied to a superseded gross-earnings specification.

Network-size receipt diagnostics were not fitted. Core-only singleton receipt is
54.39% in MMP versus 42.15–42.40% simulated; core-plus-kin size-two receipt is
65.31% versus 49.32–49.89%. These discrepancies deserve attention before using
size-specific protection estimates. Sparse cells retain explicit flags.

Persistence profiles at rho = 0, 1.5, 3 produce similar losses (approximately
0.000836–0.000876). Some fixed-rho fits improve on the selected joint fit, so
optimizer success must not be read as a global optimum or strong identification.
Carry plausible baseline fits into the pandemic exercise instead of treating
one persistence estimate as established.

Coordination and activation are distinct. Another migrant's job loss does not
enter willingness directly. Replacement can occur because another willing link
has capacity and household need remains uncovered. The present model has no
explicit shortfall-triggered willingness response and no fixed-incumbent-only
benchmark. Its recorded replacement count alone does not identify deliberate
compensation for a lost job. Comparisons that refit both mechanisms differ from
same-parameter allocation switches, which the script also exports.

## COVID need extension

By default the historical run changes **US capacity only**. It is not calibrated to explain the pandemic growth in national remittances. A constant-resource counterfactual uses identical random numbers for comparison.

To add heightened origin need, provide a CSV with `date,covid_need_index`, set `COVID_NEED_FILE`, and explicitly supply `COVID_NEED_COEFFICIENT`. The model then adds `coefficient * index` to the common need index in both theta and log perceived support. The baseline shock is zero. This preserves one unified origin-need channel while keeping US unemployment and earnings out of theta.

The COVID coefficient is intentionally not estimated from 2019 data, where the index is zero. A future pandemic calibration must specify an observed Mexican need measure and empirical targets before estimating it. The script rejects a supplied need series without a coefficient instead of silently treating a guessed value as calibrated. National Banxico growth plots are contextual diagnostics; they do not imply national representativeness or fit. Exchange rates, US income support, saving and demographic/network changes are absent.

## Outputs and diagnostic charts

All generated files are in `oct-26/outputs/remittance-minimal-cps-abm/`.

- `run_settings.json`: exact coefficients, numerical settings, CPS-workbook hash, proxies, fit flags and limitations.
- `mmp_targets.csv`, `receipt_validation.csv`, `calibration_history.csv`, `persistence_profile.csv`: target definitions, residuals, optimization and sensitivity.
- `monthly_summary.csv/.json`, `stability_checks.csv`, `mechanism_comparison.csv`: all baseline seeds, convergence and allocation comparisons.
- `cps_baseline_used.csv`, `cps_monthly_inputs.csv`, `cps_history.csv`: prepared inputs actually used and historical/counterfactual trajectories.
- `household_baseline.parquet`, `migrant_baseline.parquet`: original attributes plus demographic, eligibility and CPS-proxy fields.
- `baseline_*_household_months.parquet`, `baseline_*_migrant_months.parquet`: first-seed baseline histories including zeros, for both mechanisms.
- `history_*_household_months.parquet`, `history_*_migrant_months.parquet`: dated 2020–2022 observed-capacity paths, excluding warm-up.
- `network_size_diagnostics.csv`: additional receipt moments by network size, including sparse-cell flags.

Fourteen PNG figures are generated:

1. Baseline monthly dynamics and mean-transfer target.
2. Receipt fit across all need panels.
3. Transfer distributions and household receipts relative to perceived need.
4. Sender entry, persistence, crowding out and lack of capacity.
5. Persistence profile, with a star for the joint fit.
6. All fitted theta coefficients and calibration search.
7. CPS earnings, unemployment, simulated employment and input quality.
8. Observed-capacity history versus constant-2019 counterfactual.
9. Initialization sensitivity and same-parameter allocation comparisons.
10. Individual household paths and fitted need levels.
11. Need-response curves, showing exactly how each demographic affects desire and amounts.
12. Budget binding, earnings-to-need ratios and willingness without capacity.
13. Receipt by network size, which is not included in the fitting objective.
14. Banxico growth indices against capacity-only simulations, normalized to each series' 2019 baseline.

Rates and amounts in the baseline summaries refer to this synthetic sample. Migrants are household links: cross-household duplicates of a real relative are not identifiable, so no global shared budget across different households is enforced. Most kin characteristics are imputed in the source synthetic population. Neither seed ranges nor pooled monthly averages resolve that uncertainty.

## Verification

Focused tests verify overcoverage without coordination, capped household coverage with coordination, partial budgets, incumbent replacement after capacity loss, willingness despite unemployment, the analytical two-person receipt probability, and seeded reproducibility. Export checks reconcile migrant transfers to household receipts and monthly totals, and ensure every positive transfer has employment capacity. The raw CPS extract is not part of these checks.
