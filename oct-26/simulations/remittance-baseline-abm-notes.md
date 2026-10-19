# Monthly remittance baseline ABM

Run from the project root:

```powershell
python oct-26/simulations/remittance-baseline-abm.py
python oct-26/simulations/check-remittance-baseline-abm.py
```

The script reads the existing `synthetic-migrants-house-networks` Parquet files and original MMP household targets. It does not regenerate or change that population. The default run uses 10,000 migrant–household links and 1,019 households, 120 model months, a 36-month burn-in and three validation seeds. Change `ROOT`, paths, run length and behavioural dictionaries at the top of the script. To change population size, first regenerate the synthetic population; this ABM preserves its complete household networks.

## What an agent does each month

Each household has a recurring support target N. Each adult migrant has a sending budget C determined by an editable job dictionary. These dollar values are scenario assumptions, not measured income or calibrated remittances. Minors and migrants with unknown age cannot send. The household target uses resident size, children under 12, older residents and whether there are workers. It is not a poverty line or an estimate of essential consumption. Matched ENIGH income remains in the input attributes but is not used as a monetary target.

Previous-month senders act first. Within incumbent and other groups, order is randomized each month. Each migrant decides once, observing remaining support R after earlier contributions:

```
theta = core_or_kin_intercept
      + no_worker_coefficient * household_has_no_worker
      + 0.3 * child_share + 0.3 * older_share
      + 0.25 * log(current_sending_budget / 250)
      - network_dilution[group] * log(number_of_links_in_same_group)
      + 0.8 * sent_last_month
      + 0.6 * log(remaining_support / full_support_target)

P(send | preceding decisions, last month's state) = 1 / (1 + exp(-theta))
```

Budget and remaining support must both be positive. If the migrant accepts, the contribution is `min(budget, contribution_fraction[group] * remaining_support)`. Core attempts to cover the full remainder; kin contributes 35% by default. Consequently a successful first contribution can crowd out later contributors, but several migrants may contribute when the remainder stays positive. No one exceeds their budget, and total contributions cannot exceed the household target.

At the next month, budgets and support targets reset; previous sending status is retained. There is no accumulating debt, saving, ageing, migration or resource shock in this baseline. Individual decisions continue changing even when aggregate distributions stabilize. Age, education, residence duration, occupation and other original attributes remain attached to migrants; education and duration do not yet have separate behavioural coefficients.

The network-size penalty is an explicit assumption about responsibility dilution. It is separate from coordination through observed preceding contributions. Persistence, contribution fractions, job budgets and dilution are not identified by household receipt observations; alter them as sensitivity scenarios and recalibrate the three fitted parameters.

## Calibration and interpretation

Only three parameters are fitted: core intercept, kin intercept and the no-worker coefficient. Targets are six unweighted MMP receipt rates: core-only, kin-only and mixed networks, each split by whether the origin household has workers. Targets come from unique original household records, not repeated synthetic donors. Unknown receipt codes and incomplete network counts are excluded. Unknown simulation demographics receive flagged fallback values and are excluded from fitting. Small cells are excluded using editable minimum counts. No observed household receipt flag is used to assign that household's simulated sending state.

The optimizer minimizes sample-size-weighted squared receipt discrepancies, with weak regularization. It uses the exact probability of any receipt conditional on the simulated preceding month's state, averaged over months after a calibration burn-in. Along the path where all migrants decline, remaining support stays unchanged, giving `q = 1 - product(1 - p_j)`. This calculation does not make final sending outcomes independent. Realized decisions, including partial contributions and incumbency, are still simulated individually.

Fitted parameters are then fixed for all 120-month validation runs using different random seeds. This assesses simulation reproducibility and moment fit on the same underlying population; it is not validation against held-out survey households or identification of causal effects.

MMP receipt is treated as stationary monthly prevalence **by assumption**. Cross-sectional receipt does not identify monthly transfer frequency, spells or incumbent persistence. The population represents the existing pooled MMP sample, not a national Mexican population or a dated 2019 stock. Household connections are links: without person identifiers across households, a real relative appearing in multiple networks cannot share one global budget in this version. Most kin profiles were imputed by the population generator, so occupation-based capacity and future employment shocks require sensitivity analysis.

`REMUSSIZ` is retained as an ordinal small/intermediate/substantial benchmark among recipients with known responses. No dollar thresholds are assigned. Figure 06 presents it alongside, but does not equate it with, illustrative dollar amounts. The monetary allocation is internally consistent and stable; it is **not calibrated to MMP amounts**. A high unmet-target share describes these assumed targets, not measured household deprivation.

## Results of the default run

The six calibration cells contain 3,211 original households and 963 synthetic households with usable inputs. The remaining synthetic households are simulated but excluded from these comparisons. Estimated conditional intercepts are approximately -0.364 (core), -3.274 (kin), with a no-worker coefficient of 0.132. These values depend on the fixed behavioural assumptions.

| Network / workers | MMP receipt | ABM mean across three seeds |
|---|---:|---:|
| Core + kin / no worker | 66.0% | 69.0% |
| Core + kin / worker(s) | 64.0% | 63.1% |
| Core only / no worker | 70.0% | 67.4% |
| Core only / worker(s) | 59.0% | 59.8% |
| Kin only / no worker | 8.8% | 7.2% |
| Kin only / worker(s) | 5.8% | 5.5% |

The largest cell discrepancy in any seed is 4.96 percentage points. Comparing the final two 24-month windows, receipt changes are +0.19, -0.07 and +0.65 percentage points, and total amounts change by +1.7%, +0.4% and +3.2%. All satisfy the declared practical tolerances. Starting with 50% of migrants labelled previous senders instead of none yields the same post-burn-in receipt path under common random numbers for the tested seed. These are useful stability diagnostics, not proof of stationarity under every parameter setting.

Focused checks passed for complete crowding out, joint partial coverage, an incumbent losing capacity and another member stepping in, zero capacity, the analytical two-person receipt probability, reproducibility and agent-to-household-to-month exported accounting. The capacity-loss fixture tests the mechanism; the baseline simulations themselves contain no COVID shock.

## Files and figures

Outputs are in `oct-26/outputs/remittance-baseline-abm/`:

- `monthly_summary.csv/.json`: all seeds, monthly receipt, senders, entry, retention, turnover, amounts and unmet support.
- `household_months.parquet`: first seed, every household-month, received amount, sender count and remaining support.
- `migrant_months.parquet`: first seed, every migrant-month, including nonsenders, with migrant and linked household IDs; 1.2 million rows by default.
- `household_baseline.parquet`, `migrant_baseline.parquet`: original attributes plus support targets, budgets and eligibility flags. Join IDs to the monthly panels for subgroup analysis.
- `mmp_receipt_targets.csv`, `receipt_validation.csv/.json`, `calibration_history.csv`, `stability_checks.csv`, `alternative_initialization.csv`: fit and convergence diagnostics.
- `mmp_ordinal_size_targets.csv`: original ordinal size shares with denominators.
- `run_settings.json`, `REPORT.txt`: assumptions, fitted coefficients and diagnostic results. Full trajectories are Parquet; JSON is reserved for smaller summary tables.
- `01_monthly_dynamics.png`: receipt, active senders, amounts and unmet support over all months and seeds.
- `02_receipt_validation.png`: six MMP target cells versus simulated receipt.
- `03_sending_responsibility.png`: core/kin senders, entry, retention, complete contributor turnover and network-specific receipt.
- `04_household_histories.png`: first available household in four network categories, showing individual variation beneath stable averages; illustrative, not selected for fit.
- `05_stability_and_calibration.png`: alternative initialization and numerical calibration progress. Parameter evaluations are not months.
- `06_amount_context.png`: ordinal MMP evidence and separate provisional amount outputs.

“New senders” means inactive in the immediately preceding month, not never having sent before. “Replacement households” means receiving in consecutive months with no continuing sender, not necessarily replacement caused by a capacity loss. Seed ranges in Figure 02 are simulation variation, not survey confidence intervals. Monetary means among recipients exclude zero-receipt household-months.
