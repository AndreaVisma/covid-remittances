# Mexican migrant remittance microsimulation

A monthly individual-agent prototype with synthetic inputs and real aggregate outcomes. All new code, inputs and outputs live in this folder. The source workbook is read only.

## Run

From the repository root, using Python 3.10 or later:

```powershell
python -m pip install -r microsimulation/requirements.txt
python microsimulation/run.py
python -m unittest discover -s microsimulation -v
```

The delivered `inputs/synthetic` folder already contains generated CSVs. To generate a fresh population, change `input_dir` in a copy of `config.json` and run `python microsimulation/run.py --config path/to/config.json --generate`. Generation refuses to overwrite a nonempty input directory. Paths inside configuration files resolve relative to that configuration file; `--target` resolves relative to your working directory.

`python microsimulation/run.py --single` runs the configured mechanism switches and target mode. The default runs nine scenarios: all eight combinations of need, persistence and network support with a fixed reference-income target, plus the original current-income baseline. The agent trace always describes the configured switches and first seed, rather than every factorial run.

`python microsimulation/run.py --calibrate` fits the baseline remittance share on a grid using 2017–2019 outcomes only. It freezes that share across all scenarios. This is exploratory fitting conditional on invented inputs, not empirical identification. Use a separate `output_dir` to retain alternative runs; each run replaces outputs of the same name. `--skip-sensitivity` skips the sensitivity sweep.

Default: 3,000 agents, January 2017–December 2021, three simulation seeds. The assumed represented population is 10 million individuals, not an estimate. Each connected group is interpreted as equally replicated groups of migrants and recipients. No fitted population multiplier is applied.

## Model specification

For individual i, let y0 be reference monthly labour income, y current labour income, c essential monthly expenditure, b baseline willingness probability, tau tie strength, n recipient need, and W previous willingness.

1. **Employment:** employed individuals lose work with the sector-month job-loss probability. Unemployed individuals find work with the job-find probability. Transitions are simultaneous and use last month's employment. Income is `y = employed * y0 * wage_multiplier`. The first simulated month also applies this transition to the initial state.
2. **Transfer capacity:** `C = max(y - c, 0)`. Essential costs are individual allocations, not duplicated household income. When income cannot cover essentials, the shortfall is reported, not financed. Unspent income does not accumulate; there is no savings or borrowing state.
3. **Willingness:** `logit(p) = logit(b) + need_on * beta_need * tau * n + persistence_on * beta_persistence * W`. Draw `willing ~ Bernoulli(p)`. Exact b=0 or b=1 remains absorbing. No income appears in this probability. Persistence uses previous willingness, even if a willing migrant could not send; it is a behavioural assumption, not an estimate of observed-sender persistence.
4. **Desired contribution:** current-income baseline: `D = willing * alpha * y`. Reference-target extensions: `D = willing * alpha * y0`. Own payment is `O = min(D, C)`. Even the baseline is constrained by essentials; the fixed income share is its desired amount.
5. **Network requests:** pool `D - O` within each shared recipient household. Each migrant accepts a support request with probability `support_probability * tau`, independently of own willingness. An accepting donor offers `support_budget_share * (C - O)`. Thus a request can motivate a previously unwilling donor, without an imitation mechanism.
6. **Allocation:** recipient support equals the smaller of pooled offers and pooled shortfall. Allocate that amount proportionally to donor offers. Send support directly to Mexico. There is no US-to-US transfer counted as a remittance, and no sequential recipient competition because each migrant has one recipient link.
7. **Accounting:** individual actual sending is own payment plus support. For each recipient, delivered transfers plus unmet desired contributions equal desired contributions. Actual senders include support-only donors. Update previous willingness and retain employment for next month.

Need affects willingness only. It is a dimensionless scenario index in [0,1], not an independently estimated monetary recipient budget. The network fills unmet migrant contributions; it does not measure all unmet household needs in Mexico. Each donor reserves their own contribution before offering support. The model does not guarantee network support increases total sending relative to a different contribution-target rule.

The `reference_only` control matters: changing the contribution target is itself a mechanism. Compare `reference_need_persistence_network` against `reference_need_persistence` to isolate network support, holding the target fixed. The full factorial permits other interactions. No automatic refitting occurs when switches change.

## Inputs and replacement guide

Every CSV has a header. Dates are month-start ISO dates, monetary values are monthly nominal USD, and probabilities are in [0,1]. CSV identifiers must be consistent across files. Loader validation is shared by synthetic and future empirical inputs.

| File | Fields and meaning | Real information needed |
|---|---|---|
| `agents.csv` | Unique `agent_id`; `recipient_id` shared by connected migrants; `sector`; `reference_income_usd`; `essential_cost_usd`; `base_probability`; `tie_strength`; binary `initial_employed`, `initial_willing`; positive equal `weight` | Joint individual labour-income and employment distribution, individually allocated essentials, recipient links, participation evidence, population total |
| `employment.csv` | Unique `date,sector`; `job_loss_probability`, `job_find_probability`, `wage_multiplier` | Monthly employment transition probabilities and earnings changes among employed migrants; employment-rate levels alone are not transition probabilities |
| `needs.csv` | Unique `date,recipient_id`; `need` | Transparent recipient-level or assigned regional need measure, with a fixed normalization rule |
| `config.json` | Horizon, seeds, paths, alpha, need and persistence coefficients, support probability and budget share, switches, target mode, sensitivity values | Estimates where identifiable; external constraints or sensitivity ranges otherwise |
| Target workbook | `Sheet1.date`, `Sheet1.total_mln` | Confirm currency, units and source-population coverage; optionally supply a corridor-specific replacement |

Keep complete linked groups when resampling a weighted empirical population. The current loader rejects unequal weights because arbitrarily weighted donors and recipients would not conserve represented network flows. The weights in `agents.csv` are authoritative during simulation; `represented_population` in config is used by the synthetic generator only.

Replace CSVs in a new input directory and point a copied configuration at them. Add a `provenance.json` describing sources, transformations and which fields remain assumed. Preserve relevant joint distributions instead of independently drawing all empirical attributes. Do not regenerate the other files merely because one input becomes available.

The synthetic generator uses lognormal reference income (median $2,800, log SD 0.55), uniform essential costs ($1,000–1,700), uniform base probabilities (0.15–0.4), and ties (0.4–1). Groups contain three migrants except a possible final partial group. Initial employment is 92%; initial willingness is 27.5%. These are invented numbers. Two sectors have different shock exposure. From April 2020, synthetic crisis intensity decays exponentially over 14 months. Job loss rises and wages fall with that index; monthly job finding stays at 0.12. Recipient vulnerability varies from 0.6 to 1. No workbook outcomes enter generation. These assumptions omit seasonality, population growth, policy transfers, migration flows, exchange rates and informal/formal measurement changes.

## Outcomes and interpretation

The supplied workbook is `../data/mex_remittances.xlsx`. `Sheet1` contains monthly amounts, payment channels, transaction counts and average transaction amounts. `Sheet2` is a pivot summary and is not appended to the monthly data. The loader records actual coverage, missing values and missing months in `outputs/target_audit.json`.

`total_mln` is multiplied by one million and treated as nominal USD. The header supports a millions interpretation, but the workbook does not explicitly state currency or isolate Mexican-born migrants sending from the US. These remain declared assumptions. The aggregate comparison must not be interpreted as an exact coverage match. Transaction counts and mean transaction sizes are not unique-sender counts or average monthly amounts per sender; they are retained in the source and not used as those targets.

Outputs:

- `REPORT.md` and `comparison.png`: interpretation and a chart of levels plus indices (each series' own pre-COVID mean = 100). Shaded ranges span three seeds and are not confidence intervals.
- `monthly_runs.csv`: weighted monetary flows, willing migrants, senders, support donors, employment, new willingness, essential-cost shortfalls and conditional monthly sending amounts, per scenario and seed.
- `comparison.csv`, `metrics.csv`: seed means matched by month to observations, plus pre-COVID and holdout errors. Missing outcomes remain missing.
- `annual_by_seed.csv`, `paired_effects.csv`: yearly flows and paired effects against the reference-only control.
- `sensitivity.csv`: one-at-a-time changes in assumed need response, support acceptance and transfer share, for all seeds. These are not parameter uncertainty distributions.
- `configured_agent_trace.csv.gz`: monthly individual budgets and transfers for the configured scenario and first seed. Monetary values here are unweighted; `weight` is supplied.
- `manifest.json`: effective configuration, scenarios, population size and input hashes. `observed.csv` is the extracted target, without changing the workbook.
- `calibration.csv`: produced only with `--calibrate`; grid scores use pre-COVID data exclusively. A boundary optimum is evidence to examine assumptions, not a reason to silently expand the search.

No default parameters are estimated. The optional fitted alpha is the only fitted parameter. Aggregate outcomes do not identify network support, willingness, persistence, household needs or the synthetic population. More informative validation would need individual participation and transfer histories, recipient-linked contributions, and independent employment/income evidence. The indexed chart assesses shape but deliberately removes level differences; always inspect the level chart too.

## Implementation and checks

`data.py` handles input generation and loading, `model.py` handles monthly behaviour and accounting, and `run.py` handles experiments and reporting. `test_model.py` checks reproducibility, budgets, recipient conservation, separate willingness and income, mechanism disabling, zero-support equivalence, missing-input rejection, overwrite protection and absence of holdout leakage during calibration.

The model uses NumPy arrays for individual states, with pandas for input/output. Each run draws employment, willingness and support random numbers in the same order regardless of enabled mechanisms. Paired experiments therefore share random draws. Randomness from input generation and simulation has separate seeds. Three simulation seeds are a prototype default; increase them before assessing small effects. Population and network uncertainty require additional synthetic populations, not just more simulation seeds.
