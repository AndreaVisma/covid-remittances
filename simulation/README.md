# Monthly migrant remittance microsimulation

A first, **illustrative and uncalibrated** simulation of 1,000 Mexican migrants
in the US, monthly from January 2018 through December 2022. No data are downloaded
or read from the existing demographic files. Results describe this synthetic
population, not national remittance totals. This is independent of the aggregate
model in `../mexico/simulate_remittances.py`.

## Run

From the repository root, with Python 3.10 or newer:

```powershell
python -m pip install -r simulation/requirements.txt
python simulation/simulate.py
```

Change `parameters.json`, or supply another full configuration:

```powershell
python simulation/simulate.py --config simulation/parameters.json --population 1000 --seed 123 --output simulation/outputs_seed123
python -m unittest discover -s simulation -p "test_*.py"
```

The default output folder is relative to the script; an explicit output path is
relative to the current working directory. Re-running overwrites matching output
files. The script runs without opening windows and saves PNG figures.

## Model and timing

Each agent represents an independent adult economic unit. Initial employment,
potential monthly earnings, essential consumption, liquid savings, a fiscal
eligibility indicator, and family ties are drawn once. Monetary draws are
lognormal; a configurable fraction starts with no savings. Initial attributes
are independent, except that fiscal eligibility is reused for both support types.
These are simplifying assumptions, not empirical findings.

Every month:

1. After the initial month, draw employed / unemployed / inactive status from a
   transition matrix conditional on last month's state. Rows are previous states,
   columns are next states, in that order. The matrix is
   `(1 - labour_shock) * transition_normal + labour_shock * transition_crisis`.
2. Employed migrants receive their fixed potential earnings times
   `(1 - hours_loss)`. Other states have zero earned income.
3. Eligible migrants receive scheduled stimulus; eligible unemployed migrants
   also receive the phase's monthly unemployment support. These are stylised
   incremental pandemic payments, not an implementation of benefit law.
4. Consume essentials, capped by opening savings plus income and US transfers.
   Any shortfall is recorded. There is no borrowing or negative consumption.
5. Compute net recipient need:
   `need = max(mexico_need - mexico_support_effect * mexico_support, 0)`.
   Both Mexico inputs are dimensionless indices. Support reduces unmet financial
   need, not infections; no Mexican cash enters the sender's budget.
6. Draw willingness to remit with probability
   `logistic(logit(remit_probability) + family_ties + need_log_odds * need)`.
   `remit_probability` is the probability for zero family-ties deviation and zero
   need, before affordability constraints, not aggregate observed participation.
7. For willing migrants, the desired amount is
   `(remit_income_share * (income + US transfers) + remit_savings_share * opening_savings)`
   multiplied by `(1 + need_amount_effect * need)`.
8. Actual remittances are capped by resources after consumption and a reserve of
   `reserve_months * essential_consumption`. The reserve protects funds from
   remitting, but can be spent on consumption. Record unfulfilled remittance demand.
9. Carry closing savings into the next month:
   `closing_savings = opening_savings + income + US transfers - consumption - remittances`.

Thus desire and financial capacity are separate. Actual sender participation
counts strictly positive transfers, not willingness alone. An agent can send at
most one modelled monthly amount; this is not a count of payment transactions.
Employment transitions include persistent unemployment and inactivity. The
initial state shares need not be the stationary distribution, so pre-COVID
composition and savings can drift; inspect that period before interpreting changes.

## Parameters

All configuration values are in `parameters.json`. Money is USD with a constant
price level (no inflation, exchange rates, fees, or interest). Monthly income is
spendable income; taxes are not modelled separately.

| Parameters | Interpretation |
| --- | --- |
| `population`, `seed`, `start`, `end` | Population, random seed, inclusive monthly dates |
| `initial_state_shares`, `transition_*` | Initial state probabilities and monthly transition matrices |
| `*_median`, `*_log_sd` | Median and standard deviation of the log for earnings, essentials and positive initial savings |
| `zero_savings_share` | Fraction with zero initial savings |
| `fiscal_eligible_share` | Fixed probability of receiving the stylised US support |
| `family_ties_log_sd` | Standard deviation of the normal family-ties effect in log-odds units |
| `remit_probability`, `need_log_odds` | Baseline willingness and its response to need |
| `remit_income_share`, `remit_savings_share` | Desired fractions of current inflows and opening savings |
| `need_amount_effect` | Proportional increase in desired amount per unit of need |
| `reserve_months` | Protected savings for remittance decisions |
| `mexico_support_effect` | Conversion of Mexican support index into reduced need |
| `pandemic` | Nonoverlapping inclusive phases for labour, hours, recipient need and support |
| `stimulus` | Month-to-payment mapping for eligible agents |

Outside listed phases, all shock/support inputs are zero. The defaults are broad
synthetic phases, **not measured COVID cases or historical policy reconstructions**.
All behavioural coefficients, distributions, eligibility shares, transition
probabilities, and payment timing require calibration. The figure's March 2020
line is a historical reference; change it if adapting to a different crisis.

## Scenarios and interpretation

Every run produces:

- `baseline`: all configured mechanisms.
- `no_us_support`: removes stimulus and pandemic unemployment support.
- `no_mexico_support`: removes the Mexican support index.
- `no_altruistic_response`: switches off need effects on both probability and amount.
- `no_pandemic`: removes all pandemic labour, hours, need and policy inputs.

Scenarios use identical initial agents and random draws, reducing simulation noise
in comparisons. Effects include subsequent savings dynamics; removing mechanisms
is not an additive decomposition and does not identify real-world causal effects.
Counterfactuals do not include policy feedback on employment or infections.

## Outputs

| File | Contents |
| --- | --- |
| `01_dashboard.png` | Desired/actual remittances, participation, mean amount per sender, employment, median savings and financial strain |
| `02_resource_flows.png` | Income, US transfers, consumption, remittances and net savings change; remittances by employment state |
| `03_counterfactuals.png` | Monthly and cumulative remittances across all five scenarios |
| `monthly.csv` | Scenario-month totals, state shares, sender counts and savings indicators |
| `annual.csv` | Annual sums of flows, including net savings change (not sums of savings balances) |
| `agents_baseline.csv.gz` | Baseline agent-month panel, including initial characteristics, states, decisions and budget components |
| `inputs_<scenario>.csv` | Effective monthly exogenous inputs for each scenario |
| `parameters_used.json` | Full configuration including command-line overrides |

`depleted_share` counts balances below USD 1e-8, a numerical zero tolerance.
`mean_per_sender` is reported as zero when there are no senders.
Savings change can be negative; income + transfers equals consumption +
remittances + savings change. Desired remittances are conditional on willingness,
so are stochastic rather than an unconditional expectation.

## Scope and next steps

Use this model to explore whether recipient need and US support can offset lost
earnings, whether savings smooth flows, and which migrants bear the financial
cost. Do not interpret one simulation as evidence for altruism or extrapolate
1,000 synthetic agents to the national population without weights/calibration.
There are no confidence bands: repeat seeds to assess Monte Carlo variability,
and vary assumptions separately to assess parameter uncertainty.

Next steps are to replace synthetic agents with weighted demographic records,
calibrate employment transitions/earnings and liquid assets, add household resource
sharing and realistic policy eligibility, and use Mexican recipient exposure and
support data. Current limitations also include fixed population and wages, no
household links, no migrant entry/exit, no debt, no discretionary consumption,
no seasonality, and no recipient purchasing-power calculation. Fixed consumption
can generate strong savings accumulation for steadily employed agents. Match
pre-COVID moments before fitting or evaluating pandemic behaviour.
