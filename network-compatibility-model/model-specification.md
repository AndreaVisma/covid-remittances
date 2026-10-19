# Model specification: translocal support networks in the Mexico–US remittance corridor during COVID-19

Draft, 19 October 2026. Replaces the research-question framing in `../oct-26/research-questions-literature-review.md` (kept for its audit and annotated references). Structured loosely after the ODD protocol (Grimm et al. 2006).

## 1. Purpose

Pre-pandemic forecasts expected remittances to fall sharply during COVID-19; flows to Mexico instead rose. Survey studies attribute this to migrants responding to need at home and to relatives stepping in for migrants who lost their jobs; macro studies attribute it to aggregate drivers. The model asks:

- **RQ1 (micro → macro).** Under which combinations of migrant need-responsiveness and within-network coordination, given observed heterogeneous US labour-market shocks and benefit access, does the observed network of household–migrant links reproduce the 2020–21 aggregate remittance dynamics?
- **RQ2 (networks → household resilience).** Among the parameter regimes compatible with the aggregate data, how does household protection depend on network size, relationship composition, and migrants' occupational exposure and its correlation?
- **RQ3 (policy lab).** How would alternative benefit-access and shock scenarios have changed aggregate flows and the distribution of household protection?

**Scope: the crisis window, 2020–21.** Claims about the post-crisis period (2022 onward) are outside the paper. 2022 is held out and is expected to be underpredicted (§7).

Design principles: **few free parameters**, the decision to send separated from the amount sent, all non-behavioural inputs taken from data, and results reported as **compatibility maps** over the free parameters rather than as one best fit.

## 2. Entities and state variables

### Households `h` (origin, Mexico)

| Variable | Meaning | Source |
|---|---|---|
| `L_h` | Set of linked migrants | MMP PERS/HOUSE reconstruction |
| `size_h` | `|L_h|` | derived |
| `comp_h` | core-only / kin-only / mixed | derived from links |
| `state_h` | Mexican state | MMP community |
| `R_h,t` | USD received in month `t` | simulated |
| `N_h,t` | Number of transfers received in month `t` | simulated |

### Migrants `j` (destination, US)

| Variable | Meaning | Source |
|---|---|---|
| `r_j` | Relationship to household: core (head, spouse, child; RELHEAD 1/2/3) or kin | MMP |
| `o_j` | Occupation group | MMP; imputed where missing (flag kept) |
| `m_j` | Routine transfer amount (USD), fixed over time | drawn once, see §4.4 |
| `E_j,t` | Employed (0/1) | simulated from CPS rates, §4.1 |
| `B_j,t` | Receives a US benefit payment this month (0/1) | simulated from schedule, §4.2 |
| `elig_j` | Eligible for US benefits (0/1) | drawn once with probability `π` |
| `A_j,t` | Able to send = `E_j,t OR B_j,t` | derived |
| `D_j,t` | Sends this month (0/1) | simulated, §4.3 |

### Global (national) variables

| Variable | Meaning | Source |
|---|---|---|
| `e_o,t` | Employment rate of Mexican-born workers in occupation `o` | CPS (switch from Mexican-origin to Mexican-born) |
| `S_t` | National salience index (Mexico) | §4.3 |
| benefit schedule | Months and type of UI supplements and stimulus payments | statutory dates |
| `FX_t` | MXN/USD | `data/DEXMXUS.csv` (for MXN-denominated outputs only) |

Time step: one month. Period: 2019-01 to 2022-12, plus burn-in. The 2019 conditions continued forward form the **no-COVID counterfactual**, run with common random numbers.

## 3. Process overview: order of events within month `t`

1. Update the national inputs: `e_o,t`, `S_t`, benefit schedule.
2. **Employment:** draw `E_j,t` for every migrant (§4.1).
3. **Benefits:** set `B_j,t` (§4.2); compute ability to send `A_j,t`.
4. **Stage 1, core migrants:** every core migrant with `A_j,t = 1` decides independently whether to send, with probability `θ_core,t` (§4.3).
5. Update `N_h,t` with stage-1 transfers.
6. **Stage 2, kin migrants:** every kin migrant with `A_j,t = 1` decides with probability `θ_kin,h,t`, which depends on `N_h,t` through the coordination parameter `κ`. In households without core migrants, kin decide with `N_h,t = 0`.
7. Transfer amounts are `m_j` for every sender (§4.4). Compute `R_h,t`, `N_h,t`.
8. Record the observation outputs (§5).

## 4. Submodels

### 4.1 Employment

Each migrant's monthly employment status reproduces the occupation-month employment rate `e_o,t` of Mexican-born CPS workers. Occupation-specific rate = employed / (employed + unemployed with last job in `o`). Labour-force exits need an explicit decision (treat as non-employed, or exclude); record which.

Latent Gaussian construction, to allow correlation between job losses inside a network without changing the marginal rates:

```
z_j,t = φ z_j,t−1 + sqrt(1−φ²) · ( sqrt(ρ) F_h,t + sqrt(1−ρ) ε_j,t ),   F, ε ~ N(0,1)
E_j,t = 1  if  Φ(z_j,t) < e_o(j),t
```

- `φ` (persistence of employment status over time): set externally from CPS month-to-month transitions, not fitted.
- `ρ` (correlation of job loss between migrants in the same network): `ρ = 0` in the base case; it is the axis of the network map (§6, E3). Report the realized employment correlation, not just `ρ`.

### 4.2 Benefits (budget channel only)

`elig_j ~ Bernoulli(π)`. `π` is the share of migrants with access to US benefits (roughly, work-authorized and taking up benefits). It is a **scenario parameter**, set from external estimates of the undocumented share of Mexican-born migrants, not fitted.

- **Unemployment-insurance supplements:** an eligible, non-employed migrant has `B_j,t = 1` in months with the federal supplement (Apr–Jul 2020; Aug–Sep 2020 Lost Wages Assistance; Jan–early Sep 2021). Whether regular state UI also opens the gate is a documented sensitivity choice.
- **Stimulus payments:** an eligible migrant has `B_j,t = 1` in the month of each payment (Apr 2020; Dec 2020/Jan 2021; Mar 2021). Mixed-status exclusion from the first round can enter as a lower `π` for that payment.

Benefits change only **whether** a migrant can send. They do not change θ or the amount. Without employment or benefits, a migrant does not send (no savings in the base model; noted limitation).

### 4.3 Sending decision

```
logit θ_core,t      = α_core + β · S_t
logit θ_kin,h,t     = α_kin  + β · S_t − κ · N_h,t
```

- `α_core`, `α_kin`: **baseline propensities**, calibrated to 2019 (§7).
- `S_t`: **national salience index**, standardized so that its 2019 mean is 0 (β then has no effect in 2019):
  ```
  S_t = ½ · z(employment loss in Mexico_t) + ½ · z(deaths in Mexico_t)
  ```
  - Employment loss: fall of the employed population relative to 2019 (ENOE/ETOE). Not the unemployment rate, which barely moved because people left the labour force.
  - Deaths: COVID deaths, with excess mortality as a sensitivity because reported COVID deaths were undercounted.
  - Equal weights are fixed. Splitting into two coefficients is a later test, not part of the base model.
- `β ≥ 0`: **need responsiveness**. Axis 1 of the behavioural map.
- `κ ≥ 0`: **coordination**. Kin are less likely to send when core migrants have already sent this month. Axis 2 of the behavioural map.
  - `κ = 0`: independent decisions.
  - Large `κ`: kin send mostly when core migrants did not, i.e. they step in.
  - Under this rule, a negative COVID shock to core migrants' employment automatically raises kin sending if `κ > 0`. That is the "relatives stepping in" mechanism, with no extra parameter.

Monthly decisions are independent over time given the state (no sending persistence in the base model). Persistence is a sensitivity, needed only if the receipt reference periods cannot be matched without it.

### 4.4 Amount sent

Amount strategy "B": each migrant has a fixed routine amount.

```
m_j ~ LogNormal(μ, σ²),   E[m_j] = Banxico 2019 mean transaction amount
```

- `σ`: from the dispersion of reported remittance amounts in MMP (migrant-level) or ENIGH (household-level, adjusted for multiple senders). Set externally.
- Every transfer by `j` equals `m_j`. One transfer per sender-month.
- The simulated mean transfer can still change over time, because the mix of senders changes. Any further change in the observed mean amount is **outside the model** and is reported as the amount margin (§5).
- **Robustness check (wage indexing, no new parameter):** `m_j,t = m_j · w_t / w_2019`, where `w_t` is a real median-earnings index for Mexican-born (or Hispanic) workers. It tests whether wage growth closes the gap in total dollars. The transfer count remains the main target and is unaffected.
- **Descriptive check (no model change):** plot the observed Banxico mean transaction amount against the same real earnings index. If the two move together, the amount margin is consistent with wage growth.
- Extension "C", only if needed: `m_j,t = m_j · (1 + γ S_t)`, with one parameter `γ` fitted only to the mean-amount series after the count side is settled.

## 5. Observation model and outputs

### Aggregate (compared with Banxico)

- **Transfer count index:** `C_t / C_2019`, where `C_t = Σ_j D_j,t`. A scale factor maps simulated to national counts once, in 2019, and is never refitted. It is not a count of unique migrants.
- **Mean amount:** simulated (composition only) vs observed.
- **Decomposition of total growth:** Δ count (model) + Δ mean amount (data) + interaction.
- All series are reported both as **growth since 2019** and as **excess over the no-COVID counterfactual**.

### Household (compared with ENIGH, and the main outputs for RQ2)

- Receipt probability over the survey reference period (ENIGH: six months before the interview, to be verified against the documentation; MMP: its own reference period).
- **Interruption:** a household that received in the counterfactual month but not in the COVID run (paired with common random numbers).
- **Drop in receipt:** `1 − R_h,t / R_h,t^counterfactual`, cumulated over 2020–21.
- **Recovery time:** months until receipt returns to the counterfactual level.
- **Replacement:** share of months in which a kin transfer occurs when no core transfer does.
- In MXN using `FX_t` where needed; the main analysis is in USD.

## 6. Experiments

| ID | Description | Varies | Fixed |
|---|---|---|---|
| E0 | 2019 calibration (routine regime) | `α_core`, `α_kin` for each `κ` on the grid | everything else |
| E1 | Mechanism one: heterogeneous US job loss on real networks only | `β = 0` | `κ` grid, `π` base |
| E2 | Behavioural compatibility map | `β × κ`; panels for `π` ∈ {low, base, high} | `ρ = 0` |
| E3 | Network map | `ρ`; results split by size, composition, occupational concentration; plus a synthetic experiment that varies size while holding total employment exposure fixed | `(β, κ)` restricted to compatible cells of E2 |
| E4 | Policy lab | `π` (e.g. benefits including undocumented workers), UI timing/duration | compatible cells of E2 |

Benchmark for E3: with `κ = 0` and `ρ = 0`, receipt probability is `1 − ∏_j (1 − A_j θ_j)`, which can be computed directly. The simulation earns its place where `κ > 0` or `ρ > 0` changes the conclusion relative to that formula.

## 7. Calibration and acceptance

### 2019 (E0)

For each `κ` on the grid, fit `α_core` and `α_kin` to the 2019 receipt rates of core-only, kin-only and mixed households (MMP; ENIGH 2018 as a check). Each cell of the map therefore starts from a baseline that fits 2019. `κ` partly shows up in the 2019 cross-section through the shape of receipt by network size. Report how strongly the pre-COVID data alone constrain `κ`.

Known problem to resolve first: the current model underpredicts receipt in single-migrant core households (≈42% simulated vs 54% in MMP) and in kin-only households with older residents. Report whether the simpler model reduces or worsens these errors.

### COVID (E2)

A cell `(β, κ, π)` is **compatible** if both hold:

1. **Transfer count:** RMSE of the log count index over 2020-04 to 2021-12 ≤ `τ_C`.
2. **Household receipt:** the simulated change in receipt between ENIGH 2018 and ENIGH 2020 lies within the survey confidence interval.

**Fix `τ_C` before running E2.** Held out from fitting: 2022 counts, the mean-amount series, and March 2020 (expected to be unexplained, since salience had not moved yet).

**Expected 2022 undershoot.** The salience index falls back after 2021 and the network population is fixed, so the model will likely underpredict 2022 counts. Report this as the boundary of the crisis mechanism. Three named candidate causes, none modelled:
1. growth of the Mexican-born migrant population in 2021–22 (new arrivals not in the fixed MMP network);
2. real wage growth raising amounts (partly addressed by the wage-indexing robustness check, §4.4);
3. formalization of previously informal transfers.

### Free parameters

| Parameter | Role | How set |
|---|---|---|
| `α_core`, `α_kin` | baseline propensities | calibrated to 2019 |
| `β` | need responsiveness | map axis |
| `κ` | coordination | map axis |
| `ρ` | job-loss correlation within networks | network-map axis |
| `π` | benefit access | scenario |
| `φ`, `σ`, scale, salience weights | employment persistence, amount dispersion, national scaling, index | external / fixed |

## 8. Known limitations (state upfront)

- MMP networks are not nationally representative: they over-represent traditional western-central sending regions. Results are about this network population. The aggregate comparison uses growth indices, not national levels.
- Kin occupations are largely imputed; `ρ` is a sensitivity dimension, not an estimate.
- Wages do not affect whether a migrant sends: employment or benefits are a yes/no condition only. Wage effects appear only in the amount margin (data) and the wage-indexing robustness check.
- The migrant population is fixed: no new migration or return migration during the simulation.
- No savings, debt, or remittance-channel change (informal → formal); formalization competes as an explanation of recorded growth and is discussed, not modelled.
- National salience gives every household the same need signal. That is deliberate: household heterogeneity comes only from network structure and migrant shocks.
- Compatible cells show which mechanisms are **sufficient** to produce the data, not that they were the historical cause.

## 9. Implementation tasks

1. CPS: rebuild occupation-month employment rates for Mexican-born workers; estimate `φ` from month-to-month transitions. *Done for Mexican origin (the extract has no birthplace variable); `φ ≈ 0.91`. See `handoff.md`.*
2. Mexico: build `S_t` from ENOE/ETOE employed population and national (excess) deaths.
3. Benefit schedule table; external range for `π`.
4. Amount distribution `σ` from MMP/ENIGH; Banxico 2019 mean transaction amount.
4b. Real earnings index `w_t` (from `data/weekly-earnings-median-race.xlsx` or CPS earnings, Mexican-born where possible), deflated with US CPI.
5. Refactor the ABM: two-stage decision, yes/no ability gate replacing the surplus budget rule, fixed `m_j`, latent-factor employment.
6. E0 calibration over the `κ` grid; diagnose the singleton and kin-only misfits.
7. Fix `τ_C`; run E1, then E2.
