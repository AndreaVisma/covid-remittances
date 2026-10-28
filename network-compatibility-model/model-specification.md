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

**Population (rebuilt 19 Oct 2026, `data-prep/build_population.py`).** Every connected MMP household interviewed in Mexico in 2010–19 is used once (3,405 households). Links:
- **core:** head, spouse or child currently in the US;
- **close kin:** parents, siblings and other roster relatives currently in the US;
- **distant kin:** uncles, cousins, nephews and in-laws, collapsed into one aggregated link per household.

Close kin and the distant link are both "kin" in §4.3.

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
5. Compute the household's lost sending `L_h,−j,t`: the routine sending of the linked migrants who are unable to send this month, in core-sender units.
6. **Stage 2, kin migrants:** every kin migrant with `A_j,t = 1` decides with probability `θ_kin,j,t`, which rises by `κ · L_h,−j,t` (logit scale).
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
logit θ_kin,j,t     = α_kin  + β · S_t + κ · L_h,−j,t
L_h,−j,t = (number of eligible core migrants in h unable to send in t)
         + w · (number of other eligible kin migrants in h unable to send in t)
w = θ_kin⁰ / θ_core⁰ = expit(α_kin) / expit(α_core)   (≈ 0.07)
```

`L` is the routine sending the household has lost to inability (job loss without benefits), in core-sender units. Kin step in for core migrants and for other kin, in proportion to what each would routinely have sent. Since 28 Oct 2026 every migrant is eligible (user decision): 'Not working' on the last MMP trip is treated as unknown occupation (all-occupation CPS rate), and there is no age rule (ages are measured at the 2010–19 interview). The trigger responds to inability, not to a routine month in which an able migrant chooses not to send. Only kin raise θ; core migrants do not step in.

*Revision history (19 Oct 2026).*
1. `− κ · N_h,t` (kin held back by core transfers). It could only remove a suppression, capping stepping-in at kin's routine θ and changing transfers by ≤ 2%.
2. `+ κ · U_h,t`, with `U_h,t` = 1 if all eligible core migrants are unable. It could reach only 929 of 3,405 households (none of the 70% that are kin-only) and ignored partial core losses. In E1a the April 2020 index saturated at 0.90 even as κ → ∞ (`model/diagnose_e1_kappa.py`).
3. The current rule (user decision). Rule 2 is kept as `Params(rule='all_core')` for comparison (`model/run_e1_rule_comparison.py`).

At κ = 4, several kin can each step in for one loss, so large networks can receive more than in the no-COVID counterfactual (over-replacement). This is a property of the rule to report.

*Base rule since 28 Oct 2026 (user decision; introduced 21 Oct as an alternative): coordination among all able migrants,* `Params(rule='coordination')`, `model/run_e1_coordination.py`. The lost-sending rule above is kept for comparison; `abm.Params.rule` still defaults to it so that existing output folders keep their meaning (coordination results are in the `*-coordination` folders). The household model (`recipient-level/run_household_model.py`) uses the same idea at household level: `κ` ∈ [0, 1] is the share of the receipt probability lost to job loss that the household's able migrants replace. The household's able migrants, core and kin, share the routine sending lost to migrants who cannot send:
```
θ_j,t = min(1, θ⁰_j,t · (1 + κ · Λ_h,t / P_h,t))
Λ_h,t = Σ θ⁰ over eligible migrants in h unable to send in t   (routine sending lost)
P_h,t = Σ θ⁰ over migrants in h able to send in t              (routine sending remaining)
```
`θ⁰` is the routine probability (`α + β S_t`, by type). `κ` ∈ [0, 1] is the **replacement share**: the fraction of the household's lost expected transfers that its remaining migrants make up, before the cap at 1. Replacement is split in proportion to routine sending, so core migrants carry most of it in mixed households (387 households have two or more eligible core migrants, holding two-thirds of eligible core migrants). A household cannot receive more than without the shock. Single-link households cannot be helped, as under the base rule. Same trigger (inability, not a routine month of not sending) and same eligibility as the base rule.

- `α_core`, `α_kin`: **baseline propensities**, calibrated to 2019 (§7).
- `S_t`: **national salience index**, the reported COVID-19 deaths in Mexico in month `t`, in units of 10,000 deaths (user decisions, 20 Oct 2026: keep it simple; reported COVID deaths, not excess deaths):
  ```
  S_t = reported COVID deaths in Mexico_t / 10,000,   S_t = 0 in 2019
  ```
  - Source: OWID (`data/covid_cases.csv`), monthly sums. Peaks 2.4 (Jul 2020), 3.9 (Jan 2021), 2.2 (Aug 2021), 1.1 (Feb 2022). Annual means 1.23 (2020), 1.30 (2021), 0.23 (2022).
  - β is in logit units per 10,000 COVID deaths a month. `S_t` = 0 in 2019, so β has no effect in the calibration year.
  - Sensitivity: excess all-cause deaths (World Mortality Dataset, 2015–19 baseline with a linear trend, negative values set to 0), about 2.2 times reported deaths.
  - A labour-market component (the fall in the employed population, ENOE/ETOE) was considered and set aside for simplicity; the series is kept in `data-prep/outputs/salience_inputs_monthly.csv` in case it is added later.
- `β ≥ 0`: **need responsiveness**. Axis 1 of the behavioural map.
- **Household need (added 27 Oct 2026, `model/run_e0_need.py`):** every logit above gains `+ η_w · noworker_h + η_o · older_h`, the same shift for all migrants of household `h` (`abm.need_shift`). `noworker_h` = no resident worker (0/1); `older_h` = at least one older resident (0/1). Both are centred on the population mean, so `α` remains the propensity of an average household; households with an incomplete roster get the mean. MMP has no usable household income (`hhhincom` is empty for 2010–19), so need comes from composition only (user decisions, 27 Oct 2026: worker presence and dependants only; then older residents instead of a pooled dependency ratio, see §7). `η` is calibrated to 2019 and static: it changes *who* receives, not the COVID response (an `S_t × need_h` interaction would need a 2020 target, e.g. ENIGH 2018→2020 by composition).
- `κ ≥ 0`: **stepping in**. Kin become more likely to send when other migrants linked to the household cannot send, in proportion to the routine sending lost. Axis 2 of the behavioural map.
  - `κ = 0`: kin ignore what happens to other linked migrants.
  - Large `κ`: kin respond strongly when core migrants, or other kin, lose the ability to send.
  - A COVID job-loss shock therefore raises kin sending automatically if `κ > 0`, with no extra parameter.

Monthly decisions are independent over time given the state (no sending persistence in the base model). Persistence is a sensitivity, needed only if the receipt reference periods cannot be matched without it.

### 4.4 Amount sent

Amount strategy "B": each migrant has a fixed routine amount.

```
m_j ~ LogNormal(μ, σ²),   E[m_j] = Banxico 2019 mean transaction amount
```

- `σ`: from the dispersion of reported remittance amounts in MMP (migrant-level) or ENIGH (household-level, adjusted for multiple senders). Set externally. Estimated `σ = 0.81` from MMP (heads and other migrants, trips 2000–19, 2019 USD, within 5-year periods); `μ = 5.46` given the $326 mean.
- Kin migrants use the same distribution as core migrants. MMP records amounts for only ~10 extended-kin migrants, too few to calibrate a difference. Sensitivity: kin amounts × 0.5.
- **Level: Banxico 2019 mean per transaction ($326), decided.** MMP is used for the dispersion only. MMP's mean ($497, 2019 USD) is higher, but that reflects several factors besides transactions per month: deflation from trip start year on long trips, heads with dependants dominating the sample, rounding, recall and proxy reports, and mostly 2000s trips. It is not evidence of ~1.5 transactions per sender-month. Count indices do not depend on the level in any case.
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

For each `κ` on the grid, fit `α_core` and `α_kin` to the 2019 receipt rates of core-only, kin-only and mixed households (MMP; ENIGH 2018 as a check), so every cell of the map starts from a baseline that fits 2019. Report the `κ` profile, i.e. the best 2019 fit at each `κ`. Result: 2019 barely constrains `κ` (weighted RMSE 0.2 pp at `κ = 0`, about 0.7 pp at `κ = 8`), because few migrants are unable to send in a normal year.

**Household need (27 Oct 2026, `model/run_e0_need.py`, `outputs/e0-need/`):** `α_core, α_kin, η_w, η_o` by minimum χ² on MMP cells (SD of each cell from its network type's receipt rate). A household likelihood was tried first and rejected: it also chases the core-network size misfit and pulls `α_core` down (network-type error 3.6 pp vs 0.6 pp).

*First specification (superseded): pooled dependency ratio.* `η_d · dep_h` with `dep_h` = (children + older residents) / residents, 18 cells (network type × worker × dependency none / ≤ ½ / > ½). Result, stable across `κ` and rules: `η_w` = 0.40–0.43 (SE 0.17; odds × 1.5), `η_d` ≈ 0 (SE 0.20); joint p = 0.05–0.07. The zero was an artefact of the definition. Raw MMP logits (network type and log size as controls): children do not predict receipt (count −0.02, SE 0.06; any +0.06, SE 0.13), older residents do (count +0.14, SE 0.08; any **+0.27, SE 0.13**), and no worker is stable at +0.31 to +0.34 (SE 0.14). Children are twice as common (0.70 vs 0.33 per household) and dominate the ratio, a share dilutes older residents in large households, and three bins blur it further. The older-resident gap is largest in kin-only households (receipt 12.9% vs 4.6%; 16.7% vs 8.5% without a worker), the misfit flagged below.

*Current specification: any older resident* (user decision, 27 Oct 2026). 12 cells: network type × worker × older resident. At the E0 alphas, χ² = 66.6 on these cells (coordination, `κ` = 0.7), mostly the kin-only older-resident gap. **Result (run 27 Oct 2026):** stable across `κ` and both rules: `η_w` = 0.51–0.53 (SE 0.16; odds × 1.7), `η_o` = 0.27–0.33 (SE 0.13–0.14; odds × 1.3–1.4), correlation ≈ 0; χ² 66.6 → 56.7 (coordination, `κ` = 0.7), joint test p = 0.004–0.007. The fit is still poor (χ² 56.7 on 8 df). Validation (coordination, `κ` = 0.7, model vs MMP): no worker is fixed in core-only (63% → 72% vs 71%) and kin-only (5.5% → 9.1% vs 10.4%) but overshoots in core + kin (67% → 75% vs 67%). The kin-only older-resident gap is only partly closed (5.5% → 7.5% vs 13.5%), and core + kin with older residents, already over-predicted at `η` = 0 (72% vs 63%), worsens to 76%. RMSE by older residents rises 3.9 → 4.7 pp, by network type 0.6 → 1.0 pp; by workers unchanged (1.8). As expected, one `η` per input cannot give both the large relative effect in kin-only households and the small, mixed one in core households (core + kin without a worker: 73% without vs 59% with older residents). Interpretation caveat: in kin-only households an older resident is often an older head whose siblings or parents are the migrants, so part of `η_o` may reflect who the migrants are, not need.

**Network size (decided 19 Oct 2026):** no dilution parameter. The saturation seen in the first E0 run came from counting every distant relative as an independent potential sender. After the population rebuild (distant kin as one aggregated link), the kin-only size gradient matches MMP within ~1 pp. The remaining size misfits are in core networks (±8–12 pp in some bins) and are reported as a limitation.

Known problem to resolve first: the current model underpredicts receipt in single-migrant core households (≈42% simulated vs 54% in MMP) and in kin-only households with older residents. Report whether the simpler model reduces or worsens these errors. (The older-resident need term `η_o`, above, targets the second.)

### COVID (E2)

A cell `(β, κ, π)` is **compatible** if both hold:

1. **Transfer count:** RMSE of the log count index over 2020-04 to 2021-12 (quarterly: 2020Q2–2021Q4) ≤ `τ_C`. Model index = COVID run / no-COVID run. Data index = Banxico transactions / their 2017–19 trend (log-linear with month effects, +6.3% a year; user decision 20 Oct 2026), so pre-COVID growth is not attributed to the crisis.
2. **Household receipt:** the simulated change in receipt between ENIGH 2018 and ENIGH 2020 lies within the survey confidence interval.

**`τ_C` = 0.04** (user decision, 20 Oct 2026, fixed before running E2): quarterly log RMSE over 2020Q2–2021Q4. That is about three times the SD of Banxico's quarterly deviations from its 2017–19 trend (0.014). Held out from fitting: 2022 counts, the mean-amount series, and March 2020 (expected to be unexplained, since salience had not moved yet).

**Expected 2022 undershoot.** The salience index falls back after 2021 and the network population is fixed, so the model will likely underpredict 2022 counts. Report this as the boundary of the crisis mechanism. Three named candidate causes, none modelled:
1. growth of the Mexican-born migrant population in 2021–22 (new arrivals not in the fixed MMP network);
2. real wage growth raising amounts (partly addressed by the wage-indexing robustness check, §4.4);
3. formalization of previously informal transfers.

### Free parameters

| Parameter | Role | How set |
|---|---|---|
| `α_core`, `α_kin` | baseline propensities | calibrated to 2019 |
| `η_w`, `η_o` | household need (no resident worker, any older resident) | calibrated to 2019 |
| `β` | need responsiveness | map axis |
| `κ` | coordination | map axis |
| `ρ` | job-loss correlation within networks | network-map axis |
| `π` | benefit access | scenario |
| `φ`, `σ`, scale, salience definition | employment persistence, amount dispersion, national scaling, index | external / fixed |

## 8. Known limitations (state upfront)

- MMP networks are not nationally representative: they over-represent traditional western-central sending regions. Results are about this network population. The aggregate comparison uses growth indices, not national levels.
- Kin occupations are largely imputed; `ρ` is a sensitivity dimension, not an estimate.
- Wages do not affect whether a migrant sends: employment or benefits are a yes/no condition only. Wage effects appear only in the amount margin (data) and the wage-indexing robustness check.
- The migrant population is fixed: no new migration or return migration during the simulation.
- No savings, debt, or remittance-channel change (informal → formal); formalization competes as an explanation of recorded growth and is discussed, not modelled.
- National salience gives every household the same need *signal*. Households differ in baseline need (`η`, from composition measured at the MMP survey, 2010–19, not in 2020), but not in how strongly they respond to the crisis.
- Compatible cells show which mechanisms are **sufficient** to produce the data, not that they were the historical cause.

## 9. Implementation tasks

1. CPS: rebuild occupation-month employment rates for Mexican-born workers; estimate `φ` from month-to-month transitions. *Done for Mexican origin (the extract has no birthplace variable); `φ ≈ 0.91`. See `handoff.md`.*
2. Mexico: build `S_t` from national COVID deaths. *Done (`data-prep/salience_inputs.py`).*
3. Benefit schedule table; external range for `π`.
4. Amount distribution `σ` from MMP/ENIGH; Banxico 2019 mean transaction amount. *Done: mean $326 (Banxico), `σ ≈ 0.81` (MMP, within-period). Kin use the core distribution. See `handoff.md`.*
4b. Real earnings index `w_t` (from `data/weekly-earnings-median-race.xlsx` or CPS earnings, Mexican-born where possible), deflated with US CPI.
5. Refactor the ABM: two-stage decision, yes/no ability gate replacing the surplus budget rule, fixed `m_j`, latent-factor employment.
6. E0 calibration over the `κ` grid; diagnose the singleton and kin-only misfits.
7. Fix `τ_C`; run E1, then E2.
