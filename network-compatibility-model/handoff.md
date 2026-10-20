# Handoff: network-compatibility model

Last updated 19 October 2026. Read this first in any new session, then `model-specification.md`.

## The project in one paragraph

Pre-COVID forecasts predicted remittances would collapse; flows to Mexico rose instead. Survey studies explain this with migrants responding to need at home and relatives stepping in for migrants who lost jobs; macro studies use aggregate drivers. This paper (one chapter of a PhD on how remittance flows and networks respond to crises; the other chapters exist already) builds a parsimonious ABM on the **actual household–migrant networks** from the MMP. It produces **compatibility maps**: which combinations of need responsiveness `β` and kin coordination `κ` reproduce the 2020–21 aggregate dynamics, and what each compatible regime implies for household resilience by network size, composition and occupational exposure. The model doubles as a policy lab (benefit access `π`, shock structure).

## Decisions already made (do not reopen without the user)

- **Few parameters.** Calibrated: `α_core`, `α_kin` (2019). Map axes: `β`, `κ`. Network-map axis: `ρ`. Scenario: `π`. Everything else external.
- **Sending is a Bernoulli draw with probability θ**, two stages: core migrants (RELHEAD 1/2/3) first, then kin, with `logit θ_kin,j = α_kin + β S_t + κ L_h,−j,t`. `L` is the routine sending lost to linked migrants unable to send, in core-sender units: 1 per eligible core migrant, `w = θ_kin⁰/θ_core⁰ ≈ 0.07` per other eligible kin ("lost sending", base model since 19 Oct 2026; spec §4.3).
- **Salience `S_t` is national only and simple:** reported COVID deaths in Mexico per month (OWID), in 10,000s; 0 in 2019 (user decisions, 20 Oct 2026). Excess deaths are the sensitivity. Components such as employment can be added later.
- **Amount separated from sending.** Each migrant has a fixed routine amount `m_j` (lognormal, mean = Banxico 2019 mean transaction). The model targets the transfer **count**; the change in mean amount is reported as a data margin.
- **Income is a yes/no condition only:** a migrant can send if employed or receiving a benefit payment that month. The old $20,000 allowance / 20%-of-surplus rule is dropped.
- **Benefits (UI supplements, stimulus checks) only act through that condition.** Eligibility `π` is a scenario.
- **Wages:** not in the base model. Wage-indexed amounts are a robustness check only.
- **Scope: 2020–21.** 2022 is held out; the model is expected to undershoot it (migrant-stock growth, wages, formalization).
- **Results are maps, not a single best fit.** Acceptance tolerance `τ_C` = 0.04, quarterly log RMSE over 2020Q2–2021Q4 against Banxico relative to its 2017–19 trend (user decision, 20 Oct 2026).

## Folder layout

```
network-compatibility-model/
  handoff.md                 this file
  model-specification.md     formal spec (ODD-style), sections referenced as §
  data-prep/
    build_population.py      household-migrant population from MMP
    population/              its microdata outputs (git-ignored)
    cps_employment_rates.py  task 1
    remittance_amounts.py    task 4
  model/
    abm.py                   model library (tasks 5)
    run_e0_baseline.py       E0 calibration (task 6)
    run_e1_job_loss.py       E1a (β = 0, no benefits), base rule
    run_e1_rule_comparison.py   E0 refit + E1a under the former 'all_core' rule
    diagnose_e1_kappa.py     why κ did little under the former rule (April 2020)
    run_e2_map.py            E2 first pass: β × κ map, count criterion only
  outputs/e0-baseline/       aggregate E0 tables (base rule)
  outputs/e1-job-loss/       E1a tables (base rule)
  outputs/e1-all-core-rule/  former rule: E0 refit, E1a, κ diagnostics, rule comparison
  outputs/e2-map/            E2 first pass (map.csv, quarterly_paths.csv)
  model/make_figures.py      PNG figures (plan: figures-plan.md); --e1 / --e1-rules for E1 only
  figures/                   A1-A4 population, B1-B2 inputs, C1-C5 calibration, D1-D6 E1a,
                             R1-R6 rule comparison; captions.md
    outputs/                 aggregate CPS inputs (safe to commit)
```

Older work lives in `oct-26/` (earlier ABM `oct-26/simulations/abms/remittance-minimal-cps-abm.py`, MMP network construction `oct-26/simulations/synthetic-migrants-house-networks.py`, Banxico/ENIGH/MMP descriptives). `oct-26/research-questions-literature-review.md` has the audit of that ABM and 30 annotated references; its research questions are superseded by the spec.

## Task status (spec §9)

| # | Task | Status |
|---|---|---|
| 1 | CPS occupation-month employment rates + persistence `φ` | **Done, with Mexican-origin fallback** (see below) |
| 2 | Salience index `S_t` (reported COVID deaths in Mexico) | **Done** (`data-prep/salience_inputs.py`, see below); loaded by `abm.salience_series` |
| 3 | Benefit schedule table; external range for `π` | Not started |
| 4 | Amount dispersion `σ` (MMP/ENIGH); Banxico 2019 mean transaction | **Done** (see below) |
| 4b | Real earnings index `w_t` | Not started |
| 5 | ABM refactor (two-stage decision, ability gate, fixed `m_j`, latent-factor employment) | **Done** (`model/abm.py`) |
| 6 | E0 calibration; diagnose singleton / kin-only misfits | **Done, first pass** (`model/run_e0_baseline.py`); two open decisions below |
| 7 | Fix `τ_C`; run E1, then E2 | `τ_C` = 0.04. **E1a done** (`π = 0`, base lost-sending rule; former rule kept for comparison). **E2 first pass done** (`π = 0`, count criterion only): no compatible cell. E1b/E2 with benefits wait for task 3; the ENIGH criterion is not built |

### Recommended order from here

1. **Task 5 (ABM refactor) + task 6 (E0, 2019 calibration).** Needs only finished inputs: in 2019 `S_t = 0` and no benefits. This step carries the biggest risk (baseline misfits).
2. **Task 3 (benefit schedule, `π` range)**, then **E1** (`β = 0`: job losses on real networks only). E1 is the early read on which paper this is.
3. **Task 2 (salience `S_t`)**, fix `τ_C`, then **E2** (β × κ map).
4. Task 4b (wage index) last: robustness only.
5. User action, any time: new IPUMS CPS extract with `BPL`, then rerun task 1.

### Task 1 results

Run: `python network-compatibility-model/data-prep/cps_employment_rates.py` (~1 min; reads `data/cps_00001.dat`, 3 GB, not in git).

Outputs in `data-prep/outputs/`:
- `cps_employment_rates_monthly.csv`: one row per occupation group × month, 2017-01 to 2023-12. **Use `rate_model`** (raw where the cell is reliable, centred 3-month smooth otherwise; in practice only some Agriculture months are smoothed). `rate_exit_adjusted` treats labour-force exits since 2019 as job loss (sensitivity). Ages 18–64.
- `cps_employment_persistence_2019.csv`: 2019 month-to-month flows and fitted `φ` by occupation.
- `cps_employment_run_settings.json`: settings, population label, recommended `φ`.

Key numbers (Mexican origin, all occupations):
- `rate_model`: 0.954 (Feb 2020) → 0.940 (Mar) → **0.825 (Apr)** → 0.843 (May) → 0.891 (Jul) → 0.925 (Oct 2020) → ~0.95 by Oct 2021. Sales/personal services is hit hardest (0.763 in Apr 2020).
- Exit-adjusted: 0.780 in Apr 2020 (participation fell from ~0.76 to 0.71).
- **`φ ≈ 0.91` pooled** (0.84–0.93 by occupation). Observed 2019 job-loss hazard 1.25%/month and job-finding 41%/month; the latent AR(1) reproduces job loss exactly by construction and gives job-finding 36% (close enough for a single parameter). Raw CPS flows include misclassification, so true persistence is probably somewhat higher.

### Open issue from task 1: Mexican-born population

`cps_00001` has **no birthplace variable** (no `BPL`, `NATIVITY` or `CITIZEN`), so outputs are for **Mexican origin**, which includes US-born people. To fix: the user downloads a new IPUMS CPS extract (same samples and variables, plus `BPL`, optionally `NATIVITY`/`CITIZEN`), saves the codebook, and updates `DATA` / `CODEBOOK` at the top of the script. The script then switches to `BPL == 20000` automatically and labels outputs "Mexican-born". Expect smaller cells (roughly half), so more Agriculture months will fall back to smoothing.

### Task 2: salience inputs and `S_t` (19–20 Oct 2026)

Run: `python network-compatibility-model/data-prep/salience_inputs.py` (seconds). Output: `data-prep/outputs/salience_inputs_monthly.csv` (2019–2022; public aggregates). Raw files are in `data/` (git-ignored); the script docstring has the download URLs.
- **Employed population 15+ (monthly, not seasonally adjusted):** OECD infra-annual labour statistics (monthly ENOE/ENOEN, employed = labour force − unemployed). There is no open monthly INEGI file without an API token; FRED has Mexico only quarterly. ILOSTAT's monthly series starts in Jul 2020.
  - **Apr–Jun 2020 are missing everywhere** (ENOE suspended). They are patched from INEGI's ETOE telephone-survey releases: April 43.4 M, May 43.6 M, June 48.3 M. ETOE is calibrated to ENOE Q1 2020 but is not strictly comparable.
  - Employment loss vs the same month of 2019: 19.4% (Apr 2020), 20.1% (May), 12.5% (Jun), 9.3% (Jul), ~4–6% (Oct–Dec 2020), 1–3% (Q1 2021), ≤ 0 from Apr 2021. It is negative in 2022 because the comparison has no trend.
  - INEGI switched to a new population frame from Apr 2021, so part of the Apr 2021 jump may be rebasing.
- **Excess deaths:** World Mortality Dataset weekly all-cause deaths (Karlinsky & Kobak), converted to calendar months.
  - **Use the trend baseline:** the 2015–19 same-week mean leaves ~+3,000 "excess" a month in 2019, because mortality was rising. With a linear trend, 2019 is ≈ 0.
  - Peaks: Jan 2021 (~104k excess, trend baseline), Jul 2020 (~57k), Aug 2021 (~58k), Dec 2020 (~53k).
- **Reported COVID deaths (OWID):** about 37–41% of excess deaths (trend baseline) at the peaks. **Decision (user, 20 Oct 2026): excess deaths (trend baseline) are the base component; reported COVID deaths are the sensitivity** (spec §4.3 updated).
- **`S_t` simplified (user decisions, 20 Oct 2026):** `S_t` = reported COVID deaths (OWID) / 10,000, and 0 in 2019. A first version averaged standardized employment loss and excess deaths; it was dropped as over-engineered. An excess-deaths-only version was then replaced by reported COVID deaths (user choice). Output: `data-prep/outputs/salience_index_monthly.csv` (`S`; variant `S_excess_deaths`); figure B3.
- **`S_t`:** peaks 2.4 (Jul 2020), 3.9 (Jan 2021), 2.2 (Aug 2021), 1.1 (Feb 2022). Annual means 1.23 (2020), 1.30 (2021), 0.23 (2022, held out). The excess-deaths variant is about 2.2 times larger (means 2.76 / 2.85 / 0.55).
- **For E2:** β is in logit units per 10,000 COVID deaths a month.
- **Preview: β needed to match Banxico vs trend, quarter by quarter** (exact counts, E0 alphas; a quick calculation, not yet a script):

  | Quarter | Mean `S_t` | Banxico vs trend | Extra transfers needed (κ = 4) | β needed (κ = 4) | β needed (κ = 0) |
  |---|---|---|---|---|---|
  | 2020Q2 | 1.33 | 0.974 | 0% | 0 | 0.16 |
  | 2020Q3 | 1.88 | 1.025 | +3% | 0.03 | 0.09 |
  | 2020Q4 | 1.71 | 1.017 | +2% | 0.03 | 0.07 |
  | 2021Q1 | 2.57 | 1.049 | +5% | 0.04 | 0.06 |
  | 2021Q2 | 0.51 | 1.058 | +6% | 0.23 | 0.35 |
  | 2021Q3 | 1.58 | 1.083 | +8% | 0.10 | 0.13 |
  | 2021Q4 | 0.56 | 1.108 | +11% | 0.37 | 0.39 |

  The needed β is not constant: it is 10× larger in the low-death quarters 2021Q2 and Q4. The best single β is 0.05 at κ = 4 (quarterly log RMSE 0.044 vs 0.059 at β = 0) and 0.10 at κ = 0 (0.047 vs 0.090). Both are above the proposed `τ_C` = 0.03. Deaths alone overshoot early 2021 and undershoot late 2021. As a probability: β = 0.05 with `S_t` = 2.6 raises the core sending probability from 0.56 to about 0.59.
- The employment series (OECD + ETOE patch) stays in `salience_inputs_monthly.csv` in case a labour-market component is added later.
- Optional: an INEGI API token (free, register with an email on the API page) would let us pull the ENOE series from INEGI directly. The OECD series is the same INEGI data.

### Task 4 results

Run: `python network-compatibility-model/data-prep/remittance_amounts.py` (seconds; reads MMP DTA files, Banxico parquet, ENIGH 2018, DEXMXUS).

Outputs in `data-prep/outputs/`:
- `remittance_amounts.json`: **model inputs**. `m_j ~ LogNormal(μ = 5.46, σ = 0.81)`, mean $326 (Banxico 2019 USD per transaction), median $236; plus MMP and ENIGH diagnostics.
- `remittance_amounts_mmp_diagnostics.csv`: amounts by trip period, role, occupation group, documentation.
- `remittance_amounts_mmp_sample.csv`: positive / zero / unknown counts by source file.

Data used: MMP `remit` = average monthly remittances on the last US trip. It's in `mig174` (household heads) and `migother174` (other migrants, linked to `pers174` for `relhead`: almost all children, some spouses). Trips 2000–2019, deflated with US CPI-U, top 1% trimmed: n = 1,289 positive amounts.

Findings:
- σ is stable: 0.81 within 5-year periods, 0.81 pooled, 0.88 robust (IQR).
- Little heterogeneity by occupation (median ratios 0.83–1.09) or documentation (undocumented 0.98 vs documented 1.20). This supports one distribution for all migrants.
- **Children send less than heads** (median $238 vs $395, ratio 0.66). All are core, so the base model ignores this. It's a possible refinement if household-level fit needs it.
- Kin amounts cannot be calibrated (~10 extended-kin observations). Decision: kin use the core distribution.
- 17% of MMP migrants with a known answer sent nothing on their last trip. That is relevant to baseline θ, not to amounts.
- **Level: Banxico $326, decided by the user.** MMP's mean ($497) is higher, but MMP `remit` is a self-reported, often retrospective or proxy-reported amount for the whole last trip, deflated here from the trip start year (median trip 18 months for heads, 54 for others) and dominated by heads and 2000s trips. Do not read the ratio as transactions per month. MMP is used for σ only.
- **ENIGH 2018 (validation only):** 4.7% of households receive; recipients' median $76/month, mean $145/month (quarterly MXN at 19.22 MXN/USD); log-sd 1.35. ENIGH's national total is only **8% of Banxico's** 2018 total. The user knows ENIGH and Banxico disagree; treat this as measurement, not model misfit.

### Tasks 5-6 results (ABM + E0)

Code: `model/abm.py` (population, employment process, analytic receipt, simulation) and `model/run_e0_baseline.py` (about 20 s). Outputs in `outputs/e0-baseline/`: `run_settings.json`, `fit_network_type.csv`, `validation_network_size.csv`, `validation_composition.csv`, `kappa_2019.csv`, `enigh_style_check.csv`.

Implementation notes:
- Population (first E0 run only; since replaced, see "Population rebuilt"): `oct-26/outputs/synthetic-migrants-house-networks/`, 1,019 resampled households, 10,000 migrants. 'Not working' migrants and under-18s can never send. 'Unknown' and 'Working student/pensioner' use the all-occupation CPS rate.
- Employment: latent AR(1), `φ` from task 1, stationary start (no burn-in needed). Benefits are implemented in `simulate` but off in 2019 (`π`, UI/stimulus months unset).
- Fit uses the exact expected monthly receipt probability (`ρ = 0`), so there is no simulation noise in calibration. Stochastic runs match it (0.231 vs 0.232).
- MMP `remus` is compared with the monthly receipt probability (same convention as the old ABM; not yet checked against the MMP questionnaire wording).

Results:
- `α_core = 0.25` (θ = 0.56), `α_kin = −4.84` (θ = 0.008). Fit by network type within 1.2 pp: core only 60.8% vs 62.0% MMP, kin only 6.0% vs 6.1%, mixed 65.6% vs 64.7%.
- Singleton core households: 47.5% vs 54.5% MMP (−7 pp). Better than the old ABM (~42%), not fixed.
- Kin-only with older residents: 5.2% vs 13.6% (−8 pp). Unchanged from the old ABM; the model has no household-need terms.
- **Receipt saturates with network size in MMP but not in the model.** Kin-only: MMP 8.7% (10–19 migrants) and 8.8% (20+); model 8.1% and 18.5%. Mixed 20+: MMP 56% vs model 77%. Independent kin opportunities overstate protection in large networks.
- *(Old `− κ N_h` rule, superseded below.)* κ barely mattered: kin transfers 57.5/month at `κ = 0` vs 50.5 at `κ = 4`, out of ~320.
- ENIGH-style six-month window: 46% of connected households receive at least once; among them median $120, mean $230 per month (ENIGH 2018: $76 / $145, which under-reports).
- Banxico scale: ~29,300 national transactions per simulated transfer (2019).

### Decisions taken 19 Oct 2026 after the first E0 run

1. **Stepping-in rule changed (user approved).** Now `logit θ_kin = α_kin + β S_t + κ U_h,t`, where `U_h,t` = 1 if the household has eligible core migrants and none can send this month. α is refitted for each κ. Rerun results:
   - 2019 barely constrains κ: weighted RMSE 0.33 pp at κ = 0, 0.60 at κ = 4, 0.76 at κ = 8 (`kappa_profile.csv`);
   - ~9 households a month are triggered in 2019; kin transfers in triggered households are 0.3 a month at κ = 0 and 9.7 at κ = 4 (out of ~325 transfers);
   - analytic and simulated receipt agree (0.232 / 0.231).
   - The size and composition results above are essentially unchanged.
2. **Size saturation: no dilution parameter (user decision).** Keep the model; report saturation as a finding and a limitation.

### Population audit (19 Oct 2026, read-only review of `oct-26/simulations/synthetic-migrants-house-networks.py`)

- Households are drawn **with replacement** from the 3,405 connected MMP households (interviewed in Mexico 2010–19) until exactly 10,000 migrants are reached: 1,019 draws, only 866 distinct households. The household mix matches the source well, but the calibration uses only ~30% of the available source households and duplicates add noise.
- Core = PERS roster head/spouse/child currently on a US trip (590 agents: 544 children). Kin come almost entirely from HOUSE "in US" flags for parents and siblings, plus **counts** of uncles, cousins, nephews and in-laws in the US: 33% nieces/nephews, 32% cousins, 11% uncles/aunts, only 10% siblings. 54% of agents are in the 15% of households with 20+ links.
- **Saturation is explained by distant kin.** In MMP kin-only households, receipt rises with close kin in the US (siblings/parents: 4.2% none, 4.8% one, 9.5% two, 10.9% three or more) but not with distant kin (2.8%, 5.0%, 7.8%, 4.4% for 1–4, 5–9, 10–19, 20+ distant only).
- Kin profiles (occupation, age) are copied from roster donors, 90% of whom are household heads' children. So kin occupational exposure mirrors the core mix: differences across households are imputation noise, not data. ρ and occupational concentration can only be sensitivity dimensions.
- Links are measured at interview (survey years 2010–19, median ~2015), not in 2019. Receipt varies strongly by survey year (3% to 27%).
- 'Not working' (17%) is mostly code 20 (probably housework; unverified), mainly spouses and daughters.

### Population rebuilt (user approved, 19 Oct 2026)

`data-prep/build_population.py` (~15 s) writes `data-prep/population/` (households.parquet, migrants.parquet, run_settings.json; **MMP microdata, git-ignored**) and an aggregate `data-prep/outputs/population_summary.csv`. **The ABM now uses this population. The oct-26 synthetic population is no longer used.**
- Every connected MMP household once: 3,405 households (Mexico interviews 2010–19), 7,999 links. No resampling.
- Links:
  - **Core:** roster head/spouse/child currently in the US (1,996).
  - **Close kin:** HOUSE parent/sibling slots plus other roster relatives currently in the US (3,581).
  - **Distant kin:** HOUSE counts of uncles, cousins, nephews and in-laws, collapsed into **one aggregated link** per household (2,422 households).
  - Maximum-overlap rule between roster and questionnaire, as before.
  - Largest network: 16 links (was 169).
- Profiles for HOUSE links are copied from roster donors (relationship pool if ≥ 20 donors, else non-core kin, else origin state, else all; parent age bounds; known sex kept). 5,985 profiles are imputed, labelled `profile_imputed`. Parents with no compatible donor keep an unknown profile.
- MMP check (`population_summary.csv`): among households without core migrants, receipt rises with close kin (0: 1.5%, 1: 5.6%, 2: 9.8%, 3–4: 10.5%, 5+: 13.9%). Households with only distant kin receive 2.8–7.6%.

E0 on the new population (`outputs/e0-baseline/`, same 3,252 target households as before, now simulated one-to-one):
- κ = 0: `α_core = 0.31` (θ = 0.58), `α_kin = −3.18` (θ = 0.040); weighted RMSE 0.20 pp, 0.60 at κ = 4, 0.71 at κ = 8 (base lost-sending rule; former rule 0.48 / 0.56). So 2019 still barely constrains κ.
- **Saturation largely fixed** for kin-only households: size 1 / 2 / 3–4 / 5–9 give 3.2 / 6.5 / 10.3 / 16.5% model vs 3.8 / 6.1 / 9.7 / 15.9% MMP. All link types are within ~1 pp.
- Remaining size misfits come from core networks: core-only 3–4: +8 pp; core+kin 2: −12 pp; core+kin 5–9: +11 pp. Core-only singletons: −5 pp (49% vs 55%).
- Survey period (untargeted): 2010–13 17.2% vs 18.0%; 2014–16 23.9% vs 22.2%; 2017–19 29.2% vs 32.3%. Much of the rise in receipt over time is reproduced by network composition alone.
- Composition misfits persist (no household-need terms): kin-only with older residents 5.5% vs 13.6%; core+kin with children 59% vs 71%.
- 2019 (base rule): ~1,096 transfers/month (1,113 at κ = 4); 304 households a month have some linked migrant unable (L > 0); kin transfers in those households 17.8 (κ = 0) to 70.6 (κ = 4). Banxico scale ≈ 8,690 at κ = 0. ENIGH-style six-month window: 48% receive; median $110, mean $215 per month.

Still open from the audit: kin occupations are imputed (ρ / occupational concentration are sensitivity dimensions only); optional 2015–19 calibration sensitivity.

### E1a results (job loss only, base lost-sending rule, 19 Oct 2026)

Code: `model/run_e1_job_loss.py` (~90 s); outputs in `outputs/e1-job-loss/`. `β = 0`, `ρ = 0`, `π = 0` (no benefits, user decision: E1b adds them after task 3). E0 alphas for each κ in {0, …, 4}. Runs 2019-01 to 2022-12; the counterfactual uses the 2019 CPS rates for the same calendar month, with common random numbers.
- **Comparison (user decisions, 19–20 Oct 2026):** monthly and quarterly (quarters smooth monthly noise). Model = COVID run / no-COVID run. **Banxico = transactions / their 2017–19 trend** (log-linear with month effects, +6.3% a year; residual SD 0.019 monthly, 0.014 quarterly). The raw index (same period of 2019) is kept as `index_banxico_raw`. Fit windows 2020-04..2021-12 and 2020Q2..2021Q4; 2022 held out. `τ_C` is not fixed, so RMSE is reported only.
- Aggregate results use the exact expected counts (`abm.expected_flows`, exact for `ρ = 0`, benefits included). Household outcomes come from 50 paired stochastic runs. Simulated and exact counts agree within 0.7% (mean over the 50 runs).
- **Banxico vs trend:** 0.91 in Apr 2020 (1.09 in the March record month); quarterly 2020Q2 0.974, Q3 1.025, Q4 1.017, 2021Q1 1.049, Q2 1.058, Q3 1.083, Q4 1.108 (2022: 1.07–1.13). Annual mean 1.008 (2020), 1.074 (2021). Relative to trend, **2020 is roughly flat; the excess is a 2021 phenomenon.**
- **κ = 0:** trough 0.853 (Apr 2020); quarterly 2020Q2 0.876, 2020Q4 0.958, 2021Q4 0.994. Annual mean 0.943 (2020), 0.978 (2021). Quarterly log RMSE vs trend 0.090 (0.189 vs the raw index).
- **κ = 4:** April 2020 index 0.966 (Banxico vs trend 0.91; raw 0.96). Quarterly 2020Q2 0.975, Q3 0.994, then ~1.0. Annual mean 0.991 (2020) and 0.998 (2021). Quarterly log RMSE vs trend 0.059 (0.159 vs raw). **2020Q2 matches Banxico (0.975 vs 0.974).** In April 2020, kin transfers *fall* by 0.2 per lost core transfer at κ = 0 (kin lose jobs too) and rise by 0.71 at κ = 4. The index reaches 1.0 at κ ≈ 6 and 1.012 at κ = 8.
- **Read:** job loss on real networks only lowers counts. With strong stepping-in (κ ≈ 4) the model matches the 2020 path relative to trend. It produces no 2021 excess (model ~1.00 vs Banxico 1.05–1.11), because CPS employment recovers. The 2021 rise must come from `β`, benefits (E1b), or factors outside the model.
- **Households (κ = 0):** USD received drops ~4.3% over 2020-04..2021-12, nearly flat across network types and sizes. Group receipt bottoms out in Apr 2020 (kin-only 0.86, households with core 0.90 of counterfactual) and stays below 99% of the counterfactual until Oct 2021. Larger networks are less often interrupted (size 1: 4.5% of counterfactual receipt months; 10+: 1.6%).
- **Households (κ = 4):**
  - Core + kin households barely dip (trough 0.998).
  - Kin-only: trough 0.92.
  - Core-only, single-link and distant-kin-only households are unchanged: nobody can step in.
  - USD drop by size: 1 → 4.2%, 2 → 3.0%, 3–4 → 1.1%, 5–9 → −1.3%, 10+ → −8.2% (n = 30). Large networks receive *more* than without COVID: several kin each step in for one loss (over-replacement), a property to report.

### Why κ did little under the former rule (19 Oct 2026, `model/diagnose_e1_kappa.py`)

The former rule was `κ U_h,t`, with `U` = 1 if all eligible core migrants are unable. There was no implementation bug: exact and simulated results agree, and an independent re-implementation gives the same April 2020 index (0.889 vs 0.888 at κ = 4). The rule was narrow by design:
- **Who can trigger stepping-in:** only the 929 households with an eligible core migrant (adult, not 'Not working'). 2,393 households (70%) are kin-only and never trigger. 83 more have only core migrants who never send (338 core coded 'Not working', mostly spouses and daughters, plus 6 under 18). Only 484 households have eligible core and eligible kin.
- **All-or-nothing trigger:** it fires only when *all* eligible core migrants are unable. That is rare in the 387 households with two or more (about 3% for two in April 2020 at ρ = 0). Partial losses get no response.
- **Ceiling:** April 2020 loses 161 transfers (134 core, 27 kin); only 75 of the core losses are in households with kin. At κ → ∞, kin recover at most 50, so the April index saturates at 0.904 (0.888 at κ = 4).
- **Simulating only households with core links changes nothing:** they already make 86% of transfers. At κ = 0 every subset falls to 0.853, because kin occupations are copied from core donors. The κ = 4 index is 0.893 vs 0.888. Dropping kin-only households would cost the identification of α_kin and the kin-only size gradient.

### Lost-sending rule adopted as the base model (user decision, 19 Oct 2026)

`abm.Params.rule` now defaults to `'lost_sending'`; `rule='all_core'` keeps the former rule for comparison. E0, E1a and all figures were rerun under the new rule. Kin step in for core *and* for other kin, in proportion to the routine sending lost:
```
logit θ_kin,j = α_kin + β S_t + κ L_h,−j,t
L = (# eligible core unable) + w · (# other eligible kin unable),   w = θ_kin⁰ / θ_core⁰ ≈ 0.07
```
A single-core household losing its core migrant has L = 1, as under the base rule. Only kin raise θ; core migrants do not step in. The rule has an exact receipt and flow calculation for ρ = 0 (`abm.lost_sending_exact`, Poisson-binomial over the numbers of unable core and kin). Exact and simulated results agree: 2019 receipt 0.2311 vs 0.2312; April 2020 transfers 1,107.5 vs 1,106.4.

Order to rerun everything: `run_e0_baseline.py` (~4 min), `run_e1_job_loss.py` (~2 min), `run_e1_rule_comparison.py` (~3 min), `diagnose_e1_kappa.py`, then `make_figures.py` and `make_figures.py --e1-rules`. On Windows a PNG write occasionally fails with `OSError: [Errno 22]` (file held by another program); just rerun.

New rule vs former rule (κ = 4; κ = 0 is identical under both):

| | Lost sending (base) | All core unable (former) |
|---|---|---|
| 2019 fit error (pp), κ = 4 / 8 | 0.60 / 0.71 | 0.48 / 0.56 |
| Households with L > 0 (or U = 1) per month, 2019 | 304 | 29 |
| April 2020 index (Banxico vs trend 0.91) | 0.966 | 0.888 |
| April 2020 index as κ → large | > 1 (1.012 at κ = 8) | ceiling 0.904 |
| Kin transfers added per lost core transfer, Apr 2020 | 0.71 | 0.07 |
| Annual mean index 2020 / 2021 (Banxico vs trend 1.008 / 1.074) | 0.991 / 0.998 | 0.957 / 0.984 |
| Kin-only households, April trough | 0.92 | 0.86 (as κ = 0) |

The network-size and link-type fit for 2019 is the same under both rules within ~1 pp (except cells with n < 20). Under the new rule κ moves the 2020 aggregate, so E2's β × κ map will not be flat along κ.

### E2 first pass (20 Oct 2026, `model/run_e2_map.py`, ~5 min)

`π = 0`, `ρ = 0`, base lost-sending rule, `S_t` = reported COVID deaths / 10,000. Grid: β 0–0.40 (step 0.02) × κ {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8}, with the E0 alphas for each κ; 210 cells, exact expected counts. Criterion: quarterly log RMSE, 2020Q2–2021Q4, model (COVID / no-COVID run) vs Banxico / 2017–19 trend, `τ_C` = 0.04. The ENIGH household criterion is not implemented, so these are count-only results. Figures M1 (map) and M2 (paths).
- **No cell is count-compatible at `τ_C` = 0.04.** Best: κ = 3, β = 0.06, RMSE 0.043. A valley of near-fits runs from (κ = 0, β = 0.10) to (κ = 8, β = 0.04), with RMSE 0.043–0.049: β and κ trade off, since both raise 2020 sending.
- At β = 0 the RMSE is 0.090 (κ = 0) to 0.055 (κ ≥ 6). From β = 0.16 up, every κ fits worse than β = 0 at κ = 4 (0.059).
- **Why it fails** (best cell vs Banxico vs trend):
  - The model overshoots 2020Q3–2021Q1 (1.04 / 1.04 / 1.07 vs 1.03 / 1.02 / 1.05).
  - It undershoots 2021Q2–Q4 (1.00 / 1.04 / 1.02 vs 1.06 / 1.08 / 1.11).
  - COVID deaths come in waves (low in 2021Q2 and Q4), while the excess rises steadily. 2022 (held out) is missed by 7–13 points.
- **Read:** with deaths as the only need signal and no benefits, the network model reproduces the 2020 dip (job loss and stepping-in) but not the steady 2021 rise. The need response it implies is small: β ≈ 0.06 raises a core migrant's monthly sending probability from about 0.56 to 0.62 at the Jan 2021 peak. Candidates for the steady component: benefits (E1b, task 3), migrant-stock growth, wages, formalization. Possible next tests: benefits; a smoother or cumulative need signal; ENIGH criterion.

## Things to keep in mind

- The repository is **public**. Never commit `data/` or microdata-derived outputs (MMP/IPUMS terms forbid redistribution). `oct-26/outputs/` (321 MB, includes MMP-derived panels) is git-ignored for the same reason.
- Known baseline misfits from the old ABM: singleton core households receive in 54% of MMP cases vs ~42% simulated; kin-only households with older residents 13.5% vs ~4.5%. Check whether the simplified model fixes or worsens these (task 6).
- March 2020 (record month, before salience moved) is expected to be unexplained. Do not add a parameter for it.
- The user prefers simplicity: propose extensions as optional robustness checks, not base-model changes.
