# Handoff: network-compatibility model

Last updated 19 October 2026. Read this first in any new session, then `model-specification.md`.

## The project in one paragraph

Pre-COVID forecasts predicted remittances would collapse; flows to Mexico rose instead. Survey studies explain this with migrants responding to need at home and relatives stepping in for migrants who lost jobs; macro studies use aggregate drivers. This paper (one chapter of a PhD on how remittance flows and networks respond to crises; the other chapters exist already) builds a parsimonious ABM on the **actual household–migrant networks** from the MMP. It produces **compatibility maps**: which combinations of need responsiveness `β` and kin coordination `κ` reproduce the 2020–21 aggregate dynamics, and what each compatible regime implies for household resilience by network size, composition and occupational exposure. The model doubles as a policy lab (benefit access `π`, shock structure).

## Decisions already made (do not reopen without the user)

- **Few parameters.** Calibrated: `α_core`, `α_kin` (2019). Map axes: `β`, `κ`. Network-map axis: `ρ`. Scenario: `π`. Everything else external.
- **Sending is a Bernoulli draw with probability θ**, two stages: core migrants (RELHEAD 1/2/3) first, then kin, with `logit θ_kin = α_kin + β S_t − κ N_h,t`.
- **Salience `S_t` is national only:** ½ Mexican employment loss (employed population, not unemployment rate) + ½ deaths (excess deaths as sensitivity), standardized to mean 0 in 2019.
- **Amount separated from sending.** Each migrant has a fixed routine amount `m_j` (lognormal, mean = Banxico 2019 mean transaction). The model targets the transfer **count**; the change in mean amount is reported as a data margin.
- **Income is a yes/no condition only:** a migrant can send if employed or receiving a benefit payment that month. The old $20,000 allowance / 20%-of-surplus rule is dropped.
- **Benefits (UI supplements, stimulus checks) only act through that condition.** Eligibility `π` is a scenario.
- **Wages:** not in the base model. Wage-indexed amounts are a robustness check only.
- **Scope: 2020–21.** 2022 is held out; the model is expected to undershoot it (migrant-stock growth, wages, formalization).
- **Results are maps, not a single best fit.** Fix the acceptance tolerance `τ_C` before running E2.

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
  outputs/e0-baseline/       aggregate E0 tables
  model/make_figures.py      PNG figures (plan: figures-plan.md)
  figures/                   A1-A4 population, B1-B2 inputs, C1-C5 calibration; captions.md
    outputs/                 aggregate CPS inputs (safe to commit)
```

Older work lives in `oct-26/` (earlier ABM `oct-26/simulations/abms/remittance-minimal-cps-abm.py`, MMP network construction `oct-26/simulations/synthetic-migrants-house-networks.py`, Banxico/ENIGH/MMP descriptives). `oct-26/research-questions-literature-review.md` has the audit of that ABM and 30 annotated references; its research questions are superseded by the spec.

## Task status (spec §9)

| # | Task | Status |
|---|---|---|
| 1 | CPS occupation-month employment rates + persistence `φ` | **Done, with Mexican-origin fallback** (see below) |
| 2 | Salience index `S_t` (ENOE/ETOE employed population + national/excess deaths) | Not started |
| 3 | Benefit schedule table; external range for `π` | Not started |
| 4 | Amount dispersion `σ` (MMP/ENIGH); Banxico 2019 mean transaction | **Done** (see below) |
| 4b | Real earnings index `w_t` | Not started |
| 5 | ABM refactor (two-stage decision, ability gate, fixed `m_j`, latent-factor employment) | **Done** (`model/abm.py`) |
| 6 | E0 calibration; diagnose singleton / kin-only misfits | **Done, first pass** (`model/run_e0_baseline.py`); two open decisions below |
| 7 | Fix `τ_C`; run E1, then E2 | Not started |

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
- κ = 0: `α_core = 0.31` (θ = 0.58), `α_kin = −3.18` (θ = 0.040); weighted RMSE 0.20 pp, 0.48 at κ = 4, 0.56 at κ = 8. So 2019 still barely constrains κ.
- **Saturation largely fixed** for kin-only households: size 1 / 2 / 3–4 / 5–9 give 3.2 / 6.5 / 10.3 / 16.5% model vs 3.8 / 6.1 / 9.7 / 15.9% MMP. All link types are within ~1 pp.
- Remaining size misfits come from core networks: core-only 3–4: +8 pp; core+kin 2: −12 pp; core+kin 5–9: +11 pp. Core-only singletons: −5 pp (49% vs 55%).
- Survey period (untargeted): 2010–13 17.2% vs 18.0%; 2014–16 23.9% vs 22.2%; 2017–19 29.2% vs 32.3%. Much of the rise in receipt over time is reproduced by network composition alone.
- Composition misfits persist (no household-need terms): kin-only with older residents 5.5% vs 13.6%; core+kin with children 59% vs 71%.
- 2019: ~1,096 transfers/month, 29 triggered households/month; triggered kin transfers 0.9 (κ = 0) to 15.4 (κ = 4). Banxico scale ≈ 8,690. ENIGH-style six-month window: 48% receive; median $110, mean $215 per month.

Still open from the audit: kin occupations are imputed (ρ / occupational concentration are sensitivity dimensions only); optional 2015–19 calibration sensitivity.

## Things to keep in mind

- The repository is **public**. Never commit `data/` or microdata-derived outputs (MMP/IPUMS terms forbid redistribution). `oct-26/outputs/` (321 MB, includes MMP-derived panels) is git-ignored for the same reason.
- Known baseline misfits from the old ABM: singleton core households receive in 54% of MMP cases vs ~42% simulated; kin-only households with older residents 13.5% vs ~4.5%. Check whether the simplified model fixes or worsens these (task 6).
- March 2020 (record month, before salience moved) is expected to be unexplained. Do not add a parameter for it.
- The user prefers simplicity: propose extensions as optional robustness checks, not base-model changes.
