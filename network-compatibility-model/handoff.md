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
    cps_employment_rates.py  task 1
    outputs/                 aggregate CPS inputs (safe to commit)
```

Older work lives in `oct-26/` (earlier ABM `oct-26/simulations/abms/remittance-minimal-cps-abm.py`, MMP network construction `oct-26/simulations/synthetic-migrants-house-networks.py`, Banxico/ENIGH/MMP descriptives). `oct-26/research-questions-literature-review.md` has the audit of that ABM and 30 annotated references; its research questions are superseded by the spec.

## Task status (spec §9)

| # | Task | Status |
|---|---|---|
| 1 | CPS occupation-month employment rates + persistence `φ` | **Done, with Mexican-origin fallback** (see below) |
| 2 | Salience index `S_t` (ENOE/ETOE employed population + national/excess deaths) | Not started |
| 3 | Benefit schedule table; external range for `π` | Not started |
| 4 | Amount dispersion `σ` (MMP/ENIGH); Banxico 2019 mean transaction | Not started |
| 4b | Real earnings index `w_t` | Not started |
| 5 | ABM refactor (two-stage decision, ability gate, fixed `m_j`, latent-factor employment) | Not started |
| 6 | E0 calibration over the `κ` grid; diagnose singleton / kin-only misfits | Not started |
| 7 | Fix `τ_C`; run E1, then E2 | Not started |

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

## Things to keep in mind

- The repository is **public**. Never commit `data/` or microdata-derived outputs (MMP/IPUMS terms forbid redistribution). `oct-26/outputs/` (321 MB, includes MMP-derived panels) is git-ignored for the same reason.
- Known baseline misfits from the old ABM: singleton core households receive in 54% of MMP cases vs ~42% simulated; kin-only households with older residents 13.5% vs ~4.5%. Check whether the simplified model fixes or worsens these (task 6).
- March 2020 (record month, before salience moved) is expected to be unexplained. Do not add a parameter for it.
- The user prefers simplicity: propose extensions as optional robustness checks, not base-model changes.
