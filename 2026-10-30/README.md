# Willing-and-able model with stepping-in (30 Oct 2026)

The next version of the base model, following `model_plan.tex` (the plan: Steps 0–10). **Scripts are prepared but not run yet.** Outputs will go to `outputs/` (aggregates only).

## Files

| File | What it does | Plan step |
|---|---|---|
| `model_plan.tex` | The plan: equations, parameters, calibration, tests | all |
| `willing_able.py` | Shared library: ability, desire, the four states, stepping-in with the reserve cap, receipt and transfers (exact for ρ = 0) | 1–6 |
| `calibrate_mmp.py` | MMP 2019 calibration. Stage 1: a_c, a_k (independent senders). Stage 2: b0, b_any, b_c, b_k, b_k0 and η_w, η_o, for each κ | 8 |
| `run_shocks.py` | Shock years: κ × β compatibility map, β fitted per κ, stimulus switches, ENIGH margin measures | 7, 9 (tests 1–3) |
| `run_households.py` | Paired household paths: persistence test (2018), months with receipt 2020 vs 2018 by group, who gained and lost | 9 (test 4), 10 |
| `enigh_months_received.py` | ENIGH 2018 and 2020: distribution of months with receipt among receiving households, by group, with design-based SEs | 9 (test 4), 10 |
| `check_special_cases.py` | Checks before using any result: probabilities sum to 1, nesting in the current model at κ = 0 with full desire, monotonicity in κ, exact vs simulated | – |

## Run order (when we decide to run)

```
python 2026-10-30/check_special_cases.py
python 2026-10-30/calibrate_mmp.py            # --shift 0.5 for the a_c, a_k sensitivity
python 2026-10-30/run_shocks.py               # all three stimulus rounds on, pi = 1 (base pending)
python 2026-10-30/run_shocks.py --rounds 2021-01 2021-03 --tag no-april
python 2026-10-30/run_households.py
python -I 2026-10-30/enigh_months_received.py --ing18 data/external/enigh/unzipped/enigh2018_ns_ingresos_csv/ingresos.csv --ing20 data/external/enigh/unzipped/enigh2020_ns_ingresos_csv/ingresos.csv --fx data/external/fx/fred_DEXMXUS_full.csv
```

## Outputs (to be produced)

| Script | Outputs in `outputs/` |
|---|---|
| `calibrate_mmp.py` | `stage1.csv` (a_c, a_k, check against the migrant-level M1), `calibration.csv` (one row per κ), `calibration_shift_±0.5.csv`, `calibration_settings.json` |
| `run_shocks.py` | `shocks[-tag]/map.csv`, `beta_fit.csv`, `quarterly_paths.csv`, `monthly_paths.csv`, `run_settings.json` |
| `run_households.py` | `households[-tag]/persistence_2018.csv`, `months_2020_vs_2018.csv`, `who_gains_loses.csv`, `run_settings.json` |
| `enigh_months_received.py` | `enigh_months_received.csv` |

## Reuse and dependencies

- **Population, employment, targets:** `network-compatibility-model/model/abm.py` (population, CPS rates, employment persistence), `run_e0_baseline.py` (calibration sample), `recipient-level/compare_household_logit.py` (population loader), `recipient-level/run_household_model.py` (dates, need stock, Banxico target, quarterly RMSE, spikes, the current model for the nesting check), `model/run_e1_distribution.py` (household groups).
- **ENIGH:** `identification/enigh/enigh_pre_and_monthly.py` (monthly income records, survey design).
- **Nothing in these modules is modified.**

## Open points before running

1. Stimulus base: which rounds are switched on (default in the scripts: all three, π = 1).
2. Replacement allocation when the reserve runs out (implemented: core roles first).
3. **Decided (30 Oct):** the kin coefficient b_k is constrained to b_k ≥ 0 in the calibration (L-BFGS-B), so a kin migrant who steps in never lowers receipt. The unconstrained 2019 estimate was −0.16 (not significant).
4. Runtime: not measured. The exact recursion runs for every household for every evaluation. Stage 2 evaluates it repeatedly (outer Nelder–Mead on η), and the shock map once per (κ, β) cell; `--quick` and `--kappas` restrict the grids.
