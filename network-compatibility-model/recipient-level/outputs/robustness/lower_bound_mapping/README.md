# Robustness 6g: count mapping (base vs lower bound vs mixed)

Script: `network-compatibility-model/recipient-level/robustness/lower_bound_mapping.py` (29 Oct 2026, ~10 min).

## Mappings and what they assume about stepping-in

With r_h,t(beta) = E[able senders | someone sends] and w in [0, 1]:

    counts_t(w) = sum_h P_h,t [ w r_h,t(beta) + (1 - w) r_h,t(beta) / r_h,t(0) ]

- **Base (w = 0)**: transfers per receiving household stay at their no-shock level except for the senders
  need brings in. A kin (or core migrant) who steps in replaces **all** the lost transfers of the household:
  job loss only reduces counts through households that stop receiving.
- **Lower bound (w = 1)**: counts = receipt x expected able senders. A migrant who steps in restores
  **receipt** (one transfer) but not the other lost transfers: every job-losing sender in a multi-sender
  household is one transfer less.
- **Mixed (0 < w < 1)**: a share w of the transfers of job-losing senders is lost even when the household
  keeps receiving (stepping-in replaces 1 - w of them).

The no-COVID run uses the same w with its own r (beta = 0). w = 1 reproduces the `*_lower_bound` columns of
`household-model/map.csv` at all 11 x 25 grid points (max abs diff 2.2e-16, `reproduction_check.csv`).

## Files
- `grid_summary.csv`: (i) per kappa, the grid-best beta under base and lower bound, lower-bound RMSE / 2020Q2 at
  the base beta (grid and continuous `need_fit.csv` beta), lower-bound moments at the base beta.
- `fits.csv`: (ii)/(iii) beta refit continuously (bounded, [0, 3]) per kappa and w in {0, .25, .5, .75, 1}:
  beta, rmse_fit (2020Q2-2021Q4), rmse_holdout_2022, index_2020q2, index_2021q4, mean_index_2021,
  mean_index_2022, spike_2021_01, spike_2021_03, spike_rmse.
- `quarterly_paths.csv`: quarterly index per (w, kappa) at the refit beta; `w = lower_bound_at_base_beta` is
  the lower bound at the base-fitted beta.
- `banxico_moments.csv`: Banxico 2020Q2, 2021Q4, 2022 mean, Jan/Mar 2021 spikes.
- `paths_by_mapping.png`: kappa = 1, base / mixed 0.5 / lower bound (refit) / lower bound at base beta, Banxico.

## Results (kappa = 1, the best kappa under every w)

| w | beta | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | spike Jan | spike Mar |
|---|---|---|---|---|---|---|---|
| 0 (base) | 0.643 | 0.0123 | 0.0238 | 0.974 | 1.093 | +4.1% | +2.9% |
| 0.25 | 0.597 | 0.0139 | 0.0242 | 0.965 | 1.094 | +5.1% | +3.7% |
| 0.5 | 0.563 | 0.0161 | 0.0247 | 0.958 | 1.095 | +5.9% | +4.4% |
| 0.75 | 0.537 | 0.0184 | 0.0252 | 0.951 | 1.096 | +6.6% | +4.9% |
| 1 (lower bound) | 0.517 | 0.0207 | 0.0256 | 0.945 | 1.097 | +7.3% | +5.5% |
| Banxico | | | | 0.974 | 1.108 | +4.1% | +3.6% |

RMSE rises monotonically in w at every kappa, and kappa = 1 is best at every w. Lower bound at the base beta
(0.643): RMSE 0.026, 2020Q2 0.946, spikes +7.5% / +5.6%. 2020Q2 hardly depends on beta (the need stock is near
zero then), so its gap to Banxico is a pure mapping effect.
