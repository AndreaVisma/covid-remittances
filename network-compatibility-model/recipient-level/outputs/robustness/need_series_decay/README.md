# Robustness 6f: need series and persistence

Script: `recipient-level/robustness/need_series_decay.py`. Reuses `need_margins.Experiment` with `KAPPAS` patched to
{0, 0.7, 0.8, 1} in-process; slopes refitted per row and kappa (bounded search, xatol 1e-4). The base row uses the
slope in `outputs/household-model/need_fit.csv` and reproduces that file (asserted). Cross-check with
`outputs/need-margins/fits.csv` on the 31 shared cells: max |RMSE diff| 3.3e-08, max |slope diff| 1.6e-04.

Banxico: 2021Q4 1.108, 2022 mean 1.099, spikes
+4.1 / +3.6 log points. Gap closed = share of the no-need model's 2021Q4 log gap.
Spikes and spike RMSE in log points. Full table (all kappa, implied thetas) in `summary.csv`, paths in
`quarterly_paths.csv`, figure `quarterly_paths.png` (kappa = 1).

## kappa = 1 (base best)

| row | slope | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | 2022 mean | spikes Jan / Mar | spike RMSE | gap closed |
|---|---|---|---|---|---|---|---|---|---|
| base: 6a, E cumulative | 0.64 | 0.0123 | 0.0238 | 0.974 | 1.093 | 1.106 | +4.1 / +2.9 | 0.5 | 86% |
| 6a, S cumulative | 0.65 | 0.0121 | 0.0236 | 0.974 | 1.093 | 1.105 | +3.9 / +2.9 | 0.5 | 87% |
| 6a, L cumulative | 0.47 | 0.0229 | 0.0373 | 0.990 | 1.066 | 1.068 | +3.8 / +2.9 | 0.6 | 62% |
| 6a, E lambda 0.95 | 0.56 | 0.0182 | 0.0471 | 0.977 | 1.077 | 1.064 | +4.3 / +2.9 | 0.5 | 72% |
| 6a, E lambda 0.9 | 0.56 | 0.0240 | 0.0658 | 0.979 | 1.063 | 1.040 | +4.6 / +3.0 | 0.6 | 60% |
| 6a, E flow (lambda 0) | 0.82 | 0.0464 | 0.0927 | 1.000 | 1.010 | 1.007 | +10.4 / +1.5 | 4.7 | 10% |
| 6b kin only, E cumulative | 1.87 | 0.0107 | 0.0328 | 0.972 | 1.104 | 1.126 | +4.0 / +2.9 | 0.5 | 96% |
| 6a lower-bound mapping, E cumulative | 0.52 | 0.0207 | 0.0256 | 0.944 | 1.096 | 1.112 | +7.3 / +5.5 | 2.6 | 90% |
| H household shift, E cumulative | 0.20 | 0.0126 | 0.0237 | 0.975 | 1.091 | 1.101 | +3.6 / +2.4 | 0.9 | 85% |
| placebo: 6a, R ramp (lambda 1) | 0.71 | 0.0092 | 0.0410 | 0.980 | 1.099 | 1.144 | +3.6 / +2.8 | 0.7 | 92% |
| placebo: 6a, R step (lambda 0) | 0.39 | 0.0307 | 0.0467 | 1.017 | 1.054 | 1.056 | +3.7 / +2.7 | 0.7 | 51% |
| *no need (beta = 0)* | 0 | 0.0606 | 0.0967 | 0.969 | 0.999 | 1.001 | +3.3 / +2.4 | 1.0 | 0% |

## kappa = 0.8

| row | slope | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | 2022 mean | spikes Jan / Mar | spike RMSE | gap closed |
|---|---|---|---|---|---|---|---|---|---|
| base: 6a, E cumulative | 0.65 | 0.0137 | 0.0244 | 0.967 | 1.094 | 1.107 | +5.0 / +3.6 | 0.6 | 87% |
| 6a, S cumulative | 0.65 | 0.0135 | 0.0241 | 0.966 | 1.094 | 1.107 | +4.7 / +3.6 | 0.5 | 87% |
| 6a, L cumulative | 0.48 | 0.0223 | 0.0360 | 0.983 | 1.067 | 1.070 | +4.7 / +3.6 | 0.4 | 64% |
| 6a, E lambda 0.95 | 0.57 | 0.0190 | 0.0464 | 0.969 | 1.078 | 1.065 | +5.2 / +3.6 | 0.8 | 73% |
| 6a, E lambda 0.9 | 0.57 | 0.0245 | 0.0653 | 0.971 | 1.064 | 1.041 | +5.5 / +3.6 | 1.0 | 61% |
| 6a, E flow (lambda 0) | 0.85 | 0.0465 | 0.0924 | 0.993 | 1.010 | 1.007 | +11.5 / +2.2 | 5.3 | 11% |
| 6b kin only, E cumulative | 1.88 | 0.0130 | 0.0340 | 0.964 | 1.105 | 1.128 | +4.9 / +3.6 | 0.6 | 97% |
| 6a lower-bound mapping, E cumulative | 0.52 | 0.0234 | 0.0264 | 0.938 | 1.097 | 1.113 | +8.1 / +6.0 | 3.3 | 90% |
| H household shift, E cumulative | 0.20 | 0.0137 | 0.0240 | 0.968 | 1.092 | 1.103 | +4.5 / +3.1 | 0.5 | 86% |
| placebo: 6a, R ramp (lambda 1) | 0.72 | 0.0098 | 0.0434 | 0.973 | 1.101 | 1.147 | +4.5 / +3.4 | 0.3 | 93% |
| placebo: 6a, R step (lambda 0) | 0.40 | 0.0292 | 0.0447 | 1.011 | 1.056 | 1.059 | +4.6 / +3.4 | 0.3 | 53% |
| *no need (beta = 0)* | 0 | 0.0617 | 0.0965 | 0.961 | 0.999 | 1.001 | +4.1 / +3.1 | 0.4 | 0% |

## kappa = 0.7 and 0

See `summary.csv`.

## Implied size at Dec 2021 (M1 alpha_core -0.2325, alpha_kin -3.367)

| row | kappa | slope | core theta 2019 -> Dec 2021 | kin theta 2019 -> Dec 2021 |
|---|---|---|---|---|
| base: 6a, E cumulative | 0.8 | 0.65 | 44.2% -> 60.3% | 3.3% -> 6.2% |
| base: 6a, E cumulative | 1 | 0.64 | 44.2% -> 60.1% | 3.3% -> 6.2% |
| 6b kin only, E cumulative | 0.8 | 1.88 | 44.2% -> 44.2% | 3.3% -> 18.5% |
| 6b kin only, E cumulative | 1 | 1.87 | 44.2% -> 44.2% | 3.3% -> 18.3% |
| H household shift, E cumulative | 0.8 | 0.20 | unchanged (receipt logit +0.20) | unchanged |
| H household shift, E cumulative | 1 | 0.20 | unchanged (receipt logit +0.20) | unchanged |
