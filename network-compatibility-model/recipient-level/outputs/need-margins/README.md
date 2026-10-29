# Need on other margins: options 6, 7, 8 (experiments, 29 Oct 2026)

Script: `recipient-level/need_margins.py` (about 15 min; `--het-only` redoes only the heterogeneity table from `fits.csv`).
**These are experiments, not changes to the base model.** No existing file was modified.

## Question

Under the base household model (κ = 0.7–0.8, stimulus in Jan and Mar 2021, β = 0), the count index is flat in 2021 (2021Q4 0.999) while Banxico rises to 1.108. Can need in Mexico close that gap if it acts on a margin other than the current household-level shift β × S_t? Three margins from the literature on how migrants respond to crises:

- **Option 6, more senders per receiving household.** Need brings in migrants who normally do not send. Receipt does not change, but transfers per receiving household rise.
- **Option 7, dormant ties waking up.** Need raises receipt in households that usually receive little: kin-only households (7a) or households with no resident worker or with an older resident (7b).
- **Option 8, persistence.** Sending triggered by need keeps going after the need peaks.

## Set-up

Everything is as in `run_household_model.py`: the same 3,405 households (the index is computed over all of them, the 2019 fit uses the 3,252 in `e0.target_mask`), the same 2019 ML calibration for each κ, the same stimulus (π = 1 in Jan and Mar 2021, 0 in Apr 2020), and the same target (Banxico counts over their 2017–19 trend, quarterly log RMSE 2020Q2–2021Q4, 2022 held out). The base need term β S_t is switched off, so need enters only through the margin being tested. κ is fixed at 0, 0.7 and 0.8.

**Each option has one free parameter**, a slope fitted by minimising the quarterly RMSE. Option 8 adds λ, which is profiled on a grid {0, 0.5, 0.8, 0.9, 0.95, 1}.

**Need series** come from `data-prep/outputs/salience_inputs_monthly.csv` (public aggregates). They are set to 0 before Mar 2020 and in the no-COVID run.

- **S:** reported COVID deaths (the base S_t).
- **E:** excess deaths (World Mortality Dataset, trend baseline, negative values set to 0).
- **L:** Mexican employment loss against the same month of 2019 (OECD/ETOE, negative values set to 0).
- **R:** a **placebo**, a constant flow from Mar 2020. With λ = 0 it is a step; with λ = 1 it is a linear ramp. It carries no information on need, only on timing.

**Persistence (option 8).** The stock is N*_t = Σ_k λ^k N_{t−k}, scaled to peak 1 over 2020-03..2021-12, so each slope is the logit shift at the peak. λ = 0 is the flow and λ = 1 the cumulative stock. The cumulative excess-death series asked for as series (b) is E with λ = 1.

## Equations

Household receipt is the base form with a household-specific shift d_{h,t}:

  s_{h,t}(a, b) = expit(x'z_h(a, b) + d_{h,t}),  P_{h,t} = E[s(A_c, A_k)(1 − κ[A_c + A_k > 0])] + κ s(N_c, N_k) P(A_c + A_k > 0)

- **H (reference, the base form with other series):** d = β N*_t
- **7a:** d = β_k N*_t × [household has no core migrant]
- **7b:** d = β_v N*_t × [no resident worker or any older resident] (households with an unknown roster get 0)
- **6a / 6b:** d = 0. Senders: θ_{j,t} = expit(α_j + β_s N*_t) for all migrants (6a) or for kin only (6b), using the migrant-level M1 thetas (α_core −0.23, α_kin −3.36).

Count mapping. With r_{h,t}(θ) = E[able senders | someone sends] = Σ_j q_j / (1 − Π_j(1 − q_j)) and q_j = e_{j,t} θ_{j,t} (actual ability, including the stimulus):

- **Base mapping:** counts_t = Σ_h P_{h,t} × r_{h,t}(θ(N*)) / r_{h,t}(θ(0)). Transfers per receiver stay constant except for senders activated by need, so the multiplier is 1 for H and 7.
- **Lower bound (`*_lb`):** counts_t = Σ_h P_{h,t} r_{h,t}(θ(N*)), as in map.csv `*_lower_bound`.

**Check:** with the new slopes at 0 and the raw S_t, the script reproduces `outputs/household-model/map.csv` for κ ∈ {0, 0.7, 0.8} × β ∈ {0, 0.06, 0.2}. The largest absolute difference across RMSE, 2020Q2, 2021 mean, holdout, spikes and the lower-bound columns is 2e-10, and the script asserts it.

## Results (κ = 0.8, base mapping unless `_lb`; all rows in `fits.csv`, paths in `quarterly_paths.csv`)

Reference values. Banxico: 2020Q2 0.974, 2021Q4 1.108, 2022 mean 1.10, spikes +4.1 / +3.6 log points. Base model (no need): RMSE 0.062, holdout 0.096, 2020Q2 0.961, 2021Q4 0.999, spikes +4.1 / +3.1. Gap closed = share of the base model's 2021Q4 log gap.

| option | series, λ | slope | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | spikes Jan / Mar | gap closed |
|---|---|---|---|---|---|---|---|---|
| H household shift | S flow (λ=0) | 0.23 | 0.045 | 0.092 | 0.996 | 1.013 | +7.9 / +2.4 | 14% |
| H | S cumulative (λ=1) | 0.21 | 0.014 | 0.024 | 0.967 | 1.092 | +4.3 / +3.1 | 86% |
| H | E cumulative | 0.21 | 0.014 | 0.024 | 0.968 | 1.092 | +4.5 / +3.1 | 86% |
| H | L cumulative | 0.15 | 0.023 | 0.037 | 0.986 | 1.066 | +4.2 / +3.1 | 63% |
| 6a senders, all | S flow | 0.73 | 0.046 | 0.093 | 0.992 | 1.013 | +8.9 / +2.6 | 13% |
| 6a | S cumulative | 0.65 | 0.013 | 0.024 | 0.966 | 1.094 | +4.7 / +3.6 | 88% |
| 6b senders, kin | S cumulative | 1.88 | 0.013 | 0.033 | 0.964 | 1.104 | +4.7 / +3.6 | 97% |
| 6a_lb (lower bound) | S cumulative | 0.53 | 0.023 | 0.026 | 0.937 | 1.097 | +7.9 / +6.1 | 91% |
| 6b_lb | S cumulative | 1.93 | 0.024 | 0.038 | 0.935 | 1.108 | +7.6 / +6.0 | 100% |
| 7a kin-only households | S flow | 0.52 | 0.046 | 0.093 | 0.993 | 1.012 | +8.5 / +2.3 | 13% |
| 7a | S cumulative | 0.47 | 0.013 | 0.025 | 0.967 | 1.095 | +4.3 / +3.2 | 88% |
| 7a | E cumulative | 0.47 | 0.013 | 0.025 | 0.967 | 1.095 | +4.5 / +3.1 | 88% |
| 7b vulnerable households | S cumulative | 0.50 | 0.014 | 0.024 | 0.968 | 1.091 | +4.3 / +3.1 | 85% |
| **placebo** H | R step (λ=0) | 0.12 | 0.030 | 0.046 | 1.014 | 1.054 | +4.1 / +3.1 | 52% |
| **placebo** H | R ramp (λ=1) | 0.23 | 0.010 | 0.035 | 0.975 | 1.098 | +4.1 / +3.1 | 91% |
| **placebo** 7a | R ramp | 0.52 | 0.009 | 0.048 | 0.973 | 1.102 | +4.1 / +3.1 | 95% |

The numbers at κ = 0.7 are within 0.001–0.003 of these. At κ = 0 the RMSEs are about 0.01 higher and 2020Q2 is about 0.935.

How the fit and the 2022 holdout change with λ, for S and E and every margin under the base mapping (κ = 0.8):

| λ | 0 | 0.5 | 0.8 | 0.9 | 0.95 | 1 |
|---|---|---|---|---|---|---|
| RMSE fit | 0.045–0.050 | 0.041–0.047 | 0.031–0.037 | 0.024–0.027 | 0.018–0.019 | 0.013–0.014 |
| RMSE 2022 | 0.092–0.094 | 0.090–0.093 | 0.081–0.087 | 0.064–0.072 | 0.046–0.052 | 0.024–0.034 |

### What the table says

1. **The margin hardly matters for the time series. Persistence does.** With a flow (λ = 0) of deaths, S or E, every margin closes only 6–17% of the gap, and the L flow closes none. A flow of deaths peaks in Jan 2021, which inflates the Jan stimulus spike (+8 to +15 log points under the base mapping, against +4.1 in Banxico) and falls away in 2022. With a cumulative stock (λ = 1) every margin fits within the tolerance: RMSE 0.013–0.014, 85–97% of the 2021Q4 gap closed, 2022 holdout 0.024 against 0.096. 2020Q2 and the spikes stay at their base values, because the cumulative stock is small in Q2 2020 and smooth in early 2021. S and E give the same answer once they are cumulated. L (Mexican employment loss) is front-loaded in 2020 and does worse: 63% closed, RMSE 0.023.
2. **Option 8 is the active ingredient, and it passes the 2022 test.** The cumulative stock stops growing when deaths stop in 2022, so the index stays flat at about 1.10–1.11, close to Banxico's 1.07–1.13. A decaying stock (λ ≤ 0.95) brings the index back down in 2022 and is rejected by the holdout.
3. **The placebo fits just as well.** A linear ramp with no need content gives RMSE 0.009–0.010, as good as or better than any need series. Its 2022 holdout is a little worse (0.035–0.048 against 0.024), because it keeps rising while Banxico flattens. **The time series identifies "a persistent shift that builds through 2020–21 and stops in 2022", not need.** The cumulative death series has the right timing, and that is all it adds over the ramp.
4. **Option 6 and the stimulus spikes.** Under the base mapping, option 6 keeps the spikes at +4.7 / +3.6. Under the lower-bound mapping it also fills the 2021 gap, but the spikes remain too large (+7.6 to +7.9 / +6.0 to +6.1) and 2020Q2 too low (0.935–0.937). Its fit is about 0.023–0.024. The earlier tension does not go away: the spikes favour the base mapping, and option 6 adds nothing to the time-series fit that options H or 7 do not.
5. **Sizes.** 6a needs a +0.65 logit rise in every migrant's sending probability by end-2021 (core θ 44% → 60%). 6b needs kin θ 3.4% → 18%, about five times higher. 7a needs receipt in kin-only households to rise 42% (6.1% → 8.7% a month). Kin-only households are 70% of households but only 19% of 2019 transfers, so a 42% rise there adds about 8% in total. That is enough, but it is a large change.

## Heterogeneity predictions (`heterogeneity.csv`)

The predictions use κ = 0.8, each margin at its best need series (S cumulative for all of them), and the base mapping. "Job loss only" is the base model without need. Changes are 2021 against the no-COVID run, in %.

| group | 2019 receipt | share of 2019 transfers | job loss only | H receipt | 7a receipt | 7b receipt | 6a transfers | 6b transfers |
|---|---|---|---|---|---|---|---|---|
| all | 23% | 100% | −0.1 | +7.7 | +7.8 | +7.7 | +7.7 | +7.8 |
| no resident worker: yes / no | 41% / 20% | 24% / 73% | 0 / −0.2 | +5.8 / +8.3 | +4.0 / +9.0 | **+13.7 / +6.0** | +10.1 / +6.9 | +8.5 / +7.6 |
| any older resident: yes / no | 42% / 17% | 43% / 54% | −0.1 / −0.2 | +5.7 / +9.3 | +2.8 / +11.7 | **+13.6 / +3.4** | +10.5 / +5.5 | +5.4 / +9.7 |
| core only | 65% | 34% | −0.2 | +5.2 | −0.2 | +7.0 | +6.3 | −0.2 |
| core + kin | 63% | 47% | −0.1 | +5.9 | −0.1 | +6.7 | +11.1 | **+12.4** |
| kin only | 6% | 19% | −0.2 | +16.7 | **+42.4** | +11.2 | +1.9 | +10.9 |
| network size 1 / 2 / 3–4 / 5–9 / 10+ | 12 / 26 / 34 / 42 / 55% | | ≈0 | +9.0 … +6.0 (falls with size) | +12.0 … +4.1 | +8.7 … +6.6 | **−0.4 / +4.2 / +10.0 / +17.5 / +25.0** | −0.4 … +31.4 |

**Receipt and transfers.** For H and 7, transfers = receipt under the base mapping. For 6, receipt changes by about −0.1% (no change) by construction. Lower-bound transfers are in the csv and follow the same pattern.

**States.** States with MMP communities: Jalisco, Puebla, Querétaro, Oaxaca, Morelos, Tabasco, México, Zacatecas, San Luis Potosí, Yucatán, Guanajuato; Guanajuato has only 31 households. The need series is national, so state differences come only from network composition. Under 7a the spread is +3.8 (Oaxaca) to +12.1 (Guanajuato), following the kin-only share. Under 6 it is +2.2 to +11.3 (Jalisco highest), following network size. Under H and 7b it is narrow, +7 to +9 (7b: Guanajuato +1.6). A state-level test would need a state-level need series, such as state excess deaths, which is not in the model.

**ENIGH 2020 window** (received at least once in the six months before an Aug–Nov 2020 interview, 100 paired paths, COVID against no-COVID). Job loss only gives −1.3%. H gives −0.3%, 7a +0.9% (kin-only +3.0%), 7b −0.8%, and 6 −1.3% (receipt is unchanged). **The 2020 window cannot separate the options.** The cumulative stock is still small in mid-2020, so they differ by at most 2 points. The options separate in 2021–22.

### Distinguishing patterns

- **Option 6 (more senders):** an **intensive** margin. Receipt does not change. Transfers per receiving household rise, and they rise most in large networks (size 10+ +25%, size 1 0%) and in core + kin households. Need features play almost no part, except through network size.
- **Option 7a (kin-only dormant ties):** an **extensive** margin concentrated in kin-only households (+42% receipt). Households with core migrants are unchanged. The rise is largest for single-link households.
- **Option 7b (vulnerable households):** an extensive margin in households with no resident worker or an older resident (+14% against +3 to +6%).
- **H (common shift):** an extensive margin, roughly proportional in the logit, so largest where baseline receipt is low (kin-only +17%) and slightly smaller in households with need features.
- **Option 8:** a time pattern, not a cross-section. Under every margin, the 2022 level stays as high as the 2021 level.

### Against the ENIGH results (parallel analysis, `identification/enigh/README.md`)

The ENIGH results reported there:

- National receipt: 4.73% (2018), 4.38% (2020), 4.57% (2022), 3.95% (2024).
- 2018→2022 the rise is almost entirely intensive: amount per recipient +0.34 log points (SE 0.04), receipt −0.03 (0.03).
- Differences in changes for high-need households are small or negative:
  - receipt: no worker +0.14 pp (0.61), 65+ −2.06 pp (0.34);
  - log amount: no worker −0.17 (0.08), 65+ −0.15 (0.07).

What this means for each option:

- **H, 7a and 7b (extensive margins) are not consistent with ENIGH.**
  - Combined with option 8 they predict receipt about 8–10% higher in 2022 than without COVID. ENIGH shows −3% ± 3% 2018→2022, before netting out the normal trend.
  - 7b also predicts the largest receipt gains in no-worker and older households. ENIGH shows +0.1 pp and −2.1 pp.
  - Caveat: MMP households are a migrant-network sample, not the national population. A rise among kin-only network households (7a) could be diluted nationally, but not reversed.
- **Option 6 (intensive margin) is consistent with the main fact:** receipt flat, more per receiving household.
  - It predicts no particular advantage for need households. Under 6a they gain slightly more, +10% against +7%, only because their networks are larger. ENIGH shows smaller intensive gains for need households (−0.17, −0.15). That is consistent with need not being the driver of the intensive rise, and mildly against 6a.
  - The ENIGH intensive rise is in amounts, which include the amount per transfer (Banxico real amount per transfer +9 log points in 2021Q4, +3 in 2022Q4) and the peso. It is not only the number of transfers, so it is not direct evidence of more senders.
  - Option 6's own test is in the cross-section by network size: gains in remittance income per recipient should rise steeply with the number of linked migrants.
- **Net reading:** ENIGH points to an intensive margin that is not targeted at need households. That fits option 6 combined with persistence (8) better than option 7. It fits equally well non-need explanations of more transfers per recipient (US wages and benefits, the shift to digital channels, more frequent, smaller transfers). Membership of the no-worker cell also shifts with the shock (9.1% → 11.2% in 2020), which blurs the need-cell comparisons.

## Honest read

- **What the options can do.** Any margin (base mapping) closes the 2021 gap and keeps 2020Q2 and the stimulus spikes, **but only if need is persistent** (a cumulative stock, option 8). The same stock also explains why counts stay high in 2022 (holdout RMSE 0.024 against 0.096 in the base). The flow version of every margin fails: it closes at most 17% of the gap, breaks the Jan 2021 spike and has no 2022 persistence.
- **What they cannot do.**
  - The time series cannot tell the margins apart: RMSEs agree within 0.001.
  - It cannot tell need from any other slowly building, persistent shift: the placebo ramp fits as well.
  - Other 2021 explanations have the same timing: the channel shift to digital, US fiscal transfers and wage growth, more frequent and smaller transfers, measurement changes at Banxico. One extra parameter on a stock that rises through 2020–21 will absorb all of them.
  - Cumulative deaths also act as a proxy for "time since the pandemic began".
- **Identification has to come from the cross-section.**
  - Extensive against intensive margin, and its location (kin-only against need households against large networks).
  - Or regional variation in need, which this national series does not have. A state-level excess-death stock, together with the MMP community's state, would be the natural next test.
  - The ENIGH results now available favour an intensive margin and show no need targeting. That argues against options 7a and 7b and against reading the 2021 rise as need-driven receipt.
- **Implementation caveats.**
  - Option 6 holds receipt fixed while adding senders, a deliberate decomposition. A fully consistent version would raise receipt a little too.
  - The ENIGH-window statistic comes from model paths with independent monthly receipt draws. Its level (51% receiving within six months among MMP network households) is not comparable to ENIGH's national 4–5%; use relative changes only.
  - The slopes are fitted on 7 quarters with one parameter each. Treat the 2022 holdout as the only real out-of-sample check.

## Files

- `fits.csv`: every (margin, series, λ, κ) with its fitted slope, RMSE, holdout, 2020Q2, 2021Q4, 2021 and 2022 means, spikes and gap closed.
- `quarterly_paths.csv`: model and Banxico index by quarter for each row.
- `best_by_margin.csv`: the best need series and λ per margin at κ = 0.8.
- `heterogeneity.csv`: group predictions (aggregates only).
- `need_series.csv`: the scaled need stocks (public aggregates).
- `run_settings.json`: settings and the verification result.
