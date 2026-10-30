# Recipient-level figures

## S10_usd_accounting: USD received: counts drive the lasting rise; the real amount rise of 2020–21 fades by 2022

Banxico USD relative to the same quarter of 2019 (line) = transfer counts × average amount per transfer, in log points. Counts are split as in S9 (household model, best map cell κ = 1, β = 0.65); the observed average amount is split into US CPI-U inflation and the real amount (2019 USD). Amounts are accounting, not modelled. 2022 held out (shaded). 2021 Q4: USD +42, counts +24, inflation +7.8, real amount +9.4 log points; 2022 Q4: real amount +3.2.

## S1_receipt_curve_examples: Where households sit on the receipt curve

Fitted household logit (H3, MMP 2019, maximum likelihood). Illustrative compositions, not MMP households, with all migrants working. The first able core migrant moves a household from the flat bottom of the curve to its upper half; further core migrants and kin add little. Red arrows: one migrant losing their job. Losing an only core migrant is a large drop; losing one of three is small.

## S2_receipt_curve_households: All sample households on the fitted receipt curve

Fitted household logit (H3). Each bubble is a group of households with the same expected 2019 score (able migrant counts averaged over 2019 CPS employment), drawn on the curve and sized by the number of households; colour is network type. Diamonds: observed MMP receipt in score bands of 0.5 with at least 30 households, 95% Wilson intervals. Kin-only households cluster at the flat bottom, households with core migrants in the upper half; almost none sit on the steep middle. The curve at the expected score differs from the mean fitted probability by at most 1.0% per band. 3,252 households.

## S3_kappa_2019: In 2019 the household data cannot tell the replacement share κ apart

Household model: H3 receipt s(able core, able kin) plus stepping-in, P = s + κ·[someone can send]·(s if nobody were out of work − s), κ ∈ [0, 1] the share of the receipt lost to job loss that other migrants replace. "Not working" migrants as unknown occupation, 2019 MMP receipt, maximum likelihood with κ fixed on a grid. Left: log-likelihood relative to its maximum; dashed line: the 95% limit of a one-parameter profile (−1.92). The whole range κ = 0–1 lies within 0.14 points: ordinary unemployment in 2019 is too rare to identify κ. Right: the other coefficients refitted at each κ.

## S4_receipt_index: Job loss and stepping-in alone: 2020 is matched with strong stepping-in, 2021 is missed

β = 0 (no need), ρ = 0, stimulus π as in the base (1 in Jan and Mar 2021). Household model: expected number of receiving households, COVID run / no-COVID run (2019 CPS rates of the same month), exact; read as transfers under a constant number of transfers per receiving household. κ = replacement share; coefficients refitted to 2019 for each κ. Gray: the migrant-level model, coordination rule (κ = replacement share; transfer counts, job loss only). Banxico: transactions relative to their 2017–19 trend. 2020 Q2 at κ = 0: 0.930. 2022 held out.

## S5_compatibility_map: Household model: 161 of 275 (β, κ) cells fit within τ_C = 0.04

E2 on the household model (stimulus π as in the base, ρ = 0). Need: more senders per receiving household, θ = expit(α + β N*_t), N*_t = cumulative excess deaths in Mexico (1 in Dec 2021). Fit error of the quarterly count index against Banxico relative to its 2017–19 trend. Outlined: count-compatible cells (RMSE ≤ 0.04). Dot: best cell (κ = 1, β = 0.65). Coefficients refitted to 2019 for each κ. β is on the migrant sending scale (M1 α), and need leaves receipt unchanged.

## S6_compatible_paths: Household model: compatible cells and the best cell against Banxico

E2 on the household model. Quarterly receipt index of the best cell, the other count-compatible cells and the best β = 0 cell, against Banxico relative to its 2017–19 trend. 2022 held out.

## S7_realized_losses: Household model: 3–8% of receiving households lose months; almost none lose every month

β = 0, ρ = 0, stimulus π as in the base. Each household that receives in its no-COVID twin is compared with it (common random numbers for employment and the receipt draw), Apr 2020 – Dec 2021: change in the number of months with receipt, pooled over 200 paths. Unlike the migrant model (H1, change in USD), every household draws its receipt each month, so luck moves months both ways; the paired draw removes most of it. About 2,638 households receive in a path.

## S8_who_loses: Household model: single links and core-only networks lose most, whatever κ

β = 0, ρ = 0, stimulus π as in the base; 200 paired paths, Apr 2020 – Dec 2021. Left: expected change in months with receipt by group (exact, receipt-weighted). Right: share of paths in which a household that receives in its no-COVID twin receives in no month, averaged over the group. Main occupation: the most common job group among the household’s eligible migrants ("Not working" migrants count as Other/none).

## S9_decomposition: Why transfers grew despite job loss: trend, stepping-in and need; the household model leaves a small remainder

Banxico transactions relative to the same quarter of 2019 (line), split in log points: the 2017–19 trend continued (log-linear with month effects); the model’s job-loss path at β = 0 with stepping-in κ (COVID / no-COVID run); need, i.e. the senders that cumulative excess deaths bring in (best map cell relative to its β = 0 path; migrant model: its own need term); and what remains. κ and β are each model’s best map cell (RMSE against Banxico vs trend, 2020 Q2–2021 Q4). Left: migrant-level model, coordination rule (κ = replacement share, need β × reported COVID deaths, transfer counts). Right: household model (κ = replacement share, receipt). 2022 held out (shaded). 2021 Q4, migrant model: raw +24, trend +12, job loss -0.3, need +2.0, unexplained +10 log points.
