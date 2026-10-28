# Recipient-level figures

## S1_receipt_curve_examples: Where households sit on the receipt curve

Fitted household logit (H3, MMP 2019, maximum likelihood). Illustrative compositions, not MMP households, with all migrants working. The first able core migrant moves a household from the flat bottom of the curve to its upper half; further core migrants and kin add little. Red arrows: one migrant losing their job. Losing an only core migrant is a large drop; losing one of three is small.

## S2_receipt_curve_households: All sample households on the fitted receipt curve

Fitted household logit (H3). Each bubble is a group of households with the same expected 2019 score (able migrant counts averaged over 2019 CPS employment), drawn on the curve and sized by the number of households; colour is network type. Diamonds: observed MMP receipt in score bands of 0.5 with at least 30 households, 95% Wilson intervals. Kin-only households cluster at the flat bottom, households with core migrants in the upper half; almost none sit on the steep middle. The curve at the expected score differs from the mean fitted probability by at most 0.8% per band. 3,252 households.

## S3_kappa_2019: In 2019 the household data lean against stepping-in

Household model (H3 plus stepping-in κ·[a core migrant cannot send]·log(1 + able kin)), "Not working" migrants as unknown occupation, 2019 MMP receipt, maximum likelihood with κ fixed on a grid. Left: log-likelihood relative to its maximum; dashed line: the 95% limit of a one-parameter profile (−1.92). Right: the other coefficients refitted at each κ. The 2019 fit prefers κ ≤ 0.8; unrestricted, κ is negative (log-likelihood +7.1 over κ = 0): kin count for less where some core migrant is out of work. In 2019 that indicator mostly marks large core networks, so this is a size effect (kin matter less where core migrants are many), not evidence against stepping-in in a crisis. The whole range costs at most 2.6 log-likelihood points.

## S4_receipt_index: Households lose less receipt than the migrant model loses transfers; 2021 is still missed

β = 0, ρ = 0, no benefits (E1a on the household model). Household model: expected number of receiving households, COVID run / no-COVID run (2019 CPS rates of the same month), exact; read as transfers under a constant number of transfers per receiving household. Coefficients refitted to 2019 for each κ. Gray: the migrant-level model (transfer counts, original "Not working" coding, E1a). Banxico: transactions relative to their 2017–19 trend. 2020 Q2 at κ = 0: 0.940. 2022 held out.

## S5_compatibility_map: Household model: 0 of 189 (β, κ) cells fit within τ_C = 0.04

E2 on the household model (π = 0, ρ = 0, S_t = reported COVID deaths / 10,000). Fit error of the quarterly receipt index against Banxico relative to its 2017–19 trend. Outlined: count-compatible cells (RMSE ≤ 0.04). Dot: best cell (κ = 0, β = 0.08). Coefficients refitted to 2019 for each κ. β is on the household log-odds scale, not the migrant scale of the migrant-level map (M1).

## S6_compatible_paths: Household model: the best cell still follows the death waves

E2 on the household model. Quarterly receipt index of the best cell, the other count-compatible cells and the best β = 0 cell, against Banxico relative to its 2017–19 trend. 2022 held out.

## S7_realized_losses: Household model: 6–8% of receiving households lose months; almost none lose every month

β = 0, ρ = 0, no benefits. Each household that receives in its no-COVID twin is compared with it (common random numbers for employment and the receipt draw), Apr 2020 – Dec 2021: change in the number of months with receipt, pooled over 200 paths. Unlike the migrant model (H1, change in USD), every household draws its receipt each month, so luck moves months both ways; the paired draw removes most of it. About 2,752 households receive in a path.

## S8_who_loses: Household model: single links and core-only networks lose most; large networks gain with κ

β = 0, ρ = 0, no benefits; 200 paired paths, Apr 2020 – Dec 2021. Left: expected change in months with receipt by group (exact, receipt-weighted). Right: share of paths in which a household that receives in its no-COVID twin receives in no month, averaged over the group. Main occupation: the most common job group among the household’s eligible migrants ("Not working" migrants count as Other/none).
