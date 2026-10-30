# Calibration figures (2019, willing-and-able model)

Made by `2026-10-30/make_calibration_figures.py` from `outputs/calibration.csv` (κ = 0 calibration).

## C1_households_by_type: Who is in the data, and who receives

MMP origin households interviewed 2010–19, by the US migrants linked to them: core (head, spouse or child in the US) and kin (close kin and one aggregated distant-kin link). The calibration uses the 3,252 connected households with complete records. The 3,730 households with no linked migrant are not used in the calibration; the model gives them the baseline receipt expit(b0) = 1.0% a month, against 0.6% observed, an out-of-sample check. Model: average monthly receipt probability in 2019 (κ = 0 calibration).

## C2_able_willing: Migrants across the able and willing dimensions

Each migrant is able to send in a month if employed (CPS employment rate of their occupation group) and wants to send with the calibrated desire probability; the two are independent. Square area: share of migrants in that state in a month; labels give the share and the number of migrants. An able sender sends; a lost sender wanted to send but is out of work, the gap that stepping-in can fill; the reserve (able, not planning to send) is who can step in. April 2020 uses that month's CPS employment with 2019 desire (no crisis response, no stimulus) to show where lost senders come from. Kin migrants rarely want to send (3% a month), so nearly all of them are in the reserve.

## C3_desire_network_size: Desire and the number of core migrants

Left: monthly probability that a core migrant wants to send, expit(a_c + g·log N), with a_c = 0.41 and g = -0.97 (household of average need), and the chance that at least one of them is an able sender (employment 95%). Right: share of households receiving, MMP 2019 (95% Wilson CI), the calibrated model, and the Stage 1 fit with the same desire for every core migrant (g = 0). With the same desire for all, receipt rises far too steeply with the number of core migrants; a desire that falls with each extra core migrant (one reading: a division of labour among them) fits every group.

## C4_receipt_curve_examples: Where households sit on the receipt curve

Illustrative compositions (not MMP households), all migrants employed, 2019 calibration (κ = 0). A household's score is the log-odds of its expected monthly receipt, averaged over which migrants are willing. Red: one migrant loses their job and nobody steps in. Dashed blue: someone in the household's reserve (able but not planning to send) steps in, κ = 1. An only migrant cannot be replaced; in a large network, the loss is fully restored.

## C5_households_on_curve: All households on the receipt curve

Every MMP household interviewed 2010–19 with complete records, at its expected 2019 receipt (calibrated model, κ = 0; score = log-odds), bubbles sized by the number of households with that score. Households with no linked migrant all sit at the baseline expit(b0) and are not used in the calibration. Black diamonds: observed MMP receipt by score band (bands with at least 30 households). Bottom: number of households by score. The model separates households with a core migrant (upper half of the curve) from kin-only and unlinked households (bottom).

## C6_fit_by_need: How well the calibration fits, by household need

Share of households receiving, MMP 2019 (95% Wilson CI) and the calibrated model, by whether anyone at home works and whether a resident is 65 or older, for households with a core migrant and kin-only households (households with a complete roster). Need raises desire through two parameters (η_w, η_o) shared by all households. The clear miss: kin-only households with an older resident receive more often than the model gives them.
