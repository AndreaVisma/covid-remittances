# Figure plan: rebuilt population, inputs and E0 calibration

19 October 2026. Planned with the dataviz method: pick the form first, colour by the job it does, one y-axis per chart, categorical palette from the reference instance and validated before use.

**Status: drawn.** PNG only (user decision), from `model/make_figures.py` into `figures/`, with `figures/captions.md`. All figures in A–C are done except optional A5. C2 quotes the old-population numbers in a footnote and caption instead of plotting an old-population line (user decision).

Colour roles used throughout (reference palette, light mode):
- **Link types (categorical, 3 slots, fixed order everywhere):** Core = slot 1 blue `#2a78d6`, Close kin = slot 2 orange `#eb6834`, Distant kin = slot 3 aqua `#1baf7a`. Three slots pass all-pairs checks, so this also works in small multiples.
- **Data vs model (emphasis):** MMP data = dark ink (`text-primary`) dots with 95% Wilson intervals; model = blue line or dot. The old population, where shown, = de-emphasis gray, dashed.
- **Magnitude (sequential):** one blue ramp.
- **Model error (diverging):** blue/orange poles with a neutral gray at 0.

## A. Who is in the networks (rebuilt population)

| # | Question the figure answers | Form | Data |
|---|---|---|---|
| A1 | How large and of what type are household networks? | Column chart: households by network size (1…10+), stacked by network type (core only / core + kin / kin only); n on top of each column | `population/households.parquet` |
| A2 | What did the rebuild change? | Cumulative share of links by household network size, rebuilt (blue) vs old population (gray); direct labels "max 16" vs "max 169" | both populations |
| A3 | **Why distant kin are one link** (key justification) | Two small multiples, shared y-axis (MMP receipt rate): left by number of close kin (0, 1, 2, 3–4, 5+), right by number of distant relatives (1–4 … 20+). Bars with 95% intervals; n under each bar | `outputs/population_summary.csv` (+ counts for intervals) |
| A4 | What jobs do linked migrants hold, and how much is imputed? | 100% horizontal stacked bars, one per link type, six occupation groups (incl. Not working, Unknown); share imputed written at the bar end | `population/migrants.parquet` |

Optional: A5, map of origin states (households by state). It shows how much MMP leans on western-central Mexico, which is a limitation worth seeing once.

## B. Model inputs

| # | Question | Form | Data |
|---|---|---|---|
| B1 | How hard and how unevenly did COVID hit migrant employment? | Line chart 2019–2022, four occupation groups, direct-labelled; 2019 shaded as the calibration window; April 2020 annotated | `data-prep/outputs/cps_employment_rates_monthly.csv` (`rate_model`) |
| B2 | How much does a migrant send? | Histogram of MMP amounts (2019 USD) with the fitted lognormal (mean $326, σ 0.81); reference rules for the Banxico mean, MMP median and ENIGH recipient median, each labelled | `remittance_amounts*.{json,csv}` + MMP |
| B3 | What does the national need signal look like? | Line chart: S_t (reported COVID deaths, 10,000s/month) and excess deaths (variant); peaks labelled; 2022 shaded | `data-prep/outputs/salience_index_monthly.csv` |

## C. Calibration (E0)

| # | Question | Form | Data |
|---|---|---|---|
| C1 | **Does the baseline reproduce MMP, targeted and untargeted?** (main calibration figure) | Dot plot ("calibration ladder"): one row per cell, grouped by panel (network type [targeted] → link type → network size → survey period → household composition). MMP dot + 95% interval, model dot. Targeted rows marked; cells with n < 30 grayed | `outputs/e0-baseline/*.csv` (κ = 0) |
| C2 | Does protection rise with network size as in the data? | Three small multiples (core only / core + kin / kin only), receipt vs network size. MMP dots + intervals, model line; old-population model as gray dashed line to show the saturation fix | `validation_network_size.csv` + old run |
| C3 | Where is the model wrong, and by how much? | Diverging bar of model − MMP (pp), sorted, coloured by sign; the table view of C1 | same as C1 |
| C4 | Does 2019 tell us anything about κ? | Three small panels sharing the x-axis (κ 0–8): fit error (pp), fitted α_core / α_kin, triggered kin transfers per month. Separate panels, not a dual axis | `kappa_profile.csv`, `kappa_2019.csv` |
| C5 | Are household amounts plausible? | KPI row (stat tiles), not a chart: model vs ENIGH six-month median and mean, share receiving in six months, Banxico scale | `enigh_style_check.csv`, `run_settings.json` |

## D. E1a: job loss only (β = 0, ρ = 0, no benefits)

Run `python model/make_figures.py --e1` (E1 figures only). No suptitle on the PNG; titles and captions are in `figures/captions.md`. κ uses an ordinal blue ramp (κ = 0 light, κ = 4 dark); Banxico is primary ink; core blue, kin orange.

| # | Question | Form | Data |
|---|---|---|---|
| D1 | Does job loss alone reproduce the count growth? | Quarterly index (quarter / same quarter 2019): model κ = 0 and 4 with band, Banxico; 2022 shaded as held out | `outputs/e1-job-loss/aggregate_quarterly.csv` |
| D1b | Same, monthly | Monthly index (month / same month 2019); March 2020 annotated as excluded | `aggregate_monthly.csv` |
| D2 | Who carries the loss? | Two panels (κ = 0, 4): core and kin transfers minus counterfactual | `aggregate_monthly.csv` |
| D3 | Why did κ change so little under the former rule? | Bars: where April 2020 losses sit and how much kin recover; line: April index vs κ (ceiling) | `e1-all-core-rule/kappa_reach.csv`, `kappa_ceiling.csv` (`model/diagnose_e1_kappa.py`) |
| D4 | Does the trigger definition matter? | Dumbbell: April index at κ = 0 vs 4 for three triggers (variants not refitted, diagnostic only) | `e1-all-core-rule/kappa_rule_variants.csv` |
| D5 | How long are households below the counterfactual? | Three small multiples by network type: receipt / counterfactual, κ = 0 and 4 | `receipt_ratio_paths.csv` |
| D6 | Does network size protect? | Dot plots by size: interruption rate and USD drop, κ = 0 and 4 | `household_outcomes.csv` |

## R. E1a: base 'lost sending' rule vs former 'all core unable' rule

Run `python model/make_figures.py --e1-rules` after `model/run_e1_rule_comparison.py`. Colours: κ = 0 (same under both rules) gray, base (lost sending) κ = 4 blue, former rule κ = 4 orange, data in primary ink.

| # | Question | Form | Data |
|---|---|---|---|
| R1 | Does kin-for-kin stepping-in close the count gap? | Quarterly index, κ = 0 and κ = 4 per rule, Banxico | `aggregate_quarterly.csv` (both runs) |
| R2 | Does 2019 constrain κ, and how far can κ move April 2020? | Two panels: 2019 fit error vs κ; April 2020 index vs κ with Banxico reference | `e0-baseline/kappa_profile.csv`, `e1-all-core-rule/e0_kappa_profile.csv`, `kappa_ceiling_rules.csv` |
| R3 | Do the two rules fit 2019 network sizes equally well? | Small multiples by network type: MMP dots + Wilson intervals, model lines | `e1-all-core-rule/e0_validation.csv` |
| R4 | Which households does the new rule protect? | Small multiples by network type: receipt / counterfactual | `receipt_ratio_paths.csv` (both runs) |
| R5 | Does network size protect under each rule? | Dot plots by size: interruption rate, USD drop (negative = more than counterfactual) | `household_outcomes.csv` (both runs) |
| R6 | How many extra kin transfers does each rule produce? | Kin transfers minus counterfactual over time | `aggregate_monthly.csv` (both runs) |

## M. E2 compatibility map (first pass)

Run `python model/make_figures.py --e2` after `model/run_e2_map.py`.

| # | Question | Form | Data |
|---|---|---|---|
| M1 | Which (β, κ) fit the 2020–21 counts? | Heatmap of quarterly log RMSE (single-hue blue ramp), compatible cells outlined, best cell marked, τ_C on the colour bar | `outputs/e2-map/map.csv` |
| M2 | What do the best cells miss? | Quarterly index: best cell, other compatible cells, β = 0 at κ = 4, Banxico vs trend; 2022 shaded | `outputs/e2-map/quarterly_paths.csv` |

## Notes before drawing

- **95% intervals for MMP cells** (Wilson) need cell counts and receipt; both are already in the E0 tables.
- **C2's old-population line** needs the first E0 run, whose outputs were overwritten. To rerun it, either let `abm.POPULATION` switch back to the old population, or drop the overlay and state the old numbers in the caption.
- **Microdata:** the figures show aggregates only. Raw population rows never go into a published page.
- **Proposed core set for the paper / appendix:** A3, B1, C1, C2, C4. The rest are supplementary.
