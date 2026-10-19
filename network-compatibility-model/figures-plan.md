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

## C. Calibration (E0)

| # | Question | Form | Data |
|---|---|---|---|
| C1 | **Does the baseline reproduce MMP, targeted and untargeted?** (main calibration figure) | Dot plot ("calibration ladder"): one row per cell, grouped by panel (network type [targeted] → link type → network size → survey period → household composition). MMP dot + 95% interval, model dot. Targeted rows marked; cells with n < 30 grayed | `outputs/e0-baseline/*.csv` (κ = 0) |
| C2 | Does protection rise with network size as in the data? | Three small multiples (core only / core + kin / kin only), receipt vs network size. MMP dots + intervals, model line; old-population model as gray dashed line to show the saturation fix | `validation_network_size.csv` + old run |
| C3 | Where is the model wrong, and by how much? | Diverging bar of model − MMP (pp), sorted, coloured by sign; the table view of C1 | same as C1 |
| C4 | Does 2019 tell us anything about κ? | Three small panels sharing the x-axis (κ 0–8): fit error (pp), fitted α_core / α_kin, triggered kin transfers per month. Separate panels, not a dual axis | `kappa_profile.csv`, `kappa_2019.csv` |
| C5 | Are household amounts plausible? | KPI row (stat tiles), not a chart: model vs ENIGH six-month median and mean, share receiving in six months, Banxico scale | `enigh_style_check.csv`, `run_settings.json` |

## Notes before drawing

- **95% intervals for MMP cells** (Wilson) need cell counts and receipt; both are already in the E0 tables.
- **C2's old-population line** needs the first E0 run, whose outputs were overwritten. To rerun it, either let `abm.POPULATION` switch back to the old population, or drop the overlay and state the old numbers in the caption.
- **Microdata:** the figures show aggregates only. Raw population rows never go into a published page.
- **Proposed core set for the paper / appendix:** A3, B1, C1, C2, C4. The rest are supplementary.
