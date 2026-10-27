# Recipient-level (household) receipt logit

Exploratory comparison, 27 Oct 2026. The question: should the model be specified at the recipient level, with household receipt as a logit of the household's able migrants and need, instead of independent migrant-level senders?

Run: `python network-compatibility-model/recipient-level/compare_household_logit.py` (~30 s). Details are in the script docstring. Outputs in `outputs/` (aggregates only): `comparison.csv`, `fits.csv`, `panel_rmse.csv`, `validation.csv`, `shock_response.csv`, `run_settings.json`.

Figures: `python network-compatibility-model/recipient-level/make_figures.py` (after the comparison). `figures/S1_receipt_curve_examples.png`: the fitted curve with illustrative households and job-loss arrows. `figures/S2_receipt_curve_households.png`: all sample households at their expected 2019 score, with observed receipt by score band. Captions in `figures/captions.md`.

## Set-up

Same 3,252 MMP households (`remus` known, complete records), same 2019 CPS employment, κ = 0, all fitted by maximum likelihood. The household logit uses the exact distribution of the numbers of *able* core and kin migrants, so it keeps the employment shock and can be run on 2020–21.

| Model | Receipt | Parameters |
|---|---|---|
| M1 / M2 | migrant level, 1 − ∏(1 − e·θ) | α_core, α_kin (+ η_w, η_o) |
| M1f / M2f | as M1/M2 plus a household floor: receipt from other sources | + floor |
| H0 | household logit, linear able core and kin counts | 3 |
| H1 / H2 | log(1 + count) terms (+ need) | 3 / 5 |
| H3 | H2 + any able core migrant | 6 |

## Results

**A zero-receipt artefact in the migrant model.** 258 households have no migrant who can ever send: all their linked migrants are coded "Not working" in MMP (1,417 of 7,999 migrants). The migrant model gives them P = 0; 21 of them receive, costing −580 log points, about the whole raw likelihood gap. The floor (M1f/M2f, about 5.5%) fixes this. This affects the current ABM regardless of the comparison.

**The fair comparison:**

| | H3 (6) | M2f (5) | M1f (3) | H1 (3) |
|---|---|---|---|---|
| Log-likelihood | −1171 | −1201 | −1205 | −1229 |
| AIC | 2354 | 2412 | 2415 | 2463 |
| CV log loss (by community) | 0.363 | 0.374 | 0.373 | 0.381 |
| CV log loss, households with an eligible migrant | 0.371 | 0.381 | 0.380 | 0.390 |
| χ² on the 12 need cells | 42 | 76 | 84 | 108 |
| RMSE pp: network type / size / older / workers | 1.8 / 3.7 / 4.0 / 2.1 | 3.2 / 5.5 / 4.5 / 3.5 | 3.4 / 5.7 / 5.2 / 3.7 | 4.2 / 7.1 / 5.9 / 4.5 |

The E0 cell-mean calibrations (M1-e0, M2-chi2) fit the network-type cells better by construction (0.35, 0.84 pp), but H3 fits size (3.7 vs 4.0–4.1 pp) and the 12 need cells (42 vs 55, M2-chi2's own target) better.

**What drives the difference:**
- A log form alone does not beat independent senders: H1 is worse than M1f. What wins is H3's shape: **the first able core migrant raises the log-odds by 2.67, each further one adds little** (0.59 per log unit). Independent senders make every extra core migrant add a near-independent chance, hence the too-steep size gradient (5+ core migrants: model 96%, MMP 76%).
- Kin add a little (0.24 per log unit, SE 0.12).
- **Need is clean at the household level:** η_w = 0.42 (SE 0.15), η_o = 0.38 (SE 0.13). In the migrant-level M2f, η_o turns negative (−0.37): the independence structure distorts need.

**Implied job-loss response** (receipt index, COVID / same month of 2019, κ = 0): 2020Q2 0.940 (H3) vs 0.904 (M1) and 0.928–0.930 (M1f, M2f). The household form loses less when one of several migrants loses their job. This is receipt, not transfer counts.

## Read

The household logit fits better, clearly but not dramatically, once the zero-receipt artefact is fixed: ΔAIC 58 for one more parameter, CV log loss about 3% lower, cell errors roughly halved against the likelihood-fitted migrant model. The gain comes from one structural fact: receipt depends mostly on having *any* able core migrant. Stepping-in (κ) is not tested: 2019 identifies it in neither family. Open: the mapping from receiving households to Banxico transfer counts, and κ as a lost-core × kin term.

## Variant: "Not working" as unknown occupation (27 Oct 2026)

`compare_household_logit.py --not-working-as-unknown` (outputs in `outputs/not-working-unknown/`; figure `figures/S2_receipt_curve_households_not_working_unknown.png` via `make_figures.py --not-working-as-unknown`). Migrants coded "Not working" (1,417) get the all-occupation CPS rate instead of never being able to send. Motivation: households whose core migrants are all "Not working" still receive in 43% of cases (77 households).

- **The recoding fixes the zero-receipt artefact without a floor:** migrant model M1 log-likelihood −1716 → −1189; the floor shrinks to about 2.4% and adds little (M1f −1186).
- **Both families fit better, and the gap between them widens:** H3 vs M2f log-lik −1145 vs −1182, AIC 2303 vs 2374, CV log loss 0.357 vs 0.370, χ² need cells 39 vs 73; RMSE pp network type 0.8 vs 2.3, size 3.1 vs 5.5, older 3.4 vs 4.1, workers 1.4 vs 2.7.
- H3: any able core +2.98 (SE 0.24), log core 0.39 (SE 0.21), log kin 0.27 (SE 0.11), no worker 0.36 (SE 0.15), older 0.28 (SE 0.13).
- **The middle of the curve gets emptier, not fuller:** the "Not working" core households move up to the other core households. The clusters are a composition fact (core vs kin-only), not created by the any-core term (raw MMP: 6% with no core migrant, 59% with one, then 66–76%).
- Remaining misfit: kin-only households with high scores (many kin and/or need) receive more than predicted (band around −2.35: MMP 19% vs ~9%).
