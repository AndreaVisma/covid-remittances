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

## The household model through E0–E2 and the distribution (28 Oct 2026)

`run_household_model.py` (~3 min; outputs in `outputs/household-model/`; `--original-coding` keeps "Not working" migrants unable, E0/E1 only, in `outputs/household-model-original-coding/`). Figures S3–S8: `make_figures.py --model`. Base: H3 with "Not working" as unknown occupation, plus the two map axes:

score = H3 + β·S_t + κ·[a core migrant cannot send]·log(1 + able kin)

κ is stepping-in at household level: when an eligible core migrant is out of work, the household's able kin count for more. S_t = reported COVID deaths / 10,000, as in the migrant model. Receipt maps to Banxico counts assuming a constant number of transfers per receiving household. ρ = 0, no benefits.

**E0, κ in 2019 (S3).** The six coefficients are refitted for each κ. The κ ≥ 0 profile prefers κ = 0, and κ > 0.8 falls outside the 95% profile limit, but the whole range κ = 0–4 costs only 2.6 log-likelihood points. Unrestricted, κ = −2.67 (SE 0.88, +7.1 log-likelihood); with the original coding −2.06 (SE 0.83). In 2019 the trigger mostly marks large core networks, so this reads as a size effect (kin matter less where core migrants are many), not as evidence against crisis stepping-in. It is still a tension to state: 2019 does not support κ > 0, while the 2020 dip needs κ ≈ 2.

**E1, job loss only (β = 0, S4).** Receipt index in 2020 Q2: 0.940 (κ = 0), 0.960 (κ = 1), 0.977 (κ = 2), 0.999 (κ = 4); Banxico 0.974. The household model loses less than the migrant model at κ = 0 (transfers 0.875), because losing one of several core migrants barely moves receipt. 2021: the model returns to 0.99–1.00, while Banxico rises to 1.05–1.11. RMSE 2020 Q2–2021 Q4: 0.069 (κ = 0) to 0.057 (κ = 4). The original coding gives nearly the same paths (2020 Q2 0.940 at κ = 0, 0.985 at κ = 4).

**E2, β × κ map (S5, S6).** **0 of 189 cells are within τ_C = 0.04**; the best is κ = 0, β = 0.08, RMSE 0.043, the same RMSE as the migrant-model map (κ = 3, β = 0.06). β = 0.06–0.08 gives 0.043–0.045 for every κ ≤ 1.5, so κ is not identified once β is free. The best path follows the death waves (2021 Q1 peak, 2021 Q2 drop) instead of the steady rise. Holdout 2022: RMSE 0.09. A Gaussian likelihood on the quarterly log residuals (SD 0.0143, the 2017–19 trend residual) is in `map.csv` (`loglik_gaussian`, `delta_loglik`). Within β = 0 it favours large κ: κ = 4 is 24 points below the best cell.

**Who lost out (β = 0, κ = 0/1/2; S7, S8).** 200 paired paths, Apr 2020 – Dec 2021, using the abm employment process and common draws for employment and receipt. The outcome is months with receipt, because the household model has no amounts.
- Aggregate expected change: −2.1% (κ = 0), −1.3% (κ = 1), −0.7% (κ = 2).
- In a typical path, 8 / 7 / 6% of receiving households receive in fewer months, and 0.3 / 0.2 / 0.2% receive in no month.
- Core-only households lose −2.4% at every κ: nobody steps in. Single-link households lose −2.7%. Sales/personal services is the hardest-hit occupation (−3.0%), agriculture the least (−0.5%).
- With κ > 0, core + kin households and large networks gain: 10+ links +3.5% at κ = 2, the same over-replacement as in the migrant model.

**Read.** At the aggregate level the household model tells the same story as the migrant model:
- The 2020 dip fits with stepping-in or a mild need response.
- 2021 is not explained, and no cell is within τ_C.
- The likelihood comparison of mechanisms (colleague feedback) does not discriminate either.

For RQ2 it is coarser. It sees only whether anything arrives, so it misses losses within receiving months (fewer senders, less money). That is why it shows far fewer losers than the migrant model's USD outcome (H1: ~20% lose something, 1.5% everything). A household-level amount equation (ENIGH, decision C) would be needed to report money lost at household level.
