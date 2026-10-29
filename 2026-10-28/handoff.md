# Handoff: consolidation, 28 October 2026

**Read this first.** It replaces `network-compatibility-model/handoff.md` as the entry point. The old file is kept as the archive (full history, decisions and numbers since 19 Oct). The formal model is in `network-compatibility-model/model-specification.md`.

Rule from now on: **no new experiments until the base model is fixed and every output is rebuilt from it.** A later experiment must earn a place in the paper outline below.

## 1. The paper in one paragraph

Forecasts expected remittances to Mexico to collapse in 2020; they did not. We build a parsimonious agent-based model on the actual household–migrant networks of the Mexican Migration Project (MMP), drive it with observed US job losses by occupation (CPS), and ask what the mechanisms proposed by survey studies (kin stepping in, need response, US benefits) can and cannot explain. One chapter of a PhD on remittance networks in crises. Target: transfer **counts** (Banxico transactions) relative to their pre-COVID trend.

## 2. The story so far (what holds)

1. **The puzzle is a 2021 level shift, not 2020 growth.** Against the 2017–19 trend (+6.3% a year), Banxico counts are flat in 2020 (dip to 0.91 in Apr 2020; 0.974 in 2020Q2) and jump in 2021 (1.05 → 1.11 by 2021Q4), staying high to 2023. Annual growth: 7.3% in 2020 (= trend), 13.4% in 2021. Robust to the trend window (2010–19 … 2018–19: 2020Q2 0.97–0.98, 2021Q4 1.09–1.16). The flatness of 2020 is itself the puzzle relative to the forecasts.
2. **Job loss alone implies a ~12% dip in 2020Q2; the data show ~3%.** On the real networks, with no response, the index falls to 0.876 (2020Q2; 0.853 in Apr). Something filled the gap. Candidates: kin stepping in (κ), benefits (π), sending from savings (not in the model). **Separating them is the main open problem.**
3. **Stepping-in can close the gap, but is not identified by itself.** 2019 data do not constrain κ; the 2020Q2 dip does, but it is a single moment and benefits fill it too. Coordination rule, κ = 1 (full replacement): 2020Q2 0.975.
4. **Need (COVID deaths) adds little and has the wrong shape.** Deaths come in waves, the excess rises steadily. Best β × κ cell: RMSE 0.043, no cell within the tolerance 0.04. β ≈ 0.06 raises a core migrant's monthly sending probability by a few points at the Jan 2021 peak.
5. **The 2021 jump is mostly unexplained.** Decomposition of raw growth to 2021Q4 (+24.4 log points): +12.3 trend, −0.2 job loss net of stepping-in, +2.0 need, **+10.5 unexplained**. About half of the USD excess is a rise in the average amount, which the count target does not see. The CPS Mexican-born stock is flat, so transfers per migrant rose.
6. **March–April 2020 is a timing shift, not a mechanism to model** (user, 29 Oct). The peso fell from 18.8 (Feb, monthly mean) to 22.4 (Mar; daily peak 25.1 on 24 Mar) and 24.2 MXN/USD (Apr). Counts vs trend: 1.010 (Feb), **1.087 (Mar)**, **0.912 (Apr)**, 0.984 (May); the real amount per transfer also spikes in March (1.166 vs 2019). **March + April together are exactly on trend (1.000)**, March–May 0.994: migrants rushed to send at the weak peso (and before lockdowns) and pulled April's transfers forward. Not modelled; note it in the data section and the 2020 discussion. Consequences: the April dip is not a clean job-loss moment, which is why it is not used to fit the first-round π; the 2020Q2 target (0.974) includes the depressed April, so the underlying Q2 is, if anything, higher (robustness: Q2 with Mar–Apr pooled).
7. **Stimulus checks are visible in 2021, not in April 2020** (preview, π = 1). Banxico peaks locally in Jan 2021 (1.065 vs Feb 1.010) and Mar 2021 (1.070 vs 1.010 / 1.054), where the model spikes; in April 2020 the model spikes (1.045 at κ = 0) while Banxico has its deepest dip (0.912). Consistent with eligibility: the first check excluded ITIN filers and mixed-status couples; later rounds were wider.
8. **Households (parked):** aggregate stability hides concentrated losses (a typical path: ~20% of receiving households lose something, ~1.5% lose everything: single links whose migrant lost the job). Largely driven by assumptions (no correlation within networks, imputed kin occupations, fixed amounts) until validated against ENIGH.

## 3. The base model (decided)

- **Population:** every connected MMP household interviewed 2010–19, once: 3,405 households, 7,999 links (core = head/spouse/child in the US; close kin; one aggregated distant-kin link).
- **All migrants can work** (28 Oct): "Not working" on the last trip = unknown occupation (all-occupation CPS rate); no age rule.
- **Employment:** CPS monthly rates by occupation group, latent AR(1), φ ≈ 0.91; no correlation within networks (ρ = 0).
- **Ability to send:** employed or receiving a benefit that month (yes/no gate; a model assumption, not an MMP measurement).
- **Benefits (29 Oct):** stimulus checks only, on the ability margin: a non-employed migrant can send with probability π in a stimulus month. **π = 1 in Jan and Mar 2021, π = 0 in Apr 2020** (round-1 eligibility; April distorted by the March rush). Not a map axis. With π fixed, the Jan/Mar 2021 spikes are a second moment for κ (`spike_*` columns in `recipient-level/outputs/household-model/map.csv`). Result (β = 0, base mapping): Banxico spikes +4.1 / +3.6; model +7.8 / +5.8 at κ = 0, +4.6 / +3.4 at κ = 0.7, +3.3 / +2.4 at κ = 1; spike RMSE minimal at **κ ≈ 0.7–0.8** (0.37 log points), consistent with the quarterly fit, which also improves with κ. Under the lower-bound mapping the spikes want κ > 1. Robustness: γ shift on all migrants (observationally equivalent, γ ≈ 0.05 without π), π(Apr 2020) = 1.
- **Receipt (since 28 Oct, household level):** household receipt logit on the numbers of able core and kin migrants plus need η, calibrated to 2019 MMP receipt (§4.1). The migrant-level version (two-stage Bernoulli, `α_core`, `α_kin`) is the comparison model.
- **Stepping in: replacement share κ ∈ [0, 1]** of the household's lost sending (coordination rule in the migrant model, same κ in the household model).
- **Need:** `β × S_t`, S_t = reported COVID deaths in Mexico (10,000s); 0 in 2019.
- **Amounts:** accounting only (29 Oct): model USD = model counts × observed Banxico average amount, split into US CPI-U inflation and the real amount (`recipient-level/outputs/usd_accounting.csv`, Fig. S10; CPI from `data-prep/us_cpi.py`). Real amount per transfer vs same quarter of 2019: +1.0 (2020Q2), +9.4 (2021Q4), +3.2 (2022Q4) log points; annual real USD 1.09 (2021) → 0.98 (2024) → 0.96 (2025). Argument for the paper: the lasting change is in counts; the 2020–21 real amount rise is reported, not explained.
- **Comparison:** COVID run / no-COVID run (2019 conditions, common random numbers) vs Banxico / 2017–19 trend. Fit window 2020Q2–2021Q4 (quarterly; monthly for benefit timing); 2022 held out. Tolerance τ_C = 0.04.
- **Out of scope (user decisions):** channel shift (cash → digital) and transit migrants are not modelled; amounts are not modelled; 2022+ not claimed.

## 4. Decisions taken 28 Oct 2026 (user)

1. **Model level: household level.** The recipient-level receipt logit (`network-compatibility-model/recipient-level/run_household_model.py`, H3: any able core migrant + log able core + log able kin + need; κ = replacement share) is the base model. The migrant-level ABM becomes a comparison/robustness model.
   - Consequences: outcomes are **receipt** (household receives in a month), not transfer counts or USD. Needed: (a) the mapping from household receipt to Banxico transaction counts (currently the index is on receipt; state the assumption, e.g. transfers per receiving household constant, and check against the migrant model); (b) for the household section, an amount equation (e.g. ENIGH) or receipt-only outcomes.
   - **Kin term (29 Oct, done):** kin × [no core migrant at all] added (`log_kin_no_core`). In 2019 MMP kin matter only without core (kin −0.13 with core, +1.30 more without; ΔAIC −30). Structural, not "no *able* core": the two fit identically in 2019, but the able version would hard-wire stepping-in and compete with κ (keep it as a robustness check). Effect: 2020Q2 at κ = 0 0.940 → 0.930; recipients losing > 25% of receipt months 1.7% → 3.8%, losing all 0.3% → 0.9% (κ = 0).
   - **Count mapping (29 Oct, done):** base = transfers per receiving household constant; lower bound = receipt × expected able senders per receiving household (migrant-level M1 θ; stepping-in restores receipt, not lost transfers). Columns `*_lower_bound` in `outputs/household-model/map.csv`. 2020Q2: κ = 0 0.930 / 0.905; κ = 1 0.969 / **0.939** (Banxico 0.974). **The mapping matters:** under the bound, stepping-in alone cannot close the 2020 gap; benefits (π) or savings must contribute.
2. **Benefits: one π for all three stimulus checks, profiled** as a map axis next to κ (and β). Not fitted per round; the April 2020 vs 2021 contrast is reported as a finding, with the eligibility rules as the explanation.
   - Needed: benefits (stimulus flag, π) in the household model's ability distribution (the able core/kin counts); check that `run_household_model.py` takes `stimulus_months` and π.
3. **Savings margin: robustness check only.** Base keeps the yes/no gate. One sensitivity: unemployed migrants without benefits send with a reduced probability (e.g. 25% and 50% of routine), reporting how κ and π move.
4. **Household need η: in the base model.** In the household logit it enters directly (no resident worker, any older resident; fitted η_w ≈ 0.42, η_o ≈ 0.38 in the 27 Oct comparison).

5. **Need (29 Oct, user): option 6 is the main modelling story for 2021.** Need brings in more senders per receiving household (θ_j,t = expit(α_j + β_s N*_t)), with a cumulative, non-decaying need stock (λ = 1). Receipt is unchanged; transfers per receiving household rise. Experiments and results: `recipient-level/outputs/need-margins/README.md` (6a closes 88% of the 2021Q4 gap, 2022 holdout 0.024). Caveats: a placebo ramp fits as well (identification is by timing and by the intensive-margin evidence, not by the need series); the cross-section (ENIGH cells, state panel) shows no need gradient. Supporting fact: ENIGH receiving households +5% 2018→22 while Banxico counts +40%, so count growth is transfers per receiving household. **Chosen: 6a (all migrants; core θ 44% → 60% by end-2021) and series E (excess deaths, cumulative).** The count mapping is the base mapping (transfers per receiver constant except need-activated senders), stated as an assumption; the lower bound overshoots the stimulus spikes. Still needed: an external anchor for β_s (ENIGH 2016 pre-period) and putting 6a + E into `run_household_model.py` in place of β S_t.

**Base map:** κ × π (× β), household model, criterion as in §3 (quarterly for the headline, monthly for benefit timing).

## 5. Plan

**Step 1. Decisions: done** (§4, 28 Oct).

**Step 2. One rebuild script.** A single `run_all.py` that rebuilds every output from the base model (household level, η in, stimulus π profiled): E0 calibration → E1 (job loss, + benefits) → E2 map (κ × π × β) → decomposition → savings robustness → migrant-level comparison → figures. Delete or archive outputs of rules and variants that are not in the paper. Currently **stale (old "Not working" coding):** `outputs/e0-baseline`, `e1-job-loss`, `e1-all-core-rule`, `e2-map` (lost-sending), `e1-distribution*`, `e0-need`, `figures/`, and the lost-sending rows of `e1-coordination/kappa_ceiling_rules.csv`. Up to date: `e1-coordination`, `e2-map-coordination`.

**Step 3. Validation** (needs data; user action: download the ENIGH `ingresos` tables for 2016, 2018, 2020 from INEGI):
- ENIGH 2018 → 2020 relative change in receipt rate (4.73% → 4.38%, −7%), with 2016 → 2018 as the normal-times change.
- Model equivalent: received at least once in the six months before the interview month (fieldwork Aug–Nov), COVID vs no-COVID run.
- Cells observable in both surveys: resident worker, older resident, rural/urban, traditional sending states. Design-based standard errors (`upm`, `est_dis`, `factor`).
- If `ingresos` has the monthly breakdown (six reference months), a monthly receipt profile Feb–Oct 2020: a direct test of the April dip at household level.
- Out-of-sample check from the colleague feedback: calibrate with 2018 CPS, run 2019 forward, expect ≈ 1.0.

**Step 4. Paper outline**, then write.

## 6. Paper outline (draft)

1. **Introduction.** Forecasts of collapse; flows rose; survey vs macro explanations; what a network model can add.
2. **Data.** Banxico counts and amounts; MMP networks; CPS employment by occupation; benefit schedule; ENIGH.
3. **The puzzle restated.** Fig. 1: Banxico counts vs trend, monthly, with the trend-window band. 2020 flat, 2021 level shift. Note the March 2020 rush at the weak peso and the April payback (Mar + Apr on trend); mark it on Fig. 1, not modelled.
4. **Model.** Short ODD; Fig. 2: the networks (core / close kin / distant kin, sizes).
5. **What job loss implies.** Fig. 3: CPS employment by occupation and the model's job-loss path vs Banxico (the ~12% gap in 2020Q2).
6. **What fills the gap.** Fig. 4: compatibility map over κ, π (and β); what the 2020 dip and benefit timing can and cannot separate.
7. **What remains: 2021.** Fig. 5: decomposition of raw growth (trend, job loss net, need, benefits, unexplained); candidates for the residual (channel shift, transfers per sender, amounts), discussed, not modelled.
8. **Households.** Fig. 6: who loses, with the ENIGH validation. Only if Step 3 supports it; otherwise a short section with caveats.
9. **Limitations and conclusion.**

## 7. Where things are

- Code: `network-compatibility-model/model/abm.py` (library), `run_*.py` (experiments), `recipient-level/` (household model), `data-prep/` (inputs).
- History and all earlier numbers: `network-compatibility-model/handoff.md` (archive).
- Keep in mind: the repository is **public**; never commit `data/` or MMP/IPUMS-derived microdata.
- The user prefers simplicity: extensions are robustness checks unless they enter the outline.
