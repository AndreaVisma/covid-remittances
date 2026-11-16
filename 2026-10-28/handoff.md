# Handoff: consolidation, 28 October 2026

**Read this first.** It replaces `network-compatibility-model/handoff.md` as the entry point. The old file is kept as the archive (full history, decisions and numbers since 19 Oct). The formal model is in `network-compatibility-model/model-specification.md`.

Rule from now on: **no new experiments until the base model is fixed and every output is rebuilt from it.** A later experiment must earn a place in the paper outline below.

## Start here (state on 16 Nov 2026)

**The base model is the willing-and-able model in the simplified forms (§0a), fully recalibrated and rerun.**

- **Model:**
  - desire d = expit(α_role + δ·(other core migrants) + η·need + β·N*_t);
  - effective senders E = W + min(Bin(L, κ), R), which equals min(D, A) at κ = 1;
  - receipt s = expit(b0 + b·E);
  - transfers s·max(E, 1);
  - ability a = max(e, S_t).
- **Parameters:**
  - survey 2010–19: α_c 0.26, α_k −3.38, δ −0.35, η_w 0.33, η_o 0.16, b0 −5.04, b 8.25;
  - ENIGH 2018: ρ = 0.90;
  - COVID years: κ = 1, β = 0.153 (`outputs/shocks-simple/`).
- **The national results are the same as with the earlier forms;** the household results shift slightly (§0a).
- **LaTeX for the Methods** (all changed equations and the parameter table): `2026-10-30/methods_equations_simplified.tex`.
- **Calibration figures:** `figures/calibration/` C1–C6. **Theoretical tests:** `figures/theoretical-tests/` T1–T5 (§0b). **Paper figures:** `figures/paper/` R1–R5. All on the new forms.

**Next actions, in order (user to choose):**
1. Refresh the numbers in `paper_draft.tex` and the Methods text: the edits are listed under "Text in the Methods to change by hand" in §0a. Reconcile the no-migrant sample: 3,730 households, 0.6% observed (C1), vs 3,749 and 0.7% in the draft.
2. The 2022 margin tension. The model puts a rise in need on households receiving (+5.8%) rather than transfers per receiving household (+3.9%); ENIGH shows −2.7% / +9.3%. T5 shows this is structural. Decide: state it as a limitation, or test need raising transfers per sender.
3. Theoretical tests, next step (§0b): correlated job loss within households, network structure, occupation-specific shocks, targeted need (η).
4. Still on the earlier forms: `robustness/calibrate_own_year.py`, `robustness/calibrate_period.py`, the persistence scripts (closed extension), and `figures/S1_receipt_curve_examples_sample.png`.
5. Paper choices: whether to keep the grey log-form line in C3; where the min(D, A) identity goes (Methods or Discussion).

**Run order on the new forms** (each reads the previous one's outputs):
1. `calibrate_mmp.py`
2. `household_layer.py --calibrate-only --kappa 1`
3. `run_shocks.py --tag simple` (and `--quick --need E:0 / R:1 --tag simple-need-E0 / -R1`)
4. `household_layer.py --rho 0.8992`
5. `run_households.py`, `enigh_aligned.py`
6. Figures: `make_calibration_figures.py`, `make_kappa_effect_figure.py` (with `PYTHONIOENCODING=utf-8`), `make_transition_figures.py`, `make_transition_kappa.py`, `make_paper_results_figures.py`, `theoretical_tests.py`
7. `check_special_cases.py` checks the library (all four checks pass).

On Windows, a run can be stopped for low memory; `run_shocks.py` now saves after each κ.

## 0a. Simplified functional forms (13 Nov 2026; recalibrated and rerun 13–16 Nov)

**Why:** feedback on the Methods section asked for simpler functional forms (one desire equation for core and kin, no logs, the receipt curve in one line). Decided with the user on 13 Nov 2026. The numbers in §0 below and in `paper_draft.tex` are from the earlier forms; the new ones are in this section and in "Start here".

**The model in the new forms (equation numbers as in the Methods draft):**

- **Ability (eq. 3), unchanged in substance.** Since π = 1, write a_{j,t} = max(e_{o(j),t}, S_t): every migrant is able in a check month. π drops out of the parameter table (12 → 11 parameters). Not written as an expit: e is an observed CPS rate, not an index, and an expit would add a parameter with no clear meaning.
- **Desire (eq. 4): one equation for both roles.**
  d_{j,t} = expit(α_{r(j)} + δ·M_j + η_w·NoWorker_h + η_o·Older_h + β·N*_t), where M_j is the number of core migrants in j's household *other than j* (N^c_h − 1 for a core migrant, N^c_h for kin). This replaces g·[core]·log N^c_h.
  - One δ for both roles: kin defer to core migrants just as core migrants coordinate among themselves. Kin counts are not used, because they are partly a coding artefact (distant relatives are merged into one link) and kin desire is low.
  - α_c is the desire of a household's only core migrant; α_k is the desire of kin in a household without core migrants (it was the desire of an average kin migrant).
  - Not N^c_h − 1 for everyone: that would give kin in kin-only households M = −1, a desire boost.
- **Effective senders (eq. 6), one line with a role index:** E^r = W^r + X^r, X^r ~ Binomial(L^r, κ), r ∈ {c, k}. Replacements are capped by the reserve, core roles first, and a replacement takes the lost sender's role (all unchanged).
- **Receipt (eq. 7), one line, linear:** s_{h,t} = expit(b0 + b_c·E^c + b_k·E^k). This replaces b_any·[E^c > 0] + b_c·log(1 + E^c) + b_k0·[N^c = 0]·log(1 + E^k). Kin senders now raise receipt in every household; before, a core household whose kin sent received in about 1% of months. The expit saturates, so a linear term reproduces the old one-sender levels (97% core, 89% kin) and adds almost nothing for a second sender.
- **Tests built into the calibration:** δ_c = δ_k (Stage 1, LR test) and b_c = b_k (Stage 2, LR test). If b_c = b_k is not rejected, the receipt curve becomes expit(b0 + b·E). Roles then stop mattering for receipt, and eq. 6 becomes E = W + min(X, R), X ~ Bin(L, κ) (capping the total equals capping core first, then kin). This option is **not implemented yet**: decide after the test.
- **Paper text to rewrite after recalibration:** the paragraph "In these households kin senders do not change the chance of receiving…", the parameter table, and the 60% / 97% / 89% / 1% figures. Typos noted in the 13 Nov Methods draft: "lostt", "two component", "carry over across period", "We  core migrant… of them .".

**Code changes (13 Nov 2026, all in `2026-10-30/`):**

- `willing_able.py`
  - `desire(..., delta, delta_k=None)` uses `other_core(pop)` (M_j). `delta_k` exists only for the equality test.
  - New `receipt_prob(b, Ec, Ek)` is the single place the receipt curve is written. `receipt_grid` and `design` use it (3 columns).
  - `B_NAMES = [b0, b_c, b_k]`. New `CAL_FILE = calibration_simple.csv`, the default of `load_calibration`. `params_for` returns `delta` and fails loudly on a calibration file from the earlier forms.
  - The replacement and state tables are unchanged.
- `calibrate_mmp.py`
  - Stage 1 fits (α_c, α_k, δ), plus the δ = 0 check against M1 and the separate-δ test.
  - Stage 2 fits (η_w, η_o, b0, b_c ≥ 0, b_k ≥ 0) and refits with b_c = b_k for the LR test.
  - Outputs carry the suffix `_simple`; the earlier `calibration.csv`, `stage1*.csv` stay as the record of the old forms. Run time about 20 min.
- `check_special_cases.py`: check 2 (nesting in `run_household_model`) is replaced by an analytic check on single-migrant households. All four checks pass (13 Nov).
- `run_shocks.py` reads `calibration_simple.csv` by default.
- `p['g']` was renamed to `p['delta']`, and the copied receipt formula replaced by `wa.receipt_prob`, in: `run_households.py`, `household_layer.py`, `make_transition_figures.py`, `make_transition_kappa.py`, `make_kappa_effect_figure.py`, `persistence_a.py`, `persistence_a2.py`, `persistence_b.py`, `run_persistence_b.py`.
- **Not updated (still the earlier forms):**
  - `robustness/calibrate_own_year.py` and `robustness/calibrate_period.py` have their own Stage 1 and Stage 2 code.
  - Figure labels and the hand-coded desire in `make_calibration_figures.py` (C3, C5 text) and `sample_receipt_figure.py`.

**Calibration results (13 Nov 2026; `outputs/stage1_simple.csv`, `stage1_by_core_simple.csv`, `calibration_simple.csv`, log `outputs/calibrate_simple.log`):**

- **Stage 1:** α_c = 0.26, α_k = −3.38, δ = −0.35; loglik −1143.5 (old forms −1139.5; δ = 0: −1189.0).
  - Core desire for 1 / 2 / 3 / 4 core migrants: 56 / 48 / 39 / 31% (old 60 / 43 / 34 / 28%).
  - Kin desire: 3.3% with no core migrant, 2.4% with one.
  - Receipt by number of core migrants (1 / 2 / 3 / 4+): fitted 0.547 / 0.707 / 0.757 / 0.711 vs observed 0.588 / 0.660 / 0.692 / 0.748 (old forms 0.584 / 0.666 / 0.707 / 0.737). A linear δ cannot bend the way the data do: households with one core migrant are underpredicted and those with 2–3 overpredicted.
  - **δ_c = δ_k not rejected:** δ_c −0.35, δ_k −0.94; LR 1.01, p = 0.31. δ_k is imprecise because only mixed households identify it.
- **Stage 2 (κ = 0):** η_w 0.33, η_o 0.12; b0 −4.38, b_c 7.81, b_k 5.96; loglik −1137.8, AIC 2285.6 (old 2274.7 with one more parameter).
  - Receipt with no sender is 1.2% (no-migrant households: 0.7% observed); with one core sender 96.9%; with one kin sender 82.9%.
  - **b_c = b_k not rejected:** LR 0.66, p = 0.42. With one b: b0 −5.04, b 8.25, η_w 0.33, η_o 0.16, loglik −1138.1. Receipt with no sender is 0.6% (closer to the 0.7% observed), with one sender of either role 96.3%.
- **Caveat on the b test (13 Nov):** it has little power and is not evidence that core and kin senders are alike. Only receipt is observed, so a lower b_k is close to observationally equivalent to a lower α_k. Stage 1 estimated α_k assuming every sender delivers, which already treats core and kin senders alike. Present one b as a normalisation (core–kin differences placed in desire), not as a finding.
- **Cost of the simplification:** 6.5 loglik points, about 4 of them from making δ linear (Stage 1). The p-values ignore the two-stage set-up (Stage-1 estimates are treated as known in Stage 2).

**Decision (user, 13 Nov 2026): one b for every sender.** Receipt is s = expit(b0 + b·E), with E = E^c + E^k. This is presented as a normalisation: the survey records receipt but not who sent, so all core–kin differences are placed in desire. Eq. 6 becomes role-free: E = W + min(X, R), X ~ Bin(L, κ). Capping the total gives the same distribution of E as capping core roles first, then kin, so the code keeps its role split and the results are identical.

- **Code:**
  - `willing_able.py`: `B_NAMES = [b0, b]`; `receipt_prob` and `design` use the total E; `design_by_role` and `B_ROLE_NAMES` are kept for the test only.
  - `calibrate_mmp.py` fits one b and refits separate b_c, b_k as the test (columns `sep_*`, `lr_b`, `p_b_equal`).
  - `check_special_cases.py`: all four checks pass.
- **Calibration rerun (13 Nov):**
  - Stage 1 unchanged.
  - Stage 2: η_w 0.33, η_o 0.16, b0 −5.04, b 8.25; loglik −1138.1, AIC 2284.3.
  - Receipt: 0.6% with no sender, 96.1% with one, ≈100% with two.
  - Survey parameters are now 7 (α_c, α_k, δ, η_w, η_o, b0, b); 10 in total with ρ, κ, β (π dropped).
- **LaTeX** of every affected equation (ability, desire, effective senders, receipt, transfers) and the new parameter table: `2026-10-30/methods_equations_simplified.tex`. ρ, κ, β are marked [TBD].

**ρ recalibrated on 2018 conditions (13 Nov; `household_layer.py --calibrate-only --kappa K`, outputs `outputs/household-layer/calibration_rho_simple_k{0,1}.csv` and `rho_simple_k*.log`, 100 seeds):**

- **ρ = 0.899 at κ = 1; 0.916 at κ = 0** (old forms 0.928 at κ = 1).
- κ matters more than expected. Stepping-in already steadies 2018 receipt, so less habit is needed. The gap of 0.017 is about 3 times the uncertainty that the ENIGH target's standard error (0.011) implies for ρ.
- So ρ is final only once κ is: use 0.90 provisionally if κ stays at 1.
- Check against ENIGH 2018 at κ = 1, ρ ≈ 0.90 (grid point): one-month share 14.9% (ENIGH 20.3%; old forms 17.9%), mean months 4.48 (4.08; old 4.44). The one-month share is a little worse than with the old forms.

**COVID-years calibration on the new forms (16 Nov 2026; `run_shocks.py --tag simple`, full grid κ 0–1 by 0.1, β 0–1.2 by 0.05; outputs `outputs/shocks-simple/`, log `outputs/shocks_simple.log`).**

`run_shocks.py` now saves its outputs after each κ. A first run was stopped by Claude Code for low memory and lost everything, because the script wrote only at the end.

- **Best fit: κ = 1, β = 0.153**, with κ at the edge of the grid; the fit improves steadily with κ, as before.
  - Fit error 2020Q2–2021Q4: 0.013 (κ = 0: 0.047).
  - 2022 holdout: 0.024; 2022 mean index 1.103 (data 1.100).
  - 2020Q2 index: 0.970 (data 0.974; κ = 0 gives 0.886).
  - Stimulus jumps, not fitted: +4.0 / +2.7 (data +4.1 / +3.6; κ = 0 gives +13.6 / +10.1).
  - Core desire of an only core migrant: 56.5% → 60.2% by Dec 2021.
- **Almost identical to the earlier forms:** κ = 1, β = 0.153, the same fit error and holdout to the third decimal; 2020Q2 0.970 vs 0.971. The national index is a ratio (COVID / no-COVID run), so the normal-times functional forms largely cancel. The national results do not depend on the simplification.
- **The reserve:** at κ = 1, 1.4% of households run out of reserve in 2020Q2. 30% of the replacements drawn in 2020Q2 go unfilled (28% over Apr 2020–Dec 2021), so even full stepping-in replaces about 70% of lost senders (cf. T2).
- **ENIGH checks:**
  - 2022 receipt index +7.9% vs transfers per receiving household +2.2%, against ENIGH −2% / +9%: the margin tension persists, as T5 predicts.
  - Mar → Apr 2020 receipt −0.034 vs ENIGH +0.016 (s.e. 0.048): compatible.
- **ρ = 0.90 is now final** (calibrated at κ = 1). `methods_equations_simplified.tex` has κ = 1, β = 0.15, ρ = 0.90.
- **Downstream rerun on the new forms (16 Nov 2026).** `wa.SHOCKS = outputs/shocks-simple` is now read by `household_layer.py`, `enigh_aligned.py`, `make_transition_figures.py`, `make_transition_kappa.py`, `make_kappa_effect_figure.py` and `make_paper_results_figures.py`. `run_households.py` uses `--shocks-tag simple` by default and writes `outputs/households-simple/`.
  - Rerun in order (logs in `outputs/logs-simple/`):
    1. need series E:0 and R:1, quick grid (`outputs/shocks-simple-need-E0`, `-R1`);
    2. `household_layer.py --rho 0.8992`;
    3. `run_households.py`;
    4. `enigh_aligned.py`;
    5. `make_kappa_effect_figure.py` (needs `PYTHONIOENCODING=utf-8` on Windows: it prints κ);
    6. `make_transition_figures.py`, `make_transition_kappa.py`;
    7. `make_paper_results_figures.py` (R1–R3).
  - The persistence scripts (closed extension) still read `outputs/shocks/`.
  - **Household layer (ρ = 0.899) vs old (ρ = 0.928):**
    - ENIGH 2018 one-month share 15.0% (old 17.9%; ENIGH 20.3%), mean months 4.47 (4.44; 4.08).
    - All-six change: 2020 −1.7 pp (ENIGH −2.5), 2022 +1.8 pp (ENIGH +7.8).
    - Share of lost months borne by households losing 3+ months: 49% (old 52%); core-only 53% (56%).
    - Households losing any month 2.3%, gaining any 4.5% (old 3.7%).
    - ENIGH-aligned 2022: receiving +0.058, transfers per receiving household +0.039 (ENIGH −0.027 / +0.093).
    - mig174 (share of core migrants able and wanting in at least one of L months, MMP heads 0.70–0.93): 0.45 / 0.52 / 0.62 / 0.75 / 0.92 for 3–60 months, slightly closer than before (old 0.43 / 0.48 / 0.57 / 0.69 / 0.87).
    - National check: simulated = exact.
  - **Need series (κ = 1):** monthly excess deaths fit 0.046, 2022 holdout 0.092; placebo ramp fit 0.010, holdout 0.035, 2022 at 1.137 vs 1.100. Identical to the old forms; the argument for the cumulative series holds unchanged.
- Old-forms outputs kept: `outputs/shocks/`, `shocks-quick-base/`, `shocks-need-*/`, `households/`. Overwritten in place: `household-layer/`, `enigh_aligned.csv`, `figures/paper/R1–R3` (old versions in git HEAD); `transitions/` and `figures/paper/R4*`, `R5*` (never committed, so the old-forms versions are gone).

**Calibration figures remade on the new forms (13 Nov; `make_calibration_figures.py` → `figures/calibration/` C1–C6 + `captions.md`):**

- `make_calibration_figures.py` and `sample_receipt_figure.py` (its `household_receipt`, used by C4) are now on the new forms and read `calibration_simple.csv`.
- C3 adds the earlier log-form fit as a dotted grey reference line. The caption now states the miss: one core migrant is underpredicted, two to three overpredicted.
- C1 shows one decimal for shares under 10%.
- C4: a label was moved off the stepping-in arrow.
- Saves retry when the PNG is briefly locked (a viewer or sync tool locked files during the run).
- Results:
  - No-migrant households: model 0.6%, observed 0.6% (3,730 households with complete records). The Methods draft says 0.7% for 3,749 households, a different sample definition to reconcile.
  - C5: the lowest core band (score ≈ −0.2, mostly single-core households) sits clearly above the curve (58% observed vs ≈ 45%).
  - C6: the clear miss is still kin-only households with an older resident (17% vs 7%). With a core migrant, "no worker, older resident" is overpredicted (74% vs 66%; not compared with the earlier forms).
- `S1_receipt_curve_examples_sample.png` (in `figures/`) has not been regenerated.

**Next:** rerun the shock map (κ, β), then fix ρ at the fitted κ; fill in the [TBD] values; update the robustness and figure scripts still on the earlier forms.

## 0b. Theoretical tests of the system (13 Nov 2026; first pass, user to decide how to proceed)

**Setup** (user, 13 Nov): `2026-10-30/theoretical_tests.py` → `figures/theoretical-tests/` T1–T5, `captions.md`, `theoretical_tests.csv`.

- 2019 calibration on the new forms, exact computation, runs in about 1 minute.
- Unemployment is raised by u percentage points in every occupation: a = max(e − u, 0).
- Need is a shift of every migrant's desire on the log-odds scale (as β·N* does).
- κ ∈ {0, 0.5, 1} is treated as a dimension. No COVID shocks, network changes or correlated job loss yet.

**Key identity:** with κ = 1, E = W + min(L, R) = min(D, A), the smaller of how many want to send and how many can.

**Findings:**

- **T1, response to unemployment.**
  - Without stepping-in, transfers fall in proportion to jobs (−10.4% at +10 points, −31% at +30).
  - With full stepping-in they fall by −3.1% / −12%. About 70% of the loss is absorbed at +10 points, 61% at +30.
  - The response accelerates: the loss per 10 points grows 3.1 → 3.9 → 4.9 pp.
- **T2, the buffer.**
  - Household-months with a lost sender rise from 1.4% (2019) to 9.7% at +30 points; short ones (lost senders exceed the reserve) from 0.4% to 3.7%.
  - Share of lost senders who are replaceable: 73% → 62%. The same shock adds lost senders and removes reserve members.
- **T3, unemployment × need.** At κ = 1 the loss from +10 points rises from 2.7% to 3.8% as desire goes from −1 to +1.5 log-odds (share wanting to send 6% → 27%). Need draws the reserve into sending and erodes the buffer. Without stepping-in, need barely changes the loss (about 10%).
- **T4, who bears the loss.** Single-migrant households (236 core-only, 1,341 kin-only) are not protected: −10.4% and −8.6% at +10 points, the same with or without stepping-in. Networks of 3 or more are almost fully protected at κ = 1.
- **T5, two margins.**
  - Unemployment acts almost only on the number of households receiving.
  - Need also acts mostly there: +1 log-odds raises households receiving by 0.40 log points and transfers per receiving household by 0.13. This structure puts a need-driven rise on households receiving, against the ENIGH 2022 pattern (receiving −0.027, transfers per receiving household +0.093).
  - The desire shifts in T3/T5 are far larger than the fitted crisis effect (core desire 60 → 64%, about +0.15 log-odds).

## 0. Update 30 Oct–2 Nov 2026: the willing-and-able model (calibrated; quick shock map, need series, persistence tested)

**Direction (user, 30 Oct):** the next base model is the willing-and-able model in `2026-10-30/model_plan.tex` (Steps 0–10; script map in `2026-10-30/README.md`). **The 2019 calibration is done and verified; the quick shock map and a comparison of need series have been run (below); the full map and the robustness runs have not.** Until the new model is adopted, the results in §2–§5 (option 6a, `run_household_model.py`) remain the current base, and the paper draft (`paper/main.pdf`, 29 Oct) is written on them.

- **Structure:** each migrant has an *ability* to send (employment; stimulus checks for out-of-work migrants) and a *desire* to send, d = expit(a_type + g·[core]·log N^c_h + η·need + β·N*_t). Four states: **able sender** (able and wants to send; use this term in all write-ups, user 30 Oct), lost sender, reserve, inactive. **Stepping-in** (κ): each lost sender is replaced by a reserve migrant with probability κ, capped by the reserve, core roles first, the replacement taking the lost sender's role. Receipt s_h = expit(b0 + b_any·[any able core sender] + b_c·log(1 + able core senders) + b_k0·[no core migrant]·log(1 + able kin senders)); transfers = one per effective able sender in months with receipt. Computed exactly (recursion over each household's migrants, ρ = 0).
- **Decisions (30 Oct, user):** stimulus base = Jan and Mar 2021 on, April 2020 off, π = 1 (all three rounds as robustness, `--rounds 2020-04 2021-01 2021-03 --tag all-rounds`); replacements capped core roles first, and `run_shocks.py` reports how often the cap binds; **b_k removed** (kin matter only without core migrants; b_k was flat in the likelihood and drifted 19–84); **g added** (core desire falls with the number of core migrants; without it receipt rose far too steeply with network size and b_c went negative, making receipt *rise* when a core sender lost their job); **b_c ≥ 0** kept for its meaning; **normal times calibrated once at κ = 0** (stepping-in is a shock response; κ enters only the shock runs, in both the COVID and no-COVID run; `run_shocks.py` reports the 2019 level gap `cf_*_vs_k0`); no shared household component in desire (considered, not adopted).
- **Calibration results** (`2026-10-30/outputs/calibration.csv`, `stage1.csv`, `stage1_by_core.csv`): Stage 1 a_c = 0.41, a_k = −3.40, g = −0.97 (loglik −1139.5 vs −1189.0 with g = 0; with g = 0 it reproduces M1). Core desire 0.60 / 0.43 / 0.34 / 0.28 for 1 / 2 / 3 / 4 core migrants; kin 3.2%. Fit by core migrants 0–4+: observed 0.061 / 0.588 / 0.660 / 0.692 / 0.748, fitted 0.059 / 0.584 / 0.666 / 0.707 / 0.737. Stage 2 (κ = 0, multi-start): η_w 0.33, η_o 0.19, b0 −4.60, b_any 8.16, **b_c 0 (at its bound; the data hardly pin it, the network-size gradient is carried by g)**, b_k0 9.70; loglik −1131.3. Receipt is 1.0% with no able sender, 97% with at least one able core sender, 89% with one able kin sender in a household without core migrants. The 3,730 MMP households with no linked migrant (not used) get expit(b0) = 1.0% vs 0.7% observed: an out-of-sample check. Known miss: kin-only households with an older resident (17% observed vs 7%, n = 48). Earlier calibrations kept for reference: `outputs/with-bk/` (with b_k, no g), `outputs/per-kappa-g/` (with g, per κ, before b_c ≥ 0 and multi-start).
- **Known limitation (to state):** the household survey shows only receipt, so the split between desire (60% a month for an only core migrant) and receipt given an able sender (97%) is an assumption of the two-stage set-up. `mig174` (80% of heads sent something on their last trip) suggests desire is more persistent across months than independent draws imply; the ENIGH persistence test (Step 9.4) will tell. Kin occupations are mostly imputed (5,601 of 7,501 links copy a roster donor, almost always a core migrant of the same state).
- **Colleague's specification (30 Oct, `recipient-level/robustness/colleague_spec.py`):** a two-level model with no stepping-in fits the national series as well as the base **if out-of-work migrants keep sending at about half the employed rate**; stepping-in (κ) and continued sending (B) are not separable on the national series. Claim-1 framing pending: "the shock was absorbed", stepping-in as base, continued sending as the observationally equivalent alternative.
- **Robustness 6a–6h (29 Oct)** stay valid for the current base; to be rerun on the new model once it is verified.

### Shock years, quick map (30 Oct; `2026-10-30/outputs/shocks-quick-base/`, same as `outputs/shocks/`)

`run_shocks.py --quick`: κ ∈ {0, 0.5, 0.8, 1}, β grid 0–1.2 by 0.1 plus β fitted per κ; base stimulus (Jan and Mar 2021, π = 1); need = cumulative excess deaths; normal-times parameters from the κ = 0 calibration in both runs. Runtime 6 min 51 s.

| κ (β fitted) | 0 | 0.5 | 0.8 | **1** | Banxico / ENIGH | current base (κ = 1) |
|---|---|---|---|---|---|---|
| β | 0.18 | 0.17 | 0.16 | **0.15** | | 0.65 |
| RMSE 2020Q2–21Q4 | 0.047 | 0.027 | 0.017 | **0.013** | | 0.012 |
| RMSE 2022 holdout | 0.033 | 0.027 | 0.025 | **0.024** | | 0.024 |
| 2020Q2 / 2021Q4 | 0.886 / 1.104 | 0.929 / 1.098 | 0.954 / 1.094 | **0.971 / 1.092** | 0.974 / 1.108 | 0.975 / – |
| spikes Jan / Mar 2021 (log pts) | +13.6 / +10.1 | +8.7 / +6.3 | +5.9 / +4.2 | **+4.0 / +2.7** | +4.1 / +3.6 | +4.1 / +2.9 |
| receipt 2022 vs no-COVID | 1.09 | 1.08 | 1.08 | **1.08** | ENIGH −0.02 (s.e. 0.10) | ≈ 1 by construction |
| transfers per receiving hh 2022 | 1.03 | 1.03 | 1.03 | **1.03** | ENIGH +0.09 (s.e. 0.10) | +0.10 |
| Mar→Apr 2020 receipt | −0.11 | −0.07 | −0.05 | **−0.04** | ENIGH +0.016 (s.e. 0.048) | −0.035 |

- **κ ≈ 1 from two moments** (2020Q2 trough and the stimulus spikes; spike RMSE 0.006 at κ = 1). κ = 0 is rejected (2020Q2 0.886, spikes three times too large).
- **Need is much smaller than in the current base** (β 0.15: core desire 60% → 64% by Dec 2021), because extra desire now also raises receipt.
- **Open tension: where the 2021–22 rise lands.** The model puts it mostly on receiving households (+8%) and little on transfers per receiving household (+3%); ENIGH's point estimates are the reverse (flat, +9%). Both gaps are about 1 ENIGH s.e., so not rejected, but it is the same pattern as the colleague's specification. Prepared fallback: the ω option (Step 9, test 2: desire enters receipt only partly, ENIGH picks ω). Decision pending.
- Reserve shortages are rare (1.4% of households in 2020Q2 at κ = 1; 29% of drawn replacements unfilled, mostly in single-migrant households with nobody to step in), so the core-first rule hardly matters. 2019 level gap of the no-COVID run at κ = 1: receipt +2.8%, transfers +3.6% relative to the κ = 0 calibration (cancels largely in the index; state it). Only 6 grid cells are compatible (τ = 0.04): the grid above β = 0.3 is wasted; the full map should use β 0–0.4 by 0.025.

### Which need series (30 Oct; `run_shocks.py --quick --need CODE:λ`, outputs `2026-10-30/outputs/shocks-need-*/`)

Each series is scaled to peak 1 over Mar 2020 – Dec 2021, so β is not comparable across series; the comparison is fit and shape. All at κ = 1 (best under every series):

| need series | RMSE fit | RMSE 2022 | 2020Q2 | 2021Q4 | 2022 avg | spikes Jan / Mar | receipt 2022 | TPR 2022 |
|---|---|---|---|---|---|---|---|---|
| **excess deaths, cumulative (base, E:1)** | **0.013** | **0.024** | 0.971 | 1.092 | 1.103 | +4.0 / +2.7 | 1.076 | 1.025 |
| reported COVID deaths, cumulative (S:1) | 0.013 | 0.024 | 0.970 | 1.092 | 1.103 | +3.8 / +2.8 | 1.076 | 1.025 |
| excess deaths, decaying stock λ = 0.9 (E:0.9) | 0.024 | 0.065 | 0.975 | 1.063 | 1.042 | +4.4 / +2.7 | 1.035 | 1.011 |
| excess deaths, monthly (E:0) | 0.046 | 0.092 | 0.999 | 1.011 | 1.007 | +9.6 / +1.6 | 1.007 | 1.002 |
| reported COVID deaths, monthly (S:0) | 0.045 | 0.093 | 0.998 | 1.013 | 1.007 | +7.3 / +2.0 | 1.007 | 1.002 |
| Mexican employment loss, monthly (L:0) | 0.060 | 0.097 | 0.995 | 0.999 | 1.001 | +3.5 / +2.8 | 1.001 | 1.000 |
| placebo ramp from Mar 2020 (R:1) | 0.010 | 0.035 | 0.977 | 1.097 | 1.137 | +3.6 / +2.7 | 1.095 | 1.031 |
| *Banxico* | | | *0.974* | *1.108* | *1.100* | *+4.1 / +3.6* | *ENIGH ≈ flat* | *ENIGH +0.09* |

κ ≈ 1 under every series. The receipt-vs-TPR split (about 3 : 1) is the same under every series that produces a rise: it is a property of the model's structure, not of the need measure, so it cannot be fixed by choosing another series.

### κ above 1: full replacement, not overcompensation (2 Nov; `run_shocks.py --kappas 1 1.25 1.5 1.75 2 --tag kappa-above-1`, outputs `2026-10-30/outputs/shocks-kappa-above-1/`)

Exploratory extension (`willing_able.replacement_pmf`, `relabel`; `run_households.py` follows): for 1 < κ ≤ 2 every lost sender is replaced for sure and a second reserve migrant also sends with probability κ − 1 (so κ able senders replace each lost one on average), still capped by the reserve, core roles first. Core roles cannot exceed the household's core migrants (overflow counts as kin); this never binds for κ ≤ 1, so all κ ≤ 1 results are unchanged. Checks passed: probabilities sum to 1, drawn − unfilled = added senders, monotone in κ, exact vs simulated at κ = 1.5 (max |z| 1.37). Reading of κ > 1: a job loss mobilises more than one replacement (an alarm in the wider family, or replacement split across several smaller transfers). Runtime 17 min.

| κ (β fitted) | 1.00 | 1.25 | 1.50 | 1.75 | 2.00 | Banxico |
|---|---|---|---|---|---|---|
| RMSE 2020Q2–21Q4 | 0.013 | 0.012 | 0.014 | 0.018 | 0.022 | |
| RMSE 2022 holdout | 0.024 | 0.023 | 0.024 | 0.024 | 0.025 | |
| 2020Q2 | 0.971 | 0.983 | 0.996 | 1.008 | 1.020 | 0.974 |
| spikes Jan / Mar 2021 | +4.0 / +2.7 | +2.5 / +1.6 | +1.1 / +0.4 | −0.3 / −0.7 | −1.7 / −1.8 | +4.1 / +3.6 |
| spike RMSE | 0.006 | 0.018 | 0.031 | 0.043 | 0.056 | |
| receipt / TPR 2022 | 1.076 / 1.025 | 1.074 / 1.024 | 1.072 / 1.023 | 1.070 / 1.022 | 1.068 / 1.021 | ENIGH ≈ flat / +0.09 |

- **κ is bounded at about 1 from both sides.** Below 1 the 2020Q2 dip is too deep and the spikes too large (κ = 0.8: 0.954, +5.9 / +4.2); above 1 the dip disappears (2020Q2 passes Banxico's 0.974 at about κ = 1.05; from κ = 1.75 job loss would *raise* remittances) and the spikes shrink and then turn negative (with κ > 1 a lost sender generates more than one transfer, so a stimulus check that restores the migrant's ability removes the extra replacements). Spike RMSE is minimal at κ = 1. The quarterly fit is flat between 1 and 1.5 (0.0122 at 1.25 vs 0.0128 at 1), too small a difference to matter.
- **Claim for the paper:** lost sending was covered about one for one by others in the network: full replacement, not overcompensation.
- κ > 1 does not move the 2022 receipt-vs-TPR split (it acts only while migrants are out of work; employment had recovered by 2022).
- The option stays in the code as a robustness check.

### Persistence of sending: tests and evidence (2 Nov)

**Tests run (outputs not yet committed):**
- *Desire level ±0.5* (`calibrate_mmp.py --shift 0.5 --tag _shift`, `run_shocks.py --calibration ... --tag desire-plus0.5 / desire-minus0.5`): κ = 1 best under all three levels; at κ = 1 fit 0.012 / 0.013 / 0.013, holdout 0.024, 2020Q2 0.975 / 0.971 / 0.966, spikes +3.6/+2.4, +4.0/+2.7, +4.4/+3.0. β 0.149 / 0.153 / 0.188, but its effect on desire (β·d(1−d)) is the same, about +4 points by Dec 2021: report β as that marginal effect. −0.5 is rejected by 2019 MMP (loglik −23.5); +0.5 fits 2019 better (+4.8), so the calibrated desire sits at the low end. National conclusions do not depend on the desire level.
- *Model measured as ENIGH measures it* (`enigh_aligned.py`, 200 paired seeds): 2022 receiving households (at least once in six months) +0.046 and transfers per receiving household +0.052 (ENIGH −0.027 and +0.093, s.e. 0.095): the receipt-vs-TPR split is roughly even, both gaps under 1 s.e. 2020: model −0.007 / −0.001, ENIGH −0.073 / +0.075 (s.e. 0.06).
- *Persistence test fails* (`run_households.py`, `enigh_months_received.py`): months with receipt among receiving households, 2018: model mean 2.78, all six 5.5%, one month 35.5%; ENIGH 4.08 (0.04), 49.6% (1.1), 20.3% (0.8). ENIGH is U-shaped (occasional and every-month receivers), the model hump-shaped. Causes: kin-only households receive almost at random (80% of them in one month only), and core households rarely receive every month because desire is redrawn monthly. Receipt-level persistence (the plan's u_h in s_h) cannot fix it (receipt given an able core sender is 97%): persistence must come from desire. National monthly means are not affected as long as persistence is per migrant and migrants stay independent. `who_gains_loses.csv` is provisional.
- *MMP `mig174`, heads' last US trip 2010–19 surveys* (n = 1,779 with valid duration and amount): share who sent anything (`remit` > 0) is flat across trip length (4–6 months 0.93, 7–12 0.88, 13–24 0.89, 25–60 0.91), lower for 61+ months (0.76) and 1–3 months (0.70, n = 23). Independent monthly desire at 0.6 implies almost everyone sends in any trip of 6+ months, and a habit (Markov) model implies the never-sender share falls with trip length; the flat ~10% points to persistent non-senders (heterogeneity across migrants).
- *MMP `remus`* is labelled "HH receives remittances from U.S." (no reference period in the label; no value labels in the DTA or SAV files; MMP website unreachable on 2 Nov). `remussiz` ("Size of remittances from U.S.") is a small code (median 1, 75th percentile 3 among receivers), not an amount. **Questionnaire (resolved 2 Nov; MMP Ethnosurvey VIII and HOUSE codebook in `data/Mesoamerican Migration Core Data SAV and DAT Files/`):** table J3 "Remittances (confidential)": "Household receives remittances from U.S.? Yes / No / Unk"; if yes, "Compared to your monthly salary, what's the size of remittances from the U.S.? Small / Intermediate / Substantial" (= `remussiz` 1/2/3). Present tense, no reference period; introduced from community #72 (all 2010–19 households). **Decision:** read as a current, ongoing flow and keep matching it to the model's monthly receipt probability; with persistent senders (options A/B) monthly receipt and "currently receives" converge, so the mapping becomes more natural, not less.

**Evidence on remittance regularity (literature and surveys; what was accessed is noted):**
- *Survey report (not a journal):* Pew Hispanic Center / IDB-MIF, "Remittance Senders and Receivers: Tracking the Transnational Channels" (2003; full text). 2003 National Survey of Latinos: 42% of foreign-born Latino adults regularly send; two-thirds of senders send at least once a month, three-quarters of those in the US under five years; in Mexico, recipients whose relatives had been abroad five years or less were nearly twice as likely to receive regularly. https://www.pewresearch.org/wp-content/uploads/sites/5/reports/23.pdf
- Ventura, I.M. and García, A.S. "Resilient Remittances? Examining Immigrant Remittances from the United States to Latin America During Covid-19." *International Migration Review* (online 2023), doi 10.1177/01979183231216048 (full text). Survey of Latino Immigrant Families and Identities (n = 1,046, 27% Mexican-born; 594 remitters) and interviews in Chicago. An "expanded remittance pool": responsibility spread to other household and extended-family members (US citizens, documented, more financially stable) in response to job loss; interviews describe fixed scheduled obligations ("we used to send 100 dollars a month"; "$145–150 every two months"). Supports stepping-in and regular senders.
- Waldinger, R. "Between 'Here' and 'There': Immigrant Cross-Border Activities and Loyalties." *International Migration Review* 42(1), 3–29, 2008 (abstract). Regular cross-border activity is common but heterogeneous; only about 9% send, call weekly and travel.
- Sana, M. and Massey, D.S. "Household Composition, Family Migration, and Community Context: Migrant Remittances in Four Countries." *Social Science Quarterly* 86(2), 509–528, 2005 (abstract; MMP and LAMP). In Mexico remittances are tied to the cohesive patriarchal family as a household risk-diversification strategy.
- Amuedo-Dorantes, C. and Pozo, S. "Remittances as Insurance: Evidence from Mexican Immigrants." *Journal of Population Economics* 19(2), 227–254, 2006 (abstract). Higher US income risk raises the propensity to remit and the share of earnings sent.
- Amuedo-Dorantes, C. and Bansak, C. "Money Transfers among Banked and Unbanked Mexican Immigrants." *Southern Economic Journal* 73(2), 374–401, 2006 (abstract; MMP). Monthly remittance flows per migrant; a US bank account does not raise them but raises money brought home.
- Amuedo-Dorantes, C. and Pozo, S. "Remittance Income Volatility and Labor Supply in Mexico." *Southern Economic Journal* 79(2), 257–276, 2012 (abstract; ENIGH 2000–08). Remittance income is volatile enough to be a household income risk; labour supply buffers it.
- Massey, D.S. and Parrado, E. "Migradollars: The Remittances and Savings of Mexican Migrants to the USA." *Population Research and Policy Review* 13(1), 3–30, 1994 (abstract; MMP). Flows include money sent while abroad and savings brought home.
- Weber, R. and Massey, D.S. "Assessing the Effect of Increased Deportations on Mexican Migrants' Remittances and Savings Brought Home." *Population Research and Policy Review* 42(2), 2023 (abstract; MMP 1970–2019). Undocumented migrants shift from saving to remitting as deportations rise.
- Ambrosius, C., Campos-Vázquez, R.M. and Esquivel, G. "Migrant Remittances During a Global Shock: Evidence From the COVID-19 Pandemic in Mexico." *IZA Journal of Development and Migration* 14(1), 1–22, 2023 (abstract). Quarterly state data: remittances followed US employment; Mexican conditions had no detectable effect.
- Kpodar, K., Mlachila, M., Quayyum, S. and Gammadigbe, V. "Defying the Odds: Remittances during the COVID-19 Pandemic." *Journal of Development Studies* 59(5), 673–690, 2023 (abstract; 52 countries). Resilient remittances; rose with home-country infections, dampened by containment.
- Not verified, not relied on: a hump-shaped remitting profile in MMP (peak after about 5.5 years) attributed by a Funcas working paper to Amuedo-Dorantes and Pozo (2006).
- Summary: remitting is mostly a regular, scheduled obligation (usually monthly, fixed amounts) held by some migrants, while others send occasionally or not at all; it is organised at the household level, and others in the family take over when the usual sender cannot.

**Options for persistence (discussed with the user, nothing implemented):** (A) habit: desire follows a two-state Markov chain per migrant (persistence ρ; ρ = 0 is the current model); (B) types: a share of regular senders and occasional senders; (C) a shared household component; (D) persistence in receipt (does not work); (E) reread what the MMP `remus` answer refers to. User leans to A (it allows behaviour to change in 2020–22, e.g. stints of regular sending) with B as robustness; both to be tried, after discussion.

### Persistence round 1: ENIGH 2022, options A and B (2 Nov)

**ENIGH 2022** (`enigh_months_received.py --ing22`, output `outputs/enigh_months_received_2022.csv`; 2018/2020 rows identical to before). Among receiving households: all six months 49.6% (2018) → 47.0% (2020) → **57.4% (2022)**, +7.8 pp vs 2018 (s.e. 1.4); one month 20.3 → 21.9 → **15.0** (−5.3 pp, s.e. 1.0); mean months 4.08 → 3.98 → 4.41. Share of households receiving unchanged (4.73 / 4.38 / 4.57%). Broad across groups (not households with a resident 65+). Interview-month mix identical across waves. **In 2022 the same households received, more regularly.**

**Option A, habit** (`persistence_a.py`, outputs `outputs/persistence-a/`): core desire is a two-state chain, P(want | wanted) = d_t + ρ(1 − d_t), P(want | not) = d_t(1 − ρ); marginal m_t = (1 − ρ) d_t + ρ m_{t−1}; 2019 calibration unchanged; exact monthly computation with m_t. Checks pass. With kin iid (as first agreed) no ρ reaches ENIGH's all-six share (max ≈ 38–39% as ρ → 1; one-month share rises to 42%): kin-only households (70% of households, about half of receivers) receive in scattered single months. Diagnostic with kin persistence too: ρ ≈ 0.93 fits 2018 (all six 49.5%, one month 17.9%; slightly too U-shaped, mean 4.44 vs 4.08). Shock map: κ ≈ 1 under every variant; habit makes desire lag need (mean lag ρ/(1 − ρ) months), so with cumulative need β rises 1.5–2.4× and 2022 overshoots (holdout 0.033–0.055 vs 0.024). 2022 households: receiving +15%, TPR −2.5%, every-month share −3.5 pp (ENIGH: flat, +9%, +7.8 pp): the crisis adds new occasional starters. Who loses: losses concentrate (core-only households losing 3+ months 2.6% → 3.9%).

**Option B, types** (`persistence_b.py`, `run_persistence_b.py`, outputs `outputs/persistence-b/`): a core migrant is a regular sender with probability λ·d_j0, otherwise occasional with a rate that keeps the monthly marginal at d_j,t; national results identical to the base by construction (checked to 1e-16). Max all-six 39% at λ = 1 (kin-only again); mig174 profile flat only at 42%; 2022 every-month share falls (−5.7 pp at λ = 1) and receiving households +16%: fixed types cannot produce the 2022 pattern. Kept as robustness.

**mig174** (flat ~0.9 sending across trip lengths) is matched by neither (it would need a ~10% never-sender type).

**Decisions (user, 2 Nov):**
- **Kin persistence: yes** (habit for core and kin).
- **The crisis does not act on persistence.** Mechanisms are fixed in normal times (calibrated up to 2019/2018); in the shock years only job loss changes ability, stepping-in (κ) responds, and COVID need changes desire d_t (which enters the chain as above).
- **Try monthly need** with the habit (persistence could carry the effect forward) against cumulative need.
- **No never-sender type** (overspecification, hard to justify); mig174 reported as a limitation.
- Next: option A2 = habit for core and kin (ρ calibrated to ENIGH 2018), need cumulative vs monthly; tests: ENIGH 2022 change, national map and holdout, ENIGH-aligned margins, who loses.

### Persistence round 2: option A2 and decision (2 Nov)

**Option A2** (`persistence_a2.py`, using `persistence_a.py`; outputs `outputs/persistence-a2/`): habit for core and kin, ρ_c and ρ_k fixed in normal times; the crisis acts only through ability, stepping-in and desire d_t. Calibrated to ENIGH 2018 all-six and one-month shares: **ρ_c = 0.9725, ρ_k = 0.79** (common ρ 0.923 fits less well). Results (κ = 1 best throughout):

| | base (ρ = 0) | A2, cumulative need (E:1) | A2, monthly need (E:0) | data |
|---|---|---|---|---|
| 2018 all-six / one-month | 5.5% / 35.5% | 49.4% / 20.3% | same | 49.6% / 20.3% |
| 2020 change in all-six | +0.2 pp | −2.3 pp | −3.0 pp | −2.5 pp (1.5) |
| 2022 change in all-six | +1.8 pp | −1.8 pp | +0.4 pp | **+7.8 pp (1.4)** |
| β | 0.153 | 0.398 | 0.607 | |
| RMSE 2020Q2–21Q4 / 2022 holdout | 0.013 / **0.024** | 0.011 / 0.049 | 0.025 / 0.060 | |
| 2022 average | 1.103 | 1.153 | 1.045 | 1.100 |
| mig174: core sending at least once in 60 months | 1.00 | 0.68 | 0.68 | 0.91 |

Shape between 1 and 6 months still off (too many 5-month households; mean 4.31 vs 4.08). The habit makes desire lag need (double accumulation with the cumulative stock), implies multi-year spells without wanting to send (against mig174), and does not reproduce the 2022 regularisation (a general rise in desire adds new occasional senders as well as steadier ones). Losses concentrate: households losing 3+ months bear about half of lost months (base 30%).

**Decision (user, 2 Nov): stop the persistence work here.** No further variants (e.g. need acting only on core desire, or a mean-preserving habit, were proposed and declined). The base model stays as it is (desire drawn each month, ρ = 0). The habit (A/A2) and types (B) are documented as a tested extension; the more regular receipt of 2022 in ENIGH is reported as a finding the model does not explain. `2026-10-30/data_methodology.tex` was rewritten (2 Nov) to describe the base model with all equations, the calibration (tables and figures C2–C5) and this extension; test-compiled (9 pp.) with `paper/refs.bib`.

### Household layer: one average-preserving habit (2 Nov, user decision; `household_layer.py`, outputs `outputs/household-layer/`)

**Speed-up first (commit abf3567):** `willing_able.StateTables` groups households by size, works on valid cells only and caches the replacement mapping: quick map 6m51s → 1m08s, results identical.

**Spec:** every migrant's desire follows P(keep sending) = d_t − ρ d_{t−1} + ρ, P(start) = d_t − ρ d_{t−1} (`run_households.simulate(..., rho)`; ρ = 0 gives identical paths). Each migrant's monthly mean desire is exactly d_t, so the exact national computation is untouched: **national results = base model by construction** (simulated national index matches the exact one within MC error). Feasibility needs Δd_t ≤ (1 − ρ)(1 − d_{t−1}); slack 0.018 at the fitted β (asserted in the simulator). One ρ for core and kin, calibrated to ONE target, the ENIGH 2018 all-six share: **ρ = 0.928**.

**Out-of-sample results (κ = 1, β = 0.153, 200 paired seeds):** 2018 one-month 17.9% (ENIGH 20.3%), mean 4.44 (4.08; too many 5-month households, 15% vs 3%); 2020 change in all-six −1.8 pp (ENIGH −2.5, s.e. 1.5), mean −0.06 (−0.10); 2022 all-six +1.4 pp and one-month −1.2 pp (ENIGH +7.8 / −5.3: right direction, far too small); ENIGH-aligned 2022 receiving +0.056, TPR +0.041 (ENIGH −0.027 / +0.093, s.e. 0.095). Who loses (Apr 2020–Dec 2021): households losing 3+ months 0.3% → 0.6%, share of lost months borne by them 30% → 52% (core-only 41% → 56%); households gaining 17% → 3.7%. mig174: 0.43 / 0.48 / 0.57 / 0.69 / 0.87 for 3–60 months vs 0.70–0.93 (short trips too low; a limitation of one parameter).

**Status:** adopted as the household layer of the base model (national claims from the exact monthly model, household claims from the same model with ρ = 0.928). `data_methodology.tex` still describes the habit as an extension and needs updating.

### Paper draft (3 Nov; `2026-10-30/paper_draft.tex`)

Full paper rewritten in the style of the user's second template (accessible language, short subsections, biblatex with `bibliography.bib`): abstract; introduction (problem / what we know / what we do not know / what we do / what we find); Methods and Data (10 short subsections with the key equations, parameter table); Results in three sections (calibration; families replacing lost senders and need lifting 2021; household losses); Discussion (5 paragraphs); Supplementary (C4, C1, C6, fit table). The user copies it into their own paper project. Test-compiled (18 pp.) against the user's `bibliography.bib` (28 keys cited) plus `2026-10-30/paper_draft_new_refs.bib`, which holds the five data sources missing from it (`mmp`, `ipumscps`, `banxico`, `enigh`, `karlinsky2021`; IPUMS version and Banxico series to CHECK). `Barcaglioni2008` not cited (content not verified). Biber warns about `month = {October}` / `{May}` in the user's bib (use `oct`, `may`).

New results figures (`2026-10-30/make_paper_results_figures.py` → `figures/paper/`): R1 national fit (job loss only; κ 0 / 0.5 / 1 with fitted β; Banxico; 2022 shaded) and the stimulus spikes by κ; R2 need measures (cumulative, monthly, placebo ramp) at κ = 1; R3 household layer (months with receipt 2018: ENIGH vs model without and with habit; share of lost months borne by households losing 3+ months, by network type).

Numbers in the draft come from the quick map (κ 0, 0.5, 0.8, 1); key ones: job loss only 2020Q2 0.879 (−12%); κ = 1, β 0.153: 2020Q2 0.971 (data 0.974), RMSE 0.013, holdout 0.024, 2022 average 1.103 (1.100), spikes +4.0 / +2.7 (+4.1 / +3.6); κ = 0 / 0.5 fitted: 2020Q2 0.886 / 0.929; core desire 60% → 64% by Dec 2021. To be refreshed after the full map.

### Why the cumulative excess-deaths series (argument for the paper)

1. **The data ask for persistence, and only a stock delivers it.** Remittance counts did not rise and fall with the death waves: they stepped up through 2021 and stayed up in 2022. A monthly series (excess deaths, reported deaths, employment loss) returns the model to its baseline as soon as deaths fall, missing 2021Q4 by about 10 log points (1.01 vs 1.11) and failing the 2022 holdout (RMSE 0.09 against 0.024). Even a stock that loses 10% a month fails 2022 (1.04 vs 1.10). Only an undepreciated stock reproduces both the build-up and the plateau. The 2022 holdout is out of sample, so this is a test, not a fit.
2. **Monthly series also get 2020 and 2021 wrong.** They lift 2020Q2 to about 1.00 (Banxico 0.974), because the first wave arrives in mid-2020, and monthly excess deaths peak in January 2021, inflating the January spike to +9.6 (Banxico +4.1). The cumulative stock is still small in spring 2020, so the 2020 trough is left to job loss and stepping-in, where the evidence puts it.
3. **It is what a death does to a household's needs.** A death in the family is not a one-month shock: a lost earner, a widow or orphans to support, funeral and medical debts, a household that has to reorganise its income. Needs created in 2020–21 are still there in 2022. A stock that does not decay is the simplest way to say that the crisis accumulated obligations rather than producing a passing spike. Remittance commitments, once taken on, tend to be regular (monthly schedules, fixed beneficiaries), which also argues against fast decay.
4. **It matches the national-crisis reading the cross-section supports.** The state and municipal panels show no local need gradient (§4, item 6). A national cumulative series fits that reading: migrants respond to the scale of Mexico's crisis as it builds up, as followed in the news and through their wider networks, not only to deaths in their own family. The cumulative toll is the natural measure of that scale.
5. **It beats the placebo where it matters.** A ramp that only marks time since March 2020 fits 2020–21 slightly better (0.010) but keeps rising and overshoots 2022 (1.137 vs 1.100; holdout 0.035). The cumulative toll flattens when deaths stop, so it stops the rise where the data stop it, without being told to. Timing alone does not explain 2022; the shape of the crisis does.
6. **Excess rather than reported deaths, although the two give the same fit.** Cumulated, the two series have nearly the same shape, so the results do not depend on the choice. Excess deaths are preferred because they count deaths that the official COVID tally misses (under-testing, deaths outside hospitals) and deaths caused indirectly by the pandemic (disrupted care), which matter for households in the same way.
7. **Limits to state.** The time series cannot distinguish national need from other persistent forces that started in 2020 (US-side capacity, savings, the shift to digital channels); need is the interpretation the timing and the cross-section support, not an identified effect. β is small in the willing-and-able model (core desire 60% → 64%), so the claim is modest.

### How to resume (state at the end of 30 Oct)

**Next actions, in order (state on 10 Nov):**
1. Full national map on the base (κ 0–1 by 0.1, β 0–0.4 by 0.025; minutes now), the all-rounds stimulus robustness and robustness 6a–6h; add the willing-and-able model and `household_layer.py` to `run_all.py`.
2. Refresh the numbers and figures in `paper_draft.tex` from the full map; update `data_methodology.tex` (habit as household layer), `model_plan.tex` and `calibration_2019.tex`.
3. User to check the data bibliography entries (IPUMS CPS version, Banxico series) and decide on `Barcaglioni2008`.
4. Optional: Mexican price level as a need series.

**Where things are:**
- Plan: `2026-10-30/model_plan.tex` (Step 8 has the calibration results; "Decisions taken (30 Oct)"). Calibration write-up for the user to copy: `2026-10-30/calibration_2019.tex` (10 pp., data and method, "able senders" throughout; figures not included yet). Neither is compiled in the repo (test-compiled in the scratchpad).
- Library and scripts: `2026-10-30/willing_able.py`, `calibrate_mmp.py` (default κ = 0 only; `--tag` for parallel runs), `run_shocks.py`, `run_households.py`, `check_special_cases.py` (four checks, all pass with g and without b_k).
- Calibration figures: `2026-10-30/make_calibration_figures.py` → `2026-10-30/figures/calibration/` C1–C6 with `captions.md` (households by type incl. no linked migrant; able × willing grid 2019 vs April 2020; desire and network size; receipt-curve examples; all households on the curve; fit by need). Palettes checked with the dataviz validator. Sample figure: `2026-10-30/sample_receipt_figure.py` → `figures/S1_receipt_curve_examples_sample.png`.
- Current base and its outputs: `network-compatibility-model/recipient-level/run_household_model.py`, `outputs/household-model/`; robustness 6a–6h: `recipient-level/robustness/`, `outputs/robustness/summary.csv`; colleague comparison: `recipient-level/robustness/colleague_spec.py`, `outputs/robustness/colleague_spec/README.md`.
- Identification: `network-compatibility-model/identification/` (ENIGH incl. 2016 pre-period and monthly profile, state panel, municipal panel).
- Paper draft: `paper/main.tex` → `paper/main.pdf` (17 pp., 29 Oct; on the current base; claim 1 to be reframed; references to verify; figure titles to strip). Figure 1: `paper/make_paper_figures.py`.
- Full rebuild of the current base: `network-compatibility-model/run_all.py` (~75 min).

Commit only when the user asks; the repository is public (never `data/` or MMP/IPUMS-derived microdata; `2026-10-30/outputs/` holds aggregates only).

**Working style (user):** go point by point and end every answer with a short "Still open" list; plain language; propose and wait for approval before implementing; stepping-in is the preferred coordination mechanism (qualitative literature); robustness checks run as separate agents when asked.

## Colleagues' feedback (collected 10 Nov 2026)

All comments received so far, with their status against the **current base (willing-and-able model)**. Original write-up of the presentation comments: `network-compatibility-model/handoff.md`, section "Colleague feedback (presentation, 27 Oct 2026)". Status key: **done** (on the current base), **done on old base** (needs rerunning or restating for the willing-and-able model), **open**, **answered** (no analysis needed, a statement in the paper), **dropped**.

### A. Presentation, 27 Oct 2026

| # | Comment | What it asks for | Status | Next step |
|---|---|---|---|---|
| A1 | Out-of-sample validation: "calibrate to 2018, can you replicate the next year without adjustment?" | Calibrate with 2018 CPS employment, run 2019 forward without refitting, expect an index ≈ 1. Present the 2022 holdout explicitly as a test. | **done on old base** (robustness 6e: index within ±0.5% per quarter, placebo passed). The 2022 holdout is in the paper draft as a test. | Rerun 2018 → 2019 on the willing-and-able model (cheap). |
| A2 | Same, stronger version | Backcast the 2008–09 recession (remittances to Mexico about −15%). | **open** (optional) | Needs CPS occupation rates for 2007–09: the IPUMS extract `cps_00001.dat` starts in Jan 2010, so a new extract is needed. MMP networks would be those reported in 2010–19, which is a caveat. |
| A3 | Tipping points: are there thresholds or nonlinear responses? | Scale the employment shock (×0.5, 1, 1.5, 2) and test whether aggregate receipt and the share of households losing everything respond linearly. | **open**. Answer to state: no aggregate tipping points by construction (no feedback between households); household thresholds exist (single-migrant households, the reserve cap). | Run the shock-scaling scenario on the current base. |
| A4 | Normalisation (PPP, inflation) | Whether amounts must be deflated or converted. | **answered**: the target is counts against the Banxico trend, so no normalisation is needed; PPP not needed (all comparisons within Mexico over time). Household outputs in real MXN are moot (the model counts transfers, not amounts). | State in the Methods. |
| A5 | Baseline vs scenarios, HP filter, seasonality | Present the counterfactual design clearly; check that "2020 flat, excess in 2021" does not depend on the trend. | **partly done**: COVID vs no-COVID runs with the same index as the data (in the Methods rewrite); trend windows checked (28 Oct, §5; robustness 6h on old base: κ ≈ 1 holds, β moves). | Still open: HP filter fitted on pre-2020 data, STL/X-13, plain year-on-year. Rerun 6h on the current base. |
| A6 | Competing explanations: use maximum likelihood, not only maps | A likelihood surface over the (κ, β) grid: quarterly log residuals model vs data, Gaussian with the trend residual SD (0.014 quarterly), giving a confidence region, a principled tolerance and likelihood ratios between explanations. | **done on old base** (log residuals scaled by the trend SD in `run_household_model.py`). Current base reports log RMSE only. | Add the likelihood surface and confidence region to the full map on the current base; caveats: 7 quarters, autocorrelated residuals, report ratios rather than p-values. |
| A7 | Inequality and wealth | Distribution within households; differences between rich and poor households. | **answered** for within-household (out of scope, the household is the unit). Between households: **open**. | Optional: asset index from items in both MMP and ENIGH (amenities, housing); descriptive only (remittances fund assets). |
| A8 | ENIGH distribution vs wealth | Extend the ENIGH 2018 → 2020/2022 comparison to receipt by income decile and by the asset index. | **open** (optional) | Relative changes only (ENIGH covers ≈ 8% of the Banxico total). |
| A9 | Offers to discuss | Two colleagues offered to discuss ideas (no time for analysis). | – | Contact when the full map and paper draft are refreshed. |

### B. Proposal of a household-level receipt model, 27 Oct 2026

- **Comment:** model household receipt as a function of need and the number of linked migrants (household-level logit, roughly `K = α + β·Kin(π_send) + δ·FarFam`, 6 parameters), instead of independent migrant senders.
- **What happened:** first kept the migrant-level logit (27 Oct, need added there), then compared both (`network-compatibility-model/recipient-level/`): the household logit fits MMP receipt better (AIC 2303 vs 2374). **Adopted 28 Oct** as the receipt layer; it survives in the willing-and-able model as the receipt curve s_h (any core sender, kin only without core migrants, need in desire).
- **Status: done.**

### C. The colleague's own specification, compared 30 Oct 2026

- **Comment / model:** a two-level model with "keen" migrants, an employment effect on sending (B), need on receipt, and **no stepping-in** (`network-compatibility-model/recipient-level/robustness/colleague_spec.py`, README in `outputs/robustness/colleague_spec/`).
- **Result:** it fits the national series as well as the base if out-of-work migrants keep sending at about half the employed rate. Stepping-in (κ) and continued sending (B) cannot be separated on the national series. Its receipt equation fits 2019 MMP receipt clearly worse than the base logit. It puts the whole 2021–22 rise on the number of receiving households, the same tension as in the current base (§0, shock map).
- **Status: answered on old base.** Claim-1 framing still pending: "the shock was absorbed", with stepping-in as base and continued sending as the observationally equivalent alternative (cf. robustness 6a, savings). To rerun on the current base with the robustness set.

### Cheap items to schedule

A1 rerun (2018 → 2019), A3 shock scaling, A5 HP/STL/year-on-year, A6 likelihood surface (with the full map), C rerun. A2, A7, A8 optional.

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
- **Need (since 29 Oct):** option 6a, more senders per receiving household: θ_j,t = expit(α_j + β N*_t), N*_t = cumulative excess deaths (1 in Dec 2021); receipt unchanged. The former `β × S_t` household shift is the comparison margin H in `need_margins.py`.
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

5. **Need (29 Oct, user): option 6 is the main modelling story for 2021.** Need brings in more senders per receiving household (θ_j,t = expit(α_j + β_s N*_t)), with a cumulative, non-decaying need stock (λ = 1). Receipt is unchanged; transfers per receiving household rise. Experiments and results: `recipient-level/outputs/need-margins/README.md` (6a closes 88% of the 2021Q4 gap, 2022 holdout 0.024). Caveats: a placebo ramp fits as well (identification is by timing and by the intensive-margin evidence, not by the need series); the cross-section (ENIGH cells, state panel) shows no need gradient. Supporting fact: ENIGH receiving households +5% 2018→22 while Banxico counts +40%, so count growth is transfers per receiving household. **Chosen: 6a (all migrants; core θ 44% → 60% by end-2021) and series E (excess deaths, cumulative).** The count mapping is the base mapping (transfers per receiver constant except need-activated senders), stated as an assumption; the lower bound overshoots the stimulus spikes. **In the base model since 29 Oct** (`run_household_model.py`; `need_margins.py` reproduces it to 2e-10). Results: β fitted per κ 0.64–0.68 (core θ 44% → 61% by Dec 2021); best cell κ = 1, β = 0.65, RMSE 0.012, 2022 holdout 0.024, 2020Q2 0.975 (Banxico 0.974); κ = 0.8: 0.014 / 0.024 / 0.967; 161 of 275 cells within τ = 0.04. Spikes (Banxico +4.1 / +3.6): κ = 0.8 +5.0 / +3.6, κ = 1 +4.1 / +2.9, so they still point to κ ≈ 0.8–1. Decomposition 2021Q4 (household, best cell): raw +24.4 = trend +12.3, job loss −0.1, need +9.1, unexplained +3.1 (was +10.5). Lower bound still too low in 2020Q2 (0.946 at κ = 1) and spikes too big (+7.6). Receipt and the distribution are unchanged by need. Still needed: an external anchor for β (ENIGH 2016 pre-period).
6. **Framing of need (29 Oct, user): national crisis, not local family need.** The model's N*_t is a national series: every migrant responds to Mexico's crisis as it builds up, not only to deaths in their own community. The paper claims: *in 2021 migrants sent more through more senders per receiving household, as the national crisis accumulated*. It does **not** claim that migrants responded to their own family's local need, because the cross-section rules out a local need gradient:
   - ENIGH 2018→22: receiving households +5% vs Banxico counts +40%, so the rise is transfers per receiving household (intensive); high-need household cells grew *less* in amounts.
   - State panel (32 states, `identification/state-panel/`): 2021 effect of 1 SD excess mortality 0.006 (0.021) with controls, permutation p 0.73; the 2021 jump is common across mortality terciles.
   - Municipal panel (1,227 municipalities, state FE, `identification/municipal-panel/`): 2021 −0.014 (clustered SE 0.016), 95% CI [−0.045, +0.017], MDE 0.045; median 2021 rise 0.178 / 0.164 / 0.153 by within-state mortality tercile; no event-study response at the death peaks; small negative drift already in a 2016 placebo. Bound: local differential need response ≤ ~0.02 log points per SD (mean 2021 rise 0.14). Income loss untested (no municipal measure).
   - Timing: builds through 2020–21 and persists in 2022 (non-decaying stock; decay rejected by the 2022 holdout).
   - Limits to state in the paper: the time series cannot distinguish national need from a placebo ramp or from US-side capacity (wages, savings, channel shift); national need is the interpretation the cross-section supports, not an identified effect. The remaining anchor for β is the ENIGH 2016 pre-period (is the pre-2020 trend also intensive?).

**Base map:** κ × π (× β), household model, criterion as in §3 (quarterly for the headline, monthly for benefit timing).

## 5. Plan

**Step 1. Decisions: done** (§4, 28 Oct).

**Step 2. One rebuild script: done (29 Oct).** `network-compatibility-model/run_all.py` rebuilds every output in order: logit comparison (M1 alphas) → household model (E0, E1, E2 map κ × β, need fit, distribution) → need-margin experiments (robustness, also checks the base) → migrant-level comparison under the coordination rule (E1, E2 map) → figures. Options: `--data-prep` (inputs; needs MMP/IPUMS/Banxico locally), `--identification` (ENIGH, state, municipal), `--skip-robustness`, `--from STEP`. **Cleanup (29 Oct):** removed the lost-sending and all-core rule runs and their outputs (`outputs/e0-baseline`, `e0-need`, `e1-job-loss`, `e1-all-core-rule`, `e2-map`, `e1-distribution*`), the old migrant figures (`network-compatibility-model/figures/`; `model/make_figures.py` now draws only the data figures A1–A4, B1 into `figures/` and holds the shared style), the household-model variants `--kappa-logit` and `--original-coding`, the logit-comparison flag `--not-working-as-unknown` (its outputs are now `recipient-level/outputs/`), and the scripts `run_e0_need.py`, `run_e1_rule_comparison.py`, `diagnose_e1_kappa.py`. `run_e0_baseline.py`, `run_e1_job_loss.py` and `run_e1_distribution.py` remain as libraries. All of this is in the git history up to `a83b60c`. Kept: `e1-coordination` (now without the lost-sending comparison rows), `e2-map-coordination`. **Age rule (checked 29 Oct):** `abm.load_population` makes every migrant eligible and codes 'Not working' as unknown, so the old adults-only branch in `compare_household_logit.load_population` was a no-op except for 7 'Not working' migrants under 18 or of unknown age, which the 27 Oct M1 fits had excluded. Branch removed, M1 refitted (α_core −0.2296 → −0.2325, α_kin −3.361 → −3.367), household model rerun: changes ≤ 0.0002 (best cell κ = 1, β = 0.65, RMSE 0.0123). The default and `not-working-unknown` outputs of the comparison are now identical, and `run_household_model.py --original-coding` no longer reproduces the original coding (drop it in the cleanup).

**ENIGH 2016 pre-period and monthly profile (done 29 Oct, `identification/enigh/enigh_pre_and_monthly.py`, README section):**
- Pre-COVID growth was already intensive: 2016→18 Banxico counts +0.133 log, ENIGH receiving households +0.039, so transfers per receiving household +0.095 (0.037), about 4.7%/yr.
- 2018→22 transfers per receiving household +0.282 (0.034); excess over the 2016–18 trend **+0.09 (0.10)** counts-based (mostly Banxico's own excess, so not independent of the fit), +0.27 (0.12) amounts-based. Receiving households stay on trend (−0.02, 0.10), supporting 6a's "receipt unchanged".
- Model (6a, β ≈ 0.65): transfers per receiving household +0.100 in 2022Q1–Q3 vs no-COVID. Consistent with the counts-based anchor, below the amounts-based one; the anchor is loose (95% CI about −0.09 to +0.51).
- Monthly (ingresos, P041 = concentradohogar `remesas` for 100% of households): 2020/2018 receipt ≈ 0.90 in every month including Feb–Mar (a level difference, not COVID timing); **no April 2020 dip**: Mar→Apr 2020 minus 2018 +0.016 (0.048). Model Mar→Apr −0.035 (κ = 1), −0.044 (κ = 0.8), inside the CI; κ = 0 (−0.079) at its edge. Recall bias: the month before the interview is ~20% higher; same-month comparisons across years are protected (same interview-month mix).

**Robustness 6a–6h (done 29 Oct, `recipient-level/robustness/`, table `outputs/robustness/summary.csv`, one README or summary per test):** every test reproduces the base exactly before changing anything.
- **Robust:** κ ≈ 0.9–1; 2020Q2; 2021Q4 ≈ 1.09 and the 2022 level; need closes 76–86% of the 2021Q4 gap; remainder +3 to +4 log points.
- 6a savings (out-of-work migrants send with probability s): substitute for κ (best κ 1 → 0.9 → 0.5 at s = 0, 0.25, 0.5; fit 0.012 throughout); large s makes 2020Q2 too shallow (0.98) and the spikes too small, so the spikes and the quarterly fit disagree on κ. Small savings compatible.
- 6b π(Apr 2020) = 1: overshoots April month by month (1.012 vs 0.912; the dip is the March rush) but fits Mar+Apr pooled (1.004 vs 1.000); quarterly RMSE 0.0123 → 0.0139. Keep π = 0; April cannot identify π (0.5–1 compatible pooled).
- 6c stimulus as γ (all migrants' sending, or receipt, in Jan/Mar 2021): not separable from π; γ fits the spikes at every κ, so the spikes stop identifying κ under γ. **The quarterly fit is best without any stimulus (0.0084 vs 0.0123):** the Banxico spikes look like transfers shifted between months within 2021Q1 (low February), not extra volume. **Decision (user, 29 Oct): keep the base (π = 1 fitted to the spikes) and report this.**
- 6d Mar–Apr 2020 pooled: best κ stays 1 under every criterion; the model is about 2 log points below Banxico in Mar–Apr and May–Jun 2020 even at κ = 1 (a small early-2020 level gap that π(Apr) or small savings would fill).
- 6e out of sample (calibrate 2018 CPS, run 2019): index within ±0.5% per quarter (RMSE vs 1: 0.003 at κ = 0, 0.001 at κ = 1); coefficients unchanged; the model does not explain Banxico's normal-times ±2–3% (no better than a flat line). Placebo passed.
- 6f need specification: persistence is what matters (λ = 0.95 / 0.9 / 0 fail the 2022 holdout: 0.047 / 0.066 / 0.093); E ≈ S; L (employment loss) worse; 6b kin-only fits slightly better but overshoots 2022 and needs kin θ 3.3% → 18%; placebo ramp fits better in-sample (0.009) but fails 2022 (0.041).
- 6g count mapping (w = share of a job-losing sender's transfers lost despite stepping-in): only w = 0 (base) hits 2020Q2; the data tolerate w ≤ 0.25; κ = 1 best at every w; β 0.64 → 0.52.
- 6h Banxico trend window: κ ≈ 1 and 2020 unchanged; **β 0.50–0.88** (2015–19 to 2010–19; core θ end-2021 58–66%); remainder +2.9 to +4.2 (0.5 for 2010–19); the Banxico spike target itself depends on the window; a trend without month effects is rejected.

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
7. **2021: more senders as the national crisis builds.** Fig. 5: decomposition of raw growth (trend, job loss net, need as more senders per receiving household, unexplained +3 log points at 2021Q4). Evidence for the intensive margin (ENIGH) and against local need (state and municipal panels, §4 item 6); alternatives (placebo ramp, US capacity, channel shift) discussed as observationally equivalent in the time series.
8. **Households.** Fig. 6: who loses, with the ENIGH validation. Only if Step 3 supports it; otherwise a short section with caveats.
9. **Limitations and conclusion.**

## 7. Where things are

- Code: `network-compatibility-model/run_all.py` (rebuilds everything), `recipient-level/` (base household model, logit comparison, need-margin experiments, figures S1–S10), `model/` (`abm.py` and helper libraries; migrant-level comparison `run_e1_coordination.py`, `run_e2_map.py`; data figures), `data-prep/` (inputs), `identification/` (ENIGH, state and municipal panels).
- History and all earlier numbers: `network-compatibility-model/handoff.md` (archive).
- Keep in mind: the repository is **public**; never commit `data/` or MMP/IPUMS-derived microdata.
- The user prefers simplicity: extensions are robustness checks unless they enter the outline.
