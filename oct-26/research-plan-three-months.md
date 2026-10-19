# One paper in three months: migrant networks and household resilience

7 October 2026 — delivery deadline: 7 January 2027

**Implementation audit, 16 October 2026:** this document describes intended scope,
not completed modules. The active model currently uses MMP-centred connected
households, seven fitted baseline parameters, a 2019 baseline, and CPS capacity
paths for 2020–2022. It compares coordinated and uncoordinated allocation; the
fixed-incumbent/activation comparison, estimated COVID need response, benefits,
correlated employment shocks, and resource-inequality analysis remain planned.
Current core includes head, spouse, and children (RELHEAD 1/2/3). See the
[research questions, implementation audit, and 30-paper literature assessment](research-questions-literature-review.md)
for the updated experiment priorities and novelty assessment, and the
[current model notes](simulations/abms/remittance-minimal-cps-abm-notes.md)
for actual settings and saved results. The suggested extensions below are not
additional completed features or requirements to add every module before testing
the narrower research questions.

This plan supersedes the implementation scope of `research-plan.md`, which remains a background reference. Delivery means a submission-ready manuscript, figures, supplement and reproducible code/results; journal acceptance is outside this schedule. Dates are milestones, not scheduled automations. The plan assumes one lead researcher working regularly on the project, with timely coauthor feedback. If available time is lower, remove optional checks rather than expand the deadline implicitly.

## 1. One question and one contribution

**Can activation of additional migrant kin sustain remittances during a crisis, while leaving households with fewer migrant connections less protected?**

The paper compares this mechanism with increased contributions from existing senders under the same employment, benefits and origin-need shocks. Its contribution is the relationship between aggregate resilience and unequal household protection. It will assess plausibility and discriminatory evidence, not claim to reconstruct the actual national network or identify a unique historical mechanism.

Primary period: baseline 2018 and pandemic/recovery through 2022. Use 2019 information to anchor immediate pre-pandemic conditions where available. ENIGH 2024 is an external descriptive check, not a requirement to simulate a changing population through 2024. No new mechanisms will be introduced just to match 2024. The previously observed 2024 results are not a blind holdout.

## 2. Minimum model

Use the existing ENIGH household templates as weighted origin donors. Preserve their joint characteristics, but summarize results using rural/urban locality and broad dependent-age composition. Employment count is a state and resource attribute; it can change under shocks. Use a limited number of broad groups for calibration rather than all 16 types separately. Preserve those types for supplementary reporting.

Each household has zero, one, two or three potential migrants, with three representing a top-coded network-size class in the simplest implementation. Check a larger upper limit once as a robustness exercise. A network-size class does not imply every migrant is working or sending. Retain core/kin relationship categories, with core defined as head or child as requested; keep spouse identifiable in the inputs. Do not cross every origin category with every kin type in the main results.

MMP informs plausible conditional network structures and relationship differences. It does not establish national frequencies. Use a small set of low/central/high connectivity scenarios, all constrained by credible migrant totals and baseline receipt. Do not present their spread as a sampling confidence interval.

Each migrant is attached to exactly one Mexican household. Migrants have a broad occupation, baseline earnings, employment status, a parsimonious benefit-access class and an active/inactive sending state. They face monthly income shocks and a budget after US essential costs. They can activate when another contributor loses capacity or origin needs rise. This is a collection of household support networks, not a reconstructed nationwide kinship graph. It still permits substitution and correlated risks among contributors.

Start with no borrowing and no endogenous savings accumulation. A small bounded initial buffer can be a sensitivity scenario if liquidity timing proves important. Report the limitation rather than adding a detailed wealth model. Count each benefit once; expected per-migrant payments must already account for household-level allocations. Do not independently grant a full tax-unit payment to every migrant.

At each month:

1. Apply externally specified US resources and Mexico labour/need shocks.
2. Compute each migrant's transfer capacity and each origin household's resource gap.
3. Existing senders make budget-constrained contributions.
4. If activation is enabled, inactive linked kin may contribute to remaining need.
5. Sum receipts and record unmet household need and migrant resource strain.

Use one shared rule for additional contributions. Do not separately estimate willingness, bargaining, tie decay, learning and persistence unless a simpler version demonstrably fails a necessary accounting check. Baseline participation heterogeneity can be fixed by broad relationship class; one common response parameter is preferable to a parameter for every household type.

## 3. Keep shocks simple and externally informed

| Component | Main-paper implementation | Boundary |
|---|---|---|
| US employment and income | Four to six broad occupational groups; observed employment and earnings/hours paths with a simple persistence rule | No full state-by-occupation transition model; employment stocks do not identify flows |
| US benefits | Expected monthly payments for a few access classes, anchored to historical policy timing and plausible receipt | No full tax-benefit simulator; eligibility/access cannot be inferred from occupation or citizenship alone |
| Mexico employment | Broad locality/sector resource-loss paths, using ENOE and an explicit ETOE bridge for the 2020 break | No detailed origin labour-supply response or GDP-to-income conversion |
| COVID severity | Mortality-based regional exposure with an explicit lag/scale, used as a temporary need shifter | No individual infections, deaths, hospitalization or household dissolution |
| Prices and currency | Observed monthly exchange rates and price indices | Keep units and reference periods consistent |

Deaths calibrate the health exposure's timing and relative severity, not its financial cost. Treat the mapping into need as an externally bounded sensitivity, including zero; do not fit it jointly with several free behavioural responses. Avoid adding sickness-related lost earnings a second time if included in observed labour-income paths. Keep geographic identifiers for assigning shocks, without making state differences a paper objective.

Data acquisition is capped at two weeks. Prefer existing ENIGH summaries and MMP outputs, a narrow US labour extract, broad Mexico labour series, mortality and historical benefit schedules. Person-level ENIGH is optional unless essential for defining a cash budget or survey observation window; full relationship reconstruction is not a prerequisite. If disposable cash cannot be measured, use a clearly labelled resource-adequacy gap and do not claim to measure cash poverty. An occupation-level shock can be applied through a parsimonious process matching stocks; if matched CPS estimation is infeasible by the data cutoff, use that simpler specification and state that transitions are assumed.

## 4. What is explicitly outside this paper

No migrants shared across several origin households; no endogenous migration, return migration, births or household formation; no infection model or assigned individual deaths; no full US tax-benefit engine; no national network reconstruction; no endogenous Mexican labour supply; no complete 2024 demographic simulation; no new primary qualitative survey.

The qualitative literature grounds activation/substitution and relationship categories through a small evidence table. It is not a second systematic-review project. The longer original plan is a future-work menu, not a backlog to finish in this paper.

## 5. Three principal models and limited experiments

| Model | Rule | Question |
|---|---|---|
| A: fixed senders | Pre-existing senders remit a baseline share of available resources | What does an income-capacity account explain? |
| B: responsive existing senders | Same potential network and shocks; existing senders respond to origin needs | Can larger contributions alone reproduce the evidence? |
| C: activation | Model B plus additional linked kin responding to residual needs | Does allowing extra contributors improve joint fit and protection? |

All three use the same observed-access benefits and economic shocks. A separate C-without-additional-pandemic-benefits counterfactual measures the modelled contribution of benefits. Use paired random draws and freeze parameters for mechanism-off effects. If B and C are separately fitted as rival models, give them comparable calibration budgets and report predictive performance; do not confuse refitting with a causal decomposition.

One network experiment is central: vary shock correlation at fixed network size and baseline earning capacity. It tests whether additional kin provide effective redundancy or merely share the same exposure. Compare protection across network sizes separately, clearly distinguishing additional resources from diversification.

Only four priority robustness checks: network prevalence; correlated shocks; benefit access; zero versus positive mortality-related need. Add numerical convergence and measurement alignment checks as required validation, not substantive extensions. Avoid a large factorial design across every possible assumption.

## 6. Calibration, validation and an early stopping rule

Estimate external shocks separately. Fit no more than roughly three to five behavioural parameters overall, with the exact count justified by recovery checks. Baseline receipt/amount parameters use 2018; need response and activation use selected 2020 evidence. Test frozen specifications against 2022 ENIGH and later unused aggregate months, while providing observed shock paths. This is conditional historical validation, not an unconditional forecast. Observed 2022 patterns have already informed the project, so do not call it blind.

Primary targets: receipt within broad household groups, conditional receipt amounts, and aggregate trend shape. Include national levels as accounting diagnostics. Preserve survey reference windows, all-origin versus US coverage, and USD versus MXN units. Allow only an explicitly documented baseline observation adjustment if necessary; no freely fitted multiplier for each year. A discrepancy between administrative and survey levels must remain visible.

Prioritize design-based uncertainty for national and broad-group ENIGH rates. Pool cells where precision is inadequate. Check one synthetic recovery exercise to see whether activation and increased incumbent sending can be distinguished with the proposed moments.

**By 4 November, decide whether activation is empirically distinguishable.** If it is not, retain both plausible explanations and center the paper on the range of distributional outcomes compatible with aggregate resilience. Do not rescue identification by adding more free parameters. If no version matches the broad evidence, diagnose measurement and input failures before advancing the substantive story.

## 7. Four main figures and one compact results narrative

1. **Puzzle:** aggregate flows, within-group receipt and conditional amounts, with units and survey uncertainty clear.
2. **Model comparison:** observed versus A/B/C outcomes, marking fitted and validation targets. Include a small network schematic here or in the methods.
3. **Network protection:** resource-shortfall reduction by network size and employment-risk correlation, controlling baseline resources where appropriate.
4. **Distribution:** remaining shortfalls for households with no, few and more connections; include origin vulnerability and uncertainty across plausible networks.

Primary welfare outcome: cumulative resource shortfall relative to a declared adequacy threshold. Primary remittance outcomes: within-group receipt and conditional amount. Aggregate fit is necessary context, not the sole success criterion. Do not make Gini coefficients the main welfare result; put optional inequality measures, detailed typologies and model diagnostics in the supplement. Report migrant-side resource strain as a check, so origin protection is not implicitly costless.

Main paper target: about 6,000 words, four figures and one data/assumptions table, subject to the chosen journal's requirements. Suggested allocation: introduction 700; evidence/data 900; model/calibration 1,400; results 1,800; discussion/limitations 1,000; conclusion 200. These are working limits rather than journal rules. Supplement contains the full model specification, data transformations, parameter table and essential robustness.

## 8. Delivery schedule

| Dates | Deliverable and gate |
|---|---|
| 7–14 October | Lock question, three models, primary outcomes and target definitions. Draft introduction and empirical-puzzle figure using existing outputs. |
| 15–21 October | Finish narrow data audit and shock inputs; establish broad-group ENIGH uncertainty. Stop expanding data sources on 21 October. Record fallbacks and remaining limitations. |
| 22 October–4 November | Complete executable A/B/C prototype, budget tests, baseline calibration and first joint-fit plots. Run identification/recovery check. Decision gate on 4 November. |
| 5–18 November | Complete production calibration and 2022 validation; resolve substantive accounting or observation issues. Draft data and methods. **Freeze model features on 18 November.** |
| 19 November–2 December | Run the limited network experiments and four robustness checks; stabilize four main figures. **Freeze primary results on 2 December.** |
| 3–16 December | Finish the full manuscript and supplement; obtain coauthor feedback by 16 December. No additional experiment unless it addresses a material error or central objection. |
| 17–23 December | Revise claims and figures; complete clean-run replication and journal formatting. Aim for a submission-ready package by 23 December. |
| 24 December–7 January | Holiday/contingency buffer, final coauthor approval and delivery by 7 January. Do not plan essential new research here. |

Writing runs alongside modelling. Budget by milestones because actual time depends on data access and collaborator availability. If a milestone slips, drop optional 2024 comparisons, fine kin subdivisions and supplementary inequality statistics first. Preserve accounting, primary validation and uncertainty; do not trade these for more mechanisms.

## 9. Paper and journal decision

Working title: **Migrant kin networks and unequal household resilience during the pandemic**.

Paper argument: the empirical puzzle motivates two competing responses—larger incumbent contributions and activation; a minimal model shows which observations discriminate them and when network structure changes protection; the discussion distinguishes model-supported mechanisms from unresolved historical attribution.

Keep **PLOS Complex Systems** as the first target and **JASSS** as the alternative. Do not build extra modules to pursue a more ambitious venue. The fit rests on a clear mechanism and its emergent distributional consequences, not the number of agents or equations. Scope references: [PLOS Complex Systems](https://journals.plos.org/complexsystems/s/journal-information), [JASSS](https://jasss.org/JASSS.html). These links were checked for the original plan; recheck submission requirements at manuscript preparation.

Success on 7 January is a compact, defensible paper and reproducible analysis, including a publishable statement of what the available data cannot distinguish. It is not completion of the original comprehensive model.
