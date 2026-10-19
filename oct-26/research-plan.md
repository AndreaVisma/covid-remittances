# Transnational household networks, pandemic shocks and remittance resilience

Research plan, version 1 — 7 October 2026

**Superseded as an implementation plan by [the three-month paper plan](research-plan-three-months.md).** This document is retained as a background catalogue of possible methods and extensions. Its larger model, data requirements and experiments are not delivery commitments. The focused plan governs scope and the 7 January 2027 deadline.

## 1. Research question and contribution

**When do migrant family networks protect households against simultaneous shocks in origin and destination countries, and who remains unprotected even when aggregate remittances grow?**

The empirical case is Mexico–United States during COVID-19. Build a monthly, empirically constrained network microsimulation with Mexican households, migrants in the US, and potential support links between them. Treat 2018–2019 as the baseline, 2020–2022 as the principal shock/recovery period, and 2023–2024 as a later-period assessment. ENIGH 2018 anchors household structure; 2019 labour and population information updates the immediate pre-pandemic baseline without using pandemic outcomes.

The intended contribution is an explanation of the relationship between aggregate resilience and unequal household protection. It is not simply another model that reproduces national remittance growth. Complexity enters through shared budgets, correlated employment risks, contingent activation of dormant links, and competing demands on the same migrants. These interactions must demonstrably change outcomes relative to a simpler independent-sender model.

Three questions organize the paper:

1. Which combinations of sender capacity, origin need and network activation can jointly reproduce remittance growth and within-group receipt patterns?
2. Does protection depend more on the number of migrant links, their economic diversity, or access to US benefits?
3. Can aggregate resilience coexist with larger gaps in protection between households with strong, weak and absent migrant connections?

Claims should concern mechanisms consistent with multiple observations, and mechanisms the observations reject. Cross-sectional receipt data and aggregate flows alone cannot identify the true network or prove that activation occurred.

## 2. Starting evidence and resources already available

The existing analysis uses household summaries from ENIGH 2018, 2020, 2022 and 2024. National receipt rates are approximately 4.73%, 4.38%, 4.57% and 3.95%. Conditional quarterly receipts, approximately in 2024 pesos, are 11,250, 11,981, 15,975 and 15,149. Rural receipt is comparatively stable; urban receipt declines. These are descriptive estimates; design-based uncertainty remains to be added.

Use the files in `oct-26/outputs/enigh-households-2018-2024/`, especially `baseline_2018_templates.csv`, `simple_type_targets.csv`, `detailed_type_targets.csv`, and `household_groups.csv`. The 16 simple types combine locality, age composition and employment presence. Preserve separate attributes and the mapping to detailed types, so reporting can be simplified without rebuilding the model.

The MMP analysis in `oct-26/analysis/mmp-households.py` and associated outputs supplies household profiles and recorded migrant-kin structures. Its communities are not a national probability sample. Potential kin are not necessarily earners or remitters; incomplete observation of kin is not absence of kin. Last-trip jobs and remittances must not automatically become current employment or transfers.

The existing `microsimulation/` prototype has useful budget checks, paired experiments and mechanism switches. It currently uses invented populations, one recipient per migrant, no savings, and a dimensionless need index. Reuse its tested accounting and experiment infrastructure where appropriate, but replace these substantive assumptions. Keep `simulation/` and previous outputs as documented prototypes rather than overwriting them.

## 3. Hypotheses and discriminating predictions

| Hypothesis | Mechanism | Observable prediction or diagnostic | Main ambiguity |
|---|---|---|---|
| H0: fixed senders, income-constrained transfers | Pre-existing senders remit a fixed share of spendable resources | Sender losses reduce transfers, especially in exposed occupations | A benchmark interpretation, not a claim to replicate the World Bank forecasting model |
| H1: fiscal support sustains capacity | US benefits replace income or relax liquidity constraints | Greater protection where eligible migrants receive support; response follows payment timing | Eligibility and take-up are incompletely observed |
| H2: origin need raises contributions | Labour losses or health-related costs raise requested support | Higher conditional receipts among exposed households, subject to migrant budgets | Common pandemic timing can mimic a need response |
| H3: dormant links activate | Previously inactive kin begin contributing when need or a funding gap rises | Transfers survive losses of established senders; active links can increase without more recipient households | ENIGH does not observe individual contributors |
| H4: diversity protects | Migrants in different jobs or destinations face less correlated losses | Protection rises with diversity at fixed network size and baseline resources | More diverse networks may also be richer |
| H5: networks share scarcity | A migrant supports multiple origin households | A shock to one household can reduce support to another | Multiple-recipient links are weakly measured |
| H6: observed growth partly reflects measurement or settlement changes | Formal-channel substitution, fees and exchange rates change measured/delivered flows | Administrative growth can exceed growth in real household receipts | A flexible measurement adjustment can conceal model failure |

Do not require all hypotheses to survive. Compare a small number of interpretable combinations and report indistinguishable alternatives. Build a literature-to-mechanism evidence table before choosing priors: study, sample, place, period, reported behaviour, corresponding rule, and what is not measured. Qualitative accounts motivate rules and heterogeneous responses; they do not supply population frequencies unless the design supports that inference.

## 4. Population and data architecture

Keep immutable source identifiers separate from synthetic identifiers. Every imputed field should have a provenance flag and uncertainty specification.

| Table | Unit and essential fields |
|---|---|
| `households` | Origin household: synthetic ID, donor ID, locality/state, member counts, resources by source, needs threshold, savings assumption, simulation weight |
| `origin_people` | Resident person: household ID, age, sex, relationship, employment, occupation/industry, earnings; requires person files |
| `migrants` | Unique US person: migrant ID, age, sex, arrival cohort, occupation/industry, US region/state, earnings, employment state, US household ID, benefit-access scenario |
| `us_households` | Shared US resources, dependants, essential expenditure and tax-unit information where needed; prevents duplicated household benefits |
| `links` | Unique migrant–origin-household pair, relationship, potential-link status, baseline active status, tie strength, history of sending |
| `shocks` | Month × relevant labour/geographic/health strata; transition rates, earnings/hours changes, benefits, prices and exchange rate |
| `targets` | Estimate, period, population, currency, denominator, uncertainty, source and role: calibration or validation |

### 4.1 Mexican households

Start with weighted whole-household donor sampling from ENIGH 2018, preserving the observed joint distribution. Never independently draw size, age structure, employment and income if donor rows already preserve their dependence. Keep recipients and nonrecipients: nonreceipt is not evidence of having no migrants abroad.

Download and audit POBLACION, INGRESOS and TRABAJOS for the relevant vintages before implementing person-level shocks; confirm table names and keys against each vintage's dictionary. POBLACION is required for actual ages and relationships. The existing under-12/12–64/65+ summary is too coarse for age-specific mortality, and ages 12–64 must not be treated as all adults or all workers. INGRESOS supports receipt timing and separating cash resources; TRABAJOS supports origin employment detail. Preserve survey strata and primary sampling units.

Use separate accounts for labour income, public transfers, pensions, other cash, noncash resources and imputed rent. Neither imputed rent nor total current income is a liquid remittance-financing budget. Observed spending is not an exogenous needs threshold because remittances may finance it.

Define needs initially using a transparent locality-specific consumption threshold, scaled by household composition; compare per-capita and equivalence-scale alternatives. Label this a model adequacy threshold unless it exactly implements the official poverty concept. Add health-related exceptional costs separately. Track both cash-resource gaps and a broader welfare measure.

Use two population views: (a) a fixed pre-pandemic population for paired mechanism experiments; (b) a cross-sectional observation view with independently constrained demographic evolution or reweighting to each ENIGH vintage. Do not redraw households independently every month, or claim that repeated surveys identify household entry and exit. Employment changes belong inside the simulation, not in post-hoc reweighting that silently forces employment-related outcomes.

### 4.2 Migrants in the US

Use pre-pandemic ACS microdata to anchor Mexican-born population structure, household composition, arrival cohorts, location, occupations and earnings. Use CPS for monthly labour dynamics. Define the migrant population explicitly: Mexican-born residents are a defensible primary universe, but some US-born descendants also support Mexican relatives. Treat their inclusion as an extension with external bounds rather than assuming all Mexican-origin people remit.

Pool or partially pool sparse Mexican-born occupation cells. Start with approximately six to eight broad occupational groups; retain industry separately because job and industry exposure are different. Select the grouping after examining sample sizes and code harmonization. National foreign-born estimates alone do not identify Mexican-born transitions.

Do not interpret citizenship or country of birth as documentation status or benefit eligibility. ACS/CPS do not provide a complete work-authorization measure. Introduce uncertain eligibility classes with externally supported ranges. Distinguish labour earnings, disposable income, savings and US dependants. Shared tax-unit benefits must be allocated once, rather than independently to every migrant.

The standard 2020 ACS one-year release was disrupted; use pre-pandemic ACS for initialization and CPS for the main shock path, with any experimental 2020 estimates explicitly identified. [Census methodology](https://www2.census.gov/programs-surveys/acs/methodology/design_and_methodology/2024/acs_design_methodology_report_2024.pdf)

### 4.3 Linking households and migrants

Represent a bipartite graph: Mexican household nodes and unique migrant nodes. A potential edge is a feasible family support relation; an active edge carries a positive transfer in a particular period. A migrant can have several origin links, but has only one budget.

Use the user's reporting definition of **core = head or child**, keeping spouse/partner separate in the underlying data and mapping other relationships to kin. Headship is a survey convention and may include an absent migrant in MMP; it must not automatically imply residence in Mexico or economic authority. Preserve role, actual residence, membership and relationship as distinct fields. Retain an alternative spouse-inclusive core classification as a sensitivity check.

Construction sequence:

1. Draw ENIGH origin households and preserve their attributes.
2. Estimate broad MMP conditional distributions of recorded relationships and connection counts, pooling sparse cells and preserving community heterogeneity.
3. Generate multiple plausible networks conditional on origin structure; vary the prevalence of unobserved/no-recorded links. Seek an independent national anchor for migrant-connected households through a targeted audit of ENADID, census migration modules or other suitable surveys before treating any source as measuring current kin abroad.
4. Match migrant profiles to link roles using feasible age gaps, migration histories and destination information. Keep uncertain cross-dataset associations explicit.
5. Enforce migrant population totals and both degree distributions jointly. Check that the number of edges counted from migrants equals that counted from households. Do not create a fresh migrant for every household connection when those links represent one person.
6. Calibrate baseline active links and sending behaviour to 2018 receipt/amount distributions, while preserving variation in dormant links.

For the first executable version, use several migrants per origin household but at most one origin household per migrant. Clearly label this restricted graph. Add shared migrants before making claims about competition or network-wide propagation.

Use equal-weight synthetic copies, or consistently weighted whole connected components. Arbitrary individual ENIGH and ACS weights cannot simply be multiplied across graph edges. Report implied represented household and migrant totals; reject networks that match one by implausibly inflating the other. A small prototype may have large Monte Carlo error because only roughly 4–5% of households receive; choose production size by convergence of subgroup outcomes.

## 5. Exogenous shocks and policy modules

### 5.1 US employment, hours and earnings

Estimate monthly employed/unemployed/inactive transition probabilities from matched CPS records where feasible, conditioning on broad occupation, demographics and destination region. Keep previous occupation for displaced workers. Handle rotation matching, attrition and code changes explicitly. If matching is infeasible, fit a parsimonious transition system that reproduces observed stocks and report the nonuniqueness of its flows; employment-rate changes are not job-loss probabilities.

Model earnings through employment, hours and conditional wage changes. Selection can raise average wages among remaining workers even as everyone loses resources, so do not apply an observed mean-wage increase mechanically to all migrants. BLS documents pandemic employment-status misclassification; include an alternative treatment rather than silently ignoring it. [BLS](https://www.bls.gov/blog/2020/update-on-the-misclassification-that-affected-the-unemployment-rate.htm)

Correlate shocks through occupation/industry and geography, adding residual within-network correlation in sensitivity experiments. Compare concentrated and diversified networks at fixed degree and baseline total earning capacity.

### 5.2 US benefits

Build dated policy schedules for regular unemployment insurance, pandemic supplements/extensions, PUA and Economic Impact Payments. Benefits depend on policy date, US state where relevant, earnings/work history, eligibility, take-up and payment delays—not occupation alone. Occupation predicts exposure and earnings; it is not a legal eligibility rule.

Separate stimulus tax units from unemployment claimant units. Distinguish contemporaneous cash payments from later tax-credit recovery, including mixed-status household changes. Audit historical rules from archived IRS/DOL guidance before coding exact entitlements. Model take-up and access separately from statutory eligibility; legal status is uncertain in the input data. [IRS eligibility documentation](https://www.irs.gov/newsroom/2020-recovery-rebate-credit-topic-b-eligibility-for-claiming-a-recovery-rebate-credit-on-a-2020-tax-return); [DOL pandemic unemployment guidance](https://www.dol.gov/node/160837).

The first version can use a small number of benefit-access classes and externally constructed expected monthly payments. Later add more detailed rules if results depend on that simplification. Counterfactuals: no additional pandemic benefits, observed-access scenarios, and expanded access with an explicit fiscal cost. Do not mistake an expanded-access experiment for an estimate of an enacted policy.

### 5.3 Mexican employment and economic resources

**Preferred approach: ENOE-based labour shocks**, distinguishing employment loss, hours/earnings reductions, inactivity and informal/self-employment. Start with broad sector × locality and partially pool geography. This is more informative for household need than a uniform GDP multiplier.

The hardest period is April–June 2020: ENOE fieldwork was suspended and ETOE provided telephone-based information. Document this design break, use ETOE as an aggregate bridge, and avoid treating respondents or estimates as a seamless household panel. Compare alternative bridge paths. ENOE/its new edition then informs recovery. [INEGI ENOE](https://www.inegi.org.mx/programas/enoe/15ymas/default.html?init=2); [ETOE](https://www.inegi.org.mx/investigacion/etoe/).

Fallback if microdata processing is initially prohibitive: broad sector/locality earnings multipliers plus employment probabilities from published labour indicators. GDP or sector activity can constrain aggregate paths, but output losses cannot be translated one-for-one into household income. Use ENIGH nonremittance labour resources in 2020/2022 as additional checks, not as an unannounced residual adjustment that absorbs remittance behaviour.

Maintain pensions and domestic public transfers separately, with their own observed paths. Begin with exogenous origin labour supply; investigate remittance-induced labour responses only in a later extension, since adding this feedback immediately would weaken identification.

### 5.4 COVID mortality and household health shocks

Use INEGI registered deaths to construct monthly mortality by age, sex and state of residence, using date of occurrence rather than registration where possible. Construct hazards with population denominators. Compare certified COVID deaths with excess all-cause mortality as alternative severity measures; do not add both together. Excess mortality contains indirect effects and uncertainty about expected deaths. [INEGI mortality programme](https://www.inegi.org.mx/programas/mortalidad/).

Recommended two-layer design:

* **Regional severity exposure:** a per-capita mortality series affects perceived emergency need with a parsimonious distributed lag. This is a proxy, not a count of household illnesses; deaths often lag infections and spending.
* **Household bereavement:** assign stochastic deaths using demographic hazards and allow loss of earnings/pensions, temporary funeral costs and changed household size. Surviving needs can increase or decrease; do not force every death to raise remittances indefinitely.

Deaths alone do not identify infections, hospitalization, morbidity, medical costs or their household clustering. Estimate those from additional evidence or present explicit low/central/high mappings. Avoid fitting a large health-cost multiplier solely to remittance totals. Check whether removing the health channel leaves conclusions intact.

Prevent double counting: broad ENOE labour losses may already include sickness-related absence. Either use the reduced-form labour path plus additional documented costs, or explicitly decompose health and economic losses and match their combined total. Similarly, account for background mortality separately from pandemic excess. Household deletion, dissolution and recipient death require bookkeeping rules; do not replace deceased people automatically. Add US migrant mortality only with an appropriate separate data source and denominators.

## 6. Monthly behaviour and accounting

Begin with a small, transparent rule set and only a few fitted behavioural parameters. Use nominal USD for migrant budgets and nominal MXN for origin budgets; apply exchange rates, delivery fees and price indices explicitly.

Monthly order:

1. Update demographics/deaths, employment, hours and earnings using common exogenous shocks across experiments.
2. Pay benefits and domestic transfers; update available savings.
3. Calculate origin needs before current remittances from composition, prices, exceptional costs and nonremittance resources.
4. Existing senders decide desired contributions and feasible budgets.
5. Households request additional support from dormant or other connected migrants according to residual need.
6. Allocate migrants' limited budgets jointly across all their origin links. Use simultaneous or randomized allocation to avoid an arbitrary ID-order advantage.
7. Deliver transfers, pay consumption, record unmet need and update savings and link histories.

Illustrative quantities, to be finalized before calibration:

`need_h,t = max(essential_cost_h,t + exceptional_cost_h,t - available_nonremittance_resources_h,t, 0)`

`capacity_i,t = max(earnings_i,t + benefits_i,t + allowed_savings_draw_i,t - US_essential_cost_i,t, 0)`

`P(active_i,h,t) = logistic(intercept_relationship + persistence * previous_active_i,h + response * standardized_residual_need_h,t)`

Actual sending is positive only if willingness and budget permit it. Desired contributions can depend on baseline commitments and need; compare this with the fixed current-income-share rule while keeping every other mechanism constant. A nonzero baseline probability is needed for genuinely dormant links to activate; missing links cannot activate without a separate link-formation mechanism.

For each migrant, sum of transfers and fees cannot exceed capacity. For each origin household, receipts equal delivered transfers across its links. Contributions beyond the adequacy threshold can finance saving or other uses; need is not an absolute ceiling on voluntary transfers. If migrant savings are carried forward, drawdowns must reduce their stock. If origin savings cover a shortfall, record that separately from remittance protection.

Track individual and shared US budgets consistently. A new migrant does not appear solely because the model needs more aggregate money. Add externally constrained migration/return flows for the longer horizon; the core short-run experiment can hold the migrant stock fixed, but 2024 comparisons must acknowledge stock and family changes.

## 7. Measurement, calibration and validation

### 7.1 Two observation systems

Model economic transfers first, then map them to observations.

* **ENIGH:** within-group probability of positive receipts over the actual survey reference period; conditional mean, quantiles and dependence; repeated cross-sections with survey weights and design uncertainty. A quarterly-equivalent amount need not imply a literal three-month observation window: inspect income questionnaires and monthly records before matching simulation periods.
* **Banco de México:** monthly administrative flows in USD, by origin where available. Overall inward flows and ENIGH include non-US sources; either use a corridor target plus a separately accounted residual or explicitly model the broader universe. Match gross/net/fee conventions and formal/informal coverage. Transactions are neither unique migrants nor recipient households.

Use monthly exchange rates and appropriate price indices for precise comparisons. Replace the preliminary common-price approximation in our descriptive analysis before publication. Avoid multiplying a survey period mechanically by four and treating that as observed annual flow.

Keep reporting/coverage parameters externally bounded and parsimonious. First diagnose discrepancies with coverage and timing; do not fit an unrestricted year-specific multiplier to make both sources agree. A failure to reconcile them is a result to report, not evidence of one particular illicit or unobserved channel.

### 7.2 Estimation sequence

1. **Independent inputs:** estimate labour shocks, mortality, prices and fiscal schedules from their own sources. Freeze these before fitting remittance responses, allowing uncertainty draws later.
2. **Baseline:** fit 2018 receipt and amount distributions and pre-pandemic aggregate seasonality. Allow a short initialization period and check that savings or link states do not drift artificially before COVID.
3. **Pandemic behaviour:** fit a small parameter set to predeclared 2020/2022 moments and an explicit subset of monthly aggregates. Use simulated minimum distance initially; use likelihood-free/Bayesian methods only if the additional complexity is justified.
4. **Predictive assessment:** reserve 2023–2024 outcomes and selected unused subgroup moments from estimation. Because descriptive 2024 results have already been inspected, call this later-period validation, not a blind prospective test. If post-2022 observed shocks are supplied, it is a conditional hindcast rather than a forecast of those shocks.
5. **Robustness:** repeat across plausible networks, eligibility assumptions, death-to-cost mappings, survey bridges and measurement specifications.

Give receipt probabilities and conditional amounts separate roles. Prioritize within-group rates; use counts mainly to reconcile total population and flows. Avoid fitting every detailed household cell independently. Use sampling covariance or sensible moment-block scaling so dozens of correlated monthly observations do not overwhelm household evidence. Do not simultaneously count a total and its exact components as independent information.

Supply design-based ENIGH intervals, preferably by resampling PSUs within strata with the appropriate vintage design. MMP community resampling describes uncertainty within its selected communities, not national representativeness. Separate survey sampling, synthetic-network, parameter and simulation uncertainty in reporting.

### 7.3 Identification and failure checks

Before interpreting fitted parameters, generate synthetic data under known mechanisms and test recovery with the proposed targets. Examine objective surfaces and parameter trade-offs, especially network prevalence versus activation, benefits versus savings, and origin need versus commitment. If multiple parameter sets fit equally well, report an admissible set and the conclusions shared across it.

Reject mechanisms that cannot jointly match national and subgroup evidence within defensible uncertainty. Do not infer active-sender turnover from ENIGH cross-sections. Additional migrant surveys or linked qualitative follow-up would be especially valuable for validating activation, substitution and multi-household support.

## 8. Counterfactual experiments

Use the same population, exogenous shocks and random-number streams for paired experiments. Freeze calibrated parameters when disabling mechanisms; separately refitting rival model classes is a model-comparison exercise and must be labelled differently.

1. No-pandemic trend benchmark, with explicitly specified labour and mortality trends.
2. Sender-income shock only with fixed potential network and pre-existing sender participation.
3. Add observed-access US benefits.
4. Add domestic labour/resource shocks and responsive sending.
5. Add the health/bereavement channel.
6. Allow dormant links to activate and existing contributors to substitute.
7. Add migrant sharing across multiple origin households.

The numbered sequence is explanatory, not an order-invariant attribution. Use a factorial experiment for the principal switches—benefits, need response, activation—and report interactions. If an additive decomposition is desired, use a clearly defined averaging over orders, with Monte Carlo uncertainty.

Network-specific experiments should hold total migrants, degree, baseline resources and donor capacity fixed where feasible: diversify occupational/destination exposure; concentrate the same capacity in fewer links; randomize household assignment within matched strata; vary overlap in migrants serving several households. Separate diversification effects from simply adding more income. Explore a grid of shock correlation and network redundancy, including synchronized crisis scenarios.

A simple intuition to test, not assume: with k independent equally reliable links each available with probability p, at least one is available with probability 1-(1-p)^k. Correlated job losses and shared financial obligations reduce that protection. The simulation determines whether an available migrant can actually cover the household shortfall.

## 9. Outcomes and selection of results

Predeclare a small primary outcome set:

* Within-group probability of receiving and its percentage-point change, including all households in each denominator.
* Conditional receipt distribution and unconditional receipts per household, in real MXN.
* Remaining resource shortfall, months below the adequacy threshold and cumulative shortfall; report absolute gaps and proportional shock buffering.
* Protection by baseline network size, relationship structure, diversity and benefit access, with origin composition and initial resources held comparable.
* Aggregate USD flows and recipient-country purchasing power.

Secondary outcomes: active versus potential links, sender substitution, fiscal dollars transmitted abroad, migrant consumption shortfalls/savings depletion, and inequality. Separate remittance inequality among all households (many zeros) from inequality among recipients and inequality in total resources. Report rural/urban and network-poor groups even when aggregate fit is good. Network size is an explanatory dimension, not the primary definition of household welfare.

Do not select only parameters, seeds or household groups where activation helps. Report the distribution across admissible parameter/network ensembles and include cases of failure. More remittances need not imply better welfare if needs rose faster or migrants depleted their own buffers.

Main-text figure plan:

1. **Empirical puzzle:** aggregate USD and real-MXN trends alongside ENIGH within-group receipt and conditional amounts; show uncertainty and distinguish observation windows.
2. **Model and data:** bipartite network schematic, shared budgets, shocks and the two observation systems.
3. **Joint fit and later-period performance:** observed versus simulated rates and amount distributions; identify calibration and validation periods explicitly.
4. **Mechanism contributions:** paired effects and interactions on aggregate flows and household shortfalls; uncertainty across networks/parameters.
5. **When networks protect:** protection surface over redundancy and employment-shock correlation, at fixed resources; supplement with shared-donor competition.
6. **Who remains exposed:** resource-gap changes and receipt by origin type and network access, alongside migrant-side costs.

Keep six as a maximum; combine figures if the story is clearer. Append full diagnostics, target tables, alternative needs thresholds and negative results. A mechanism paper needs at least one result that cannot be obtained simply by assigning each household an independent income multiplier.

## 10. Development stages and completion criteria

| Stage | Deliverable | Completion criterion |
|---|---|---|
| A. Evidence and target audit | Source registry, literature-to-rule matrix, harmonized target table | Units, universes, reference periods and uncertainty roles documented; baseline/validation split frozen |
| B. Origin and migrant populations | Validated household/person and migrant/US-household tables | Reproduce selected source margins and joint distributions; no double counting of resources |
| C. Network ensemble | Several plausible graphs plus provenance and diagnostics | Feasible relationships, balanced edge counts, credible household/migrant totals; uncertainty retained |
| D. Independent shock modules | US labour/benefits, Mexico labour, mortality, prices/FX | Match source series without fitting remittance outcomes; pandemic survey breaks addressed |
| E. Minimal dynamic model | Budget-consistent simulator with activation switch | Reproducible, no negative stocks, conserved transfers, stable pre-pandemic baseline |
| F. Calibration and recovery | Parameter/admissible-set estimates and synthetic recovery report | Known-mechanism recovery assessed; weakly identified parameters clearly marked |
| G. Experiments and validation | Predeclared figures and robustness ensemble | Main claims survive plausible data/network assumptions or are appropriately narrowed |
| H. Paper and replication | Manuscript, model description, runnable release | Every principal result reproducible from recorded inputs/configuration; source licence restrictions respected |

Suggested first implementation scope: national model with rural/urban origin strata; broad US occupations and regions; one origin household per migrant; monthly employment, benefits, monetary need, savings and activation. Then add mortality heterogeneity and shared migrants. State identifiers remain available for shock assignment even if state comparisons are not presented.

Use plain Python, NumPy and pandas with configuration files and tidy CSV/Parquet outputs. Add optimization tools only when calibration requires them. Tests should target budget conservation, edge consistency, benefit nonduplication, zero-shock/zero-activation limits, period alignment and known-mechanism recovery. Establish numerical convergence by enlarging population and ensemble sizes until Monte Carlo error is small relative to survey uncertainty, rather than choosing an arbitrary number of runs.

## 11. Paper structure

Working title: **Transnational family networks and unequal resilience to pandemic shocks**.

1. **Introduction:** the aggregate remittance puzzle; why receipt and protection differ; three questions and bounded contribution.
2. **Empirical setting and evidence:** US/Mexico shocks, observed ENIGH patterns, aggregate measurement and qualitative motivations.
3. **Model and empirical construction:** origin households, migrants, links, budgets, shock processes and source limitations. Include a concise causal/process diagram; full specification in supplement.
4. **Calibration and evaluation:** target selection, uncertainty, later-period assessment and identification limits.
5. **Results:** fit first; mechanism comparisons second; network diversity/competition third; distributional and migrant-side costs fourth.
6. **Discussion:** general principle about redundancy under correlated risk, implications for transnational social protection, alternative explanations and what new linked data would resolve.
7. **Conclusion:** only claims supported across plausible specifications; no claim that a unique historical mechanism has been identified if alternatives remain observationally equivalent.

Supplement: complete model protocol, code/data provenance, variable crosswalks, survey designs, historical benefit schedules, network-construction diagnostics, parameter recovery, sensitivity and all calibration targets. Release synthetic inputs only after checking disclosure and source licences; a synthetic donor population is not automatically safe to redistribute.

## 12. Journal strategy

These are fit judgments based on current stated scopes, not predictions of acceptance. Recheck policies and publication charges when a manuscript exists.

| Journal | Fit and proposed framing | What the paper would need |
|---|---|---|
| **PLOS Complex Systems — recommended first target** | Multidisciplinary scope explicitly includes economics, networks, risk and social networks | Show how network structure and correlated shocks generate unequal resilience; strong empirical constraints, transparent uncertainty and reproducible simulation |
| **npj Complexity — alternative ambitious target** | Cross-field complex systems and emergent behaviour across scales | A clear general result beyond Mexico, e.g. when redundant ties fail under correlated risk and shared obligations; convincing empirical validation |
| **Journal of Artificial Societies and Social Simulation (JASSS) — strong methodological fit** | Interdisciplinary explanation of social processes through simulation | A carefully specified, evaluated social simulation with explicit behavioural grounding; particularly suitable if the model and evidence integration become the main contribution |

Sources: [PLOS scope](https://journals.plos.org/complexsystems/s/journal-information), [npj Complexity scope](https://www.nature.com/npjcomplex/aims), [JASSS](https://jasss.org/JASSS.html).

Prefer PLOS Complex Systems for the current concept. Consider npj Complexity if the experiments establish a general relationship between redundancy, correlation and competition that is robust across plausible network constructions. Prefer JASSS if careful model validation is stronger than the claim to a broadly general complexity result. Do not add technical machinery merely to fit a journal. A substantive migration/development outlet can be reconsidered if the eventual contribution is mainly empirical rather than mechanistic.

## 13. Immediate next work and decisions

1. Audit person/job/income ENIGH files and construct design-based uncertainty around existing targets; retain the current minimal plots.
2. Audit MMP recorded relationships, current residence and occupational timing; produce a network constraint table with observed versus assumed fields.
3. Assemble pre-pandemic ACS and monthly CPS/ENOE input specifications and a historical benefit schedule, before choosing detailed parameter values.
4. Build a small restricted-network prototype using existing household templates, replacing dimensionless need with a monetary resource gap.
5. Test whether benefits and responsive existing senders already explain the joint target patterns. Introduce activation as a testable addition, not a required explanation.
6. Expand to demographic mortality assignment and shared migrants once the simpler model is stable.

The principal unresolved empirical issue is the national distribution of potential support networks. The principal identification issue is distinguishing more money from established contributors from activation of additional kin. Both should remain visible throughout the project rather than being hidden inside a single fitted population.
