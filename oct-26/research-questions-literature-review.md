# Research questions, implementation audit, and literature assessment

Prepared 16 October 2026. Scope: US-connected Mexican household support networks during COVID, with 2019 as the current ABM baseline and 2020–2022 as the historical exercise.

## 1. Recommended contribution

**How much additional origin-household need is required to reconcile pandemic remittance growth with observed US labour capacity, and how does the answer—and the distribution of protection—depend on access to substitute contributors?**

The defensible contribution is a quantitative bridge between observed family-level substitution and aggregate remittance growth, followed by a controlled assessment of unequal household protection. It is not the discovery that remittances provide insurance, that relatives substitute for one another, or that networks affect inequality. Those ideas have substantial precedents.

The closest empirical foundation is Ventura and García [P01]: their survey/interview evidence explicitly describes remittance responsibilities spreading across household and extended-family members after job loss and income instability. Treat that mechanism as the empirical starting point. Agarwal and Horowitz [P30] already distinguish multiple- from single-migrant households in remittance theory and evidence. Jack and Suri [P19] connect insurance improvements to remittances and sender diversity. A stronger paper therefore quantifies **when substitution changes aggregate feasibility and who remains insufficiently protected**, conditional on empirically constrained networks and budgets.

Novelty assessments below are judgments from a targeted search, not proof of an unoccupied research area. No claim of being the first ABM of remittances, the first network-insurance model, or the first explanation of COVID remittance resilience is supported.

## 2. What currently exists in this repository

The active empirical implementation is [the minimal CPS ABM](simulations/abms/remittance-minimal-cps-abm.py); [its updated notes](simulations/abms/remittance-minimal-cps-abm-notes.md) describe the actual saved specification. The [three-month plan](research-plan-three-months.md) remains a scope document, with several modules still unimplemented. Older models in `simulation/`, `microsimulation/`, and `oct-26/simulations/simul-income-*.py` are separate prototypes; their features must not be attributed to the active ABM.

| Component | Audited status | Implication |
|---|---|---|
| Administrative remittances | Banxico monthly inputs and monthly/quarterly count–amount decompositions exist | Can constrain aggregate shape and components, subject to coverage and transaction assumptions |
| Recipient evidence | ENIGH 2018/2020/2022/2024 household descriptions, receipt, amounts, dependence, and templates exist | Provides an external distributional check; survey windows and uncertainty require alignment |
| Potential support networks | MMP PERS/HOUSE reconciliation; 10,000 links and 1,019 connected synthetic households | Population is MMP-like, not nationally representative; no zero-connection households in this ABM |
| Personal profiles | 9,408 of 10,000 profiles imputed; 9,300 borrow a core profile | Kin earnings/job composition is a major uncertainty, not resolved by adding simulation seeds |
| Capacity | Mexican-origin CPS occupation means/unemployment; fixed lognormal earnings ranks; 20% of surplus above $20,000 annual allowance | Capacity is assumption-dependent; Mexican origin is not Mexican-born migrant status |
| Baseline behaviour | Seven fitted parameters; 18 overlapping receipt moments plus 2019 Banxico mean transfer | Baseline has useful empirical constraints, but important fit/identification limitations |
| Allocation | Coordinated and uncoordinated rules, each fitted; same-parameter allocation switches also exported | Isolates allocation under fixed parameters when used correctly; not an activation-off comparison |
| Pandemic | CPS capacity history and paired constant-2019 capacity counterfactual | No estimated COVID need shift; no benefits, savings, or FX in this implementation |
| Network-size outcomes | Unfitted receipt diagnostics by network type and size | Descriptive check exists; controlled redundancy and risk-correlation experiments do not |
| Welfare/inequality | Receipts and perceived need panels are exported | No validated resource-adequacy budget, national income inequality result, or causal migration counterfactual |

Saved output evidence is in `outputs/remittance-minimal-cps-abm/`: `run_settings.json`, `receipt_validation.csv`, `stability_checks.csv`, `persistence_profile.csv`, `network_size_diagnostics.csv`, and the household/migrant panels. Population construction is recorded in `outputs/synthetic-migrants-house-networks/run_settings.json`.

Both baseline mechanisms pass the saved stability and amount tolerances, but neither passes the receipt tolerance in every fitted cell: maximum errors are 9.11/9.00 percentage points against an 8-point criterion. Core-only singleton receipt is 54.39% in MMP versus approximately 42% simulated. Kin-only households with older residents receive in 13.50% of source observations versus approximately 4.4–4.6% simulated. These are substantive weaknesses for size-specific protection claims.

The current monetary ceiling binds for 94.62%/94.37% of positive baseline contributions, pooled across post-burn-in months and seeds. That makes inferred pandemic need particularly sensitive to the assumed surplus share, allowance, and earnings dispersion. The fitted need intercept is approximately $1,169/$1,266; it is not average household need or an independently measured subsistence requirement. The large simulated unmet-need fraction must not be described as measured poverty.

Persistence is weakly constrained by current cross-sections. Fixed-persistence profiles have similar losses, and some improve on the selected joint fit. Optimizer convergence is not evidence of a unique/global optimum. Different stochastic seeds test Monte Carlo stability, not held-out empirical prediction.

### Maintenance performed

- Reconciled the companion notes with current source and saved results: $20,000 allowance, Mexican-origin default, surplus-budget rule, earnings heterogeneity, fitted coefficients, fit failures, budget binding, and sender counts.
- Removed superseded gross-budget conclusions and the obsolete $52,909 allowance from the current description; preserved the distinction between earlier results and current evidence.
- Flagged unresolved provenance: the code/report's MIT label does not document how the current $20,000 assumption was derived.
- Clarified that coordination permits replacement but does not make willingness respond directly to another contributor's job loss.
- Linked the earlier paper plan to this implementation audit rather than silently presenting planned features as completed work.
- Did not alter behavioural code, recalibrate parameters, replace outputs, or adjust tolerances to manufacture a passing fit.

## 3. Questions and novelty assessment

### Q1. What increase in need is required under different support mechanisms?

**Priority: central. Relevance: high. Potential novelty: moderate, conditional on joint constraints and explicit alternatives.**

Estimate the smallest admissible pandemic log-need shift, or a low-dimensional time profile, that reconciles observed remittance growth with frozen baseline behaviour and supplied labour capacity. In the current equation, an additive shift `delta` multiplies monetary perceived need by `exp(delta)` and multiplies willingness odds by the same factor. Report `100*(exp(delta)-1)` as the model-implied percentage increase in perceived need. This coupling is a structural restriction, not evidence that real monetary need and willingness have identical elasticities.

Compare fixed incumbents, responsive incumbents, and explicit shortfall-triggered activation. Retain the present coordinated/uncoordinated allocation comparison as a separate dimension. First conduct capacity-only and monetary-need-only diagnostic sweeps; if amounts barely respond because budgets bind, the data may constrain willingness/capacity rather than monetary need itself.

**Literature check:** altruism during Mexico's health crisis is already proposed by Mora-Rivera et al. [P04]; origin shock insurance by Yang and Choi [P11]; pandemic macro explanations by Babii et al. [P03]; substitution by Ventura and García [P01]. The contribution would be the *conditional range of required need across empirically constrained mechanisms*, not a newly identified altruism effect.

**Targets and rejection:** fit selected 2020 aggregate growth observations, constrain count/amount components, and check unused 2021–2022 periods and compatible ENIGH moments. If no nonnegative shift fits both flow and component patterns within declared tolerances, report infeasibility. If many shifts/mechanisms fit, report an admissible range. Never interpret a residual need shift as a uniquely identified historical increase.

### Q2. Does activation add protection beyond larger incumbent contributions?

**Priority: central. Relevance: high. Potential novelty: moderate to high for a calibrated quantitative assessment; low for the qualitative mechanism.**

Define incumbents from each household's pre-pandemic simulated history, using a declared reference window. Freeze the eligible contributor set in the incumbent-only model while allowing incumbent amounts and, in a separate responsive variant, willingness to respond to need. The activation model additionally allows previously inactive linked migrants to respond to the residual gap after incumbent contributions. Use one parsimonious response parameter before considering relationship interactions.

Do not define incumbents anew after the shock or remove ordinary baseline occasional senders without quantifying the resulting baseline change. A sustained pre-pandemic remitter set and a last-month-sender flag are different objects. Record both, and examine sensitivity to the reference window.

Report paired changes in cumulative transfers, household receipt interruptions, coverage, replacement after a documented capacity loss, new contributors, and donor concentration. Decompose changes in sending into incumbent amounts, incumbent participation, and newly activated contributions, using a declared accounting convention and reporting interactions.

**Literature check:** [P01] directly establishes expanded remittance pools; [P30] analyses multiple-migrant contributions; [P22] shows participation persistence matters. The open quantitative question is how much explicit activation changes feasible aggregate growth and protection once incumbent responsiveness is allowed.

**Discriminating test:** separately fitted rival models receive comparable fit budgets; frozen-parameter mechanism switches measure conditional model effects. A mechanism is not necessary merely because switching it off reduces an outcome in a model built to use it. If incumbent-only and activation models fit equally well, retain both rather than claiming historical identification.

### Q3. How much protection comes from additional connections rather than additional resources?

**Priority: central. Relevance: high. Potential novelty: moderate for the controlled Mexico-COVID application; low for the general diversification principle.**

Compare network sizes 1, 2, 3–4, 5–9, and 10+ where sample support permits. Present the observed-composition comparison first, then a standardized experiment with comparable household characteristics, relationship mix where feasible, and fixed total baseline migrant earning capacity. Define an equal-total-earnings intervention separately from an equal-total-remittable-capacity intervention: splitting earnings across more people changes the aggregate living-cost allowance and can mechanically reduce capacity.

Measure the probability of any receipt, the probability of a large receipt drop, interruption spells, conditional amounts, and cumulative uncovered need. Distinguish potential migrants from employed, positive-capacity, willing, and actual contributing migrants. A singleton with high stable capacity need not be less protected on every outcome than several low-capacity kin.

**Literature check:** single versus multiple migrants is already explicit in [P30]; network insurance and asymmetric outcomes are established in [P14–P18]; sender diversity matters in [P19]. The useful extension is the resource-standardized protection gradient under simultaneous origin/destination shocks, with empirical relationship structure.

**Discriminating test:** if the size advantage disappears after capacity standardization, the result is primarily resource access. If it remains, quantify the role of independent opportunities to contribute, composition, and replacement. Do not label cross-sectional receipt differences causal effects of adding a migrant.

### Q4. When do common employment shocks eliminate network redundancy?

**Priority: central companion to Q3. Relevance: high. Potential novelty: moderate in application, low as general theory.**

Vary within-household correlation of migrant employment shocks while preserving each occupation-month's marginal unemployment rate. A latent-factor/copula construction with explicit marginal checks is preferable to arbitrarily raising all unemployment rates in large networks. Separate cross-migrant correlation from temporal persistence; the current independent monthly draws estimate neither employment spells nor a joint household risk process.

Hold network size and total baseline capacity fixed. Compare concentrated versus diversified occupation profiles. Report the achieved employment correlation, not only the latent correlation parameter; with unequal marginal rates they differ. Map protection across size and common-risk exposure, including cases where all contributors lose capacity together.

**Literature check:** risk-pooling limits and network structure are established in [P14–P18], and immigrant pandemic exposure is documented in [P06]. The contribution would be a quantified boundary for this remittance system: which network structures cease to provide useful redundancy under plausible common shocks. Kin job correlation is imputed, so this is a sensitivity experiment unless external joint data become available.

### Q5. Can the same aggregate remittance growth conceal widening protection gaps?

**Priority: central distributional result. Relevance: high. Potential novelty: moderate to high as an integrated mechanism result.**

Retain multiple mechanism/need combinations that satisfy the same macro constraints and compare their household distributions. Report unconditional receipts including zeros, receipt interruptions, coverage relative to declared perceived need, and lower-tail outcomes by network size and relationship composition. Compare remittance inequality among all connected households and among recipients separately.

Only call an outcome total-resource inequality or poverty after defining nonremittance household resources, a price/FX conversion, an adequacy threshold, consistent equivalence scales, and appropriate weights. Matching an ENIGH economic donor does not establish a real MMP household budget. Adding zero-connection households requires a justified network-prevalence scenario, not treating every nonrecipient as unconnected.

**Literature check:** [P01, P05] already warn that aggregate resilience masks strain; [P23–P25] analyse migration/remittance inequality and counterfactual income. The new result would be showing how macro-compatible substitution mechanisms imply different distributions of crisis protection. It would not identify the total causal effect of migration or reproduce a national Gini with the existing connected sample.

### Q6. Does relationship composition matter more than raw network size?

**Priority: secondary, with baseline fit diagnostic first. Relevance: high. Potential novelty: moderate.**

At fixed size and capacity, compare core-only, kin-only, and mixed networks. Distinguish an additional core tie from an additional extended-kin tie and report how often either becomes a replacement contributor. Keep source classifications explicit: current core includes head, spouse, and children (RELHEAD 1/2/3), unlike the older plan's proposed definition.

**Literature check:** relational obligation, social control, and transnational scripts are established in [P08, P20, P27]; co-national destination networks in [P21] are not equivalent to household-linked kin networks. A useful contribution is a controlled composition comparison grounded in MMP. The strong fitted core/kin intercept difference is not itself proof of relationship effects because profiles, network size, and measurement differ.

**Gate:** first diagnose the existing kin-only/older-household and singleton receipt errors. Evaluate a parsimonious alternative or carry a misspecification sensitivity. Do not add numerous interactions merely to match all source cells.

### Q7. How sensitive is the inferred need increase to formalization, benefits, and budget assumptions?

**Priority: required robustness for Q1, not a separate large model. Relevance: very high. Novelty: limited alone; essential for credibility.**

Separate latent household support from the observation process that produces recorded remittances. Evaluate a small, declared set of externally motivated changes in recording/formalization, capacity allowances/surplus shares, earnings dispersion, and plausible nonlabour support. Include zero changes. Avoid estimating all of them jointly from the same aggregate target.

**Literature check:** Dinarte-Diaz et al. [P02] provide a directly competing Mexico explanation based on formalization; their study tests and rejects particular CARES/altruism explanations in its design, which does not establish that such mechanisms never mattered. Babii et al. [P03] study fiscal, labour, and transmission channels. Yang [P12] establishes why FX matters for migrant-household outcomes. The need residual can absorb omitted resources, exchange-rate effects, population changes, and measurement changes.

**Discriminating test:** plot the inferred need range across these scenarios. A zero-need fit under plausible alternatives is a substantive result. The high recorded electronic-transfer share does not bound unrecorded informal flows, so it cannot by itself dismiss [P02]. Keep transaction frequency distinct from the number of unique remitters.

### Q8. What does maintaining remittances cost contributors?

**Priority: supplementary. Relevance: high. Potential novelty: moderate when linked to replacement roles.**

Report transfers/gross earnings, transfers/available surplus, the distribution of post-transfer resources, concentration of the household burden across donors, and the attributes of replacement contributors. More receipt protection can coincide with a narrower or more strained contributor pool.

**Literature check:** migrant-side strain is central in [P01, P05, P27]. The current cap mechanically protects the fixed allowance and retains 80% of positive surplus, so it cannot establish actual consumption sacrifice, savings depletion, or debt. Add benefits/buffers only if needed for a central inference; otherwise describe these outcomes as model budget burdens.

## 4. Minimum experiment and identification plan

### Stage A — establish a credible baseline

1. Treat saved outputs as a versioned reference. Record code, population, workbook and configuration hashes in the next production run. Preserve current outputs before recalibration.
2. Diagnose receipt errors by size, relationship, demographics, eligibility, capacity-zero status, and source sample counts. Check whether source-period/observation-window assumptions explain systematic discrepancies.
3. Compare a small set of justified baseline specifications and starting values. Retain admissible alternatives; report weak identification and failed cells. Revisit the fixed $20,000 allowance and 20% surplus assumption using documented sources or explicitly labelled ranges.
4. Audit the observation mapping: MMP receipt reference period, pooled 2010–2019 interview populations versus 2019 calibration, Banxico all-origin versus US-connected coverage, and one contribution per link-month versus potentially multiple actual transactions.

**Gate:** no distributional claim should rely silently on cells the model does not reproduce. An accepted approximation needs a documented scope and a sensitivity check; relaxing the tolerance is not a repair.

### Stage B — define the narrow historical inverse problem

Use one common pandemic need amplitude multiplying a fixed, justified time profile, or a few prespecified phases. A completely free monthly need residual would fit variation without explaining it. Freeze baseline parameters. If an external health-need profile is unavailable, describe the phases as inferred support-demand scenarios rather than measured Mexican need. Do not require a health series as the only possible way to answer the inverse question.

Use an observation equation such as `recorded_USD_t = represented_scale * recorded_fraction_t * latent_support_USD_t`, with the scale anchored once and the recording assumptions externally bounded. If only indexed series are defensible, use indices and explicitly give up a claim to national dollar levels. The current ABM has no national expansion weights. Do not interpret a fitted scale as a count of unique migrants.

Fit a subset of 2020 observations; evaluate later months/2022 with frozen behaviour and supplied shocks. Document that later-period evidence has already informed project development and is not blind validation. Baseline seasonality and pre-existing growth need an explicit counterfactual: report total growth since 2019 separately from excess growth above the pre-pandemic trajectory. Do not attribute all 2020–2022 growth to COVID.

Total remittances equal transaction count times mean transaction amount, so those three statistics are not three independent targets. Fit two independent components or use totals plus component ratios with transparent weighting; evaluate household receipt separately. Reconcile monthly and annual weighting rather than averaging monthly means to obtain annual mean transaction size.

### Stage C — test mechanisms and recoverability

| Comparison | Freeze | Vary | Primary interpretation |
|---|---|---|---|
| Capacity-only versus added need | Baseline behaviour, networks, capacity path, random draws | Need amplitude | Conditional required demand shift |
| Incumbents versus activation | Need path, labour resources, allocation, baseline history | Eligibility/shortfall activation | Conditional effect of additional contributors |
| Coordinated versus uncoordinated | Parameters and random draws | Allocation rule | Crowding out and coverage, not deliberate activation |
| Small versus large networks | Household attributes and declared aggregate capacity | Number/composition of links | Redundancy beyond resource differences |
| Independent versus common shocks | Marginals, network size, capacity | Cross-migrant dependence | Failure of diversification |
| Macro-compatible models | Fit criteria and observed targets | Admissible mechanisms/parameters | Distributional ambiguity despite aggregate fit |

Run one synthetic recovery exercise before full calibration: generate targets under known incumbent response/activation/need, fit rival models using the actual proposed moments, and inspect whether the mechanism and need are recovered. Distinct true mechanisms that fit equally well imply nonidentification, even if the optimizer reports success. Seed ranges describe numerical noise; baseline parameter, imputation, and network scenarios describe different uncertainties and should be reported separately.

For the network argument, include the analytical no-persistence benchmark `P(receipt)=1-product(1-p_j)` for positive-capacity contributors. An increasing receipt curve alone can arise from independent opportunities. The ABM earns its complexity through dynamic replacement, capacity/coverage interactions, or correlated shocks that change the quantitative conclusion beyond that benchmark.

### Stage D — four paper figures

1. **Empirical puzzle:** aggregate dollars/counts/amounts and ENIGH receipt/conditional amounts, with compatible units and observation windows.
2. **Need and mechanism feasibility:** required need ranges, fitted/unused moments, and infeasible or indistinguishable mechanisms.
3. **Network protection:** receipt interruption or cumulative uncovered need across size and common-shock exposure, with capacity-standardized comparisons.
4. **Unequal protection:** lower-tail outcomes and gaps across network groups among models compatible with the same macro growth; show donor burden in a supplementary panel.

Main outcomes: unconditional receipt, positive amounts, interruption duration, and cumulative uncovered perceived need. A validated resource gap can replace the last outcome once economic resources are defined. Gini coefficients are secondary. Freeze this scope before adding legal-status classes, shared donors across households, savings, or endogenous migration.

## 5. Search method and limits

Targeted web search conducted 16 October 2026 using combinations of: Mexico/US remittances, COVID/pandemic, expanded remittance pool, family/kin networks, multiple migrants, insurance, sender diversity, correlated shocks, inequality, formal/informal channels, and agent-based modelling. Searches included Spanish terms and a recent-paper check yielding the 2025-online/2026-volume study [P07]. Citation chaining used the reference list of [P01] and the publisher page for [P10], including the particularly relevant [P30].

The 30 distinct papers below are not duplicate working/published versions counted twice. P03, P06, P24 and P26 are working papers in the cited versions; P02 is also explicitly a working paper. P29 is methodological rather than remittance evidence. The remaining items are journal articles. Priorities reflect relevance to this project's questions, not journal ranking.

Evidence depth: bibliographic metadata and publisher/institution/author abstracts were screened; selected accessible text was examined for the closest studies, notably [P02, P04, P05, P07]. This is an annotated scoping assessment, not 30 full-paper critical appraisals or a systematic review with exhaustive database coverage. Before manuscript claims, read the full methods/results of the closest papers, check later published versions of working papers, and forward-search [P01, P02, P19, P30]. Conclusions below avoid numerical elasticities imported across settings.

Primary publisher, author, university, central-bank, World Bank, IMF and NBER sources underpin the references. Search aggregators were discovery aids, not the basis for substantive claims when a primary source was available. DOI links identify the work even where full text is restricted.

## 6. Annotated references: 30 papers

### Closest pandemic and Mexico studies

**[P01] Ventura, Ilana M., and Angela S. García (2024; online 2023).** “Resilient Remittances? Examining Immigrant Remittances from the United States to Latin America During Covid-19.” *International Migration Review*, 58(3). [Publisher/DOI](https://doi.org/10.1177/01979183231216048).

Survey/interview evidence documents expanded contributor pools after remitters' income instability and the strain underneath aggregate resilience. This is the direct empirical antecedent for activation/substitution, not an ABM calibration or an estimate of the necessary Mexican need increase. **Use:** Q1, Q2, Q5, Q8; indispensable framing.

**[P02] Dinarte-Diaz, Lelys, David Jaume, Eduardo Medina-Cortina, and Hernan Winkler (2022).** “Neither by Land nor by Sea: The Rise of Electronic Remittances during COVID-19.” *World Bank Policy Research Working Paper* 10057. [Institutional record](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099434205232231209); [paper](https://documents1.worldbank.org/curated/en/099434205232231209/pdf/IDU0b9463f130c07f040160bb26018c9daadd997.pdf).

Uses Mexican geographic variation to investigate substitution from informal to formal channels. Challenges the assumption that administrative growth measures an equal increase in real family support; tests alternative explanations in its setting. **Use:** Q1/Q7 and the observation equation. Treat its conclusions as design-specific, not a universal rejection of altruism.

**[P03] Babii, Aleksandra, Alina Carare, Dmitry Vasilyev, and Yorbol Yakhshilikov (2022).** “Evolution of Remittances to CAPDR Countries and Mexico During the COVID-19 Pandemic.” *IMF Working Paper* 2022/092. [DOI](https://doi.org/10.5089/9798400210150.001).

Examines pandemic remittance dynamics and the roles of US labour conditions, fiscal stimulus and transmission modes. Aggregate macro analysis is a competing explanatory layer, not evidence identifying within-family substitution. **Use:** Q1/Q7; establishes why a need-only residual requires qualification.

**[P04] Mora-Rivera, Jorge, Martha Cecilia García Amador, and Luis David Sosa Rodríguez (2021).** “Remesas internacionales y altruismo en el contexto de la pandemia de COVID-19.” *Economía: Teoría y Práctica*, 54. [DOI](https://doi.org/10.24275/etypuam/ne/542021/mora); [article](https://www.scielo.org.mx/scielo.php?pid=S0188-33802021000100161&script=sci_arttext).

Uses ENIGH 2016/2018 probit models linking health expenditures and receipt to motivate an altruistic interpretation of pandemic resilience. Its microdata precede COVID; it does not directly estimate pandemic behavioural change. **Use:** Q1; origin health needs are established motivation, while the proposed inverse estimate remains a different exercise.

**[P05] Lindley, Anna, Kavita Datta, Elaine Chase, Kaltun Fadal, Laura Hammond, Gabriela Loureiro, and Saliha Majeed-Hajaj (2024).** “Remitting through crisis: Looking beyond resilience in UK migrant and diaspora communities.” *Migration Studies*, 12(3), mnae026. [DOI](https://doi.org/10.1093/migration/mnae026).

Mixed-methods UK evidence describes rising origin needs, unequal capacity, and the burdens concealed by sustained flows. Different corridor from Mexico-US. **Use:** Q5/Q8 and the distinction between aggregate resilience and welfare; mechanisms motivate sensitivities rather than transplantable parameters.

**[P06] Borjas, George J., and Hugh Cassidy (2020).** “The Adverse Effect of the COVID-19 Labor Market Shock on Immigrant Employment.” *NBER Working Paper* 27243, revised June 2020. [Paper and abstract](https://www.nber.org/papers/w27243); [DOI](https://doi.org/10.3386/w27243).

CPS evidence documents immigrant employment exposure and differences related to remote-work capacity and immigration status. **Use:** Q4/Q7; supports heterogeneity in capacity risk. The present Mexican-origin aggregate inputs do not themselves identify legal status or recreate this study's population.

**[P07] Flores, Miguel, and Sebastián Gutiérrez Bernal (2026; online 2025).** “Estimation of state-to-state remittance flows from the US to Mexico.” *Quality & Quantity*, 60, 3301–3320. [DOI](https://doi.org/10.1007/s11135-025-02239-y).

Constructs geographic migration-based links to estimate US-state/Mexican-state remittance flows and examine regional shock transmission. **Use:** Q4; recent geographic-network precedent. This is an estimated regional flow network, distinct from household-linked substitute contributors; avoid claiming network-based Mexico-COVID analysis is new.

### Motivations, insurance, and recipient shocks

**[P08] Lucas, Robert E. B., and Oded Stark (1985).** “Motivations to Remit: Evidence from Botswana.” *Journal of Political Economy*, 93(5), 901–918. [DOI](https://doi.org/10.1086/261341).

Foundational model and evidence on remittance motivations and implicit family arrangements beyond pure altruism. **Use:** Q1/Q6; do not equate a positive need response with identification of a unique motive. Publisher metadata and primary abstract record screened; no numerical parameter imported.

**[P09] Stark, Oded, and David E. Bloom (1985).** “The New Economics of Labor Migration.” *American Economic Review*, 75(2), 173–178. [Author-affiliated institutional record](https://cris.tau.ac.il/en/publications/the-new-economics-of-labor-migration/); [article text](https://newdoc.nccu.edu.tw/teasyllabus/1102265813001/5_NewEconomicsOfLaborMigratin.pdf).

Frames migration within household strategies and interactions. **Use:** theoretical background for family risk diversification. This project holds migration and networks fixed; it does not test an endogenous migration decision model.

**[P10] Amuedo-Dorantes, Catalina, and Susan Pozo (2006).** “Remittances as insurance: evidence from Mexican immigrants.” *Journal of Population Economics*, 19(2), 227–254. [Publisher/DOI](https://doi.org/10.1007/s00148-006-0079-6).

Links host-economy income risk to remittances as family-provided insurance and self-insurance. The direction of insurance is conceptually different from replacing origin income losses. **Use:** Q1/Q7; destination risk need not operate solely through a mechanical earnings ceiling.

**[P11] Yang, Dean, and HwaJung Choi (2007).** “Are Remittances Insurance? Evidence from Rainfall Shocks in the Philippines.” *The World Bank Economic Review*, 21(2), 219–248. [DOI](https://doi.org/10.1093/wber/lhm003); [author-hosted paper](https://sites.lsa.umich.edu/deanyang/wp-content/uploads/sites/205/2014/12/yangchoi_insurance.pdf).

Uses origin rainfall/income shocks to assess compensating overseas remittances. **Use:** Q1 and shortfall buffering. Provides a precedent for remittance insurance, not evidence about simultaneous US-Mexico shocks or extended-kin activation.

**[P12] Yang, Dean (2008).** “International Migration, Remittances and Household Investment: Evidence from Philippine Migrants’ Exchange Rate Shocks.” *The Economic Journal*, 118(528), 591–630. [DOI](https://doi.org/10.1111/j.1468-0297.2008.02134.x); [NBER working version](https://www.nber.org/papers/w12325).

Uses migrant destination exchange-rate shocks to study remittances and household investment. **Use:** Q7 and currency/purchasing-power alignment. Exchange-rate-induced gains are not equivalent to a rise in Mexican household need; model USD and real MXN outcomes separately.

**[P13] Ambrosius, Christian, and Alfredo Cuecuecha (2013).** “Are Remittances a Substitute for Credit? Carrying the Financial Burden of Health Shocks in National and Transnational Households.” *World Development*, 46, 143–152. [DOI](https://doi.org/10.1016/j.worlddev.2013.01.032); [institutional abstract](https://www.sfb-governance.de/en/publikationen/articles/Ambrosius_Cuecuech_Are-Remittances-a-Substitute-for-Credit_-Carrying-the-Financial-Burden-of-Health-Shocks-in-National-and-Transnational-Households/index.html).

Mexican panel evidence compares health-shock financing in households with and without nuclear family in the US. **Use:** Q1/Q5; household insurance can appear as less debt, not just greater remittance receipt. Debt outcomes are outside the current ABM.

### Networks, diversification, and participation

**[P14] Townsend, Robert M. (1994).** “Risk and Insurance in Village India.” *Econometrica*, 62(3), 539–591. [Author publication record](https://www.robertmtownsend.net/content/risk-and-insurance-village-india).

Foundational full-insurance benchmark for consumption under household and common risk. **Use:** Q3/Q4; protection against individual shocks must be distinguished from aggregate shocks. Current perceived-need coverage is not a consumption-insurance test.

**[P15] Fafchamps, Marcel, and Susan Lund (2003).** “Risk-sharing networks in rural Philippines.” *Journal of Development Economics*, 71(2), 261–287. [DOI](https://doi.org/10.1016/S0304-3878(03)00029-4); [author-hosted working text](https://web.stanford.edu/~fafchamp/risk.pdf).

Studies gifts and loans within friend/relative networks rather than assuming village-wide insurance. **Use:** Q3/Q6; network quality and the form of support matter. Its loan/enforcement mechanisms are not implemented in this remittance ABM.

**[P16] Fafchamps, Marcel, and Flore Gubert (2007).** “Risk Sharing and Network Formation.” *American Economic Review*, 97(2), 75–79. [DOI](https://doi.org/10.1257/aer.97.2.75); [author-hosted paper](https://web.stanford.edu/~fafchamp/aerc.pdf).

Examines trade-offs between diversification benefits and the costs/proximity of support ties in the Philippines. **Use:** Q4; empirical kin networks need not maximize occupational diversification. Do not assume independent imputation is an observed joint occupational distribution.

**[P17] Bramoullé, Yann, and Rachel Kranton (2007).** “Risk-sharing networks.” *Journal of Economic Behavior & Organization*, 64(3–4), 275–294. [DOI](https://doi.org/10.1016/j.jebo.2006.10.004); [author-hosted text](https://sites.duke.edu/rachelkranton/files/2016/12/risksharingnetworks-jebo-bramoulle-and-kranton.pdf).

Theory connects bilateral links, network formation, insurance and unequal outcomes. **Use:** Q3–Q5; broad theoretical network-insurance novelty is unavailable. The current fixed household stars do not implement endogenous formation or general paths through a community network.

**[P18] Ambrus, Attila, Markus Mobius, and Adam Szeidl (2014).** “Consumption Risk-Sharing in Social Networks.” *American Economic Review*, 104(1), 149–182. [DOI](https://doi.org/10.1257/aer.104.1.149); [author-hosted paper](https://adamszeidl.com/papers/risksharing_published.pdf).

Models social collateral and the relationship between network structure and insurance. **Use:** Q3–Q5; topology and enforcement have established theoretical roles. This project can investigate local support redundancy without claiming to estimate social collateral or global network expansiveness.

**[P19] Jack, William, and Tavneet Suri (2014).** “Risk Sharing and Transactions Costs: Evidence from Kenya’s Mobile Money Revolution.” *American Economic Review*, 104(1), 183–223. [DOI](https://doi.org/10.1257/aer.104.1.183); [MIT record](https://dspace.mit.edu/entities/publication/5768dd69-f029-48f9-8284-9b4cded2ae3f).

Studies mobile-money access, consumption smoothing, remittance receipts and sender diversity. **Use:** Q2–Q4/Q7; a particularly close precedent for broader sender access improving insurance. Domestic mobile-money mechanisms differ from international kin networks under a simultaneous pandemic shock.

**[P20] Chort, Isabelle, Flore Gubert, and Jean-Noël Senne (2012).** “Migrant networks as a basis for social control: Remittance incentives among Senegalese in France and Italy.” *Regional Science and Urban Economics*, 42(5), 858–874. [DOI](https://doi.org/10.1016/j.regsciurbeco.2012.02.001).

Connects network services and social control to remittance obligations. **Use:** Q6; networks can shape willingness and information, beyond providing spare earnings. The current coordination rule does not model monitoring, sanctions, or reciprocity.

**[P21] Aparicio Fenoll, Ainhoa, and Zoë Kuehn (2018).** “Immigrant networks and remittances: Cheaper together?” *World Development*, 111, 225–245. [DOI](https://doi.org/10.1016/j.worlddev.2018.07.008); [publisher page](https://www.sciencedirect.com/science/article/abs/pii/S0305750X18302377).

Studies migrant networks in Spain and remitting behaviour/costs. **Use:** Q6/Q7; specify the network unit carefully. Co-national local density, household migrant connections, and actual contributors are different measures; none can substitute automatically for another.

**[P22] Bettin, Giulia, and Riccardo Lucchetti (2016).** “Steady streams and sudden bursts: persistence patterns in remittance decisions.” *Journal of Population Economics*, 29(1), 263–292. [DOI](https://doi.org/10.1007/s00148-015-0565-9); [author-affiliated abstract](https://iris.univpm.it/handle/11566/227665).

Longitudinal German data distinguish state dependence and individual heterogeneity in remitting. **Use:** Q2 and baseline identification; cross-sectional MMP receipt does not pin down monthly sending persistence. Do not directly import an annual German coefficient into a monthly Mexico-US model.

### Inequality, counterfactuals, and interpretation

**[P23] McKenzie, David, and Hillel Rapoport (2007).** “Network effects and the dynamics of migration and inequality: Theory and evidence from Mexico.” *Journal of Development Economics*, 84(1), 1–24. [DOI](https://doi.org/10.1016/j.jdeveco.2006.11.003); [published paper](https://s3.amazonaws.com/ssrc-cdn1/crmuploads/new_publication_3/network-effects-and-the-dynamics-of-migration-and-inequality-theory-and-evidence-from-mexico.pdf).

Studies migration networks, migration costs, selection and inequality dynamics. **Use:** Q5; network-inequality links in Mexico are established. This ABM's short-run contributor redundancy differs from long-run migration diffusion and endogenous access to migration.

**[P24] Koczan, Zsoka, and Franz Loyola (2018).** “How Do Migration and Remittances Affect Inequality? A Case Study of Mexico.” *IMF Working Paper* 2018/136. [DOI](https://doi.org/10.5089/9781484361634.001); [institutional record](https://www.imf.org/en/publications/wp/issues/2018/06/14/how-do-migration-and-remittances-affect-inequality-a-case-study-of-mexico-45926).

Examines household remittances and inequality with attention to counterfactual resources and crisis periods. **Use:** Q5; subtracting remittances from observed income is not a no-migration counterfactual. Here, fixed-migration mechanism comparisons should remain explicitly conditional.

**[P25] Acosta, Pablo, Cesar Calderón, Pablo Fajnzylber, and Humberto Lopez (2008).** “What is the Impact of International Remittances on Poverty and Inequality in Latin America?” *World Development*, 36(1), 89–114. [DOI](https://doi.org/10.1016/j.worlddev.2007.02.016); [publisher](https://www.sciencedirect.com/science/article/pii/S0305750X07001945).

Combines cross-country and household approaches, including imputation of potential home earnings. **Use:** Q5; broad remittance/inequality questions are established and sensitive to counterfactual definitions. The proposed contribution is crisis protection by support-network access rather than another unconditional inequality association.

**[P26] McKenzie, David, and Marcin J. Sasin (2007).** “Migration, Remittances, Poverty, and Human Capital: Conceptual and Empirical Challenges.” *World Bank Policy Research Working Paper* 4272. [Paper](https://documents.worldbank.org/curated/en/864521468313743648/pdf/wps4272.pdf).

Reviews selection, endogeneity and interpretation challenges in migration/remittance welfare analysis. **Use:** Q5 and all causal language. Empirically constrained simulation is not a substitute for an identified migration treatment effect.

**[P27] Carling, Jørgen (2014).** “Scripting Remittances: Making Sense of Money Transfers in Transnational Relationships.” *International Migration Review*, 48(S1), S218–S262. [DOI](https://doi.org/10.1111/imre.12143); [PRIO record](https://www.prio.org/publications/7498).

Synthesizes ethnographic work on social meanings and obligations in transfers. **Use:** Q6/Q8; relationship intercepts are reduced descriptions of richer practices. Need responsiveness should not be labelled a directly observed psychological altruism parameter.

### Modelling precedents and reporting

**[P28] Walsh, Stephen J., George P. Malanson, Barbara Entwisle, Ronald R. Rindfuss, Peter J. Mucha, Benjamin W. Heumann, Philip M. McDaniel, Brian G. Frizzelle, Ashton M. Verdery, Nathalie E. Williams, Xiaozheng Yao, and Deng Ding (2013).** “Design of an agent-based model to examine population–environment interactions in Nang Rong District, Thailand.” *Applied Geography*, 39, 183–198. [DOI](https://doi.org/10.1016/j.apgeog.2012.12.010); [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3838637/).

An empirically grounded ABM includes household migration, remittances, assets and social-network dynamics. **Use:** methodological precedent; ABM plus remittances is not by itself novel. This work addresses population/environment interactions rather than pandemic contributor substitution.

**[P29] Grimm, Volker, et al. (2006).** “A standard protocol for describing individual-based and agent-based models.” *Ecological Modelling*, 198(1–2), 115–126. [DOI](https://doi.org/10.1016/j.ecolmodel.2006.04.023); [USGS publication record and complete author list](https://pubs.usgs.gov/publication/70161470).

Introduces the ODD reporting protocol. **Use:** reproducible model description: entities, state variables, process order, initialization, assumptions and submodels. This is reporting guidance, not substantive evidence for a remittance mechanism.

**[P30] Agarwal, Reena, and Andrew W. Horowitz (2002).** “Are International Remittances Altruism or Insurance? Evidence from Guyana Using Multiple-Migrant Households.” *World Development*, 30(11), 2033–2044. [DOI](https://doi.org/10.1016/S0305-750X(02)00118-3); [publisher](https://www.sciencedirect.com/science/article/pii/S0305750X02001183).

Develops different predictions for single- and multiple-migrant households under altruism and risk-sharing motivations. **Use:** Q2/Q3/Q6; essential prior art against claiming that contributor multiplicity is a new idea. The proposed extension is dynamic capacity-loss substitution under common shocks, not a generic comparison of one versus several migrants.

## 7. Decision for the paper

Lead with Q1, Q2, and Q5; use Q3–Q4 as the experiment that explains when the network mechanism changes protection. Q6 helps interpret heterogeneity; Q7 is necessary to keep the need estimate defensible; Q8 exposes migrant-side burdens without requiring a full savings model.

A suitable working claim is: **“We quantify the range of increases in support demand compatible with pandemic remittance growth under alternative family contribution mechanisms, and show how network redundancy and shared labour risk distribute protection across households.”** Use “show” only after the experiments produce that result. If mechanisms remain observationally equivalent, that ambiguity and its distributional consequences can itself be the central finding.
