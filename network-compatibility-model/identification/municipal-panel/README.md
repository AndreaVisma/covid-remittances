# Municipal test of the need hypothesis (within states)

**Question.** Within a state, did municipalities hit harder by COVID mortality receive more remittances in 2020–2023 than their own 2017–19 trend predicts? This is the state-panel test (`../state-panel/`) moved to about 1,200 municipalities. State fixed effects absorb anything common to a state, such as US-side capacity of the state's migrant network, state lockdown policy and the southern boom.

**Short answer.** No. Within states, excess mortality does not predict the 2021 rise:
- The 2021 coefficient is −0.014 per 1 SD (state-cluster SE 0.016). The 95% CI [−0.045, 0.017] rules out a positive effect above about 0.02 log points per SD, against a sample mean 2021 excess of 0.14.
- Every specification is ≤ 0.02 and none is positive and significant. This covers weights, winsorizing, three excess-mortality definitions, COVID-certified deaths, excluding the southern states, and size cuts.
- The quarterly coefficients are slightly negative from 2020Q2 on, with no 2021 spike.
- In all years, high-mortality municipalities track slightly *below* low-mortality ones in the same state.
- The 2016 pre-period placebo is negative by about the same amount (−0.013), so most of this negative tilt predates COVID. The best reading is a zero need effect plus a pre-existing negative drift.

## Files

| Script | What it does |
|---|---|
| `build_municipal.py` | Matches Banxico CE166 municipal names to INEGI codes, aggregates death certificates by municipality of residence, joins census 2020 and CONAPO IIM 2020, computes trend deviations |
| `analysis.py` | Within-state regressions (HC3 + state-clustered SE, within-state permutation), robustness, placebo, quarterly event study, within-state terciles, 3 figures |

Run (from the repo root): `python -I -X utf8 network-compatibility-model/identification/municipal-panel/build_municipal.py` (about 3 min, mostly reading the death certificates; `--skip-deaths` reuses the cached table), then `python -I -X utf8 .../analysis.py`. Input locations are set with `--state-raw` and `--mun-raw`; the defaults are `data/external/state-panel` and `data/external/municipal-panel`, both gitignored. Large intermediates go to `data/external/municipal-panel/derived/`:
- `deaths_mun_year.csv`
- `municipal_panel.csv` (one row per unit)
- `municipal_quarterly_dev.csv`

Outputs (`outputs/`, aggregates only, about 110 KB):

| File | Content |
|---|---|
| `match_summary.csv`, `match_report.csv` | Match rates; the Banxico names not matched exactly, with their INEGI target |
| `sample_description.csv` | Sample size, coverage, distributions |
| `regressions_need.csv` | year (2020–24) x outcome (raw / winsorized) x sample x weight x need (excess mortality / COVID deaths) x spec (no FE / state FE / state FE + controls); HC3 and state-clustered SE, within-state permutation p (headline cells), MDE |
| `robustness_2021.csv` | 2021 variants and the 2016 placebo |
| `event_study_quarterly.csv` | Quarter-by-quarter coefficients 2019Q1–2024Q4 |
| `terciles_within_state_quarterly.csv`, `terciles_within_state_annual.csv` | Paths by within-state tercile of excess mortality; high-minus-low with state FE |

## Data and sources (accessed 29 Oct 2026)

| Variable | Source | Notes |
|---|---|---|
| Remittances by municipality | Banxico SIE cuadro **CE166** "Ingresos por remesas, distribución por municipio" ([page](https://www.banxico.org.mx/SieInternet/consultarDirectorioInternetAction.do?sector=1&accion=consultarCuadro&idCuadro=CE166&locale=es)), parsed by `../state-panel/build_panel.py` | Quarterly, USD millions, 2013Q1+. **USD only; no transaction counts by municipality.** Used through 2024Q4. |
| Municipal catalogue, population, urban share, share 65+ | INEGI Censo 2020, ITER national table ([iter_00_cpv2020_csv.zip](https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip)) → `data/external/municipal-panel/iter/` | 2,469 municipalities. Municipal totals are the `LOC=0000` rows. Urban share = population in localities of 2,500 or more. Land area is not in ITER, so density is not used; urban share is the control. |
| Migration intensity | CONAPO, Índice de Intensidad Migratoria México–Estados Unidos 2020, municipal base `iim_base2020m.csv` (IIM-DP2 value, % of households receiving remittances, with emigrants, etc.), linked from [gob.mx/conapo](https://www.gob.mx/conapo/documentos/indices-de-intensidad-migratoria-mexico-estados-unidos-2020); file at `https://conapo.segob.gob.mx/work/models/CONAPO/IIM/iim_base2020m.csv` → `data/external/municipal-panel/conapo/` | Reachable this time, unlike for the state panel. Missing for 2 sample municipalities, which are dropped. |
| Deaths | DGIS death certificates `DEFUN_2015–2023.zip` (same files as the state panel) | Counted by `ent_resid` x `mun_resid` x year of occurrence. 99.5–99.97% of deaths per year have a known municipality of residence. COVID = underlying cause U07.1/U07.2. |

**Labour-income-loss proxies at the municipal level are not available here.** ENOE/ITLP are representative only by state and city, and nothing like CONEVAL ITLP exists by municipality. IMSS insured-employment by municipality (datos abiertos IMSS) would be a formal-sector proxy, but it was not downloaded or built. The census is from March 2020, before the shock. No income proxy is used.

## Matching Banxico names to INEGI codes

- Banxico lists 2,488 municipal rows. 32 of them are per-state **"No identificado"** rows: USD 20.7 m in 2019, 0.06% of the national total. They are excluded.
- The remaining **2,456 names all match (100%)**:
  - 2,437 after normalizing accents, case and punctuation, and expanding "Dr."/"Gral.";
  - 19 by manual alias (`match_report.csv`). Examples: "Silao" → Silao de la Victoria, "Tlaquepaque" → San Pedro Tlaquepaque, "Zapotitlán del Río" → San Mateo Yucutindoo (renamed), and the two duplicated Oaxacan names "San Juan/San Pedro Mixtepec – Distr. NN", assigned by district.
- **Matched USD = 99.94% of the 2019 total** (100% of identified USD).
- 13 INEGI municipalities created after 2013 are not reported separately by Banxico: San Quintín, Seybaplaya, 6 new Chiapas municipalities, 3 Morelos indigenous municipalities, Bacalar and Puerto Morelos. Each is merged back into its parent, for population, deaths and IIM as well, because pre-split death certificates also code them to the parent. This gives **2,456 analysis units**.

## Sample and variables

- **Sample:** positive remittances in every quarter 2017Q1–2019Q4 (1,540 units) and **2019 USD ≥ 1 million** (threshold; `--min-usd`), with IIM non-missing:
  - **N = 1,227 municipalities in all 32 states**;
  - 99.5% of 2019 USD and 91% of population;
  - median 2019 receipts USD 12 m, median population 32 k;
  - excluding the southern transit core (CHS, TAB, OAX, VER): N = 966; the extended set (+ GRO, QROO, CAM): N = 886.
- **Outcome `exc1719_y`:** log of annual USD in year y minus the log of the value predicted by the unit's own log-linear quarterly trend with quarter dummies, fitted on 2017Q1–2019Q4 (as in the state panel).
  - Mean (median) 2020 0.026 (0.021), 2021 0.139 (0.164), 2022 0.115 (0.152), 2023 −0.07 (0.06).
  - The tails are heavy: the 1st percentile in 2021 is −1.3 and in 2023 it is −3.8. From 2022Q3, 25–38 sample-eligible municipalities report zero in some quarter, which suggests a change in Banxico's municipal assignment.
  - Every regression is therefore also run with the outcome **winsorized at 1/99%**.
  - Two units with zero 2023 receipts are dropped from the 2023 regressions.
- **Need:**
  - `excess_pc_2020_21` = deaths 2020+2021 minus 2 x the 2015–19 average, per 1,000 population (2020). Mean 6.0, SD 2.2, **within-state SD 1.94**.
  - `covid_pc_2020_21` = certified COVID deaths per 1,000 (mean 2.8). Its correlation with excess mortality is 0.58.
  - Both are z-scored over the sample, so coefficients are per 1 SD = 2.2 excess deaths per 1,000.
  - Variants: excess vs the 2017–19 mean, vs the 2015–19 linear trend, in % of baseline deaths, 2020 only, 2021 only.
- **Controls:** pre-trend slope (2017–19), log 2019 remittances per capita, log population, urban share, CONAPO IIM-DP2 2020, plus **state fixed effects**.
- **Inference:**
  - HC3 and **SEs clustered by state** (32 clusters).
  - Randomization inference that **permutes the need variable within state** (2,000 draws; statistic |b| from the FWL regression).
  - MDE = 2.8 x max(HC3, cluster SE), for 80% power at 5% two-sided.

## Results

### 1. Main table: need coefficient by year (state FE + controls, unweighted, all states)

Coefficient per 1 SD of excess mortality, with state-clustered SE in parentheses; [HC3 SE]; p (cluster) / p (within-state permutation); MDE.

| outcome | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| raw | −0.023 (0.012) [0.014]; p .05 / .01; MDE .040 | **−0.014 (0.016)** [0.013]; p .37 / .19; MDE .045 | −0.045 (0.025) [0.024]; p .07 / .001; MDE .071 | −0.040 (0.029) [0.027]; p .16 / .09 |
| winsorized 1/99 | −0.009 (0.008); p .29 / .24 | **−0.006 (0.015)**; p .69 / .52; MDE .043 | −0.026 (0.018); p .14 / .02 | −0.036 (0.025); p .14 / .05 |
| weighted by 2019 USD (raw) | 0.003 (0.006) | **0.002 (0.014)** | −0.032 (0.020) | −0.057 (0.028) |
| COVID deaths, raw | −0.013 (0.010) | −0.021 (0.019) | −0.022 (0.024) | −0.010 (0.033) |
| excess, excl. southern ext. (raw) | −0.034 (0.017) | −0.017 (0.026) | −0.028 (0.032) | −0.026 (0.044) |
| excess, excl. southern ext., weighted | 0.006 (0.008) | 0.013 (0.018) | −0.013 (0.024) | −0.025 (0.031) |

Without controls the estimates look the same:
- **No FE:** 2021 −0.010 (0.013).
- **State FE only:** 2021 −0.019 (0.014).

The positive bivariate cross-state slope seen in the state panel does not appear within states.

### 2. Robustness, 2021 (`robustness_2021.csv`)

All variants lie between −0.017 and +0.024, and none is positive and significant (cluster p ≥ 0.06):
- excess mortality vs the 2017–19 mean: −0.015 (0.014);
- vs the 2015–19 trend: −0.010 (0.011);
- in % of baseline: −0.003 (0.012);
- 2020 deaths only: −0.004; 2021 deaths only: −0.017;
- adding the share aged 65+: −0.003 (0.017);
- joint with COVID deaths: −0.005 (0.011); weighted 0.024 (0.013), p = 0.06;
- 2015–19 trend window: −0.003 (0.017);
- raw Δlog 2019–21 with 2016–19 growth: −0.002 (0.015);
- population ≥ 10k: −0.016 (0.019);
- 2019 USD ≥ 5m: −0.010 (0.022);
- excluding the southern core states: −0.016 (0.022); the extended set: −0.017 (0.026).

### 3. Placebo (pre-period)

The 2016 deviation from each unit's own 2013–15 trend (N = 1,203), with the same controls and 2013–15 slope:
- −0.013 (0.008), permutation p = 0.16;
- winsorized −0.014 (0.007), p = 0.05;
- weighted −0.021 (0.007).

High-2020–21-mortality municipalities were already running below their own trends in 2016, by about as much as in 2020–22. The event study shows the same in the 2019 in-window residuals: 2019Q3–Q4 coefficients are −0.011 and −0.015. The small negative post-2020 coefficients are therefore not a need effect of the wrong sign. They are a pre-existing drift.

### 4. Event study (Fig. F1)

Quarterly coefficient on excess mortality (1 SD), state FE + controls, clustered CI:
- 2019 quarters: about 0 (in the trend window);
- 2020Q1–2022Q4: between −0.04 and 0.00 raw, and −0.03 to 0.00 winsorized;
- 2021Q1 0.00 (0.02);
- 2023–24: drifting to about −0.07 (SE 0.03–0.04).

There is no positive divergence when deaths peaked (2020Q2–Q3, 2021Q1, 2021Q3). COVID certified deaths show the same pattern: |coef| ≤ 0.02 in 2020–22.

### 5. Within-state terciles (Fig. F2)

Terciles are defined within each state, so every state contributes to every tercile. Mean excess deaths per 1,000 by tercile: low 4.0, mid 5.9, high 8.0.

| tercile | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| low (mean / median) | 0.044 / 0.037 | 0.166 / 0.178 | 0.156 / 0.186 | −0.033 / 0.057 |
| mid | 0.039 / 0.019 | 0.154 / 0.164 | 0.137 / 0.151 | −0.046 / 0.066 |
| high | −0.004 / 0.008 | 0.098 / 0.153 | 0.056 / 0.140 | −0.127 / 0.055 |
| high − low, state FE (cluster SE) | −0.049 (0.019) | −0.069 (0.030) | −0.103 (0.045) | −0.101 (0.072) |
| same, winsorized | −0.033 (0.017) | −0.042 (0.027) | −0.064 (0.036) | −0.072 (0.062) |

- The 2021 rise is present in every tercile, with medians of +0.15 to +0.18.
- The high tercile is slightly *lower*, and the gap in means is larger than the gap in medians, so the left tail drives it.

### 6. Binned scatter (Fig. F3)

The 2021 deviation (winsorized) and excess mortality, both residualized on state FE (left) or state FE + controls (right), are flat. Slopes are −0.005 and −0.003 per excess death per 1,000.

### Power

- 2021 (headline, cluster SE 0.016): **MDE ≈ 0.045 log points per SD** (0.043 winsorized), i.e. about 0.02 per excess death per 1,000. The MDE is similar to the state panel's (0.06), but the variation is now within states, the point estimate is negative, and the upper end of the CI is +0.017 (+0.024 winsorized).
- What this excludes: a need response in which a municipality with one SD more mortality (2.2 deaths per 1,000) receives 2+ log points more than its same-state neighbours.
- As an illustration: going from the low to the high within-state tercile (+4 deaths per 1,000) would have to raise 2021 remittances by more than about 0.03 to be detectable. The estimate is −0.04 to −0.07.
- As in the state panel, a common need effect shared by all municipalities is not identified, because it is absorbed by the FE and the intercept.

## Interpretation

- Within the same state, municipalities with more COVID deaths did not get a larger 2021 remittance rise relative to their own trend. If anything they got slightly less, but that tilt was already there before COVID (2016 placebo, late-2019 residuals).
- The 2021 surge is broad-based: medians of +15 to +18 log points in every mortality tercile.
- This is the pattern expected from a driver common to origin areas (US-side capacity, a channel shift to recorded transfers, the exchange rate) rather than from local need.
- It agrees with the state panel (about 0 per SD), the ENIGH result (amounts grew less in high-need cells), and the household model (need adds little).
- **For the model:** a β > 0 need channel that predicts larger receipts where mortality was higher is bounded at about ≤ 0.02 log points per SD *differentially* within states. This is a sharper falsification moment than the 32-state version.

## Caveats

- **Banxico municipal assignment.** CE166 assigns transfers to the municipality reported by the paying institution: the paying point, branch or the recipient's registered location. It is not necessarily where the recipient household lives. Recipients who collect in the municipal seat or a nearby city shift receipts toward urban centres. That adds noise and could attenuate within-state need effects, especially between neighbouring municipalities. Weighting by USD, dropping small units and winsorizing do not change the result.
- **Heavy tails and reassignments.** Some municipalities show multi-log-point jumps or drop to zero, especially from 2022Q3. These look like reporting or assignment changes, not behaviour. Hence the winsorized and median versions; the 2023–24 results are the least reliable.
- **USD only.** There are no counts or recipient numbers by municipality, so frequency and amount cannot be separated.
- **Excess mortality is crude.** It is not age-standardized and the baseline is a 2015–19 average. Adding the 65+ share or using a trend baseline changes nothing. Small municipalities have noisy death counts, which attenuates estimates. The ≥ USD 1 m threshold already removes most of the smallest; the pop ≥ 10k result is similar.
- **Need ≠ mortality only.** No municipal labour-income-loss measure is available (see above), so the income channel is not tested within states.
- New municipalities (13) are merged into their 2013 parents. Deaths occurring in 2022–23 slightly undercount late registrations.

## Figures

- `figures/F1_event_study.png`: quarterly coefficient on excess mortality (1 SD), state FE + controls, 95% CI clustered by state; raw and winsorized, all states and excluding the southern transit states.
- `figures/F2_terciles_within_state.png`: mean (winsorized) and median deviation by within-state tercile of excess mortality.
- `figures/F3_binscatter_2021_within_state.png`: 2021 deviation vs excess mortality, residualized on state FE (± controls), 20 bins.
