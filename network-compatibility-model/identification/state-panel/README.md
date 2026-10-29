# State-level test of the need hypothesis

**Question.** Did remittances grow more in 2020–2023 in the Mexican states that were hit harder by need shocks (COVID deaths, excess mortality, labour-income losses), once we control for each state's own pre-trend and migration intensity? The test uses the 32 states and is independent of the ABM.

**Short answer.** No, not in a way we can use. The 2021 excess over each state's own trend is unrelated to excess mortality or labour-income losses once the pre-trend is controlled for. Point estimates are near zero, and the confidence intervals rule out large effects but not small ones. The one significant result, registered COVID deaths outside the southern states, already appears in 2020Q1, before there were any deaths. It weakens with a CDMX/Estado de México dummy, so it looks like a centre/urban effect rather than need. Labour-income losses have the wrong sign. The cross-state variation is mostly a common level shift plus idiosyncratic state paths, which includes a pre-2020 boom in Chiapas and Tabasco.

## Files

| Script | What it does |
|---|---|
| `build_deaths.py` | Aggregates the DGIS/INEGI death-certificate microdata (registration years 2015–2023) to state of residence x year of occurrence, all causes and COVID (U07.1/U07.2) |
| `build_panel.py` | Builds the state panel: Banxico remittances, trend deviations, need shocks, controls, transit variables |
| `analysis.py` | Regressions, randomization inference, leave-one-out, quarterly timing, transit checks, ENIGH comparison, figures |

Run order: `python -I build_deaths.py`, then `python -I build_panel.py`, then `python -I analysis.py`. All three read from `data/external/state-panel/`, which is gitignored. Two environment notes:
- `statsmodels.api` is broken in this environment (numpy ABI), so `analysis.py` imports `OLS`/`WLS` directly.
- Use `-X utf8` on a Windows console.

Outputs (`outputs/`, public aggregates only):

| File | Content |
|---|---|
| `remittances_state_quarterly.csv` | Banxico, state x quarter, USD m, 2013Q1–2026Q2 |
| `data/external/state-panel/derived/remittances_municipal_quarterly.csv` | Banxico, municipality x quarter (4.5 MB, not committed; rebuilt by `build_panel.py`) |
| `remittances_state_quarterly_dev_trend1719.csv` | Quarterly log deviation from each state's own 2017–19 trend |
| `deaths_state_year.csv` (+ `_by_regyear`) | All-cause and COVID deaths, state x year of occurrence |
| `state_panel.csv` | One row per state: all outcomes, shocks and controls |
| `regressions_need.csv` | Every year x sample x weight x need x spec cell; HC3 SE, permutation p (headline cells), MDE |
| `robustness_2021.csv` | Alternative outcomes and controls for 2021 |
| `transit_checks.csv` | Transit-intensity regressions and group dummies |
| `leave_one_out_2021.csv`, `leave_one_out_2021_covid_excl_south.csv` | Leave-one-out estimates |
| `timing_quarterly_coef.csv`, `timing_need_terciles.csv` | Quarter-by-quarter estimates and tercile paths |
| `chiapas_transit_hubs.csv`, `chiapas_top_municipalities.csv` | Within-Chiapas breakdown |
| `enigh_comparison.csv`, `enigh_vs_banxico_states.csv` | Comparison with the ENIGH state table |

## Data and sources (accessed 29 Oct 2026)

| Variable | Source | Notes |
|---|---|---|
| Remittances by state and municipality | Banxico SIE, cuadro **CE166** "Ingresos por remesas, distribución por municipio". [Page](https://www.banxico.org.mx/SieInternet/consultarDirectorioInternetAction.do?sector=1&accion=consultarCuadro&idCuadro=CE166&locale=es); exported with `&formatoXLS.x=1&fechaInicio=<ms>&fechaFin=<ms>` (no token needed) | State series **SE29670–SE29701** (quarterly, USD millions, flows, 2003Q1–2026Q2); municipal series SE41187–SE43674 (2013Q1+). The export starts at 2013Q1. **USD only:** Banxico publishes the number of operations nationally (CE81), not by state, so the outcome here is USD, not counts. |
| Deaths (all-cause, COVID) | Secretaría de Salud DGIS open data, death certificates (same certificates as INEGI EDR): `http://www.dgis.salud.gob.mx/descargas/datosabiertos/defunciones/registro/DEFUN_<year>.zip`, 2015–2023 | Aggregated by `ent_resid` x `anio_ocur` across registration years 2015–2023. Occurrences in 2022 miss registrations made in 2024+ (small). The INEGI EDR datos-abiertos zip URLs returned 404 on the access date. |
| COVID deaths (alt.) | SSA datos abiertos via the [mexicovid19/Mexico-datos](https://github.com/mexicovid19/Mexico-datos) mirror (`covid19_mex_muertes.csv`, to Feb 2022) | Downloaded but not used; the certificate counts are used instead |
| Labour income, labour poverty | CONEVAL ITLP, `Indicadores_ITLP_Febrero_2025.zip` → `Cuadro_de_indicadores_ITLP_4T2024.xlsx` (Cuadro 10: % in labour poverty; Cuadro 11: real labour income per capita, INPC-deflated), from [CONEVAL ITLP](https://www.coneval.org.mx/Medicion/Paginas/ITLP-IS_resultados_a_nivel_nacional.aspx) | **2020Q2 not collected** (ENOE suspended), so the 2020 shock is 2020H2 vs 2019H2 |
| Population, median age | INEGI Censo 2020, `cpv2020_b_eum_01_poblacion.xlsx`, sheet 04 | Fixed 2020 denominator |
| Migration intensity | CONAPO IIM 2020 components (households with emigrants, circular migrants, return migrants, remittance receipt, %), from Table 2 of BBVA Research, *Migración y Remesas*, 1 Oct 2021 ([pdf](https://www.bbvaresearch.com/wp-content/uploads/2021/10/Remesas_e_Indice_de_Intensidad_Migratoria.pdf)); CONAPO degree (5 levels) from the same text | conapo.gob.mx and the gob.mx document page were unreachable or redirected; the index value itself (DP2) is not in hand, only its components and degree |
| Transit intensity | UPM/SEGOB Boletín estadístico, Cuadro 3.1 "Eventos de personas en situación migratoria irregular, según entidad federativa", 2019–2023 (`https://portales.segob.gob.mx/work/models/PoliticaMigratoria/CEM/Estadisticas/Boletines_Estadisticos/<y>/Cuadros<y>/cuadro3.1.xls`) | Apprehension events by state, per 1,000 residents. A proxy for transit-migrant presence. |
| ENIGH states | `../enigh/outputs/states.csv` (built in the ENIGH analysis) | Relative patterns only |

**Not obtained:**
- **US-side exposure (shift-share).** We need the Mexican state of origin x US state of residence of migrants. IME matrícula consular statistics (ime.gob.mx) did not respond, and ACS/CPS do not record the Mexican state. The MMP networks cover only 11 origin states (Jalisco, Puebla and Querétaro dominate), too few for a 32-state shift-share.
  - **How to get it:** download the IME "Estadísticas de mexicanos en el exterior / matrículas consulares" tables by entidad de origen x estado de residencia for 2018 or 2019 (xlsx) from a browser, or request them via INFOMEX. Combine them with BLS LAUS or QCEW state employment changes 2019→2020/21; FRED series `<ST>NA` and `<ST>UR` download without a key.
  - **Fallback used here:** controls for the 2017–19 own pre-trend and migration intensity.
- **State GDP (ITAEE):** skipped; ITLP labour income carries the income shock.
- **State transaction counts:** not published (see above).

## Variable definitions

- **Outcome `exc1719_y`:** log of state annual remittances (USD) in year y minus the log of the value predicted by the state's own log-linear quarterly trend with quarter dummies, fitted on 2017Q1–2019Q4. The national analogue in the handoff is counts vs a 2017–19 trend.
  - Alternatives: `exc1519`, `exc1319` (longer trend windows), and the raw `dlog_y` = log(R_y / R_2019) with 2016–19 growth as a control.
  - Mean `exc1719_2021` = 0.16 (SD 0.086). Every state is above trend in 2021 except Sinaloa (−0.015), Chihuahua is at 0.002, and the rest are above.
- **Need shocks** (z-scored across the 32 states in the regressions):
  - `excess_pc_2020_21`: all-cause deaths 2020+2021 minus the state's 2015–19 linear trend, per 1,000 residents (mean 4.96, SD 1.10).
  - `covid_pc_2020_21`: certificate COVID deaths per 1,000 (mean 3.25).
  - `labinc_loss_2020`: −Δlog real labour income per capita, 2020H2 vs 2019H2 (mean 0.04, SD 0.09).
- **Controls:** `pretrend_1719` (the fitted slope), `log_hh_remit` (log % of households receiving remittances, Censo 2020/CONAPO; correlation 0.96 with 2019 remittances per capita).
- **Samples:**
  - all 32;
  - excluding the **southern transit core** (Chiapas, Tabasco, Oaxaca, Veracruz: the southern border entry points and the Gulf/Pacific routes north);
  - excluding the **extended southern set** (+ Guerrero, Quintana Roo, Campeche). These are low-intensity emigration states on or next to the routes, where transfers to transit migrants and asylum seekers (COMAR caseload is concentrated in Tapachula) are a large share of a small base. Guerrero is included on the user's list, but it is a traditional sending state with little transit, so it is the weakest member.
  - additionally excluding the **northern border** (BC, Sonora, Chihuahua, Coahuila, Nuevo León, Tamaulipas), where migrants waiting under MPP/Title 42 in 2020–23 could also receive transfers.
- **Inference:** HC3 robust SEs. Randomization-inference p-values for the headline cells (5,000 permutations of the need variable across states, |t| statistic). MDE = 2.8 x SE, the effect detectable with 80% power at 5% two-sided.

## Results

### 1. Need coefficient by year (outcome: log deviation from own 2017–19 trend; need = 1 SD; HC3 SE in parentheses)

With controls (pre-trend, log % of households receiving remittances), unweighted:

| need | sample | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|
| excess mortality | all 32 | −0.006 (0.023) | **0.006 (0.021)** | −0.021 (0.023) | 0.011 (0.030) |
| excess mortality | excl. southern (25) | 0.002 (0.017) | **0.017 (0.030)** | −0.009 (0.036) | 0.031 (0.046) |
| COVID deaths | all 32 | 0.025 (0.012) | 0.009 (0.030) | −0.035 (0.066) | −0.040 (0.127) |
| COVID deaths | excl. southern | 0.031 (0.020) | 0.048 (0.017) | 0.025 (0.021) | 0.065 (0.036) |
| labour income loss | all 32 | 0.020 (0.016) | −0.017 (0.021) | −0.016 (0.042) | −0.045 (0.061) |
| labour income loss | excl. southern | 0.009 (0.021) | −0.029 (0.023) | 0.004 (0.030) | −0.000 (0.038) |

2021, headline cells:

| need | sample | coef | SE | p (HC3) | p (perm.) | MDE (80%) |
|---|---|---|---|---|---|---|
| excess mortality | all | 0.006 | 0.021 | 0.77 | 0.73 | 0.060 |
| excess mortality | excl. southern | 0.017 | 0.030 | 0.56 | 0.52 | 0.083 |
| excess mortality | excl. southern + north | 0.017 | 0.041 | 0.67 | – | 0.114 |
| COVID deaths | all | 0.009 | 0.030 | 0.77 | 0.72 | 0.084 |
| COVID deaths | excl. southern | 0.048 | 0.017 | 0.005 | 0.015 | 0.047 |
| labour income loss | all | −0.017 | 0.021 | 0.42 | 0.39 | 0.059 |
| labour income loss | excl. southern + north | −0.045 | 0.026 | 0.08 | – | 0.072 |

Without controls (bivariate), excess mortality in 2021 is 0.022 (0.013) for all 32 and 0.029 (0.011) excluding the southern states. Weighted by 2019 remittances it is 0.032 (0.015) and 0.035 (0.015). The positive bivariate slope (Fig. F1) is absorbed by the pre-trend. High-mortality states (CDMX, Edo. México, Tlaxcala, Morelos, Puebla) had slow 2017–19 trends (correlation −0.27), and slow-trend states sit above their extrapolated trend everywhere (mean reversion). With raw growth plus 2016–19 growth as the control, the estimate is again 0.009 (0.029).

Robustness for 2021, excess mortality (`robustness_2021.csv`): every variant lies between −0.004 and 0.027 with SE 0.02–0.04. The variants are the 2015–19 or 2013–19 trend windows, excess in % rather than per capita, 2020 deaths only, adding the labour-poverty change, adding log population and median age, and adding transit intensity. Weighted by 2019 remittances with controls: 0.020 (0.020) all, 0.024 (0.028) excluding the southern states.

**The COVID-death result is not need:**
1. It is significant only outside the southern states.
2. It falls to 0.027 (0.019) with a CDMX/Edo. México dummy and to 0.036 (0.028) with log population and median age.
3. It **rises** when excess mortality is also included (0.067). It is therefore the certified share of deaths, which is higher in the centre and in urban states, not mortality itself.
4. Its quarterly coefficient is already 0.026–0.036 in **2020Q1–Q2**, before COVID deaths accumulated (Fig. F3 data, `timing_quarterly_coef.csv`).

Urban, central states also saw the largest shift from cash to recorded channels, which would produce exactly this pattern.

**Labour income** has the opposite sign to the need prediction: states with bigger labour-income losses had slightly *less* excess in 2021. With weights, excluding the southern and northern states: −0.069 (0.025). This matches the ENIGH finding that amounts grew less in high-need cells.

**Leave-one-out** (2021, excess mortality, controls, Fig. F5): the coefficient ranges from −0.011 (dropping CDMX) to 0.013. No single state drives a hidden effect. For COVID deaths excluding the southern states it ranges from 0.036 (dropping CDMX) to 0.058.

### 2. Timing

The quarter-by-quarter coefficient on excess mortality (Fig. F3) is flat at about 0 from 2020Q1 through 2022, with |coef| ≤ 0.03 in every quarter. It becomes positive (0.04–0.08) only from 2023Q3, after the mortality shock was over, and the CIs are wide. High-need states therefore do **not** diverge in 2020, and not clearly in 2021 either. The late divergence is more consistent with state-specific trends (catch-up in the centre) than with a need response.

By tercile of excess mortality (Fig. F4), mean deviations in 2020/2021/2022/2023 are:

| tercile | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| high | 0.04 | 0.20 | 0.22 | 0.22 |
| mid | −0.02 | 0.12 | 0.13 | 0.06 |
| low | 0.03 | 0.15 | 0.20 | 0.13 |

The pattern is non-monotonic, and the 2021 level shift is present in every tercile.

Labour income loss: a small positive coefficient in 2020Q2–Q3 (+0.03–0.04, n.s.), possibly an immediate response, which turns negative from 2021Q3.

### 3. Transit migrants

- **Transit intensity** (log UPM irregular-migrant events per 1,000, 2021) is *negatively* related to 2020–2021 excess: −0.033 (0.016) in 2020 and −0.034 (0.015) in 2021, with controls.
- Chiapas and Tabasco had the steepest pre-trends of all states (+22% and +21% a year in 2017–19), so their boom predates COVID. Relative to their own trends they are unremarkable in 2021 (Chiapas 0.19, Tabasco 0.13). In raw growth they are the top two (Chiapas +0.63, Tabasco +0.53 log points 2019→21).
- The extended southern dummy is +0.07 (0.05) in 2021 and +0.15 (0.10) in 2023, not significant.
- The northern border states are **below** the others (−0.06 in 2021, −0.23 in 2023, p < 0.03), so there is no sign of a transfer-to-waiting-migrants surge there.
- **Within Chiapas**, the transit hubs (Tapachula, Suchiate, Tuxtla Chico, Frontera Comalapa, Palenque) *lose* share: 19% of state remittances in 2019, 16% in 2021, 15% in 2023. Growth is concentrated in the highlands and the interior: San Cristóbal ×6.5 2019→23, Comitán ×4.6, Ocosingo ×13, Simojovel ×10. That points to new emigration from the Chiapas highlands, or to non-family flows recorded as remittances (press reports since 2023 have questioned the Chiapas figures), more than to transfers to transit migrants at the border.
- The transit story may still matter for the southern *levels*, but it does not create a need-shaped cross-state pattern.

### 4. ENIGH vs Banxico (relative patterns, 2018→2022)

- State log changes in ENIGH total remittances (real MXN) track Banxico's (USD): slope 1.18 (0.41). The two sources agree on which states grew.
- The agreement comes through receipt (slope 0.92, SE 0.56), not through the amount per recipient (0.08, SE 0.66).
- Neither excess mortality nor labour-income loss predicts the ENIGH changes: total 0.085 (0.134); receipt 0.038 (0.046); amount 0.030 (0.079), per 1 SD with controls.

### Power

With 32 states, the 2021 SE on a 1 SD need shock is about 0.02, so effects of at least 0.06 log points per SD are detectable with 80% power (about 0.08 excluding the southern states). The 95% CI for excess mortality (all, controls) is [−0.035, 0.048] per SD, against a mean 2021 excess of 0.16.

What this rules out:
- A need channel strong enough to make a 1 SD more-affected state receive 5+ log points more is unlikely.
- It does **not** bound the aggregate need effect. The state cross-section only identifies differential responses, and the common 2021 shift (every tercile +0.12 to +0.20) is absorbed by the intercept.
- As an illustration only: taken linearly through zero mortality, the point estimate (0.006 per SD = 0.0055 per excess death per 1,000) implies a need contribution of about 0.03 log points at the mean excess of 4.96 per 1,000. The CI on that extrapolation spans −0.16 to +0.22, which is uninformative.

## Read: need vs capacity vs transit

- **Need:** not supported by the state variation.
  - Excess mortality, the cleanest need measure, has no conditional association with the 2021 excess and no timing that matches the deaths.
  - Labour-income losses have the wrong sign.
  - The only positive association, certified COVID deaths, starts before the deaths and loads on the urban centre.
  - This agrees with the model result that need adds little (+2 of the 24 log points) and with the ENIGH finding that amounts grew less in high-need cells.
- **Capacity (US side):** cannot be tested directly without the IME origin x destination matrix. Two indirect points:
  - The level shift is common to all states and all need terciles, which is what a US-side or channel driver shared by all origins would produce.
  - The northern border states, whose migrants are concentrated in Texas, California and Arizona, grew least. This is suggestive only; it is the natural next test once IME data are in hand.
- **Transit migrants:** they explain little of the cross-section after own pre-trends. The southern boom started before 2020, transit intensity is negatively related to the 2020–21 excess, and Chiapas growth is not in the transit hubs.
- **Channel shift (cash → recorded):** consistent with the COVID-death/urban pattern from 2020Q1, but not identified here.

**Usable for identification?** Not as a need instrument: the state need shocks do not move remittances differentially, and with 32 states the design has power only against large differential effects. Two uses remain:
1. **A falsification moment for the model.** The household model with β > 0 predicts larger 2021 excess where mortality was higher, and the data show about 0 per SD (CI ±0.04), which bounds the *differential* need response.
2. **A capacity shift-share**, once IME origin x US-state shares are obtained. That is where cross-state variation is more likely to have power.

The municipal panel (2,488 municipalities, 2013+) is in hand and could support a much better-powered design if municipal need measures are built (death certificates have `mun_resid`).

## Figures

- `figures/F1_scatter_excess_mortality.png`: excess vs own trend, 2020–2023, against excess mortality, with OLS lines for all states and excluding the southern states.
- `figures/F2_scatter_income_transit.png`: 2021 excess vs labour-income loss and vs transit intensity.
- `figures/F3_timing_coefficients.png`: quarterly coefficients (95% CI) on excess mortality and labour-income loss, with controls.
- `figures/F4_need_terciles_quarterly.png`: quarterly mean deviation by excess-mortality tercile.
- `figures/F5_leave_one_out_2021.png`: leave-one-out 2021 coefficient.
- `figures/F6_enigh_vs_banxico.png`: state growth 2018–22, ENIGH vs Banxico.
