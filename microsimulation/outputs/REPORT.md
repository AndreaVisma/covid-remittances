# Prototype results

Synthetic inputs and assumed behavioural parameters; these runs are not an empirical explanation.

Target: `C:\git-projects\covid-remittances\data\mex_remittances.xlsx`, 1995-01-01 through 2024-05-01 (353 monthly observations).
Units: USD; not explicitly documented in workbook. Values multiplied by 1,000,000.
No sending-country or migrant-population identification in workbook; aggregate Mexico comparison only.

Population represented by equal agent weights: 10,000,000.0.
No outcome-based population scaling is applied. The indexed chart separately divides each series by its own pre-COVID mean.

Default parameters are not calibrated. Optional calibration fits only the baseline remittance share before the holdout cutoff.
The same parameters are retained across ablations; changes in baseline levels are part of the experiment.

See comparison.png, metrics.csv, annual_by_seed.csv and paired_effects.csv. Sensitivity results are in sensitivity.csv.
The seed range captures simulation randomness only. Network structure, population composition and shocks remain assumed.
Aggregate fit cannot separately identify activation, persistence, transfer norms or support.
