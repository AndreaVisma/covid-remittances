# Minimal remittance simulation

Run `python mexico/simulate_remittances.py` from the project folder. Install
dependencies with `python -m pip install -r mexico/requirements.txt` if needed.
The six `# %%` sections can also run sequentially in VS Code/Spyder or be
pasted into a console opened in the project or mexico folder. Inspect
`population`, `inputs`, `agents`, `monthly` and `fig` as you go.

The model uses the supplied population, January 2010–January 2025. A logistic
probability depends on NTA surplus, asymmetry and the supplied normalized GDP
covariates. Independent binomial draws give the number sending in each cohort.
Each sender sends `rho * monthly_income_usd`. Expected flow uses `n * p` instead
of a draw. No individual identities or household budgets persist across months.

All coefficients and the USD 4,000 monthly income are illustrative assumptions.
This is a runnable baseline, not a calibrated reproduction of the paper.
The dataset lacks destination income, disaster exposure, CPI and exchange rates.
It therefore uses constant assumed income, zero shocks and no inflation
adjustment or imposed seasonal pattern. Edit `inputs` to supply monthly income
or experiment with shifts in sending propensity; an imposed shock does not
explain the observed pandemic response. A disaster response kernel is not fitted.

The 15–19 cohort is assumed uniformly distributed across ages: four fifths are
eligible. Cohort NTA values are retained, and nonpositive NTA implies no sending.
The file has no sex column, so its existing asymmetry is used directly.
Higher asymmetry is treated as a proxy for having migrated alone, not a measured
household probability. GDP normalization provenance is absent from the pickle;
`gdp_diff_norm` and `gdp_origin_norm` are used as supplied. The methods' printed
GDP equation and its claimed bounded range are inconsistent, so US income is
not reconstructed from `relative_diff`.

Outputs: monthly flows, cohort calculations (including probabilities), and a
self-contained Plotly HTML. These are USA-to-Mexico simulated flows. The earlier
Banco de Mexico workbook measures total inflows from all origins and is not
silently used as a bilateral calibration target. This model allows at most one
transfer per person per month and does not identify observed sender frequency.

Reference: supplied methods text and
[Methods, equations 1–8](https://arxiv.org/html/2512.16373v1#S4).
