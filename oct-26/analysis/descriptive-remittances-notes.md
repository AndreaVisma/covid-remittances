# Descriptive analysis and model experiments

Run `descriptive-remittances.py` from top to bottom or run its numbered `# %%` sections in order. There are no custom functions or classes. `PRE_START` and `PRE_END` control the baseline; HTML figures are saved automatically. `SHOW_FIGURES = True` also opens them. Dependencies: pandas, numpy, openpyxl, plotly.

## Source and methods

Source: `data/mexico_remittances_2026.xlsx`, sheet Hoja1, series identifiers in Excel row 18. Actual observations span January 2010–July 2026 (199 months), despite the metadata's January 1995 availability date. SE27803 is million USD, SE27808 thousand transactions, and SE28647 average USD. All monetary results here are nominal USD.

The published average is rounded to dollars. The script derives the unrounded average as total USD / transactions; its maximum difference from the published figure is $0.499. This permits an exact identity, R = T × A.

The exact growth decomposition splits the interaction equally: count contribution = 100 gT(1 + gA/2), amount contribution = 100 gA(1 + gT/2). Their sum equals total percentage growth. This is a symmetric accounting allocation, not causal attribution. Annual totals and counts are summed; annual average size is their ratio, not the unweighted mean of monthly sizes. Partial 2026 is excluded from annual comparisons but retained in monthly charts.

Default trend: an ordinary least-squares straight line in log levels on a monthly time index, fitted over 2015–2019 without seasonal terms. Extrapolated trends are descriptive benchmarks. Monthly growth deviations compare each month's month-on-month percentage growth with the arithmetic average for that calendar month in 2015–2019. Each calendar month has five baseline observations. The difference is in percentage points, not percent or year-on-year growth. March 2020 begins the displayed pandemic-and-later comparison; later differences are not necessarily caused by the pandemic.

## Findings in this workbook vintage

| Year | Total growth (%) | Count contribution (pp) | Amount contribution (pp) |
|---|---:|---:|---:|
| 2020 | 11.96 | 7.43 | 4.52 |
| 2021 | 25.94 | 14.10 | 11.84 |
| 2022 | 12.08 | 8.91 | 3.17 |
| 2023 | 7.56 | 6.60 | 0.96 |
| 2024 | 2.68 | 3.00 | -0.32 |
| 2025 | -3.91 | -4.40 | 0.48 |

The 2015–2019 trend annualizes to 10.72% for totals, 7.36% for counts and 3.12% for average amounts. Against this benchmark, average monthly total levels are about 13.6%, 15.1% and 11.8% above trend in 2021, 2022 and 2023. The gap narrows to 3.7% in 2024 and becomes negative in 2025. High nominal levels are therefore distinct from permanently faster growth or permanently positive trend deviations.

Baseline sensitivity is substantial: fitting 2010–2019 gives a 5.97% annual total trend and a 45.1% average 2023 total-level gap, versus 10.72% and 11.8% using 2015–2019. Do not interpret either gap as a causal pandemic effect. The annual count/amount decomposition does not depend on this trend choice.

## Meaningful next model experiments

1. **Participation/frequency versus transfer amounts.** Test count-only, amount-only and joint responses against both observed margins. The sizable 2021 contributions from both margins make a model matching totals alone insufficient. Count and amount changes are not independent of composition: changing which agents send can change the average even with a fixed individual rule.
2. **Temporary intensive response and longer-lived count response.** Allow crisis-related desired amounts to rise temporarily while sending participation or frequency changes more persistently. Target the large 2021 amount contribution followed by the more count-driven growth in 2022–2024. Compare this with persistent amount growth; do not assume the former explanation is proven.
3. **Desire versus capacity protection.** Under the same employment shock, compare increased desired transfers alone, savings/benefit support alone, and their combination. Evaluate transaction numbers and sizes jointly. These aggregate series cannot themselves identify benefits, savings, or motives; external evidence must anchor those mechanisms.
4. **Persistence with exit.** Evaluate whether the same model can produce the subsequent slowdown and count-driven 2025 decline. Permanent upward adjustments to propensity or population may fit the rise while failing later. Use 2024–2026 as a held-out challenge where feasible, and evaluate partial 2026 only on matched months.
5. **Separate senders from transactions in sensitivity experiments.** Compare one transfer per sender per month with variable frequency under externally plausible constraints. More transactions do not identify more unique migrants; a higher mean does not identify richer senders.

First establish the pre-pandemic count and amount dynamics. Then calibrate a small set of behavioral parameters to selected pandemic patterns, assess remaining years out of sample, and run multiple random seeds. Keep descriptive baseline sensitivity separate from simulation uncertainty. The same-calendar-month growth comparison controls for the historical average monthly pattern but is not an estimated seasonal trend model.

The notes above describe the current input vintage and default settings; rerunning with revised data or settings regenerates the charts and tables, not this narrative.
