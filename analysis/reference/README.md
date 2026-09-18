# Supplementary national series

The original files in `data/` are unchanged. These reference series fill gaps in national coverage, not individual missing observations in those files.

## Philippines

`philippines_national_monthly.csv` is extracted by `scripts/fetch_reference_data.py` from archived BSP releases. Each observation carries its source URL and vintage. The July 2023 release takes precedence where both tables contain an observation. Values are **cash remittances through the banking system, nominal USD millions**, not the broader personal-remittances measure. Monthly values are rounded to whole millions; annual results use the reported December cumulative value, avoiding rounding discrepancies.

- https://www.bsp.gov.ph/Media_And_Research/Media%20Releases/2022_05/news-05162022a1.aspx
- https://www.bsp.gov.ph/Media_And_Research/Media%20Releases/2023_07/news-07172023a1.aspx

## Pakistan

`pakistan_national_fiscal_months.csv` transcribes **Table 9.7, printed page 112 (PDF page 10)** of the State Bank of Pakistan's Statistical Supplement 2022–23. Values are nominal USD millions. FY2019 means July 2018–June 2019. The analysis converts these to calendar months before any comparison.

Source: https://www.sbp.org.pk/reports/annual/arFY23/Stats/Eng/Chapter-9.pdf

Published fiscal totals, used as independent checks: FY2019 21,739.4; FY2020 23,132.3; FY2021 29,449.9; FY2022 31,278.8; FY2023 27,027.6 million USD. Monthly rounding may produce a discrepancy up to 0.3 million.

This is a frozen **2023 vintage**, not a claim to be the latest revised series. The supplied Pakistan pickle extends to 2025, so differences between it and this reference may reflect both omitted countries and revisions. The national reference is never spliced into the local corridor sums. The publication notes an original-remitter-country basis from July 2019 and inclusion of current-transfer conversions from Roshan Digital Accounts from September 2020.

References accessed 2026-09-14. See the main report for interpretation and comparability limitations.
