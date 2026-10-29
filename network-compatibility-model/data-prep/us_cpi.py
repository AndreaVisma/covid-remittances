"""US consumer prices for the amount accounting (29 Oct 2026).

Run: python network-compatibility-model/data-prep/us_cpi.py  (seconds; downloads from FRED)
Output: data-prep/outputs/us_cpi_monthly.csv (public series, safe to commit)

CPIAUCSL: CPI-U, all items, US city average, seasonally adjusted, 1982-84 = 100 (BLS via FRED).
Used to split the Banxico average amount per transaction (nominal USD) into US inflation and the
real amount: the decomposition in recipient-level/make_figures.py (S10). The model targets counts;
amounts enter only as observed accounting (user decision, 29 Oct 2026).
"""
from pathlib import Path
import pandas as pd

OUT = Path(__file__).resolve().parent / 'outputs'
URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL'


def main():
    cpi = pd.read_csv(URL, parse_dates=['observation_date'])
    cpi = cpi.rename(columns={'observation_date': 'date', 'CPIAUCSL': 'cpi_u'})
    cpi = cpi[cpi.date >= '2010-01-01']
    OUT.mkdir(exist_ok=True)
    cpi.to_csv(OUT / 'us_cpi_monthly.csv', index=False)
    print(f'{len(cpi)} months, {cpi.date.min():%Y-%m} to {cpi.date.max():%Y-%m}')


if __name__ == '__main__':
    main()
