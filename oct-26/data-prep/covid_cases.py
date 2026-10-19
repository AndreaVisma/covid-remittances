"""
Analyse and plot covid cases
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

#
covid_file = "C:\\git-projects\\covid-remittances\\data\\covid_cases.csv"
df_large = pd.read_csv(covid_file)

#
df = df_large[df_large.continent.isin(['North America', 'South America'])][["date",
    "country", "total_cases", "new_cases", "total_cases_per_million", "new_cases_per_million",
    "total_deaths", "new_deaths", "total_deaths_per_million", "new_deaths_per_million"]]
df['date'] = pd.to_datetime(df['date'])
df = df[df.date.dt.year < 2024]
df = df.set_index('date')

agg_rules = {
    'new_cases': 'sum',
    'total_cases': 'max',
    'new_deaths': 'sum',
    'total_deaths': 'max',
    'new_cases_per_million': 'sum',
    'total_cases_per_million': 'max',
    'new_deaths_per_million': 'sum',
    'total_deaths_per_million': 'max',
}

weekly_df = df.groupby('country').resample('W').agg(agg_rules).reset_index()
monthly_df = df.groupby('country').resample('M').agg(agg_rules).reset_index()
quarterly_df = df.groupby('country').resample('QE').agg(agg_rules).reset_index()

## save
weekly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\america-covid-weekly.parquet")
monthly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\america-covid-monthly.parquet")
quarterly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\america-covid-quarter-end.parquet")

## mexico
df_mex = quarterly_df[quarterly_df.country == "Mexico"]

#plot

df_rem = pd.read_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\mexico-rem-quarterly-total.parquet")
df_rem = df_rem[(df_rem.date.dt.year > 2017) & (df_rem.date.dt.year < 2025)]

# --- 3. Create Subplots with a Secondary Y-Axis ---
fig = make_subplots(specs=[[{"secondary_y": True}]])

# --- 4. Add COVID Trace to the Primary Y-Axis (Left) [New Deaths Removed] ---
fig.add_trace(
    go.Scatter(
        x=df_mex["date"],
        y=df_mex["new_deaths_per_million"],
        name="New Deaths per Million",
        line=dict(color="blue")
    ),
    secondary_y=False,
)

# --- 5. Add Remittances Trace to the Secondary Y-Axis (Right) ---
fig.add_trace(
    go.Scatter(
        x=df_rem["date"],
        y=df_rem["total_usd"],
        name="Total Remittances (Absolute)",
        line=dict(color="green", dash="dot")
    ),
    secondary_y=True,  # <--- Pushes it to the right axis
)

# --- 6. Update Layout & Axis Labels ---
fig.update_layout(
    title_text="Mexico: COVID-19 Cases vs. Remittances Over Time",
    xaxis_title="Date",
    legend=dict(x=0.01, y=0.99)
)

# Set y-axis titles explicitly (adjusted since deaths were removed)
fig.update_yaxes(title_text="COVID-19 Cases (per million)", secondary_y=False)
fig.update_yaxes(title_text="Remittances (Absolute USD)", secondary_y=True)

# --- 7. Save as SVG ---
# Note: Saving static image formats like SVG/PNG in Plotly requires 'kaleido' (pip install kaleido)
fig.write_image("C:\\git-projects\\covid-remittances\\oct-26\\outputs\\general\\mexico_covid_remittances_deaths.svg")

fig.show()