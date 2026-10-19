"""
Self-standing agent-based simulation for remittances, incorporating median earnings,
quintile income distribution, COVID channel impacts, and exportable Plotly SVG charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

parquet_path = "C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\mexico-rem-monthly-total.parquet"
df_actual = pd.read_parquet(parquet_path)
df_actual['date'] = pd.to_datetime(df_actual['date']).dt.to_period('M').dt.to_timestamp()
df_actual = df_actual.sort_values('date')

monthly_df = pd.read_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\monthly-median-earnings-latinx.parquet")
monthly_df['date'] = pd.to_datetime(monthly_df['date']).dt.to_period('M').dt.to_timestamp()
monthly_df = monthly_df.sort_values('date').reset_index(drop=True)
if monthly_df['date'].duplicated().any():
    raise ValueError("Earnings data must have one observation per month.")

df_merged = pd.merge(df_actual, monthly_df, on = "date", how= "inner")
df_merged["monthly_earnings_nominal"] *= 0.2

df_merged["needed_senders"] = df_merged["total_usd"] / df_merged["monthly_earnings_nominal"]
df_merged["pct_migrant_senders"] = df_merged["needed_senders"] / 11_000_000

fig = px.line(df_merged, x="date", y="pct_migrant_senders")
fig.show()