"""
peso dollar exchange rate
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

##
ex_file = "C:\\git-projects\\covid-remittances\\data\\DEXMXUS.csv"
df = pd.read_csv(ex_file).dropna()

df["date"] = pd.to_datetime(df["observation_date"])
df = df[(df.date.dt.year > 2018) & (df.date.dt.year < 2025)]
df = df.set_index("date")
df.drop(columns = "observation_date", inplace = True)

monthly_df = df.resample('ME').agg("mean").reset_index()
monthly_df.rename(columns = {"DEXMXUS" : "ex_rate"}, inplace = True)

monthly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\dollar-peso-exchange-rate.parquet")


# --- 6. Plotting with Plotly ---
fig = go.Figure()

# Add the interpolated monthly spline line
fig.add_trace(go.Scatter(
    x=monthly_df['date'],
    y=monthly_df['ex_rate'],
    mode='lines',
    name='exchange rate dollar peso',
    line=dict(color='royalblue', width=2)
))


# Update layout & labels
fig.update_layout(
    title="Exchnage rate dollar peso - 2019 to 2023",
    xaxis_title="Date",
    yaxis_title="Mexican Pesos to 1 US dollar",
    legend=dict(x=0.01, y=0.99)
)

# --- 7. Save as SVG ---
fig.write_image("C:\\git-projects\\covid-remittances\\oct-26\\outputs\\general\\dollar-peso-exchnage-rate.svg")

fig.show()