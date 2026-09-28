"""
Analyse and prepare historical flows
"""

import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

##
rem_file = "C:\\git-projects\\covid-remittances\\data\\mex_remittances.xlsx"
df_large = pd.read_excel(rem_file, sheet_name="Sheet1")

df = df_large[df_large.year > 2017][["date", "total_mln"]]
df["total_mln"] *= 1_000_000
df["date"] = pd.to_datetime(df["date"])

# plot
fig = px.line(df, x="date", y="total_mln", title="Mexico Remittances Over Time")

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Remittances (Absolute Value)"
)

fig.show()

## save
df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\mexico-rem-monthly-total.parquet")