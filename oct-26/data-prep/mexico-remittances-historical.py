"""
Analyse and prepare historical flows
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

##
rem_file = "C:\\git-projects\\covid-remittances\\data\\mexico_remittances_2026.xlsx"
df_large = pd.read_excel(rem_file, sheet_name='Hoja1', header=17)
df = df_large[['Fecha', 'SE27803', 'SE27808', 'SE28647']].copy()
df.columns = ['date', 'total_million_usd', 'transactions_thousand', 'published_average_usd']
df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
df['total_usd'] = df['total_million_usd'] * 1_000_000
df['transactions'] = df['transactions_thousand'] * 1_000
df = df.set_index('date')

agg_rules = {
    'total_usd': 'sum',
    'transactions': 'sum'
}

quarterly_df = df[['total_usd', 'transactions']].resample('QE').agg(agg_rules).reset_index()
quarterly_df['average_usd_per_transaction'] = quarterly_df['total_usd'] / quarterly_df['transactions']
df['average_usd_per_transaction'] = df['total_usd'] / df['transactions']
df.reset_index(inplace = True)

# plot
fig = px.line(df, x="date", y="total_usd", title="Mexico Remittances Over Time")

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Remittances (Absolute Value)"
)
fig.add_trace(go.Line(x = quarterly_df["date"], y=quarterly_df["total_usd"]))

fig.show()

## save
df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\mexico-rem-monthly-total.parquet")
quarterly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\mexico-rem-quarterly-total.parquet")