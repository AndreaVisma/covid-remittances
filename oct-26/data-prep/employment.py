"""
employment rate and median wages by ethnicity
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

df = pd.read_excel("C:\\git-projects\\covid-remittances\\data\\weekly-earnings-median-race.xlsx")
formatted_quarters = df['Quarter'].apply(lambda x: f"{x.split()[1]}{x.split()[0]}")
df['date'] = pd.PeriodIndex(formatted_quarters, freq='Q').to_timestamp()
df = df.sort_values('date')

df.rename(columns = {"Hispanic or Latino" : "weekly_earnings"}, inplace = True)
df = df[(df['date'].dt.year.between(2019, 2023))][["date", "weekly_earnings"]]
df['monthly_earnings'] = np.ceil(df['weekly_earnings'] * (52 / 12))

# 3. Create a Monthly Date Range (from Jan 2019 to Dec 2023)
monthly_dates = pd.date_range(start='2019-01-01', end='2023-12-01', freq='MS')

# Convert dates to numeric format (ordinal days) for the spline function
x_original = df['date'].map(pd.Timestamp.toordinal)
y_original = df['monthly_earnings']

# 4. Fit a Cubic Spline to get smooth monthly values
cs = PchipInterpolator(x_original, y_original)

x_monthly = monthly_dates.map(pd.Timestamp.toordinal)
interpolated_monthly_earnings = cs(x_monthly)

# 5. Build the Final Reworked DataFrame
monthly_df = pd.DataFrame({
    'date': monthly_dates,
    'monthly_earnings_nominal': interpolated_monthly_earnings
})

# save
monthly_df.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\monthly-median-earnings-latinx.parquet")

# --- 6. Plotting with Plotly ---
fig = go.Figure()

# Add the interpolated monthly spline line
fig.add_trace(go.Scatter(
    x=monthly_df['date'],
    y=monthly_df['monthly_earnings_nominal'],
    mode='lines',
    name='Monthly Spline (Nominal)',
    line=dict(color='royalblue', width=2)
))

# Add the original quarterly values as markers (dots)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['monthly_earnings'],
    mode='markers',
    name='Original Quarterly Data',
    marker=dict(color='darkorange', size=8)
))

# Update layout & labels
fig.update_layout(
    title="Latinx Median Monthly Earnings (Nominal) - 2019 to 2023",
    xaxis_title="Date",
    yaxis_title="Nominal Earnings (USD)",
    legend=dict(x=0.01, y=0.99)
)

# --- 7. Save as SVG ---
fig.write_image("C:\\git-projects\\covid-remittances\\oct-26\\outputs\\general\\latinx_median_monthly_earnings.svg")

fig.show()

###################
# unemployment rate

df_emp = pd.read_csv("C:\\git-projects\\covid-remittances\\data\\employment_table_ethnicity_monthly.csv")
df_emp["date"] = pd.to_datetime(df_emp["Date"])
df_emp.rename(columns = {"Hispanic or Latino" : "unemployment_hispanic"}, inplace = True)
df_emp = df_emp[["date", "unemployment_hispanic"]]
df_emp["unemployment_hispanic"] *= 0.01

df_emp.to_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\monthly-unemployment-latinx.parquet")