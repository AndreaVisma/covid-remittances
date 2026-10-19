"""Mexico remittances: quarterly version.

Requires pandas, numpy, openpyxl and plotly.
"""

# %% 1. Settings and imports
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path('C:/git-projects/covid-remittances')
INPUT = ROOT / 'data/mexico_remittances_2026.xlsx'
OUTPUT = ROOT / 'oct-26/outputs/descriptive-remittances-quarterly'
OUTPUT.mkdir(parents=True, exist_ok=True)

PRE_START = pd.Timestamp('2015-01-01')       # 2015Q1
PRE_END = pd.Timestamp('2019-10-01')         # 2019Q4
PANDEMIC_START = pd.Timestamp('2020-01-01')  # 2020Q1; use 2020-04-01 for 2020Q2
SHOW_FIGURES = False

labels = {
    'total_usd': 'Total remittances',
    'transactions': 'Transaction count',
    'average_usd': 'Average transfer',
}
colors = {
    'total_usd': '#222222',
    'transactions': '#277DA1',
    'average_usd': '#D97706',
}
pre_label = (
    f'{PRE_START.year}Q{((PRE_START.month - 1) // 3) + 1}'
    f'–{PRE_END.year}Q{((PRE_END.month - 1) // 3) + 1}'
)

# %% 2. Read monthly workbook, validate it, then aggregate to complete quarters
raw = pd.read_excel(INPUT, sheet_name='Hoja1', header=17)
monthly = raw[['Fecha', 'SE27803', 'SE27808', 'SE28647']].copy()
monthly.columns = [
    'date', 'total_million_usd', 'transactions_thousand', 'published_average_usd'
]
monthly['date'] = pd.to_datetime(monthly['date']).dt.to_period('M').dt.to_timestamp()
monthly = monthly.sort_values('date').set_index('date')

if monthly.index.has_duplicates or not monthly.index.equals(
    pd.date_range(monthly.index.min(), monthly.index.max(), freq='MS')
):
    raise ValueError('The input must contain one observation for every consecutive month.')

if not np.isfinite(monthly.to_numpy(dtype=float)).all() or not monthly.gt(0).all().all():
    raise ValueError('Missing or nonpositive input: inspect the workbook before calculating growth.')

monthly['total_usd'] = monthly['total_million_usd'] * 1_000_000
monthly['transactions'] = monthly['transactions_thousand'] * 1_000
monthly['average_usd'] = monthly['total_usd'] / monthly['transactions']
monthly['average_rounding_difference_usd'] = (
    monthly['average_usd'] - monthly['published_average_usd']
)

print(f'Observed monthly data: {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}; {len(monthly)} months')
print('Maximum difference from published average (USD):',
      monthly['average_rounding_difference_usd'].abs().max())

# Aggregate to quarterly sums, then recompute the average from the sums.
q = monthly.resample('QS').agg(
    total_usd=('total_usd', 'sum'),
    transactions=('transactions', 'sum'),
    months=('total_usd', 'size'),
)
q = q.loc[q['months'].eq(3)].copy()
q['average_usd'] = q['total_usd'] / q['transactions']
df = q.drop(columns='months')

if df.index.has_duplicates or not df.index.equals(
    pd.date_range(df.index.min(), df.index.max(), freq='QS')
):
    raise ValueError('Quarterly input must contain consecutive complete quarters.')

if not np.isfinite(df.to_numpy(dtype=float)).all() or not df.gt(0).all().all():
    raise ValueError('Missing or nonpositive quarterly input.')

print(
    f'Observed quarterly data: {df.index.min().to_period("Q")} '
    f'to {df.index.max().to_period("Q")}; {len(df)} quarters'
)

pre = (df.index >= PRE_START) & (df.index <= PRE_END)
if pre.sum() < 8 or pd.Timestamp(PRE_END) >= PANDEMIC_START:
    raise ValueError('Choose at least two years of pre-pandemic quarters.')

# %% 3. Quarter-on-quarter growth decomposition
# R = T * A implies g_R = g_T + g_A + g_T*g_A.
growth = df[list(labels)].pct_change(fill_method=None)

decomposition = pd.DataFrame(index=df.index)
decomposition['Count contribution'] = (
    100 * growth['transactions'] * (1 + growth['average_usd'] / 2)
)
decomposition['Amount contribution'] = (
    100 * growth['average_usd'] * (1 + growth['transactions'] / 2)
)
decomposition['Total growth'] = 100 * growth['total_usd']

np.testing.assert_allclose(
    decomposition.iloc[1:, :2].sum(axis=1),
    decomposition['Total growth'].iloc[1:],
    atol=1e-10,
)

fig_growth = go.Figure()
for name, color in [
    ('Count contribution', colors['transactions']),
    ('Amount contribution', colors['average_usd']),
]:
    fig_growth.add_bar(x=df.index, y=decomposition[name], name=name, marker_color=color)

fig_growth.add_scatter(
    x=df.index,
    y=decomposition['Total growth'],
    name='Total QoQ growth (%)',
    line=dict(color='#222222'),
)
fig_growth.update_layout(
    template='plotly_white',
    barmode='relative',
    hovermode='x unified',
    title='Quarterly growth: transaction-count and average-amount contributions',
    yaxis_title='QoQ growth / contribution (%, percentage points)',
    xaxis_title='Quarter',
    legend=dict(orientation='h', y=-0.2),
    height=550,
)
fig_growth.write_html(OUTPUT / '01_quarterly_growth_decomposition.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_growth.show()

# %% 4. Year-over-year quarterly decomposition
# This replaces the annual section while keeping the frequency quarterly.
yoy_growth = df[list(labels)].pct_change(4, fill_method=None)

yoy_decomp = pd.DataFrame(index=df.index)
yoy_decomp['Count contribution'] = (
    100 * yoy_growth['transactions'] * (1 + yoy_growth['average_usd'] / 2)
)
yoy_decomp['Amount contribution'] = (
    100 * yoy_growth['average_usd'] * (1 + yoy_growth['transactions'] / 2)
)
yoy_decomp['Total growth'] = 100 * yoy_growth['total_usd']

np.testing.assert_allclose(
    yoy_decomp.iloc[4:, :2].sum(axis=1),
    yoy_decomp['Total growth'].iloc[4:],
    atol=1e-10,
)

fig_yoy = go.Figure()
fig_yoy.add_bar(
    x=df.index,
    y=yoy_decomp['Count contribution'],
    name='Count contribution',
    marker_color=colors['transactions'],
)
fig_yoy.add_bar(
    x=df.index,
    y=yoy_decomp['Amount contribution'],
    name='Amount contribution',
    marker_color=colors['average_usd'],
)
fig_yoy.add_scatter(
    x=df.index,
    y=yoy_decomp['Total growth'],
    name='Total YoY growth (%)',
    line=dict(color='#222222'),
)
fig_yoy.update_layout(
    template='plotly_white',
    barmode='relative',
    hovermode='x unified',
    title='Year-over-year quarterly growth decomposition',
    xaxis=dict(title='Quarter', dtick=1),
    yaxis_title='YoY growth / contribution (%, percentage points)',
    legend=dict(orientation='h', y=-0.2),
    height=550,
)
fig_yoy.write_html(OUTPUT / '02_yoy_quarterly_growth_decomposition.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_yoy.show()

# %% 5. Pre-pandemic trend: log(level) = intercept + slope * quarter
# No quarter dummies. Extrapolation is a descriptive benchmark.
t = np.arange(len(df))
trend = pd.DataFrame(index=df.index)
trend_rows = []

fig_trend = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=list(labels.values()),
    vertical_spacing=0.09,
)

for row, (column, label) in enumerate(labels.items(), start=1):
    slope, intercept = np.polyfit(t[pre], np.log(df.loc[pre, column]), 1)
    trend[column] = np.exp(intercept + slope * t)

    trend_rows.append({
        'series': label,
        'quarterly_trend_pct': 100 * np.expm1(slope),
        'annualized_trend_pct': 100 * np.expm1(4 * slope),
    })

    anchor = trend.loc[PRE_END, column]

    fig_trend.add_scatter(
        x=df.index,
        y=100 * df[column] / anchor,
        name='Observed',
        legendgroup='observed',
        showlegend=row == 1,
        line=dict(color=colors[column]),
        row=row,
        col=1,
    )
    fig_trend.add_scatter(
        x=df.index[pre],
        y=100 * trend.loc[pre, column] / anchor,
        name='Fitted trend',
        legendgroup='fit',
        showlegend=row == 1,
        line=dict(color='#666666'),
        row=row,
        col=1,
    )

    after = df.index > PRE_END
    fig_trend.add_scatter(
        x=df.index[after],
        y=100 * trend.loc[after, column] / anchor,
        name='Trend extrapolation',
        legendgroup='projection',
        showlegend=row == 1,
        line=dict(color='#666666', dash='dash'),
        row=row,
        col=1,
    )

    fig_trend.update_yaxes(title_text='Index', row=row, col=1)

fig_trend.update_layout(
    template='plotly_white',
    height=950,
    hovermode='x unified',
    title=(
        f'Pre-pandemic log-linear trends ({pre_label}); '
        f'fitted {PRE_END.year}Q{((PRE_END.month - 1) // 3) + 1} = 100'
    ),
    legend=dict(orientation='h', y=-0.08),
)
fig_trend.write_html(OUTPUT / '03_pre_pandemic_trends.html', include_plotlyjs=True)

trend_summary = pd.DataFrame(trend_rows)
print('\nPre-pandemic quarterly trend rates (%)\n', trend_summary.round(2).to_string(index=False))
if SHOW_FIGURES:
    fig_trend.show()

# %% 6. Quarterly growth compared with the SAME quarter's pre-pandemic average
quarterly_growth_pct = growth * 100

baseline = (
    quarterly_growth_pct.loc[pre]
    .groupby(quarterly_growth_pct.loc[pre].index.quarter)
    .mean()
)
baseline_n = (
    quarterly_growth_pct.loc[pre]
    .groupby(quarterly_growth_pct.loc[pre].index.quarter)
    .count()
)

benchmark = pd.DataFrame(index=df.index)
for column in labels:
    benchmark[column] = df.index.quarter.map(baseline[column])

excess_growth_pp = quarterly_growth_pct - benchmark
post = df.index >= PANDEMIC_START

fig_deviation = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=list(labels.values()),
    vertical_spacing=0.09,
)

for row, (column, label) in enumerate(labels.items(), start=1):
    fig_deviation.add_bar(
        x=df.index[post],
        y=excess_growth_pp.loc[post, column],
        name='Growth minus same-quarter baseline (pp)',
        marker_color=colors[column],
        showlegend=row == 1,
        row=row,
        col=1,
    )
    fig_deviation.add_scatter(
        x=df.index[post],
        y=quarterly_growth_pct.loc[post, column],
        name='Observed QoQ growth (%)',
        line=dict(color='#222222', width=1),
        showlegend=row == 1,
        row=row,
        col=1,
    )
    fig_deviation.add_scatter(
        x=df.index[post],
        y=benchmark.loc[post, column],
        name=f'Same-quarter mean ({pre_label}, %)',
        line=dict(color='#777777', dash='dot'),
        showlegend=row == 1,
        row=row,
        col=1,
    )
    fig_deviation.add_hline(y=0, line_width=1, line_color='#AAAAAA', row=row, col=1)
    fig_deviation.update_yaxes(title_text='% / pp', row=row, col=1)

fig_deviation.update_layout(
    template='plotly_white',
    height=1000,
    hovermode='x unified',
    title=f'Quarterly growth deviations from the same-quarter {pre_label} average',
    legend=dict(orientation='h', y=-0.08),
)
fig_deviation.write_html(OUTPUT / '04_same_quarter_growth_deviations.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_deviation.show()

# %% 7. Persistent level differences from the pre-pandemic trend
trend_gap_pct = (df[list(labels)] / trend - 1) * 100

fig_gap = go.Figure()
for column, label in labels.items():
    fig_gap.add_scatter(
        x=df.index[post],
        y=trend_gap_pct.loc[post, column],
        name=label,
        line=dict(color=colors[column]),
    )

fig_gap.add_hline(y=0, line_color='#999999')
fig_gap.update_layout(
    template='plotly_white',
    height=550,
    hovermode='x unified',
    title=f'Levels relative to extrapolated {pre_label} trends',
    yaxis_title='Difference from trend (%)',
    xaxis_title='Quarter',
    legend=dict(orientation='h', y=-0.2),
)
fig_gap.write_html(OUTPUT / '05_level_deviations.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_gap.show()

# %% 8. Save transparent tables and print useful summaries
df.to_csv(OUTPUT / 'quarterly_data.csv')
decomposition.to_csv(OUTPUT / 'quarterly_growth_decomposition.csv')
yoy_decomp.to_csv(OUTPUT / 'yoy_quarterly_growth_decomposition.csv')
trend_summary.to_csv(OUTPUT / 'trend_rates.csv', index=False)
baseline.to_csv(OUTPUT / 'same_quarter_growth_baseline.csv', index_label='calendar_quarter')
baseline_n.to_csv(OUTPUT / 'same_quarter_baseline_sample_sizes.csv', index_label='calendar_quarter')
excess_growth_pp.to_csv(OUTPUT / 'quarterly_excess_growth_pp.csv')
trend_gap_pct.to_csv(OUTPUT / 'quarterly_trend_gaps_pct.csv')

print(
    '\nYoY quarterly growth and contributions (%, pp)\n',
    yoy_decomp.loc[PANDEMIC_START:, [
        'Total growth', 'Count contribution', 'Amount contribution'
    ]].round(2),
)
print(
    '\nAverage quarterly level gaps from trend (%)\n',
    trend_gap_pct.loc[post].groupby(trend_gap_pct.loc[post].index.year).mean().round(2),
)
print('\nHTML figures and CSV tables saved to:', OUTPUT)

# %% 9. Check how much the trend benchmark depends on the starting year
sensitivity_rows = []
for start in [df.index.min(), pd.Timestamp(PRE_START)]:
    fit_period = (df.index >= start) & (df.index <= PRE_END)
    for column, label in labels.items():
        slope, intercept = np.polyfit(t[fit_period], np.log(df.loc[fit_period, column]), 1)
        gaps = (df[column] / np.exp(intercept + slope * t) - 1) * 100
        sensitivity_rows.append({
            'start': str(start.to_period('Q')),
            'end': str(PRE_END.to_period('Q')),
            'series': label,
            'annualized_trend_pct': 100 * np.expm1(4 * slope),
            'mean_2023_level_gap_pct': gaps.loc['2023'].mean(),
        })

sensitivity = pd.DataFrame(sensitivity_rows).drop_duplicates()
sensitivity.to_csv(OUTPUT / 'trend_baseline_sensitivity.csv', index=False)
print('\nTrend-baseline sensitivity\n', sensitivity.round(2).to_string(index=False))