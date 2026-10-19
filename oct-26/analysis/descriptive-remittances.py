"""Mexico remittances: run each numbered # %% section in the Python console.

Requires pandas, numpy, openpyxl and plotly. No model fitting beyond a straight
line in log levels. HTML figures are self-contained and work offline.
"""

# %% 1. Settings and imports
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path('C:/git-projects/covid-remittances')
INPUT = ROOT / 'data/mexico_remittances_2026.xlsx'
OUTPUT = ROOT / 'oct-26/outputs/descriptive-remittances'
OUTPUT.mkdir(parents=True, exist_ok=True)
PRE_START = '2015-01-01'  # Try 2010-01-01 to check baseline sensitivity.
PRE_END = '2019-12-01'
PANDEMIC_START = pd.Timestamp('2020-03-01')
SHOW_FIGURES = False    # HTML is always saved; True also opens browser tabs.
labels = {'total_usd': 'Total remittances', 'transactions': 'Transaction count',
          'average_usd': 'Average transfer'}
colors = {'total_usd': '#222222', 'transactions': '#277DA1', 'average_usd': '#D97706'}
pre_label = f'{PRE_START[:4]}–{PRE_END[:4]}'

# %% 2. Read the raw workbook and check the accounting identity
# Hoja1, Excel row 18: series identifiers. Row 14 gives the units.
# SE27803 = million USD; SE27808 = thousand transactions;
# SE28647 = published average USD, rounded to whole dollars.
raw = pd.read_excel(INPUT, sheet_name='Hoja1', header=17)
df = raw[['Fecha', 'SE27803', 'SE27808', 'SE28647']].copy()
df.columns = ['date', 'total_million_usd', 'transactions_thousand', 'published_average_usd']
df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
df = df.sort_values('date').set_index('date')
if df.index.has_duplicates or not df.index.equals(pd.date_range(df.index.min(), df.index.max(), freq='MS')):
    raise ValueError('The input must contain one observation for every consecutive month.')
if not np.isfinite(df.to_numpy(dtype=float)).all() or not df.gt(0).all().all():
    raise ValueError('Missing or nonpositive input: inspect the workbook before calculating growth.')
df['total_usd'] = df['total_million_usd'] * 1_000_000
df['transactions'] = df['transactions_thousand'] * 1_000
df['average_usd'] = df['total_usd'] / df['transactions']
df['average_rounding_difference_usd'] = df['average_usd'] - df['published_average_usd']
print(f'Observed data: {df.index.min():%Y-%m} to {df.index.max():%Y-%m}; {len(df)} months')
print('Maximum difference from published average (USD):', df['average_rounding_difference_usd'].abs().max())
pre = (df.index >= PRE_START) & (df.index <= PRE_END)
if pre.sum() < 24 or pd.Timestamp(PRE_END) >= PANDEMIC_START:
    raise ValueError('Choose at least two years of pre-pandemic observations.')

# %% 3. Decompose month-on-month growth in totals into count and amount effects
# R = T * A implies g_R = g_T + g_A + g_T*g_A.
# Split the interaction equally: an exact, symmetric accounting decomposition.
# Contributions are percentage points of ordinary percentage growth, not causal effects.
growth = df[list(labels)].pct_change(fill_method=None)
decomposition = pd.DataFrame(index=df.index)
decomposition['Count contribution'] = 100 * growth['transactions'] * (1 + growth['average_usd'] / 2)
decomposition['Amount contribution'] = 100 * growth['average_usd'] * (1 + growth['transactions'] / 2)
decomposition['Total growth'] = 100 * growth['total_usd']
np.testing.assert_allclose(decomposition.iloc[1:, :2].sum(axis=1), decomposition['Total growth'].iloc[1:], atol=1e-10)
fig_growth = go.Figure()
for name, color in [('Count contribution', colors['transactions']), ('Amount contribution', colors['average_usd'])]:
    fig_growth.add_bar(x=df.index, y=decomposition[name], name=name, marker_color=color)
fig_growth.add_scatter(x=df.index, y=decomposition['Total growth'], name='Total MoM growth (%)', line=dict(color='#222222'))
fig_growth.update_layout(template='plotly_white', barmode='relative', hovermode='x unified',
    title='Monthly growth: transaction-count and average-amount contributions',
    yaxis_title='Growth / contribution (%, percentage points)', xaxis_title='Month',
    legend=dict(orientation='h', y=-0.2), height=550)
fig_growth.write_html(OUTPUT / '01_monthly_growth_decomposition.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_growth.show()

# %% 4. Annual decomposition: sum flows and counts; use a weighted average amount
# Keep complete years only. A partial 2026 must not be compared with all of 2025.
annual = df.groupby(df.index.year)[['total_usd', 'transactions']].sum()
annual['months'] = df.groupby(df.index.year).size()
annual = annual.loc[annual['months'].eq(12)].copy()
annual['average_usd'] = annual['total_usd'] / annual['transactions']
annual_growth = annual[list(labels)].pct_change(fill_method=None)
annual['count_contribution_pp'] = 100 * annual_growth['transactions'] * (1 + annual_growth['average_usd'] / 2)
annual['amount_contribution_pp'] = 100 * annual_growth['average_usd'] * (1 + annual_growth['transactions'] / 2)
annual['total_growth_pct'] = 100 * annual_growth['total_usd']
fig_annual = go.Figure()
fig_annual.add_bar(x=annual.index, y=annual['count_contribution_pp'], name='Count contribution', marker_color=colors['transactions'])
fig_annual.add_bar(x=annual.index, y=annual['amount_contribution_pp'], name='Amount contribution', marker_color=colors['average_usd'])
fig_annual.add_scatter(x=annual.index, y=annual['total_growth_pct'], name='Total annual growth (%)', line=dict(color='#222222'))
fig_annual.update_layout(template='plotly_white', barmode='relative', hovermode='x unified',
    title='Annual growth decomposition — complete years only', xaxis=dict(title='Year', dtick=1),
    yaxis_title='Growth / contribution (%, percentage points)', legend=dict(orientation='h', y=-0.2), height=550)
fig_annual.write_html(OUTPUT / '02_annual_growth_decomposition.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_annual.show()

# %% 5. Pre-pandemic trend: log(level) = intercept + slope * month
# No month dummies or seasonal adjustment. Extrapolation is a descriptive
# benchmark, not an identified no-pandemic counterfactual or a forecast.
t = np.arange(len(df))
trend = pd.DataFrame(index=df.index)
trend_rows = []
fig_trend = make_subplots(rows=3, cols=1, shared_xaxes=True,
    subplot_titles=list(labels.values()), vertical_spacing=0.09)
for row, (column, label) in enumerate(labels.items(), start=1):
    slope, intercept = np.polyfit(t[pre], np.log(df.loc[pre, column]), 1)
    trend[column] = np.exp(intercept + slope * t)
    trend_rows.append({'series': label, 'monthly_trend_pct': 100 * np.expm1(slope),
                       'annualized_trend_pct': 100 * np.expm1(12 * slope)})
    # Normalize by the fitted December 2019 level, so all panels have readable units.
    anchor = trend.loc[pd.Timestamp(PRE_END), column]
    fig_trend.add_scatter(x=df.index, y=100 * df[column] / anchor, name='Observed',
        legendgroup='observed', showlegend=row == 1, line=dict(color=colors[column]), row=row, col=1)
    fig_trend.add_scatter(x=df.index[pre], y=100 * trend.loc[pre, column] / anchor, name='Fitted trend',
        legendgroup='fit', showlegend=row == 1, line=dict(color='#666666'), row=row, col=1)
    after = df.index > PRE_END
    fig_trend.add_scatter(x=df.index[after], y=100 * trend.loc[after, column] / anchor, name='Trend extrapolation',
        legendgroup='projection', showlegend=row == 1, line=dict(color='#666666', dash='dash'), row=row, col=1)
    fig_trend.update_yaxes(title_text='Index', row=row, col=1)
fig_trend.update_layout(template='plotly_white', height=950, hovermode='x unified',
    title=f'Pre-pandemic log-linear trends ({pre_label}); fitted {PRE_END[:7]} = 100',
    legend=dict(orientation='h', y=-0.08))
fig_trend.write_html(OUTPUT / '03_pre_pandemic_trends.html', include_plotlyjs=True)
trend_summary = pd.DataFrame(trend_rows)
print('\nPre-pandemic trend rates (%)\n', trend_summary.round(2).to_string(index=False))
if SHOW_FIGURES:
    fig_trend.show()

# %% 6. Monthly growth compared with the SAME month's pre-pandemic average
# Example: March 2021 vs February 2021 growth, minus the average March-vs-
# February growth in 2015–2019. This is NOT year-on-year growth.
# January baseline observations use December of the previous year.
monthly_growth_pct = growth * 100
baseline = monthly_growth_pct.loc[pre].groupby(monthly_growth_pct.loc[pre].index.month).mean()
baseline_n = monthly_growth_pct.loc[pre].groupby(monthly_growth_pct.loc[pre].index.month).count()
benchmark = pd.DataFrame(index=df.index)
for column in labels:
    benchmark[column] = df.index.month.map(baseline[column])
excess_growth_pp = monthly_growth_pct - benchmark
fig_deviation = make_subplots(rows=3, cols=1, shared_xaxes=True,
    subplot_titles=list(labels.values()), vertical_spacing=0.09)
post = df.index >= PANDEMIC_START
for row, (column, label) in enumerate(labels.items(), start=1):
    fig_deviation.add_bar(x=df.index[post], y=excess_growth_pp.loc[post, column],
        name='Growth minus same-month baseline (pp)', marker_color=colors[column],
        showlegend=row == 1, row=row, col=1)
    fig_deviation.add_scatter(x=df.index[post], y=monthly_growth_pct.loc[post, column],
        name='Observed MoM growth (%)', line=dict(color='#222222', width=1),
        showlegend=row == 1, row=row, col=1)
    fig_deviation.add_scatter(x=df.index[post], y=benchmark.loc[post, column],
        name=f'Same-month mean ({pre_label}, %)', line=dict(color='#777777', dash='dot'),
        showlegend=row == 1, row=row, col=1)
    fig_deviation.add_hline(y=0, line_width=1, line_color='#AAAAAA', row=row, col=1)
    fig_deviation.update_yaxes(title_text='% / pp', row=row, col=1)
fig_deviation.update_layout(template='plotly_white', height=1000, hovermode='x unified',
    title=f'Monthly growth deviations from the same-month {pre_label} average',
    legend=dict(orientation='h', y=-0.08))
fig_deviation.write_html(OUTPUT / '04_same_month_growth_deviations.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_deviation.show()

# %% 7. Persistent level differences from the pre-pandemic trend
# Growth can return to its historical norm while LEVELS remain elevated.
# This chart complements, rather than replaces, the requested growth comparison.
trend_gap_pct = (df[list(labels)] / trend - 1) * 100
fig_gap = go.Figure()
for column, label in labels.items():
    fig_gap.add_scatter(x=df.index[post], y=trend_gap_pct.loc[post, column],
        name=label, line=dict(color=colors[column]))
fig_gap.add_hline(y=0, line_color='#999999')
fig_gap.update_layout(template='plotly_white', height=550, hovermode='x unified',
    title=f'Levels relative to extrapolated {pre_label} trends',
    yaxis_title='Difference from trend (%)', xaxis_title='Month', legend=dict(orientation='h', y=-0.2))
fig_gap.write_html(OUTPUT / '05_level_deviations.html', include_plotlyjs=True)
if SHOW_FIGURES:
    fig_gap.show()

# %% 8. Save transparent tables and print useful summaries
df.to_csv(OUTPUT / 'monthly_data.csv')
decomposition.to_csv(OUTPUT / 'monthly_growth_decomposition.csv')
annual.to_csv(OUTPUT / 'annual_summary.csv', index_label='year')
trend_summary.to_csv(OUTPUT / 'trend_rates.csv', index=False)
baseline.to_csv(OUTPUT / 'same_month_growth_baseline.csv', index_label='calendar_month')
baseline_n.to_csv(OUTPUT / 'same_month_baseline_sample_sizes.csv', index_label='calendar_month')
excess_growth_pp.to_csv(OUTPUT / 'monthly_excess_growth_pp.csv')
trend_gap_pct.to_csv(OUTPUT / 'monthly_trend_gaps_pct.csv')
print('\nAnnual growth and contributions (%, pp)\n', annual.loc[2019:, [
    'total_growth_pct', 'count_contribution_pp', 'amount_contribution_pp']].round(2))
print('\nAverage monthly level gaps from trend (%)\n', trend_gap_pct.loc[post].groupby(
    trend_gap_pct.loc[post].index.year).mean().round(2))
print('\nHTML figures and CSV tables saved to:', OUTPUT)

# %% 9. Check how much the trend benchmark depends on the starting year
# This does not change the selected baseline or the charts above.
sensitivity_rows = []
for start in [df.index.min(), pd.Timestamp(PRE_START)]:
    fit_period = (df.index >= start) & (df.index <= PRE_END)
    for column, label in labels.items():
        slope, intercept = np.polyfit(t[fit_period], np.log(df.loc[fit_period, column]), 1)
        gaps = (df[column] / np.exp(intercept + slope * t) - 1) * 100
        sensitivity_rows.append({'start': start.strftime('%Y-%m'), 'end': PRE_END[:7],
            'series': label, 'annualized_trend_pct': 100 * np.expm1(12 * slope),
            'mean_2023_level_gap_pct': gaps.loc['2023'].mean()})
sensitivity = pd.DataFrame(sensitivity_rows).drop_duplicates()
sensitivity.to_csv(OUTPUT / 'trend_baseline_sensitivity.csv', index=False)
print('\nTrend-baseline sensitivity\n', sensitivity.round(2).to_string(index=False))
