"""
Simple income-based simulation with optional COVID, unemployment, persistence and exchange-rate channels.
Run the whole script, or run its numbered sections from the console.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

income_rng = np.random.default_rng(42)
n_agents = 1000
n_quintiles = 5
if n_agents <= 0 or n_agents % n_quintiles:
    raise ValueError("n_agents must be positive and divisible by five.")
agents_per_quintile = n_agents // n_quintiles  # 200 agents per quintile

# --- Remittance assumptions and channel switches ---
remittance_probability = 0.40  # Baseline probability, before channel effects
pct_earnings_remitted = 0.20   # Share of income sent, conditional on sending
use_covid = True             # Set True to activate the Mexico COVID channel
use_unemployment = True      # Set False to give every agent earnings each month
use_persistence = False       # Previous-month senders receive a log-odds boost
beta_persistence = 0.7      # Illustrative log-odds boost for previous-month senders
use_exchange_rate = True     # Exchange rate affects sending probability only
beta_exchange_rate = 0.2    # Illustrative log-odds effect per 10% above Jan 2019
run_all_versions = True      # Run the final eight-model comparison as well
beta_covid = 0.1             # Illustrative, NOT estimated: effect on log-odds
                             # per 1,000 monthly new infections per million

if not 0 <= remittance_probability <= 1:
    raise ValueError("remittance_probability must be between zero and one.")
if not 0 <= pct_earnings_remitted <= 1:
    raise ValueError("pct_earnings_remitted must be between zero and one.")

# 1. Define the target share of aggregate income for each quintile
quintile_shares = {
    'Q1': 0.030,
    'Q2': 0.082,
    'Q3': 0.139,
    'Q4': 0.225,
    'Q5': 0.524,
}

# 2. load dataframe of monthly median earnings
monthly_df = pd.read_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\monthly-median-earnings-latinx.parquet")
monthly_df['date'] = pd.to_datetime(monthly_df['date']).dt.to_period('M').dt.to_timestamp()
monthly_df = monthly_df.sort_values('date').reset_index(drop=True)
if monthly_df['date'].duplicated().any():
    raise ValueError("Earnings data must have one observation per month.")

# Unemployment rates are already fractions (0.05 means 5%) in this file.
# Treat the simulated agents as labor-force participants. Unemployment affects
# earnings; it does not add a term to the sending-probability equation.
if use_unemployment or run_all_versions:
    df_emp = pd.read_parquet("C:/git-projects/covid-remittances/oct-26/data-prep/outputs/monthly-unemployment-latinx.parquet")
    df_emp = df_emp[['date', 'unemployment_hispanic']].copy()
    df_emp['date'] = pd.to_datetime(df_emp['date']).dt.to_period('M').dt.to_timestamp()
    monthly_df = monthly_df.merge(df_emp, on='date', how='left', validate='one_to_one')
    if not monthly_df['unemployment_hispanic'].between(0, 1).all():
        raise ValueError("Each simulation month needs an unemployment rate between zero and one.")

# COVID data uses month-end dates; match it to earnings by calendar month.
# Load data when needed by the selected model or the final comparison.
if use_covid or run_all_versions:
    covid_df = pd.read_parquet("C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\america-covid-monthly.parquet")
    df_mex = covid_df.loc[covid_df['country'].eq('Mexico'),
                          ['date', 'new_cases_per_million']].copy()
    df_mex['date'] = pd.to_datetime(df_mex['date']).dt.to_period('M').dt.to_timestamp()
    monthly_df = monthly_df.merge(df_mex, on='date', how='left', validate='one_to_one')
    # Explicit assumption: no COVID effect before 2020. Do not fill later gaps.
    monthly_df.loc[monthly_df['date'] < '2020-01-01', 'new_cases_per_million'] = 0.0
    if monthly_df['new_cases_per_million'].isna().any():
        raise ValueError("Missing Mexico COVID data from 2020 onward.")
    # The source preparation should preserve missing reports rather than sum
    # all-missing months to zero (use sum(min_count=1) in data preparation).

# Exchange rate: Mexican pesos per US dollar. Align month-end observations.
# Use the deviation from January 2019, NOT the month-to-month growth rate.
# A value of 1 in exchange_rate_change means 10% more pesos per dollar.
if use_exchange_rate or run_all_versions:
    df_exchange = pd.read_parquet("C:/git-projects/covid-remittances/oct-26/data-prep/outputs/dollar-peso-exchange-rate.parquet")
    df_exchange = df_exchange[['date', 'ex_rate']].copy()
    df_exchange['date'] = pd.to_datetime(df_exchange['date']).dt.to_period('M').dt.to_timestamp()
    monthly_df = monthly_df.merge(df_exchange, on='date', how='left', validate='one_to_one')
    if not (np.isfinite(monthly_df['ex_rate']) & monthly_df['ex_rate'].gt(0)).all():
        raise ValueError("Each simulation month needs a finite, positive exchange rate.")
    exchange_base = monthly_df.loc[monthly_df['date'].eq(pd.Timestamp('2019-01-01')), 'ex_rate']
    if len(exchange_base) != 1:
        raise ValueError("January 2019 is required as the exchange-rate reference.")
    monthly_df['exchange_rate_change'] = (monthly_df['ex_rate'] / exchange_base.iloc[0] - 1) / 0.10

# --- Step 1: Create Static Agent Profiles ---
agent_ids = np.arange(1, n_agents + 1)
quintiles = []
for q in quintile_shares.keys():
    quintiles.extend([q] * agents_per_quintile)

agents_df = pd.DataFrame({'agent_id': agent_ids, 'quintile': quintiles})

# --- Step 2: Assign Fixed Relative Incomes ---
# Generate random variations within each quintile to simulate intra-quintile dispersion,
# then normalize them so each group captures its exact target income share.
agent_weights = np.zeros(n_agents)
start_idx = 0

for q, share in quintile_shares.items():
  end_idx = start_idx + agents_per_quintile
  # Random distribution shape within the quintile (e.g., uniform spread)
  raw_draws = income_rng.uniform(0.8, 1.2, size=agents_per_quintile)
  # Scale weights to match the quintile's share of total income
  agent_weights[start_idx:end_idx] = raw_draws * (share / raw_draws.sum())
  start_idx = end_idx

# Multiplying these by each month's median earnings reproduces the original
# base-income * growth-factor calculation, with fewer intermediate variables.
agents_df['income_multiplier'] = agent_weights / np.median(agent_weights)

# Reuse these same two steps for the selected model and all eight versions.
def run_model(use_covid, use_unemployment, persistence_on=None, exchange_rate_on=None):
    if exchange_rate_on is None:
        exchange_rate_on = use_exchange_rate
    if exchange_rate_on and not np.isfinite(beta_exchange_rate):
        raise ValueError("beta_exchange_rate must be finite.")
    if persistence_on is None:
        persistence_on = use_persistence
    if persistence_on:
        if not np.isfinite(beta_persistence) or beta_persistence < 0:
            raise ValueError("beta_persistence must be finite and nonnegative.")
        months = pd.DatetimeIndex(monthly_df['date']).to_period('M')
        if not months.equals(pd.period_range(months.min(), months.max(), freq='M')):
            raise ValueError("Persistence requires consecutive, ordered months.")
    # --- Step 3: Evolve Income Over Time ---
    # Expand agent profiles across all months using a cross-join style loop
    employment_rng = np.random.default_rng(456)  # Independent of income/sending draws
    panel_data = []
    for _, row in monthly_df.iterrows():
      df_t = agents_df.copy()
      df_t['date'] = row['date']
      df_t['monthly_income_nominal'] = df_t['income_multiplier'] * row['monthly_earnings_nominal']
      df_t['is_employed'] = True
      if use_unemployment:
          df_t['unemployment_hispanic'] = row['unemployment_hispanic']
          # Fix the unemployed count, rather than draw a Bernoulli for each agent.
          # Round to the nearest agent; randomly reassign unemployment each month,
          # independently of income quintile and previous employment status.
          n_unemployed = int(np.floor(n_agents * row['unemployment_hispanic'] + 0.5))
          unemployed_indices = employment_rng.choice(df_t.index, size=n_unemployed, replace=False)
          df_t.loc[unemployed_indices, 'is_employed'] = False
          df_t.loc[unemployed_indices, 'monthly_income_nominal'] = 0.0
      if use_covid:
          df_t['new_cases_per_million'] = row['new_cases_per_million']
      if exchange_rate_on:
          df_t['ex_rate'] = row['ex_rate']
          df_t['exchange_rate_change'] = row['exchange_rate_change']
      panel_data.append(df_t)

    population_panel = pd.concat(panel_data, ignore_index=True)

    # Clean up final columns
    population_panel = population_panel.drop(columns='income_multiplier')

    # --- Step 4: Add Remittance Behavior ---
    # Each enabled probability channel adds one column of contributions to theta.
    # Add future channels as another `if use_...` block here and append their column
    # names. The probability calculation and sending decision below stay the same.
    probability_channels = []
    if use_covid:
        population_panel['theta_covid'] = beta_covid * population_panel['new_cases_per_million'] / 1000
        probability_channels.append('theta_covid')

    if exchange_rate_on:
        population_panel['theta_exchange_rate'] = beta_exchange_rate * population_panel['exchange_rate_change']
        probability_channels.append('theta_exchange_rate')

    # Persistence requires simulating one month at a time: this month's
    # realised transfer determines next month's boost for the SAME agent.
    # Initial condition: no one is assumed to have sent before the first month.
    sending_rng = np.random.default_rng(123)
    sent_last_month = pd.Series(False, index=agents_df['agent_id'])
    use_logistic = bool(probability_channels) or persistence_on
    if use_logistic:
        if not 0 < remittance_probability < 1:
            raise ValueError("Enabled probability channels require a baseline strictly between 0 and 1.")
        baseline_theta = np.log(remittance_probability / (1 - remittance_probability))

    remittance_panels = []
    for date, df_t in population_panel.groupby('date', sort=True):
        df_t = df_t.copy()
        df_t['sent_last_month'] = df_t['agent_id'].map(sent_last_month)
        df_t['theta_persistence'] = (
            beta_persistence * df_t['sent_last_month'] if persistence_on else 0.0
        )
        df_t['sending_probability'] = remittance_probability
        if use_logistic:
            df_t['theta'] = (
                baseline_theta + df_t[probability_channels].sum(axis=1)
                + df_t['theta_persistence']
            )
            df_t['sending_probability'] = np.exp(-np.logaddexp(0, -df_t['theta']))

        # Probability is conditional on having earnings. Unemployed agents
        # cannot remit; they also lose the persistence boost next month.
        df_t['sends_remittance'] = (
            (sending_rng.random(len(df_t)) < df_t['sending_probability'])
            & df_t['monthly_income_nominal'].gt(0)
            & (pct_earnings_remitted > 0)
        )
        df_t['remittance_amount'] = np.where(
            df_t['sends_remittance'],
            pct_earnings_remitted * df_t['monthly_income_nominal'], 0.0
        )
        sent_last_month = df_t.set_index('agent_id')['remittance_amount'].gt(0)
        remittance_panels.append(df_t)

    population_panel = pd.concat(remittance_panels, ignore_index=True)

    aggregate_remittances = population_panel.groupby('date')['remittance_amount'].sum().reset_index()
    return population_panel, aggregate_remittances


# Run the selected model; its results remain available in the console.
population_panel, aggregate_remittances = run_model(use_covid, use_unemployment)

print(population_panel.head(10))
print("\nTotal Remittances per Month:")
print(aggregate_remittances.head())

## PLOT

# --- 1. Load Actual Remittances from Parquet ---
parquet_path = "C:\\git-projects\\covid-remittances\\oct-26\\data-prep\\outputs\\mexico-rem-monthly-total.parquet"
df_actual = pd.read_parquet(parquet_path)
df_actual['date'] = pd.to_datetime(df_actual['date']).dt.to_period('M').dt.to_timestamp()
df_actual = df_actual.sort_values('date')

# --- 2. Compute Total Simulated Remittances from Agent Panel ---
df_simulated = aggregate_remittances.copy()
df_simulated.rename(columns={'remittance_amount': 'simulated_remittances'}, inplace=True)

# Merge both time series and filter for 2019-2023
df_merged = pd.merge(df_actual, df_simulated, on='date', how='inner', validate='one_to_one')
df_merged = df_merged[(df_merged['date'].dt.year.between(2019, 2023))].copy()

# --- 3. Normalize Both to an Index (Jan 2019 = 100) for a Shared Y-Axis ---
# This lets you compare growth and trends side-by-side without scale distortion
base_month = df_merged.loc[df_merged['date'].eq(pd.Timestamp('2019-01-01'))]
if base_month.empty:
    raise ValueError("January 2019 is required for the comparison index.")
base_actual = base_month['total_mln'].iloc[0]
base_simulated = base_month['simulated_remittances'].iloc[0]
if min(base_actual, base_simulated) <= 0:
    raise ValueError("Cannot index remittances with a zero or negative base value.")

df_merged['actual_indexed'] = (df_merged['total_mln'] / base_actual) * 100
df_merged['simulated_indexed'] = (df_merged['simulated_remittances'] / base_simulated) * 100

# ==========================================
# PLOT 2: Probability of Sending Over Time
# ==========================================
# df_prob = population_panel[population_panel['agent_id'] == 1][['date', 'sending_probability']].copy()
# df_prob = df_prob.sort_values('date')
# df_prob = df_prob[df_prob['date'].dt.year.between(2019, 2023)]
#
# fig2 = go.Figure()
#
# fig2.add_trace(go.Scatter(
#     x=df_prob['date'],
#     y=df_prob['sending_probability'],
#     mode='lines',
#     name='Probability of Sending Given Employment',
#     line=dict(color='firebrick', width=2)
# ))
#
# fig2.update_layout(
#     title_text=f"Remittance Probability Given Employment ({scenario})",
#     xaxis_title="Date",
#     yaxis_title="Probability (0 to 1)",
#     yaxis=dict(range=[0, 1.05]),
#     legend=dict(x=0.01, y=0.99)
# )
#
# output_svg_path = f"C:/git-projects/covid-remittances/oct-26/outputs/simulations/remittance_probability_evolution-{scenario}.svg"
# fig2.write_image(output_svg_path)
# fig2.show()


# ==========================================
# FINAL COMPARISON: All Eight Model Versions
# ==========================================
# All runs share agent incomes and random seeds. Unemployment assignments are
# identical in the two unemployment models; sending draws match in all eight.
if run_all_versions:
    persistence_label = 'persistence on' if use_persistence else 'persistence off'
    # The persistence switch applies to every version in this comparison.
    model_versions = {
        'Income only': (False, False, False),
        'Income + COVID': (True, False, False),
        'Income + unemployment': (False, True, False),
        'Income + unemployment + COVID': (True, True, False),
        'Income + exchange rate': (False, False, True),
        'Income + COVID + exchange rate': (True, False, True),
        'Income + unemployment + exchange rate': (False, True, True),
        'Income + unemployment + COVID + exchange rate': (True, True, True),
    }
    comparison_totals = {}
    comparison_sender_percentages = {}
    df_comparison = df_actual.loc[
        df_actual['date'].dt.year.between(2019, 2023), ['date', 'total_mln']
    ].rename(columns={'total_mln': 'Realised remittances'})

    for model_name, (covid_on, unemployment_on, exchange_on) in model_versions.items():
        model_panel, model_totals = run_model(covid_on, unemployment_on, exchange_rate_on=exchange_on)
        comparison_totals[model_name] = model_totals
        # Count positive transfers among ALL migrants, including unemployed
        # agents in the denominator. These are realised simulation outcomes.
        comparison_sender_percentages[model_name] = (
            model_panel['remittance_amount'].gt(0)
            .groupby(model_panel['date']).mean().mul(100)
        )
        df_comparison = df_comparison.merge(
            model_totals.rename(columns={'remittance_amount': model_name}),
            on='date', how='inner', validate='one_to_one'
        )

    # Index each series to its own January 2019 value: simulated totals for
    # 1,000 agents and realised national totals have different scales/units.
    df_comparison = df_comparison.sort_values('date').set_index('date')
    if pd.Timestamp('2019-01-01') not in df_comparison.index:
        raise ValueError('January 2019 is required for the comparison index.')
    comparison_base = df_comparison.loc['2019-01-01']
    if not np.isfinite(comparison_base).all() or (comparison_base <= 0).any():
        raise ValueError('All comparison series need a positive January 2019 value.')
    df_comparison_indexed = df_comparison.div(comparison_base).mul(100)

    fig_comparison = go.Figure()
    colors = ['black', 'steelblue', 'darkorange', 'forestgreen', 'purple', 'deepskyblue', 'coral', 'olive', 'deeppink']
    for column, color in zip(df_comparison_indexed.columns, colors):
        fig_comparison.add_trace(go.Scatter(
            x=df_comparison_indexed.index,
            y=df_comparison_indexed[column],
            mode='lines', name=column,
            line=dict(color=color, width=3 if column == 'Realised remittances' else 2)
        ))
    fig_comparison.update_layout(
        title_text=f'Realised Remittances vs. Eight Model Versions (Jan 2019 = 100; {persistence_label})',
        xaxis_title='Date', yaxis_title='Index (Jan 2019 = 100)',
        legend=dict(orientation='h', y=-0.2), hovermode='x unified'
    )
    fig_comparison.write_image('C:/git-projects/covid-remittances/oct-26/outputs/simulations/remittances-all-models-exchange-rate.svg')
    fig_comparison.show()

    # --- Percentage of Migrants Sending: All Eight Model Versions ---
    df_sender_percentages = pd.DataFrame(comparison_sender_percentages).sort_index()
    df_sender_percentages = df_sender_percentages.loc[
        (df_sender_percentages.index.year >= 2019)
        & (df_sender_percentages.index.year <= 2023)
    ]
    fig_senders = go.Figure()
    for model_name, color in zip(model_versions, colors[1:]):
        fig_senders.add_trace(go.Scatter(
            x=df_sender_percentages.index,
            y=df_sender_percentages[model_name],
            mode='lines', name=model_name,
            line=dict(color=color, width=2),
            hovertemplate='%{y:.1f}%<extra>%{fullData.name}</extra>'
        ))
    fig_senders.update_layout(
        title_text=f'Percentage of Migrants Sending: Eight Model Versions ({persistence_label})',
        xaxis_title='Date', yaxis_title='Migrants sending (% of all migrants)',
        yaxis=dict(range=[0, 100], ticksuffix='%'),
        legend=dict(orientation='h', y=-0.2), hovermode='x unified'
    )
    fig_senders.write_image('C:/git-projects/covid-remittances/oct-26/outputs/simulations/senders-percentage-all-models-exchange-rate.svg')
    fig_senders.show()
