# %% 1. Imports and settings (run cells in order, or run the whole file)
"""Minimal USA -> Mexico model based on the supplied methods section.

Run: python mexico/simulate_remittances.py
Use # %% cells in VS Code/Spyder, or paste each section into a Python console.
All coefficients and income below are illustrative, NOT calibrated estimates.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

# Works as a script, or in a console opened in the project or mexico folder.
if "__file__" in globals():
    folder = Path(__file__).resolve().parent
else:
    folder = Path.cwd() if (Path.cwd() / "data/mexico-migrants-usa-detail.pkl").exists() else Path.cwd() / "mexico"

seed = 42

alpha = 0.02                       # constant
beta_surplus = 1.1                  # param_nta
beta_asymmetry = -4.654337290049305  # param_asy
beta_gdp_gap = 2.836343162643659     # param_gdp
beta_origin_income = -3.6797532393725803  # param_inc
rho = 0.18                         # rem_pct

# Disaster-response parameters (not yet used by the baseline script)
height = 0.15481668467510104
shape = 0.18889740060639804
shift = -0.9813255797340747

monthly_income_usd = 4000.0  # Still a placeholder; not included in params.

# %% 2. Load the existing monthly age-group population
population = pd.read_pickle(folder / "data/mexico-migrants-usa-detail.pkl")
population["mean_age"] = pd.to_numeric(population.mean_age)
population["date"] = pd.to_datetime(population.date).dt.to_period("M").dt.to_timestamp()
population = population.sort_values(["date", "mean_age"]).reset_index(drop=True)
numeric = ["mean_age", "n_people", "nta", "asymmetry", "gdp_diff_norm", "gdp_origin_norm"]
dates = pd.DatetimeIndex(population.date.unique())

## total population
df_pop = population.groupby("date")["n_people"].sum()
fig = px.line(df_pop)
fig.show()


# Uniform single-year ages within each five-year bin: 4/5 of ages 15-19
# are eligible (16-19). Other cohorts are wholly eligible or ineligible.
age_bounds = population.age_group.str.split("-", expand=True).astype(int)
eligible_share = ((age_bounds[1] - np.maximum(age_bounds[0], 16) + 1)
                  / (age_bounds[1] - age_bounds[0] + 1)).clip(0, 1)
population["eligible_people"] = np.rint(population.n_people * eligible_share).astype(int)


# %% 3. Monthly inputs: edit these before running the next cell
inputs = pd.DataFrame(index=dates)
inputs.index.name = "date"
inputs["monthly_income_usd"] = monthly_income_usd
inputs["shock_log_odds"] = 0.0  # No disaster data supplied: baseline only.


# %% 4. Sending probabilities and binomial decisions for each age group
agents = population.join(inputs, on="date", validate="many_to_one")
if agents[["monthly_income_usd", "shock_log_odds"]].isna().any().any():
    raise ValueError("Monthly inputs must cover every population month.")
# Use supplied normalized covariates unchanged. The preprocessing of
# gdp_diff_norm is undocumented; do not infer US GDP from relative_diff.
theta_factors = {
    "theta_constant": "Constant",
    "theta_nta": "NTA surplus",
    "theta_asymmetry": "Demographic asymmetry",
    "theta_gdp_gap": "GDP gap",
    "theta_origin_income": "Origin income",
    "theta_shock": "Shock",
}
agents["theta_constant"] = alpha
agents["theta_nta"] = beta_surplus * agents.nta
agents["theta_asymmetry"] = beta_asymmetry * agents.asymmetry
agents["theta_gdp_gap"] = beta_gdp_gap * agents.gdp_diff_norm
agents["theta_origin_income"] = beta_origin_income * agents.gdp_origin_norm
agents["theta_shock"] = agents.shock_log_odds
agents["theta"] = agents[list(theta_factors)].sum(axis=1)
agents["p_send"] = np.exp(-np.logaddexp(0, -agents.theta))
agents.loc[agents.nta <= 0, "p_send"] = 0.0
agents["expected_senders"] = agents.eligible_people * agents.p_send
rng = np.random.default_rng(seed)
agents["simulated_senders"] = rng.binomial(agents.eligible_people, agents.p_send)
# A binomial draw sums independent Bernoulli decisions without allocating
# millions of individual records. Each sender makes one transfer per month.
agents["amount_per_sender_usd"] = rho * agents.monthly_income_usd
agents["expected_usd"] = agents.expected_senders * agents.amount_per_sender_usd
agents["simulated_usd"] = agents.simulated_senders * agents.amount_per_sender_usd


# %% 5. Aggregate the USA -> Mexico monthly flow and save inspectable results
monthly = agents.groupby("date")[[
    "n_people", "eligible_people", "expected_senders", "simulated_senders",
    "expected_usd", "simulated_usd",
]].sum()
monthly["expected_usd_million"] = monthly.expected_usd / 1e6
monthly["simulated_usd_million"] = monthly.simulated_usd / 1e6
output = folder / "outputs"
output.mkdir(exist_ok=True)
monthly.to_csv(output / "simulation_monthly.csv")
agents.to_csv(output / "simulation_age_groups.csv", index=False)
print("Illustrative, uncalibrated simulation; no inflation adjustment.")
print(monthly[["expected_usd_million", "simulated_usd_million"]].head())


# %% 6. Interactive figure (self-contained HTML)
fig = go.Figure()
for column, label in [("expected_usd_million", "Expected flow"),
                      ("simulated_usd_million", "One stochastic simulation")]:
    fig.add_scatter(x=monthly.index, y=monthly[column], name=label, mode="lines",
                    hovertemplate="%{x|%b %Y}: %{y:,.2f} million USD<extra>%{fullData.name}</extra>")
fig.update_layout(
    title="USA to Mexico remittances — illustrative, uncalibrated baseline",
    template="plotly_white", yaxis_title="Monthly flow (million USD; assumed income)",
    xaxis_title="Month", hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(orientation="h", y=1.1),
)
fig.write_html(output / "simulation_remittances.html", include_plotlyjs=True)
# fig.show()  # Uncomment to open the figure from your console.


# %% 7. Theta decomposition over time (rerun after changing model inputs)
# Average over potential senders: age-eligible people with positive NTA.
# These contributions add in log-odds space, not in dollars or probabilities.
weights = agents.eligible_people.where(agents.nta > 0, 0)
weight_totals = weights.groupby(agents.date).sum()
theta_monthly = (
    agents[list(theta_factors) + ["theta"]].mul(weights, axis=0)
    .groupby(agents.date).sum()
    .div(weight_totals.replace(0, np.nan), axis=0)
)
theta_monthly["potential_senders"] = weight_totals
theta_monthly.to_csv(output / "simulation_theta_monthly.csv")

theta_fig = go.Figure()
for column, label in theta_factors.items():
    theta_fig.add_scatter(
        x=theta_monthly.index, y=theta_monthly[column], name=label, mode="lines",
        hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>%{fullData.name}</extra>",
    )
theta_fig.add_scatter(
    x=theta_monthly.index, y=theta_monthly.theta, name="Total theta",
    mode="lines", line=dict(color="black", width=3, dash="dash"),
    hovertemplate="%{x|%b %Y}: %{y:.3f}<extra>Total theta</extra>",
)
theta_fig.add_hline(y=0, line_color="gray", line_width=1)
theta_fig.update_layout(
    title="Theta contributions — weighted by potential senders",
    template="plotly_white", yaxis_title="Contribution to theta (log-odds)",
    xaxis_title="Month", hovermode="x unified",
    xaxis=dict(rangeslider=dict(visible=True)),
    legend=dict(orientation="h", y=1.15),
    margin=dict(t=130),
)
theta_fig.write_html(output / "simulation_theta.html", include_plotlyjs=True)
# theta_fig.show()
# Cohort-level contributions are also in agents and simulation_age_groups.csv.
# Logistic(mean theta) is NOT the population's mean sending probability.
