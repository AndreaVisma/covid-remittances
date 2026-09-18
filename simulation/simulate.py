"""Illustrative monthly migrant microsimulation; no external data required."""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STATES = np.array(["employed", "unemployed", "inactive"])
SCENARIOS = ("baseline", "no_us_support", "no_mexico_support", "no_altruistic_response", "no_pandemic")


def validate(p):
    """Reject invalid probabilities, dates, and overlapping shock periods."""
    for key in ("population", "seed"):
        if not isinstance(p[key], int) or isinstance(p[key], bool) or p[key] < (1 if key == "population" else 0):
            raise ValueError(f"{key} must be a valid nonnegative integer (population > 0)")
    if pd.Period(p["start"], "M") > pd.Period(p["end"], "M"):
        raise ValueError("start must precede end")
    for key, shape in (("initial_state_shares", (3,)), ("transition_normal", (3, 3)), ("transition_crisis", (3, 3))):
        a = np.asarray(p[key], dtype=float)
        if a.shape != shape or not np.isfinite(a).all() or (a < 0).any() or not np.allclose(a.sum(axis=-1), 1):
            raise ValueError(f"{key} must contain nonnegative probabilities summing to one")
    fractions = ("zero_savings_share", "fiscal_eligible_share", "remit_income_share", "remit_savings_share")
    for key in fractions:
        if not 0 <= p[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not 0 < p["remit_probability"] < 1:
        raise ValueError("remit_probability must be strictly between zero and one")
    for key, value in p.items():
        if isinstance(value, (float, int)) and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("earnings_median", "consumption_median", "savings_median"):
        if p[key] <= 0:
            raise ValueError(f"{key} must be positive")
    occupied = set()
    for phase in p["pandemic"]:
        start, end = pd.Period(phase["start"], "M"), pd.Period(phase["end"], "M")
        if start > end:
            raise ValueError("Pandemic phase start must precede end")
        months = set(pd.period_range(start, end, freq="M"))
        if occupied & months:
            raise ValueError("Pandemic phases cannot overlap")
        occupied |= months
        for key in ("labour_shock", "hours_loss", "mexico_need", "mexico_support", "ui_monthly"):
            if not np.isfinite(phase[key]) or phase[key] < 0:
                raise ValueError(f"Invalid phase value: {key}")
        if phase["labour_shock"] > 1 or phase["hours_loss"] > 1:
            raise ValueError("labour_shock and hours_loss must be in [0, 1]")
    for month, amount in p["stimulus"].items():
        pd.Period(month, "M")
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("Stimulus must be finite and nonnegative")


def monthly_inputs(p):
    dates = pd.period_range(p["start"], p["end"], freq="M").to_timestamp()
    inputs = pd.DataFrame(0.0, index=dates, columns=["labour_shock", "hours_loss", "mexico_need", "mexico_support", "ui_monthly", "stimulus"])
    for phase in p["pandemic"]:
        mask = (dates >= pd.Timestamp(phase["start"])) & (dates <= pd.Timestamp(phase["end"]))
        for key in inputs.columns.drop("stimulus"):
            inputs.loc[mask, key] = phase[key]
    for month, amount in p["stimulus"].items():
        if pd.Timestamp(month) in inputs.index:
            inputs.loc[pd.Timestamp(month), "stimulus"] = amount
    inputs.index.name = "date"
    return inputs


def simulate(p, scenario="baseline"):
    """Return agent-month records and the effective exogenous input path.

    Identical seeds use the same initial population and random draws across
    scenarios, so differences are not caused by independent simulation noise.
    """
    validate(p)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(p["seed"])
    n = p["population"]
    state = rng.choice(3, n, p=p["initial_state_shares"])
    wage = rng.lognormal(np.log(p["earnings_median"]), p["earnings_log_sd"], n)
    essentials = rng.lognormal(np.log(p["consumption_median"]), p["consumption_log_sd"], n)
    savings = rng.lognormal(np.log(p["savings_median"]), p["savings_log_sd"], n)
    savings[rng.random(n) < p["zero_savings_share"]] = 0
    eligible = rng.random(n) < p["fiscal_eligible_share"]
    ties = rng.normal(0, p["family_ties_log_sd"], n)
    inputs = monthly_inputs(p)
    if scenario == "no_us_support":
        inputs[["ui_monthly", "stimulus"]] = 0
    elif scenario == "no_mexico_support":
        inputs["mexico_support"] = 0
    elif scenario == "no_pandemic":
        inputs.loc[:, :] = 0
    frames = []
    for t, (date, x) in enumerate(inputs.iterrows()):
        # Draw even in the initial month to keep the random stream aligned.
        transition_draw, remit_draw = rng.random((2, n))
        if t:
            matrix = (1 - x.labour_shock) * np.array(p["transition_normal"]) + x.labour_shock * np.array(p["transition_crisis"])
            state = (transition_draw[:, None] > matrix[state].cumsum(axis=1)[:, :2]).sum(axis=1)
        opening = savings.copy()
        income = wage * (state == 0) * (1 - x.hours_loss)
        transfers = eligible * (x.stimulus + x.ui_monthly * (state == 1))
        resources = opening + income + transfers
        consumption = np.minimum(essentials, resources)
        available = np.maximum(resources - consumption - p["reserve_months"] * essentials, 0)
        need = max(x.mexico_need - p["mexico_support_effect"] * x.mexico_support, 0)
        if scenario == "no_altruistic_response":
            need = 0.0
        logit = np.log(p["remit_probability"] / (1 - p["remit_probability"])) + ties + p["need_log_odds"] * need
        probability = np.exp(-np.logaddexp(0, -logit))
        willing = remit_draw < probability
        target = (p["remit_income_share"] * (income + transfers) + p["remit_savings_share"] * opening) * (1 + p["need_amount_effect"] * need)
        desired = willing * target
        remittances = np.minimum(desired, available)
        savings = resources - consumption - remittances
        frame = pd.DataFrame({
            "date": date, "scenario": scenario, "agent_id": np.arange(n),
            "state": STATES[state], "fiscal_eligible": eligible,
            "potential_earnings": wage, "essential_consumption": essentials,
            "family_ties": ties, "opening_savings": opening, "income": income,
            "us_transfers": transfers, "consumption": consumption,
            "consumption_shortfall": essentials - consumption,
            "remit_probability": probability, "willing": willing,
            "desired_remittances": desired, "remittances": remittances,
            "unmet_remittances": desired - remittances, "closing_savings": savings,
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), inputs


def aggregate(agents):
    flows = ["income", "us_transfers", "consumption", "consumption_shortfall", "desired_remittances", "remittances", "unmet_remittances", "opening_savings", "closing_savings"]
    rows = []
    for (scenario, date), g in agents.groupby(["scenario", "date"], sort=False):
        row = {"scenario": scenario, "date": date, **g[flows].sum().to_dict()}
        senders = (g.remittances > 0).sum()
        row.update(senders=senders, participation=senders / len(g),
                   mean_per_sender=g.remittances.sum() / senders if senders else 0,
                   median_savings=g.closing_savings.median(),
                   depleted_share=(g.closing_savings < 1e-8).mean(),
                   consumption_shortfall_share=(g.consumption_shortfall > 1e-8).mean(),
                   savings_change=(g.closing_savings - g.opening_savings).sum())
        for state in STATES:
            row[f"{state}_share"] = (g.state == state).mean()
            row[f"{state}_remittances"] = g.loc[g.state == state, "remittances"].sum()
        rows.append(row)
    return pd.DataFrame(rows)


def figures(monthly, out):
    b = monthly[monthly.scenario == "baseline"].set_index("date")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True, constrained_layout=True)
    specs = [
        (["remittances", "desired_remittances"], "Remittance flows", "USD / month", 1),
        (["participation"], "Actual sender participation", "% of migrants", 100),
        (["mean_per_sender"], "Amount per actual sender", "USD / month", 1),
        ([f"{s}_share" for s in STATES], "Labour force composition", "% of migrants", 100),
        (["median_savings"], "Median closing liquid savings", "USD", 1),
        (["depleted_share", "consumption_shortfall_share"], "Financial strain", "% of migrants", 100),
    ]
    for ax, (columns, title, unit, scale) in zip(axes.flat, specs):
        for col in columns:
            ax.plot(b.index, b[col] * scale, label=col.replace("_", " "))
        ax.set(title=title, ylabel=unit)
        ax.legend(fontsize=8)
        ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle="--", linewidth=1)
    fig.suptitle("Illustrative migrant microsimulation — baseline (uncalibrated)")
    fig.savefig(out / "01_dashboard.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for col in ("income", "us_transfers", "consumption", "remittances", "savings_change"):
        axes[0].plot(b.index, b[col] / 1000, label=col.replace("_", " "))
    axes[0].set(title="Monthly resource flows: income + transfers = consumption + remittances + savings change", ylabel="Thousand USD / month")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].stackplot(b.index, *[b[f"{s}_remittances"] / 1000 for s in STATES], labels=STATES)
    axes[1].set(title="Remittances by sender's current employment state", ylabel="Thousand USD / month")
    axes[1].legend()
    fig.savefig(out / "02_resource_flows.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for scenario, g in monthly.groupby("scenario", sort=False):
        axes[0].plot(g.date, g.remittances / 1000, label=scenario.replace("_", " "))
        axes[1].plot(g.date, g.remittances.cumsum() / 1e6, label=scenario.replace("_", " "))
    axes[0].set(title="Scenario comparison — conditional model outcomes, not causal estimates", ylabel="Monthly remittances (thousand USD)")
    axes[1].set(ylabel="Cumulative remittances (million USD)")
    axes[0].legend(fontsize=9)
    fig.savefig(out / "03_counterfactuals.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "parameters.json")
    parser.add_argument("--population", type=int, help="Override population size")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    p = copy.deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    for key in ("population", "seed"):
        if getattr(args, key) is not None:
            p[key] = getattr(args, key)
    validate(p)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scenario in SCENARIOS:
        agents, inputs = simulate(p, scenario)
        summaries.append(aggregate(agents))
        inputs.to_csv(args.output / f"inputs_{scenario}.csv")
        if scenario == "baseline":
            agents.to_csv(args.output / "agents_baseline.csv.gz", index=False, compression="gzip")
    monthly = pd.concat(summaries, ignore_index=True)
    monthly.to_csv(args.output / "monthly.csv", index=False)
    annual = monthly.assign(year=monthly.date.dt.year).groupby(["scenario", "year"])[["income", "us_transfers", "consumption", "remittances", "savings_change", "consumption_shortfall"]].sum()
    annual.to_csv(args.output / "annual.csv")
    (args.output / "parameters_used.json").write_text(json.dumps(p, indent=2) + "\n", encoding="utf-8")
    figures(monthly, args.output)
    print("Illustrative, uncalibrated simulation; all monetary values in USD.")
    print(annual.loc["baseline"].round().to_string())
    print(f"\nOutputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
