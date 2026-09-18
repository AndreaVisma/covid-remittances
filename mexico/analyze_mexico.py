"""Reproduce the descriptive analysis of the Banco de Mexico remittance export.

Install: python -m pip install pandas openpyxl plotly
Run:     python mexico/analyze_mexico.py
Optional: python mexico/analyze_mexico.py --input path/to/export.xlsx
"""

import argparse
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CHANNELS = ["total", "money_orders", "personal_cheques", "electronic", "cash_in_kind"]
METRICS = ["value_usd_million", "transactions_thousand", "reported_average_usd"]
HERE = Path(__file__).resolve().parent


def read_export(path):
    """Read Hoja1; columns B:F are values, G:K counts, L:P averages."""
    workbook = openpyxl.load_workbook(path, data_only=True)
    try:
        sheet = workbook["Hoja1"]
        # This export uses three five-column blocks in the order above.
        rows = list(sheet.values)
    finally:
        workbook.close()
    columns = [f"{channel}_{metric}" for metric in METRICS for channel in CHANNELS]
    records = [
        [row[0], number, *row[1:16]]
        for number, row in enumerate(rows, 1)
        if isinstance(row[0], datetime)
    ]
    if not records:
        raise ValueError("No monthly date rows found in Hoja1.")
    frame = pd.DataFrame(records, columns=["month", "source_row", *columns])
    frame = frame.set_index("month").sort_index()
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="MS")
    if not frame.index.equals(expected):
        raise ValueError("Dates must be unique, consecutive month starts.")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if frame[columns].isna().any().any() or (frame[columns] < 0).any().any():
        raise ValueError("Missing or negative observations in the monthly series.")
    if (frame[[f"total_{m}" for m in METRICS[:2]]] <= 0).any().any():
        raise ValueError("Total values and transaction counts must be positive.")
    for metric, tolerance in zip(METRICS[:2], [0.0003, 0.003]):
        residual = frame[f"total_{metric}"] - frame[
            [f"{channel}_{metric}" for channel in CHANNELS[1:]]
        ].sum(axis=1)
        frame[f"reconciliation_{metric}"] = residual
        if residual.abs().max() > tolerance:
            raise ValueError(f"Channel totals do not reconcile for {metric}.")
    return frame


def add_ratios(frame):
    frame["average_usd"] = (
        1000 * frame.total_value_usd_million / frame.total_transactions_thousand
    )
    frame["electronic_value_share_pct"] = (
        100 * frame.electronic_value_usd_million / frame.total_value_usd_million
    )
    return frame


def analyze(monthly):
    monthly = add_ratios(monthly.copy())
    measures = {
        "value": "total_value_usd_million",
        "transactions": "total_transactions_thousand",
        "average": "average_usd",
    }
    for label, column in measures.items():
        monthly[f"{label}_yoy_pct"] = monthly[column].pct_change(12, fill_method=None) * 100
    # Exact identity: g_value = g_count + g_average + g_count * g_average.
    monthly["interaction_pp"] = monthly.transactions_yoy_pct * monthly.average_yoy_pct / 100
    additive = [f"{c}_{m}" for m in METRICS[:2] for c in CHANNELS]
    annual = monthly[additive].groupby(monthly.index.year).sum()
    annual.index.name = "year"
    annual["months"] = monthly.groupby(monthly.index.year).size()
    annual["complete_year"] = annual.months.eq(12)
    annual = add_ratios(annual)
    annual["value_usd_billion"] = annual.total_value_usd_million / 1000
    annual["transactions_million"] = annual.total_transactions_thousand / 1000
    for label, column in measures.items():
        growth = annual[column].pct_change(fill_method=None) * 100
        comparable = annual.complete_year & annual.complete_year.shift(1, fill_value=False)
        annual[f"{label}_yoy_pct"] = growth.where(comparable)
    return monthly, annual


def save_figure(monthly, path, source_name):
    """Save a self-contained interactive figure, with no internet dependency."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
        subplot_titles=("Monthly remittance growth", "Electronic share of recorded value"),
    )
    for column, label, color in [
        ("value_yoy_pct", "Total value", "#2563eb"),
        ("transactions_yoy_pct", "Transaction count", "#d97706"),
        ("average_yoy_pct", "Average transfer", "#059669"),
    ]:
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly[column], name=label,
            mode="lines", line=dict(color=color, width=2),
            hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=monthly.index, y=monthly.electronic_value_share_pct,
        name="Electronic share", mode="lines", line=dict(color="#7c3aed", width=2),
        hovertemplate="%{x|%b %Y}: %{y:.2f}%<extra>Electronic share</extra>",
    ), row=2, col=1)
    for row in [1, 2]:
        fig.add_vrect(x0="2020-03-01", x1="2022-01-01", fillcolor="#94a3b8",
                      opacity=0.15, line_width=0, row=row, col=1)
    fig.add_hline(y=0, line_color="#64748b", line_width=1, row=1, col=1)
    fig.update_yaxes(title_text="Year-over-year change (%)", row=1, col=1)
    fig.update_yaxes(title_text="Share (%)", row=2, col=1)
    fig.update_xaxes(range=["2018-01-01", "2022-01-01"])
    buttons = []
    for label, start, end in [
        ("2018–2021", "2018-01-01", "2022-01-01"),
        ("2020–2021", "2020-01-01", "2022-01-01"),
        ("Full history", monthly.index.min().isoformat(), monthly.index.max().isoformat()),
    ]:
        buttons.append(dict(label=label, method="relayout", args=[{
            "xaxis.range": [start, end], "xaxis2.range": [start, end],
            "xaxis.autorange": False, "xaxis2.autorange": False,
        }]))
    fig.update_layout(
        template="plotly_white", height=850, hovermode="x unified",
        title=dict(text="Mexico remittances: transactions, transfer size and channels", font_size=21),
        margin=dict(l=85, r=35, t=155, b=120),
        legend=dict(orientation="h", y=1.14, x=0),
        updatemenus=[dict(type="buttons", direction="right", buttons=buttons,
                         x=0, y=1.24, xanchor="left", yanchor="top")],
    )
    fig.add_annotation(
        x=0, y=-0.15, xref="paper", yref="paper", showarrow=False, align="left",
        text=f"Source: Banco de Mexico, {source_name}, Hoja1.<br>"
             "Growth compares with the same month one year earlier. Average = value / count.<br>"
             "Shading: March 2020–December 2021. Descriptive comparisons; total Mexico inflows.",
        font=dict(size=11, color="#475569"),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"responsive": True, "displaylogo": False,
                           "toImageButtonOptions": {"format": "png", "scale": 2}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "Consulta_20260915-073135560.xlsx")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    monthly, annual = analyze(read_export(args.input))
    if not all(y in annual.index and annual.loc[y, "complete_year"] for y in [2019, 2020, 2021]):
        raise ValueError("The summary requires complete 2019, 2020 and 2021 data.")
    pandemic = monthly.loc["2020":"2021", [
        "total_value_usd_million", "total_transactions_thousand", "average_usd",
        "value_yoy_pct", "transactions_yoy_pct", "average_yoy_pct", "interaction_pp",
    ]]
    summary_columns = ["value_usd_billion", "value_yoy_pct", "transactions_million",
                       "transactions_yoy_pct", "average_usd", "average_yoy_pct",
                       "electronic_value_share_pct"]
    change = annual.loc[2020, "total_value_usd_million"] - annual.loc[2019, "total_value_usd_million"]
    cash_change = annual.loc[2020, "cash_in_kind_value_usd_million"] - annual.loc[2019, "cash_in_kind_value_usd_million"]
    report = "\n".join([
        f"Source: {args.input.resolve()}, Hoja1 (export vintage is in the filename).",
        f"Coverage: {len(monthly)} consecutive months, {monthly.index.min():%Y-%m} to {monthly.index.max():%Y-%m}.",
        "Channel values and counts reconcile within source rounding.",
        "\nAnnual summary (growth and shares in percent):",
        annual.loc[2019:2021, summary_columns].to_string(float_format=lambda x: f"{x:.2f}"),
        "\nMonthly pandemic comparisons (year over year):",
        pandemic.to_string(float_format=lambda x: f"{x:.2f}"),
        f"\n2020 total value change: USD {change:.2f} million.",
        f"2020 recorded cash/in-kind value change: USD {cash_change:.2f} million.",
        f"Cash/in-kind decline relative to total increase: {-cash_change / change * 100:.2f}%.",
        "\nMethods and interpretation:",
        "Average USD = 1,000 * value (USD million) / transactions (thousand).",
        "Annual averages use annual totals, not the mean of monthly averages.",
        "Growth uses the previous calendar year or the same month a year earlier.",
        "2021 monthly growth therefore compares with 2020, not 2019.",
        "Count growth + average growth + interaction equals value growth (percentage points).",
        "Partial years are labeled and have no annual growth rate.",
        "These are descriptive comparisons, not causal COVID effects.",
        "Transaction counts cannot distinguish more senders from greater sending frequency.",
        "Switching between recorded channels cannot itself increase the recorded total.",
        "Previously unrecorded flows cannot be measured with this export.",
        "The series covers Mexico's total inflows, not an explicitly US-only corridor.",
        "Historical values reflect this export vintage; forecast comparisons must track revisions.",
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output_dir / "monthly.csv", date_format="%Y-%m", float_format="%.6f")
    annual.to_csv(args.output_dir / "annual.csv", float_format="%.6f")
    pandemic.to_csv(args.output_dir / "pandemic_monthly.csv", date_format="%Y-%m", float_format="%.6f")
    (args.output_dir / "summary.txt").write_text(report + "\n", encoding="utf-8")
    save_figure(monthly, args.output_dir / "mexico_remittances.html", args.input.name)
    print(report)
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
