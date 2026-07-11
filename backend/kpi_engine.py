"""
KPI Engine — computes core manufacturing KPIs from raw production records.

Formulas (matching the project spec):
    Efficiency %  = units_produced / target_units * 100
    Defect Rate % = defective_units / units_produced * 100
    Downtime %    = downtime_minutes / shift_duration_min * 100

Also produces daily and machine-level rollups, since per-shift granularity
is too noisy to reason about directly (and Power BI will want both levels).
"""

import pandas as pd


def load_production_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df


def compute_shift_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-record KPI columns to the raw shift-level data."""
    out = df.copy()
    out["efficiency_pct"] = (out["units_produced"] / out["target_units"] * 100).round(2)
    out["defect_rate_pct"] = (out["defective_units"] / out["units_produced"].replace(0, pd.NA) * 100).round(2)
    out["downtime_pct"] = (out["downtime_minutes"] / out["shift_duration_min"] * 100).round(2)
    out["defect_rate_pct"] = out["defect_rate_pct"].fillna(0)
    return out


def daily_rollup(shift_kpis: pd.DataFrame) -> pd.DataFrame:
    """Aggregates shift-level records to one row per machine per day."""
    grouped = shift_kpis.groupby(["date", "machine_id"]).agg(
        target_units=("target_units", "sum"),
        units_produced=("units_produced", "sum"),
        defective_units=("defective_units", "sum"),
        downtime_minutes=("downtime_minutes", "sum"),
        shift_duration_min=("shift_duration_min", "sum"),
        failure_flag=("failure_flag", "max"),
    ).reset_index()

    grouped["efficiency_pct"] = (grouped["units_produced"] / grouped["target_units"] * 100).round(2)
    grouped["defect_rate_pct"] = (grouped["defective_units"] / grouped["units_produced"] * 100).round(2)
    grouped["downtime_pct"] = (grouped["downtime_minutes"] / grouped["shift_duration_min"] * 100).round(2)
    return grouped.sort_values(["machine_id", "date"])


def machine_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per machine: overall averages across the full period."""
    summary = daily.groupby("machine_id").agg(
        avg_efficiency_pct=("efficiency_pct", "mean"),
        avg_defect_rate_pct=("defect_rate_pct", "mean"),
        avg_downtime_pct=("downtime_pct", "mean"),
        total_units_produced=("units_produced", "sum"),
        total_defective_units=("defective_units", "sum"),
        total_downtime_minutes=("downtime_minutes", "sum"),
        failure_days=("failure_flag", "sum"),
    ).round(2).reset_index()
    return summary.sort_values("avg_efficiency_pct")


if __name__ == "__main__":
    df = load_production_data("../data/production_data.csv")
    shift_kpis = compute_shift_kpis(df)
    daily = daily_rollup(shift_kpis)
    summary = machine_summary(daily)

    shift_kpis.to_csv("../data/shift_kpis.csv", index=False)
    daily.to_csv("../data/daily_kpis.csv", index=False)
    summary.to_csv("../data/machine_summary.csv", index=False)

    print("KPI engine run complete.")
    print("\nMachine summary (worst efficiency first):")
    print(summary.to_string(index=False))
