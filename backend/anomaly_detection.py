"""
Anomaly Detection Engine.

Approach: for each machine, build a rolling baseline (mean + std over a
trailing window, excluding the current day) for efficiency, defect rate,
and downtime %. Flag a day as anomalous if any metric deviates from the
baseline by more than a z-score threshold. This mirrors how the spec's
example works: "Production dropped by 37%, Defects increased by 263%"
are both deviation-from-normal statements, not fixed thresholds -- a
machine that's always a bit noisy shouldn't trigger constantly, and a
machine that's normally rock-solid should trigger on smaller deviations.

Severity levels:
    LOW      : 1 metric breaches threshold
    MEDIUM   : 2 metrics breach threshold
    HIGH     : 3 metrics breach threshold, or any single metric > 2x threshold
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW_DAYS = 14
Z_THRESHOLD = 2.0

METRICS = ["efficiency_pct", "defect_rate_pct", "downtime_pct"]
# efficiency: anomaly = drop below baseline. defect/downtime: anomaly = rise above baseline.
DIRECTION = {"efficiency_pct": "below", "defect_rate_pct": "above", "downtime_pct": "above"}


def detect_anomalies(daily: pd.DataFrame, window: int = ROLLING_WINDOW_DAYS, z_thresh: float = Z_THRESHOLD) -> pd.DataFrame:
    df = daily.sort_values(["machine_id", "date"]).copy()
    results = []

    for machine_id, group in df.groupby("machine_id"):
        group = group.reset_index(drop=True)
        for metric in METRICS:
            roll_mean = group[metric].rolling(window, min_periods=5).mean().shift(1)
            roll_std = group[metric].rolling(window, min_periods=5).std().shift(1)
            z = (group[metric] - roll_mean) / roll_std.replace(0, np.nan)

            if DIRECTION[metric] == "below":
                breach = z <= -z_thresh
            else:
                breach = z >= z_thresh

            group[f"{metric}_baseline"] = roll_mean.round(2)
            group[f"{metric}_zscore"] = z.round(2)
            group[f"{metric}_breach"] = breach.fillna(False)

        results.append(group)

    out = pd.concat(results, ignore_index=True)

    breach_cols = [f"{m}_breach" for m in METRICS]
    out["breach_count"] = out[breach_cols].sum(axis=1)

    def severity(row):
        if row["breach_count"] == 0:
            return "NONE"
        max_z = max(
            abs(row["efficiency_pct_zscore"]) if not pd.isna(row["efficiency_pct_zscore"]) else 0,
            abs(row["defect_rate_pct_zscore"]) if not pd.isna(row["defect_rate_pct_zscore"]) else 0,
            abs(row["downtime_pct_zscore"]) if not pd.isna(row["downtime_pct_zscore"]) else 0,
        )
        if row["breach_count"] >= 3 or max_z >= 2 * Z_THRESHOLD:
            return "HIGH"
        elif row["breach_count"] == 2:
            return "MEDIUM"
        else:
            return "LOW"

    out["severity"] = out.apply(severity, axis=1)
    out["is_anomaly"] = out["breach_count"] > 0
    return out


def pct_change_vs_baseline(row, metric):
    """Human-readable % change vs baseline, for recommendation text."""
    baseline = row[f"{metric}_baseline"]
    current = row[metric]
    if pd.isna(baseline) or baseline == 0:
        return None
    return round((current - baseline) / baseline * 100, 1)


if __name__ == "__main__":
    daily = pd.read_csv("../data/daily_kpis.csv", parse_dates=["date"])
    anomalies = detect_anomalies(daily)
    anomalies.to_csv("../data/anomaly_flags.csv", index=False)

    flagged = anomalies[anomalies["is_anomaly"]]
    print(f"Total daily records: {len(anomalies)}")
    print(f"Anomalous records flagged: {len(flagged)}")
    print(f"\nSeverity breakdown:\n{flagged['severity'].value_counts()}")

    print("\nSample HIGH severity anomalies:")
    high = flagged[flagged["severity"] == "HIGH"].head(8)
    for _, row in high.iterrows():
        eff_chg = pct_change_vs_baseline(row, "efficiency_pct")
        def_chg = pct_change_vs_baseline(row, "defect_rate_pct")
        down_chg = pct_change_vs_baseline(row, "downtime_pct")
        print(f"  {row['date'].date()} | {row['machine_id']} | "
              f"efficiency {eff_chg}% vs baseline | defects {def_chg}% vs baseline | downtime {down_chg}% vs baseline")
