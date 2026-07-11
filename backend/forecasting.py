"""
AI Forecasting Engine — uses Prophet to predict next-week production
and defect counts per machine.

Why Prophet over a plain moving average:
    - Handles weekly seasonality automatically (we baked a weekend dip
      into the synthetic data, so this is a real pattern, not noise)
    - Produces uncertainty intervals (yhat_lower/yhat_upper), which is
      useful for the recommendation engine later ("high confidence" vs
      "wide uncertainty" forecasts should be treated differently)
    - Robust to the anomaly events in the data (the breakdown days act
      like outliers Prophet naturally down-weights over a long history)

Two forecasts are produced per machine:
    1. units_produced  -> next 7 days
    2. defective_units -> next 7 days

Output: a single tidy CSV with forecast + actual history for Power BI.
"""

import pandas as pd
from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

FORECAST_HORIZON_DAYS = 7


def forecast_metric(daily: pd.DataFrame, machine_id: str, metric: str, horizon: int = FORECAST_HORIZON_DAYS) -> pd.DataFrame:
    """Fits Prophet on one machine's history for one metric, returns forecast rows."""
    hist = daily[daily["machine_id"] == machine_id][["date", metric]].copy()
    hist = hist.rename(columns={"date": "ds", metric: "y"})
    hist = hist.sort_values("ds")

    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.85,
        changepoint_prior_scale=0.1,  # a bit more flexible, to track post-breakdown recovery
    )
    model.fit(hist)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
    result["machine_id"] = machine_id
    result["metric"] = metric
    result = result.rename(columns={"ds": "date"})
    result[["yhat", "yhat_lower", "yhat_upper"]] = result[["yhat", "yhat_lower", "yhat_upper"]].round(1)
    return result


def run_all_forecasts(daily: pd.DataFrame) -> pd.DataFrame:
    all_forecasts = []
    machines = daily["machine_id"].unique()

    for machine_id in machines:
        for metric in ["units_produced", "defective_units"]:
            print(f"  Forecasting {metric} for {machine_id}...")
            fc = forecast_metric(daily, machine_id, metric)
            all_forecasts.append(fc)

    return pd.concat(all_forecasts, ignore_index=True)


if __name__ == "__main__":
    daily = pd.read_csv("../data/daily_kpis.csv", parse_dates=["date"])

    # collapse shift-level rows to one row per machine per day
    # (daily_kpis.csv from Day 2 is already daily-rolled-up, so this is a no-op safety check)
    daily = daily.groupby(["machine_id", "date"], as_index=False).agg(
        units_produced=("units_produced", "sum"),
        defective_units=("defective_units", "sum"),
    )

    print("Running forecasts for all machines...")
    forecasts = run_all_forecasts(daily)
    forecasts.to_csv("../data/forecasts.csv", index=False)

    print(f"\nDone. {len(forecasts)} forecast rows written to forecasts.csv")
    print("\nSample: next 7 days for M3 (units_produced)")
    sample = forecasts[(forecasts.machine_id == "M3") & (forecasts.metric == "units_produced")]
    print(sample[["date", "yhat", "yhat_lower", "yhat_upper"]].to_string(index=False))
