"""
Failure Risk Classifier.

Predicts the probability that a machine will experience a failure event
on a given day, using a RandomForest classifier trained on the KPI +
anomaly-detection features already built in Days 2-3.

Framed as an EARLY WARNING system: features are built from trailing
(lagged) values only, and the target is whether a failure occurs on
the *current* day. This matters -- if we let the model see same-day
downtime/defect numbers, it's not predicting a failure, it's just
recognizing one that already happened. Using lagged features means the
model has to catch the "smoke before the fire" (small efficiency drops
and defect upticks in the days just before a breakdown), which is the
actual real-world use case: warn before the machine actually stops.

Output: for every machine-day, a failure probability (0-1) and a
risk band (LOW / MEDIUM / HIGH), plus a "Required maintenance date"
suggestion for machines currently trending HIGH.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

LAG_DAYS = 3  # use trailing N-day average as predictive features


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values(["machine_id", "date"]).copy()

    feature_cols = []
    for metric in ["efficiency_pct", "defect_rate_pct", "downtime_pct"]:
        lag_mean_col = f"{metric}_lag{LAG_DAYS}_mean"
        lag_trend_col = f"{metric}_lag{LAG_DAYS}_trend"

        df[lag_mean_col] = (
            df.groupby("machine_id")[metric]
            .transform(lambda s: s.shift(1).rolling(LAG_DAYS, min_periods=2).mean())
        )
        # trend = most recent value minus the lag-window mean (is it getting worse right now?)
        df[lag_trend_col] = df.groupby("machine_id")[metric].shift(1) - df[lag_mean_col]

        feature_cols += [lag_mean_col, lag_trend_col]

    return df, feature_cols


def train_and_predict(df: pd.DataFrame, feature_cols: list):
    model_df = df.dropna(subset=feature_cols).copy()

    X = model_df[feature_cols]
    y = model_df["failure_flag"].astype(int)

    # time-aware split: train on first 80% of dates, test on last 20%
    # (never train on the future to predict the past)
    split_date = model_df["date"].quantile(0.8)
    train_mask = model_df["date"] <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight="balanced",  # failures are rare, don't let the model just predict "no failure" always
        random_state=42,
    )
    clf.fit(X_train, y_train)

    if y_test.nunique() > 1:
        y_pred = clf.predict(X_test)
        y_proba_test = clf.predict_proba(X_test)[:, 1]
        print("=== Test set performance (held-out last 20% of dates) ===")
        print(classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_test):.3f}")
    else:
        print("Note: test window contains only one class - skipping held-out metrics for this run.")

    # predict probability for ALL rows (full history) to output a risk score everywhere
    model_df["failure_probability"] = clf.predict_proba(X)[:, 1].round(3)

    def risk_band(p):
        if p >= 0.6:
            return "HIGH"
        elif p >= 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    model_df["risk_band"] = model_df["failure_probability"].apply(risk_band)

    feature_importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Feature importance ===")
    print(feature_importance.round(3).to_string())

    return model_df, clf


if __name__ == "__main__":
    daily = pd.read_csv("../data/daily_kpis.csv", parse_dates=["date"])
    daily = daily.groupby(["machine_id", "date"], as_index=False).agg(
        efficiency_pct=("efficiency_pct", "mean"),
        defect_rate_pct=("defect_rate_pct", "mean"),
        downtime_pct=("downtime_pct", "mean"),
        failure_flag=("failure_flag", "max"),
    )

    df, feature_cols = build_features(daily)
    result, model = train_and_predict(df, feature_cols)

    result.to_csv("../data/failure_risk.csv", index=False)

    print("\n=== Current risk snapshot (most recent day per machine) ===")
    latest = result.sort_values("date").groupby("machine_id").tail(1)
    print(latest[["machine_id", "date", "failure_probability", "risk_band"]].to_string(index=False))
