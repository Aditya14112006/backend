"""
Train an iceberg trajectory prediction model.

Validation philosophy: a RandomForest is only worth shipping if it beats
simple, honest baselines on a proper held-out set (each iceberg's most
recent hop, never seen during training). We report all three side by side
rather than a single flattering number.

Baselines:
  - Persistence: "the iceberg doesn't move" -> predict current position.
  - Linear extrapolation: "it keeps drifting at its last known velocity."

Metric: haversine distance (km) between predicted and actual next position.
This is a physical error metric, not an accuracy percentage - there is no
"confidence score" fabricated here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ml.iceberg_trajectory.data_pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    build_transition_dataset,
    chronological_split,
    haversine_km,
    load_raw,
    wrapped_lon_delta,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "iceberg_trajectory_model.joblib"
META_PATH = MODEL_DIR / "iceberg_trajectory_meta.json"


def persistence_baseline(test_df):
    """Predict next position = current position (no movement)."""
    return test_df["current_lat"].values, test_df["current_lon"].values


def linear_extrapolation_baseline(test_df):
    """Predict next position = current position + (last velocity * days_ahead)."""
    pred_lat = test_df["current_lat"].values + test_df["prev_vel_lat"].values * test_df["days_ahead"].values
    pred_lon = test_df["current_lon"].values + test_df["prev_vel_lon"].values * test_df["days_ahead"].values
    pred_lon = (pred_lon + 180) % 360 - 180
    return pred_lat, pred_lon


def error_summary(name, actual_lat, actual_lon, pred_lat, pred_lon) -> dict:
    errors_km = haversine_km(actual_lat, actual_lon, pred_lat, pred_lon)
    return {
        "name": name,
        "mean_error_km": round(float(np.mean(errors_km)), 2),
        "median_error_km": round(float(np.median(errors_km)), 2),
        "max_error_km": round(float(np.max(errors_km)), 2),
        "n_samples": int(len(errors_km)),
    }


def train_and_evaluate():
    raw = load_raw()
    transitions = build_transition_dataset(raw)
    train_df, test_df = chronological_split(transitions)

    X_train = train_df[FEATURE_COLUMNS].values
    X_test = test_df[FEATURE_COLUMNS].values

    # Predict DISPLACEMENT (delta lat/lon), not absolute position.
    # Tree ensembles can't extrapolate outside the coordinate ranges seen in
    # training, so predicting absolute lat/lon fails badly on unseen icebergs.
    # Predicting the movement instead keeps the target range small and
    # centered near zero, which is what a persistence baseline already assumes.
    y_train_delta = np.column_stack([
        train_df["target_lat"].values - train_df["current_lat"].values,
        wrapped_lon_delta(train_df["current_lon"].values, train_df["target_lon"].values),
    ])

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train_delta)

    delta_pred = model.predict(X_test)
    rf_pred_lat = test_df["current_lat"].values + delta_pred[:, 0]
    rf_pred_lon = (test_df["current_lon"].values + delta_pred[:, 1] + 180) % 360 - 180

    persist_lat, persist_lon = persistence_baseline(test_df)
    linear_lat, linear_lon = linear_extrapolation_baseline(test_df)

    actual_lat = test_df["target_lat"].values
    actual_lon = test_df["target_lon"].values

    results = [
        error_summary("persistence_baseline", actual_lat, actual_lon, persist_lat, persist_lon),
        error_summary("linear_extrapolation_baseline", actual_lat, actual_lon, linear_lat, linear_lon),
        error_summary("random_forest_model", actual_lat, actual_lon, rf_pred_lat, rf_pred_lon),
    ]

    print("\n=== Validation results (held-out last hop per iceberg, n={}) ===".format(len(test_df)))
    for r in results:
        print(f"  {r['name']:<32s} mean={r['mean_error_km']:>7.2f} km  "
              f"median={r['median_error_km']:>7.2f} km  max={r['max_error_km']:>7.2f} km")

    best = min(results, key=lambda r: r["mean_error_km"])
    print(f"\nBest by mean error: {best['name']}")
    if best["name"] != "random_forest_model":
        print("WARNING: the trained model did NOT beat a simple baseline on this "
              "dataset. This is expected with only ~13 weekly snapshots per "
              "iceberg over a single 3-month window - saving it anyway for the "
              "pipeline to be wired end-to-end, but the API response will surface "
              "this comparison rather than claim the model is reliable.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_icebergs": int(raw["Iceberg"].nunique()),
        "date_range": {
            "start": raw["Last Update"].min().strftime("%Y-%m-%d"),
            "end": raw["Last Update"].max().strftime("%Y-%m-%d"),
        },
        "validation_split": "last observed hop per iceberg held out; trained on all earlier hops",
        "validation_results": results,
        "beats_simple_baseline": best["name"] == "random_forest_model",
        "known_limitations": [
            "Trained on a single ~3-month window (Apr-Jul 2026) in one season - "
            "will not generalize to other seasons without more data.",
            "Only 33 icebergs, all pre-identified/named (e.g. A76C) - the model "
            "cannot detect NEW icebergs, only forecast drift of tracked ones.",
            "Weekly observation cadence limits precision of short-horizon "
            "predictions (e.g. a 1-day-ahead forecast).",
        ],
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {META_PATH}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate()
