"""
Inference wrapper for iceberg trajectory prediction.

Loads the trained RandomForest once at import time, and always returns it
alongside the persistence and linear-extrapolation baselines with the
validation metadata attached - because the validation run showed the
trained model does NOT clearly beat "the iceberg hasn't moved" on this
dataset (see data/models/iceberg_trajectory_meta.json). The API must not
hide that.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from ml.iceberg_trajectory.data_pipeline import (
    FEATURE_COLUMNS,
    haversine_km,
    initial_bearing_deg,
    load_raw,
    wrapped_lon_delta,
)

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "iceberg_trajectory_model.joblib"
META_PATH = MODEL_DIR / "iceberg_trajectory_meta.json"


class IcebergTrajectoryService:
    def __init__(self):
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run "
                f"`python -m ml.iceberg_trajectory.train` first."
            )
        self.model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            self.metadata = json.load(f)
        self._raw_cache: Optional[pd.DataFrame] = None

    def _raw(self) -> pd.DataFrame:
        if self._raw_cache is None:
            self._raw_cache = load_raw()
        return self._raw_cache

    # -----------------------------------------------------------------
    # Listing tracked icebergs (for the frontend to pick from / show on map)
    # -----------------------------------------------------------------

    def list_icebergs(self) -> list[dict]:
        raw = self._raw()
        out = []
        for iceberg_id, g in raw.groupby("Iceberg"):
            g = g.sort_values("Last Update")
            latest = g.iloc[-1]
            out.append(
                {
                    "iceberg_id": iceberg_id,
                    "latest_date": latest["Last Update"].strftime("%Y-%m-%d"),
                    "latitude": float(latest["Latitude"]),
                    "longitude": float(latest["Longitude"]),
                    "length_nm": float(latest["Length (NM)"]),
                    "width_nm": float(latest["Width (NM)"]),
                    "area_sqkm": float(latest["Area (sqKM)"]),
                    "observation_count": int(len(g)),
                }
            )
        return sorted(out, key=lambda r: r["iceberg_id"])

    # -----------------------------------------------------------------
    # Feature building shared by both entry points
    # -----------------------------------------------------------------

    @staticmethod
    def _build_features(
        current_lat, current_lon, prev_lat, prev_lon, prev_days,
        length_nm, width_nm, area_sqkm, days_ahead,
    ) -> dict:
        prev_speed_kmday = haversine_km(prev_lat, prev_lon, current_lat, current_lon) / prev_days
        prev_bearing = initial_bearing_deg(prev_lat, prev_lon, current_lat, current_lon)
        prev_vel_lat = (current_lat - prev_lat) / prev_days
        prev_vel_lon = wrapped_lon_delta(prev_lon, current_lon) / prev_days

        return {
            "current_lat": current_lat,
            "current_lon": current_lon,
            "length_nm": length_nm,
            "width_nm": width_nm,
            "area_sqkm": area_sqkm,
            "prev_speed_kmday": float(prev_speed_kmday),
            "prev_bearing_deg": float(prev_bearing),
            "prev_vel_lat": prev_vel_lat,
            "prev_vel_lon": prev_vel_lon,
            "days_ahead": days_ahead,
        }

    def _predict_all_methods(self, features: dict) -> dict:
        current_lat, current_lon = features["current_lat"], features["current_lon"]
        days_ahead = features["days_ahead"]

        X = np.array([[features[c] for c in FEATURE_COLUMNS]])
        delta = self.model.predict(X)[0]
        model_lat = current_lat + float(delta[0])
        model_lon = (current_lon + float(delta[1]) + 180) % 360 - 180

        persistence_lat, persistence_lon = current_lat, current_lon

        linear_lat = current_lat + features["prev_vel_lat"] * days_ahead
        linear_lon = current_lon + features["prev_vel_lon"] * days_ahead
        linear_lon = (linear_lon + 180) % 360 - 180

        val = {r["name"]: r for r in self.metadata["validation_results"]}

        return {
            "current_position": {"latitude": current_lat, "longitude": current_lon},
            "days_ahead": days_ahead,
            "predictions": {
                "random_forest_model": {
                    "latitude": round(model_lat, 4),
                    "longitude": round(model_lon, 4),
                    "validated_mean_error_km": val["random_forest_model"]["mean_error_km"],
                },
                "persistence_baseline": {
                    "latitude": round(persistence_lat, 4),
                    "longitude": round(persistence_lon, 4),
                    "validated_mean_error_km": val["persistence_baseline"]["mean_error_km"],
                },
                "linear_extrapolation_baseline": {
                    "latitude": round(linear_lat, 4),
                    "longitude": round(linear_lon, 4),
                    "validated_mean_error_km": val["linear_extrapolation_baseline"]["mean_error_km"],
                },
            },
            "recommended_method": min(
                val, key=lambda k: val[k]["mean_error_km"]
            ),
            "disclaimer": (
                "'validated_mean_error_km' is the average error each method made "
                "on 33 held-out real forecasts, not a per-prediction confidence "
                "score. On this dataset the trained model did not clearly beat "
                "the simple baselines - see beats_simple_baseline in model metadata."
            ),
            "model_metadata": {
                "trained_at": self.metadata["trained_at"],
                "n_train_rows": self.metadata["n_train_rows"],
                "n_icebergs": self.metadata["n_icebergs"],
                "beats_simple_baseline": self.metadata["beats_simple_baseline"],
                "known_limitations": self.metadata["known_limitations"],
            },
        }

    # -----------------------------------------------------------------
    # Predict from a tracked iceberg's own history
    # -----------------------------------------------------------------

    def predict_for_iceberg(self, iceberg_id: str, days_ahead: int = 7) -> dict:
        raw = self._raw()
        g = raw[raw["Iceberg"].str.upper() == iceberg_id.upper()].sort_values("Last Update")
        if len(g) < 2:
            raise ValueError(
                f"Iceberg '{iceberg_id}' has fewer than 2 tracked observations - "
                f"cannot compute a velocity feature to predict from."
            )

        prev, cur = g.iloc[-2], g.iloc[-1]
        prev_days = (cur["Last Update"] - prev["Last Update"]).days
        if prev_days <= 0:
            raise ValueError(f"Non-positive day gap for iceberg '{iceberg_id}' - bad data.")

        features = self._build_features(
            current_lat=float(cur["Latitude"]),
            current_lon=float(cur["Longitude"]),
            prev_lat=float(prev["Latitude"]),
            prev_lon=float(prev["Longitude"]),
            prev_days=prev_days,
            length_nm=float(cur["Length (NM)"]),
            width_nm=float(cur["Width (NM)"]),
            area_sqkm=float(cur["Area (sqKM)"]),
            days_ahead=days_ahead,
        )
        result = self._predict_all_methods(features)
        result["iceberg_id"] = iceberg_id.upper()
        result["last_observed_date"] = cur["Last Update"].strftime("%Y-%m-%d")
        return result

    # -----------------------------------------------------------------
    # Predict from manually supplied state (iceberg not in our dataset)
    # -----------------------------------------------------------------

    def predict_manual(
        self, current_lat, current_lon, prev_lat, prev_lon, prev_days,
        length_nm, width_nm, area_sqkm, days_ahead=7,
    ) -> dict:
        if prev_days <= 0:
            raise ValueError("prev_days must be a positive number of days.")
        features = self._build_features(
            current_lat, current_lon, prev_lat, prev_lon, prev_days,
            length_nm, width_nm, area_sqkm, days_ahead,
        )
        return self._predict_all_methods(features)


_service: Optional[IcebergTrajectoryService] = None


def get_service() -> IcebergTrajectoryService:
    global _service
    if _service is None:
        _service = IcebergTrajectoryService()
    return _service
