"""
Data pipeline for iceberg trajectory prediction.

Input: a set of weekly snapshot CSVs, each with columns
    Iceberg, Length (NM), Width (NM), Latitude, Longitude,
    Area (sqMI), Area (sqNM), Area (sqKM), Last Update

These are NOT sea-ice concentration data - they are point-in-time
positions of individually named icebergs. The pipeline turns this into
a supervised "given iceberg state at time t, predict position at t+1"
dataset by pairing up consecutive observations of the same iceberg.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "iceberg_tracks"

EARTH_RADIUS_KM = 6371.0


def load_raw(data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate every weekly snapshot CSV into one long dataframe."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No iceberg CSVs found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.strip() for c in combined.columns]
    combined["Last Update"] = pd.to_datetime(combined["Last Update"], format="%m/%d/%Y")
    combined = combined.sort_values(["Iceberg", "Last Update"]).reset_index(drop=True)

    # De-duplicate: same iceberg + same date appearing in more than one file
    combined = combined.drop_duplicates(subset=["Iceberg", "Last Update"], keep="last")

    return combined


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def wrapped_lon_delta(lon_from, lon_to) -> np.ndarray:
    """
    Shortest signed longitude difference (lon_to - lon_from), wrapped to
    [-180, 180]. Without this, an iceberg crossing the antimeridian
    (e.g. 179.7 -> -179.0) looks like it jumped ~360 degrees instead of ~1.3.
    """
    raw_delta = lon_to - lon_from
    return (raw_delta + 180) % 360 - 180


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def build_transition_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each iceberg, build one row per (t-1 -> t -> t+1) triple:
      - features describe the iceberg's state and recent drift velocity at time t
      - target is its actual position at t+1

    Requires at least 3 observations of an iceberg (to have both a "previous
    velocity" feature and a "next position" target). Icebergs with fewer
    observations are skipped rather than padded with invented values.
    """
    rows = []

    for iceberg_id, g in df.groupby("Iceberg"):
        g = g.sort_values("Last Update").reset_index(drop=True)
        n = len(g)
        if n < 3:
            continue  # not enough history to compute a velocity feature + target

        for i in range(1, n - 1):
            prev, cur, nxt = g.iloc[i - 1], g.iloc[i], g.iloc[i + 1]

            prev_days = (cur["Last Update"] - prev["Last Update"]).days
            next_days = (nxt["Last Update"] - cur["Last Update"]).days
            if prev_days <= 0 or next_days <= 0:
                continue  # guard against duplicate/out-of-order timestamps

            prev_speed_kmday = haversine_km(
                prev["Latitude"], prev["Longitude"], cur["Latitude"], cur["Longitude"]
            ) / prev_days
            prev_bearing = initial_bearing_deg(
                prev["Latitude"], prev["Longitude"], cur["Latitude"], cur["Longitude"]
            )
            prev_vel_lat = (cur["Latitude"] - prev["Latitude"]) / prev_days
            prev_vel_lon = wrapped_lon_delta(prev["Longitude"], cur["Longitude"]) / prev_days

            rows.append(
                {
                    "iceberg": iceberg_id,
                    "obs_date": cur["Last Update"],
                    "target_date": nxt["Last Update"],
                    "current_lat": cur["Latitude"],
                    "current_lon": cur["Longitude"],
                    "length_nm": cur["Length (NM)"],
                    "width_nm": cur["Width (NM)"],
                    "area_sqkm": cur["Area (sqKM)"],
                    "prev_speed_kmday": prev_speed_kmday,
                    "prev_bearing_deg": prev_bearing,
                    "prev_vel_lat": prev_vel_lat,
                    "prev_vel_lon": prev_vel_lon,
                    "days_ahead": next_days,
                    "target_lat": nxt["Latitude"],
                    "target_lon": nxt["Longitude"],
                }
            )

    return pd.DataFrame(rows)


def chronological_split(transitions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out each iceberg's LAST transition as test data (forecasting its most
    recent hop), train on every earlier transition across all icebergs.
    This avoids leaking future positions of an iceberg into its own training
    rows while still using all icebergs for training signal.
    """
    test_idx = transitions.groupby("iceberg")["obs_date"].idxmax()
    test = transitions.loc[test_idx].reset_index(drop=True)
    train = transitions.drop(index=test_idx).reset_index(drop=True)
    return train, test


FEATURE_COLUMNS = [
    "current_lat",
    "current_lon",
    "length_nm",
    "width_nm",
    "area_sqkm",
    "prev_speed_kmday",
    "prev_bearing_deg",
    "prev_vel_lat",
    "prev_vel_lon",
    "days_ahead",
]
TARGET_COLUMNS = ["target_lat", "target_lon"]


if __name__ == "__main__":
    raw = load_raw()
    print(f"Loaded {len(raw)} raw observations across {raw['Iceberg'].nunique()} icebergs")
    print(f"Date range: {raw['Last Update'].min().date()} to {raw['Last Update'].max().date()}")

    transitions = build_transition_dataset(raw)
    print(f"\nBuilt {len(transitions)} training transitions "
          f"from {transitions['iceberg'].nunique()} icebergs "
          f"(icebergs with <3 observations were skipped)")

    train, test = chronological_split(transitions)
    print(f"Train: {len(train)} rows | Test (last hop per iceberg): {len(test)} rows")
