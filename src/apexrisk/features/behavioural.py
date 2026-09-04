"""Convert synthetic trip summaries into exposure-normalised risk features."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_TRIP_COLUMNS = {
    "trip_id",
    "driver_id",
    "start_timestamp_utc",
    "distance_km",
    "duration_seconds",
    "is_night_trip",
    "hard_brake_count",
    "hard_acceleration_count",
    "harsh_cornering_count",
    "speeding_duration_seconds",
    "mean_speeding_excess_pct",
}


def _validate_trips(trips: pd.DataFrame) -> None:
    missing = REQUIRED_TRIP_COLUMNS.difference(trips.columns)
    if missing:
        raise ValueError(f"Trips are missing required columns: {sorted(missing)}")
    if trips.empty:
        raise ValueError("Trips must not be empty")
    if (trips["distance_km"] <= 0).any():
        raise ValueError("All trips must have positive distance_km")
    if (trips["duration_seconds"] <= 0).any():
        raise ValueError("All trips must have positive duration_seconds")


def build_trip_features(trips: pd.DataFrame) -> pd.DataFrame:
    """Add exposure-normalised behavioural features to each trip."""

    _validate_trips(trips)
    features = trips.copy()
    features["year_month"] = pd.to_datetime(features["start_timestamp_utc"], utc=True).dt.strftime(
        "%Y-%m"
    )
    features["duration_hours"] = features["duration_seconds"] / 3_600.0
    features["hard_brakes_per_100km"] = (
        features["hard_brake_count"] / features["distance_km"] * 100.0
    )
    features["hard_accelerations_per_100km"] = (
        features["hard_acceleration_count"] / features["distance_km"] * 100.0
    )
    features["harsh_cornering_per_100km"] = (
        features["harsh_cornering_count"] / features["distance_km"] * 100.0
    )
    features["speeding_time_fraction"] = np.clip(
        features["speeding_duration_seconds"] / features["duration_seconds"],
        0.0,
        1.0,
    )
    return features


def build_driver_month_features(
    trips: pd.DataFrame, *, minimum_distance_km: float = 1.0
) -> pd.DataFrame:
    """Aggregate trip observations into driver-month modelling records."""

    if minimum_distance_km <= 0:
        raise ValueError("minimum_distance_km must be positive")
    features = build_trip_features(trips)
    features["night_driving_seconds"] = features["duration_seconds"] * features[
        "is_night_trip"
    ].astype(int)
    features["speeding_severity_seconds"] = (
        features["speeding_duration_seconds"] * features["mean_speeding_excess_pct"]
    )

    monthly = (
        features.groupby(["driver_id", "year_month"], as_index=False)
        .agg(
            trip_count=("trip_id", "nunique"),
            distance_km=("distance_km", "sum"),
            driving_seconds=("duration_seconds", "sum"),
            night_driving_seconds=("night_driving_seconds", "sum"),
            speeding_duration_seconds=("speeding_duration_seconds", "sum"),
            speeding_severity_seconds=("speeding_severity_seconds", "sum"),
            hard_brake_count=("hard_brake_count", "sum"),
            hard_acceleration_count=("hard_acceleration_count", "sum"),
            harsh_cornering_count=("harsh_cornering_count", "sum"),
        )
        .sort_values(["driver_id", "year_month"])
        .reset_index(drop=True)
    )
    monthly["driving_hours"] = monthly["driving_seconds"] / 3_600.0
    monthly["night_driving_fraction"] = (
        monthly["night_driving_seconds"] / monthly["driving_seconds"]
    )
    monthly["speeding_time_fraction"] = (
        monthly["speeding_duration_seconds"] / monthly["driving_seconds"]
    )
    monthly["mean_speeding_excess_pct_when_speeding"] = np.where(
        monthly["speeding_duration_seconds"] > 0,
        monthly["speeding_severity_seconds"] / monthly["speeding_duration_seconds"],
        0.0,
    )

    credible_distance = monthly["distance_km"].where(monthly["distance_km"] >= minimum_distance_km)
    monthly["hard_brakes_per_100km"] = monthly["hard_brake_count"] / credible_distance * 100.0
    monthly["hard_accelerations_per_100km"] = (
        monthly["hard_acceleration_count"] / credible_distance * 100.0
    )
    monthly["harsh_cornering_per_100km"] = (
        monthly["harsh_cornering_count"] / credible_distance * 100.0
    )

    return monthly.drop(
        columns=[
            "driving_seconds",
            "night_driving_seconds",
            "speeding_duration_seconds",
            "speeding_severity_seconds",
        ]
    )
