import pandas as pd
import pytest

from apexrisk.features import build_driver_month_features, build_trip_features


def example_trips() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trip_id": ["T-1", "T-2"],
            "driver_id": ["D-1", "D-1"],
            "start_timestamp_utc": pd.to_datetime(
                ["2025-01-02T23:00:00Z", "2025-01-04T12:00:00Z"], utc=True
            ),
            "distance_km": [10.0, 30.0],
            "duration_seconds": [600, 1_800],
            "is_night_trip": [True, False],
            "hard_brake_count": [1, 2],
            "hard_acceleration_count": [0, 3],
            "harsh_cornering_count": [1, 0],
            "speeding_duration_seconds": [60, 180],
            "mean_speeding_excess_pct": [20.0, 10.0],
        }
    )


def test_trip_features_normalise_events_by_distance() -> None:
    result = build_trip_features(example_trips())

    assert result.loc[0, "hard_brakes_per_100km"] == pytest.approx(10.0)
    assert result.loc[1, "hard_accelerations_per_100km"] == pytest.approx(10.0)
    assert result.loc[0, "speeding_time_fraction"] == pytest.approx(0.1)


def test_driver_month_features_are_exposure_weighted() -> None:
    result = build_driver_month_features(example_trips()).iloc[0]

    assert result["trip_count"] == 2
    assert result["distance_km"] == pytest.approx(40.0)
    assert result["night_driving_fraction"] == pytest.approx(0.25)
    assert result["speeding_time_fraction"] == pytest.approx(0.1)
    assert result["mean_speeding_excess_pct_when_speeding"] == pytest.approx(12.5)
    assert result["hard_brakes_per_100km"] == pytest.approx(7.5)
