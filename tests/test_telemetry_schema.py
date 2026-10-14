"""Tests for the telemetry data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexrisk.telemetry.schema import (
    TELEMETRY_COLUMNS,
    check_trace_invariants,
    summarise_valid_exposure,
    validate_telemetry_frame,
    wrapped_heading_change_degrees,
)


def build_constant_speed_trace(speed_kmh: float, duration_seconds: int) -> np.ndarray:
    """Return a flat speed trace, the simplest input satisfying the distance invariant."""

    return np.full(duration_seconds, speed_kmh, dtype=np.float64)


def build_valid_telemetry_frame(trip_id: str = "TRIP-000000001", duration: int = 5) -> pd.DataFrame:
    """Return a minimal telemetry table that satisfies every schema rule."""

    return pd.DataFrame(
        {
            "trip_id": pd.array([trip_id] * duration, dtype="string"),
            "second_index": np.arange(duration, dtype="uint32"),
            "speed_kmh": np.full(duration, 50.0, dtype="float32"),
            "heading_degrees": np.full(duration, 90.0, dtype="float32"),
            "is_gps_valid": np.ones(duration, dtype="bool"),
        }
    )


def test_valid_frame_passes_validation() -> None:
    validate_telemetry_frame(build_valid_telemetry_frame())


def test_invalid_fix_uses_nan_without_removing_the_sample() -> None:
    frame = build_valid_telemetry_frame()
    frame.loc[2, ["speed_kmh", "heading_degrees"]] = np.nan
    frame.loc[2, "is_gps_valid"] = False

    validate_telemetry_frame(frame)
    assert frame["second_index"].tolist() == [0, 1, 2, 3, 4]


def test_invalid_fix_with_placeholder_measurements_is_rejected() -> None:
    frame = build_valid_telemetry_frame()
    frame.loc[2, "is_gps_valid"] = False

    with pytest.raises(ValueError, match="Invalid GPS samples must store NaN"):
        validate_telemetry_frame(frame)


def test_valid_fix_with_missing_measurement_is_rejected() -> None:
    frame = build_valid_telemetry_frame()
    frame.loc[2, "speed_kmh"] = np.nan

    with pytest.raises(ValueError, match="Valid GPS samples must have finite"):
        validate_telemetry_frame(frame)


def test_missing_column_is_rejected() -> None:
    frame = build_valid_telemetry_frame().drop(columns=["heading_degrees"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_telemetry_frame(frame)


def test_negative_speed_is_rejected() -> None:
    frame = build_valid_telemetry_frame()
    frame.loc[2, "speed_kmh"] = -0.5
    with pytest.raises(ValueError, match="non-negative"):
        validate_telemetry_frame(frame)


def test_heading_outside_circle_is_rejected() -> None:
    frame = build_valid_telemetry_frame()
    frame.loc[1, "heading_degrees"] = 360.0
    with pytest.raises(ValueError, match=r"\[0, 360\)"):
        validate_telemetry_frame(frame)


def test_dropped_sample_must_be_flagged_not_removed() -> None:
    """A gap in second_index means a sample was deleted rather than flagged invalid."""

    frame = build_valid_telemetry_frame().drop(index=2).reset_index(drop=True)
    with pytest.raises(ValueError, match="non-contiguous"):
        validate_telemetry_frame(frame)


def test_column_order_is_stable() -> None:
    """Guards against a column being renamed without the contract document being updated."""

    assert TELEMETRY_COLUMNS == (
        "trip_id",
        "second_index",
        "speed_kmh",
        "heading_degrees",
        "is_gps_valid",
    )


def test_heading_change_wraps_across_north() -> None:
    """359 degrees to 1 degree is a +2 degree turn, not a -358 degree one."""

    changes = wrapped_heading_change_degrees(np.array([359.0, 1.0, 359.0]))
    assert changes.tolist() == pytest.approx([2.0, -2.0])


def test_valid_exposure_excludes_dropout_seconds_and_distance() -> None:
    speed = np.array([36.0, np.nan, 36.0, 36.0])
    validity = np.array([True, False, True, True])

    exposure = summarise_valid_exposure(speed, validity)

    assert exposure.gps_valid_seconds == 3
    assert exposure.gps_valid_distance_km == pytest.approx(0.03)
    assert exposure.gps_coverage_fraction == pytest.approx(0.75)


def test_empty_trace_has_zero_valid_exposure() -> None:
    exposure = summarise_valid_exposure(
        np.array([], dtype=float),
        np.array([], dtype=bool),
    )

    assert exposure.gps_valid_seconds == 0
    assert exposure.gps_valid_distance_km == 0.0
    assert exposure.gps_coverage_fraction == 0.0


def test_trace_matching_its_summary_passes() -> None:
    duration_seconds = 3_600
    speed = build_constant_speed_trace(36.0, duration_seconds)
    check_trace_invariants(
        speed,
        np.zeros(duration_seconds),
        distance_km=36.0,
        duration_seconds=duration_seconds,
        maximum_speed_kmh=36.0,
    )


def test_trace_of_wrong_length_is_rejected() -> None:
    speed = build_constant_speed_trace(36.0, 100)
    with pytest.raises(ValueError, match="samples but the trip summary"):
        check_trace_invariants(
            speed,
            np.zeros(100),
            distance_km=1.0,
            duration_seconds=200,
            maximum_speed_kmh=36.0,
        )


def test_true_trace_with_missing_measurement_is_rejected() -> None:
    speed = build_constant_speed_trace(36.0, 10)
    speed[4] = np.nan

    with pytest.raises(ValueError, match="True speed and heading traces"):
        check_trace_invariants(
            speed,
            np.zeros(10),
            distance_km=0.1,
            duration_seconds=10,
            maximum_speed_kmh=36.0,
        )


def test_trace_not_reconciling_with_distance_is_rejected() -> None:
    """The integral of speed must equal the recorded distance at one sample per second."""

    duration_seconds = 3_600
    speed = build_constant_speed_trace(36.0, duration_seconds)
    with pytest.raises(ValueError, match="km but the trip summary records"):
        check_trace_invariants(
            speed,
            np.zeros(duration_seconds),
            distance_km=50.0,
            duration_seconds=duration_seconds,
            maximum_speed_kmh=36.0,
        )


def test_implausible_yaw_rate_is_rejected() -> None:
    """Catches unwrapped angle arithmetic, which would manufacture phantom corner events."""

    duration_seconds = 10
    speed = build_constant_speed_trace(36.0, duration_seconds)
    heading = np.zeros(duration_seconds)
    heading[5] = 120.0
    with pytest.raises(ValueError, match="yaw rate"):
        check_trace_invariants(
            speed,
            heading,
            distance_km=0.1,
            duration_seconds=duration_seconds,
            maximum_speed_kmh=36.0,
        )
