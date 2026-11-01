"""Focused tests for task-04 telemetry signal preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.detection import (
    lateral_acceleration_ms2,
    longitudinal_acceleration_ms2,
    smooth_speed_kmh,
    trim_segment_edges,
    valid_sample_segments,
)


def preprocessing_config(**overrides: object) -> EventDetectionConfig:
    values: dict[str, object] = {
        "smoothing_window_seconds": 3,
        "edge_trim_seconds": 2,
        "minimum_event_duration_seconds": 2,
        "minimum_event_separation_seconds": 3,
        "cornering_minimum_speed_kmh": 20.0,
        "speeding_tolerance_kmh": 5.0,
        "braking_thresholds_ms2": {"moderate": 2.0, "harsh": 3.0, "severe": 5.0},
        "acceleration_thresholds_ms2": {"moderate": 2.0, "harsh": 3.0, "severe": 5.0},
        "cornering_thresholds_ms2": {"moderate": 3.0, "harsh": 4.0, "severe": 6.0},
    }
    values.update(overrides)
    config = EventDetectionConfig(**values)  # type: ignore[arg-type]
    config.validate()
    return config


@pytest.mark.parametrize(
    ("validity", "expected"),
    [
        ([False, True, True, True, False, True, True], [(1, 4), (5, 7)]),
        ([True, True, True], [(0, 3)]),
        ([False, False, False], []),
        ([], []),
    ],
)
def test_valid_sample_segments_return_half_open_ranges(
    validity: list[bool],
    expected: list[tuple[int, int]],
) -> None:
    assert valid_sample_segments(np.asarray(validity, dtype=bool)) == expected


def test_valid_sample_segments_reject_non_boolean_or_non_vector_input() -> None:
    with pytest.raises(ValueError, match="boolean"):
        valid_sample_segments(np.array([1, 0, 1]))
    with pytest.raises(ValueError, match="one-dimensional"):
        valid_sample_segments(np.ones((2, 2), dtype=bool))


def test_segmentation_matches_a_simple_reference_for_random_masks() -> None:
    rng = np.random.default_rng(2026)

    for length in range(51):
        validity = rng.random(length) > 0.35
        expected: list[tuple[int, int]] = []
        start: int | None = None
        for index, is_valid in enumerate(validity):
            if is_valid and start is None:
                start = index
            elif not is_valid and start is not None:
                expected.append((start, index))
                start = None
        if start is not None:
            expected.append((start, length))

        assert valid_sample_segments(validity) == expected


def test_median_smoothing_removes_an_isolated_speed_spike() -> None:
    speed = np.array([30.0, 30.0, 100.0, 30.0, 30.0])

    smoothed = smooth_speed_kmh(speed, preprocessing_config())

    assert smoothed == pytest.approx([30.0, 30.0, 30.0, 30.0, 30.0])


def test_smoothing_does_not_mutate_its_input() -> None:
    speed = np.array([20.0, 21.0, 22.0])
    original = speed.copy()

    smoothed = smooth_speed_kmh(speed, preprocessing_config())
    smoothed[0] = -1.0

    assert np.array_equal(speed, original)


def test_smoothing_matches_an_edge_padded_median_reference() -> None:
    rng = np.random.default_rng(44)
    config = preprocessing_config()

    for length in range(1, 21):
        speed = rng.uniform(0.0, 120.0, size=length)
        padded = np.pad(speed, (1, 1), mode="edge")
        expected = np.array([np.median(padded[index : index + 3]) for index in range(length)])

        assert smooth_speed_kmh(speed, config) == pytest.approx(expected)


def test_smoothing_rejects_an_invalid_sample_inside_a_valid_segment() -> None:
    with pytest.raises(ValueError, match="valid speed segment"):
        smooth_speed_kmh(np.array([20.0, np.nan, 20.0]), preprocessing_config())


def test_longitudinal_acceleration_is_central_and_index_aligned() -> None:
    # 36, 39.6 and 43.2 km/h are 10, 11 and 12 m/s. The central derivative is 1 m/s^2.
    acceleration = longitudinal_acceleration_ms2(np.array([36.0, 39.6, 43.2]))

    assert np.isnan(acceleration[[0, 2]]).all()
    assert acceleration[1] == pytest.approx(1.0)


def test_lateral_acceleration_wraps_heading_across_north() -> None:
    # At 36 km/h, a two-degree change in one second is about 0.349 m/s^2.
    lateral = lateral_acceleration_ms2(
        np.array([36.0, 36.0]),
        np.array([359.0, 1.0]),
    )

    assert np.isnan(lateral[0])
    assert lateral[1] == pytest.approx(10.0 * np.deg2rad(2.0))


def test_lateral_acceleration_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        lateral_acceleration_ms2(np.array([30.0, 30.0]), np.array([90.0]))


def test_constant_heading_has_zero_lateral_acceleration() -> None:
    lateral = lateral_acceleration_ms2(
        np.full(5, 50.0),
        np.full(5, 180.0),
    )

    assert np.isnan(lateral[0])
    assert lateral[1:] == pytest.approx(np.zeros(4))


def test_edge_trim_is_applied_to_each_segment_without_mutating_input() -> None:
    signal = np.arange(7, dtype=float)
    original = signal.copy()

    trimmed = trim_segment_edges(signal, preprocessing_config())
    trimmed[0] = -1.0

    assert np.array_equal(signal, original)
    assert trimmed[1:] == pytest.approx([3.0, 4.0])


def test_edge_trim_returns_empty_for_a_segment_with_no_interior() -> None:
    trimmed = trim_segment_edges(np.arange(4), preprocessing_config())

    assert trimmed.size == 0


def test_dropout_prevents_a_derivative_between_different_speed_levels() -> None:
    speed = np.array([36.0, 36.0, np.nan, 72.0, 72.0])
    validity = np.array([True, True, False, True, True])

    accelerations = []
    for start, stop in valid_sample_segments(validity):
        smoothed = smooth_speed_kmh(speed[start:stop], preprocessing_config())
        accelerations.append(longitudinal_acceleration_ms2(smoothed))

    assert all(np.isnan(acceleration).all() for acceleration in accelerations)



def test_lateral_acceleration_does_not_overflow_when_averaging_finite_speeds() -> None:
    maximum = np.finfo(np.float64).max
    with np.errstate(over="raise", invalid="raise"):
        lateral = lateral_acceleration_ms2(
            np.full(3, maximum), np.array([0.0, 180.0, 0.0])
        )

    assert np.isnan(lateral[0])
    assert np.isfinite(lateral[1:]).all()
    np.testing.assert_allclose(lateral[1:], (maximum / 3.6) * np.pi)
