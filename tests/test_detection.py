"""Acceptance and regression tests for 1 Hz event detection.

Controlled traces and explicit test thresholds make expected counts independent of the
project's configurable assumptions. Direct signal tests isolate duration and merging rules
from smoothing; complete-trace tests exercise the detector's composition.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.detection import (
    EventCounts,
    count_banded_events,
    count_speeding_seconds,
    detect_events,
)
from apexrisk.telemetry.schema import KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND

NO_EVENTS = EventCounts(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Baseline thresholds for tests. Deliberately round, so that a reader can see why an
# expected count is what it is without doing arithmetic, and deliberately DIFFERENT between
# the three signals: if the braking thresholds are ever passed where the acceleration ones
# belong, identical numbers would hide the mistake and these make a test fail immediately.
# Complete-trace examples use values inside bands; direct signal tests cover boundaries.
TEST_BRAKING_THRESHOLDS_MS2 = {"moderate": 2.0, "harsh": 4.0, "severe": 6.0}
TEST_ACCELERATION_THRESHOLDS_MS2 = {"moderate": 3.0, "harsh": 6.0, "severe": 9.0}
TEST_CORNERING_THRESHOLDS_MS2 = {"moderate": 1.0, "harsh": 2.0, "severe": 3.0}


def detection_config(**overrides: object) -> EventDetectionConfig:
    """Build a config for one test, with explicit thresholds and optional overrides.

    Tests must never read configs/telemetry.yaml. Values there are researched assumptions
    that will change; a test asserting on them would break for reasons unrelated to the
    detector being wrong. The round numbers below make each expected count obvious instead.

    Pass any field as a keyword argument to vary it for a single test, for example
    ``detection_config(minimum_event_separation_seconds=3)``, which also documents what
    that test is actually about.
    """

    baseline = EventDetectionConfig(
        # Smallest usable odd window. Wider windows flatten the short events these tests
        # construct, which makes expected counts harder to reason about; the noise test can
        # override this if it wants a stronger smoother.
        smoothing_window_seconds=3,
        # No trimming by default, so traces do not need padding and index arithmetic in the
        # tests stays obvious. Override in whichever test exercises trimming.
        edge_trim_seconds=0,
        # Two seconds, so a single-sample spike is rejected without needing an override.
        minimum_event_duration_seconds=2,
        # Comfortably above the "one second apart merges" case and below the "ten seconds
        # apart are separate" case, so neither test sits near a boundary.
        minimum_event_separation_seconds=5,
        cornering_minimum_speed_kmh=20.0,
        speeding_tolerance_kmh=5.0,
        braking_thresholds_ms2=dict(TEST_BRAKING_THRESHOLDS_MS2),
        acceleration_thresholds_ms2=dict(TEST_ACCELERATION_THRESHOLDS_MS2),
        cornering_thresholds_ms2=dict(TEST_CORNERING_THRESHOLDS_MS2),
    )
    config = replace(baseline, **overrides)

    # A directly constructed EventDetectionConfig is unchecked. Validate here so this
    # helper cannot hand a test a config the loader would reject, such as an even window.
    config.validate()
    return config


def build_trace(segments: list[tuple[int, float]], initial_speed_kmh: float) -> np.ndarray:
    """Build a speed trace from (duration_seconds, acceleration_ms2) segments.

    Lets each test state its input in the terms the detector cares about, for example
    ``[(10, 0.0), (2, -4.0), (10, 0.0)]`` for a two-second hard brake between two steady
    stretches. Returns speeds in km/h. Clamp at zero so no segment produces negative speed.
    """

    if not segments:
        raise ValueError("Segments must not be empty")
    if any(duration < 1 for duration, _ in segments):
        raise ValueError("Every segment must last at least one second or cannot be recorded")

    per_second_acceleration_ms2 = np.concatenate(
        [np.full(duration, acceleration) for duration, acceleration in segments]
    )

    initial_speed_ms = initial_speed_kmh * KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND
    speed_ms = initial_speed_ms + np.concatenate([[0], np.cumsum(per_second_acceleration_ms2)[:-1]])

    speed_ms = np.maximum(speed_ms, 0.0)
    return speed_ms / KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND


def detect_trace(
    speed: np.ndarray,
    *,
    heading: np.ndarray | None = None,
    validity: np.ndarray | None = None,
    speed_limit: float = 300.0,
    config: EventDetectionConfig | None = None,
) -> EventCounts:
    """Default to straight, valid driving below the speeding limit for event tests."""

    return detect_events(
        speed,
        np.zeros(speed.size) if heading is None else heading,
        np.ones(speed.size, dtype=bool) if validity is None else validity,
        speed_limit,
        detection_config() if config is None else config,
    )


# --- single events -----------------------------------------------------------------


def test_sustained_deceleration_is_one_event() -> None:
    """A clean two-second brake beyond the harsh threshold counts exactly once."""

    speed = build_trace([(8, 0.0), (2, -5.0), (8, 0.0)], initial_speed_kmh=100.0)

    assert detect_trace(speed) == replace(NO_EVENTS, brake_events_harsh=1)


def test_long_deceleration_is_still_one_event() -> None:
    """A ten-second brake is one event, not ten. This is the debounce rule."""

    # 240 km/h leaves enough speed for ten seconds at -5 m/s² without hitting zero.
    speed = build_trace([(8, 0.0), (10, -5.0), (8, 0.0)], initial_speed_kmh=240.0)

    assert detect_trace(speed) == replace(NO_EVENTS, brake_events_harsh=1)


def test_deceleration_below_threshold_is_not_an_event() -> None:
    """Gentle braking produces no event at all."""

    speed = build_trace([(8, 0.0), (8, -1.0), (8, 0.0)], initial_speed_kmh=80.0)

    assert detect_trace(speed) == NO_EVENTS


def test_single_sample_spike_is_not_an_event() -> None:
    """A crossing shorter than the minimum event duration is rejected as noise."""

    config = detection_config()
    assert count_banded_events(
        np.array([0.0, 0.0, 7.0, 0.0, 0.0]), TEST_BRAKING_THRESHOLDS_MS2, config
    ) == {"moderate": 0, "harsh": 0, "severe": 0}

    speed = np.full(30, 60.0)
    speed[15] = 150.0
    assert detect_trace(speed, speed_limit=60.0) == NO_EVENTS


# --- separation and banding --------------------------------------------------------


@pytest.mark.parametrize("gap_seconds", [1, 4])
def test_closely_spaced_crossings_merge_into_one_event(gap_seconds: int) -> None:
    """Gaps shorter than the five-second separation merge qualifying runs."""

    signal = np.concatenate(([0.0, 5.0, 5.0], np.zeros(gap_seconds), [5.0, 5.0, 0.0]))

    assert count_banded_events(signal, TEST_BRAKING_THRESHOLDS_MS2, detection_config()) == {
        "moderate": 0,
        "harsh": 1,
        "severe": 0,
    }


@pytest.mark.parametrize("gap_seconds", [5, 10])
def test_widely_spaced_crossings_are_separate_events(gap_seconds: int) -> None:
    """A gap exactly at the separation limit, or longer, leaves two events."""

    signal = np.concatenate(([0.0, 5.0, 5.0], np.zeros(gap_seconds), [5.0, 5.0, 0.0]))

    assert count_banded_events(signal, TEST_BRAKING_THRESHOLDS_MS2, detection_config()) == {
        "moderate": 0,
        "harsh": 2,
        "severe": 0,
    }


@pytest.mark.parametrize(
    ("samples", "expected_band"),
    [([3.0, 3.0], "moderate"), ([3.0, 5.0], "harsh"), ([3.0, 7.0, 3.0], "severe")],
)
def test_severity_bands_are_mutually_exclusive(samples: list[float], expected_band: str) -> None:
    """The peak of a qualifying event determines its one and only severity band."""

    expected = {"moderate": 0, "harsh": 0, "severe": 0}
    expected[expected_band] = 1

    assert (
        count_banded_events(
            np.array([0.0, *samples, 0.0]), TEST_BRAKING_THRESHOLDS_MS2, detection_config()
        )
        == expected
    )


def test_hard_brake_count_sums_harsh_and_severe() -> None:
    """The legacy column keeps its meaning: harsh + severe, excluding moderate."""

    # Three separated brakes peak in moderate, harsh and severe respectively.
    speed = build_trace(
        [(8, 0.0), (4, -3.0), (10, 0.0), (4, -5.0), (10, 0.0), (4, -7.0), (8, 0.0)],
        initial_speed_kmh=240.0,
    )

    counts = detect_trace(speed)

    assert counts == replace(
        NO_EVENTS, brake_events_moderate=1, brake_events_harsh=1, brake_events_severe=1
    )
    assert counts.hard_brake_count == 2


# --- robustness --------------------------------------------------------------------


def test_noise_at_constant_speed_produces_no_events() -> None:
    """The acceptance criterion for smoothing.

    A jittery trace at unchanging true speed must yield zero events. If this fails, every
    event rate the pipeline reports afterwards is noise, and no amount of calibration
    downstream will fix it.
    """

    rng = np.random.default_rng(2026)
    speed = 60.0 + rng.normal(0.0, 1.8, 600)
    speed[300] += 50.0  # An isolated GPS spike must also disappear under the median.

    assert detect_trace(speed, speed_limit=60.0) == NO_EVENTS


def test_no_event_is_detected_across_a_dropout() -> None:
    """Differencing across an invalid stretch would invent a huge phantom event."""

    # Even a one-second dropout is a hard boundary, regardless of merge separation.
    speed = np.concatenate((np.full(10, 100.0), [np.nan], np.full(10, 20.0)))
    heading = np.concatenate((np.zeros(10), [np.nan], np.full(10, 90.0)))

    assert detect_trace(speed, heading=heading, validity=np.isfinite(speed)) == NO_EVENTS


def test_segments_either_side_of_a_dropout_are_still_searched() -> None:
    """A dropout must not discard the valid data around it."""

    segment = build_trace([(5, 0.0), (4, -5.0), (5, 0.0)], initial_speed_kmh=100.0)
    speed = np.concatenate((segment, [np.nan], segment))
    heading = np.where(np.isfinite(speed), 0.0, np.nan)

    assert detect_trace(speed, heading=heading, validity=np.isfinite(speed)) == replace(
        NO_EVENTS, brake_events_harsh=2
    )


# --- cornering ---------------------------------------------------------------------


def test_heading_change_across_north_is_not_a_corner_event() -> None:
    """359 to 1 degree is a two-degree turn. Unwrapped arithmetic reads it as -358."""

    # Multiple consecutive wraps exercise duration filtering as well as angle wrapping.
    heading = np.array([359.0, 1.0] * 10)

    assert detect_trace(np.full(heading.size, 36.0), heading=heading) == NO_EVENTS


def test_sharp_turn_below_minimum_speed_is_not_a_corner_event() -> None:
    """A tight turn at walking pace is a parking manoeuvre, not harsh cornering."""

    heading = np.arange(12) * 60.0 % 360.0
    # At 5 km/h this turn exceeds the 1 m/s² moderate threshold, so the gate matters.
    assert detect_trace(np.full(heading.size, 5.0), heading=heading) == NO_EVENTS


@pytest.mark.parametrize("speed_kmh", [20.0, 36.0])
def test_same_turn_above_minimum_speed_is_a_corner_event(speed_kmh: float) -> None:
    """The same turn counts at and above the gate, with one severe corner event."""

    heading = np.arange(12) * 60.0 % 360.0

    assert detect_trace(np.full(heading.size, speed_kmh), heading=heading) == replace(
        NO_EVENTS, corner_events_severe=1
    )


# --- speeding ----------------------------------------------------------------------


def test_speeding_is_counted_only_above_limit_plus_tolerance() -> None:
    """Sitting exactly on the limit with noise must not register as speeding."""

    config = detection_config()
    # Equality at 55 does not qualify; the next representable value above it does.
    speed = np.array([49.0, 50.0, 54.0, 55.0, np.nextafter(55.0, np.inf), 56.0, np.nan])
    assert count_speeding_seconds(speed, 50.0, config) == 2

    assert detect_trace(np.full(20, 55.0), speed_limit=50.0) == NO_EVENTS


def test_sustained_speeding_seconds_are_counted() -> None:
    """A clear overspeed stretch is counted second by second, not as one event."""

    speed = np.concatenate((np.full(8, 50.0), np.full(12, 60.0), np.full(8, 50.0)))

    # Disable derivative events to isolate the speeding exposure count.
    config = detection_config(minimum_event_duration_seconds=100)
    assert detect_trace(speed, speed_limit=50.0, config=config) == replace(
        NO_EVENTS, speeding_seconds=12
    )


# --- contract ----------------------------------------------------------------------


def test_detection_is_deterministic() -> None:
    """Identical input gives identical output. The detector holds no state and no RNG."""

    speed = build_trace([(8, 0.0), (4, -5.0), (8, 0.0)], initial_speed_kmh=100.0)
    config = detection_config()
    first = detect_trace(speed, config=config)

    assert first == replace(NO_EVENTS, brake_events_harsh=1)
    # An intervening different trip must not leak counts into the next call.
    assert detect_trace(np.full(20, 60.0), config=config) == NO_EVENTS
    assert detect_trace(speed, config=config) == first


def test_detection_does_not_mutate_its_inputs() -> None:
    """Arrays passed in are unchanged afterwards; smoothing must not write in place."""

    speed = build_trace([(8, 0.0), (4, -5.0), (8, 0.0)], initial_speed_kmh=100.0)
    heading = np.arange(speed.size, dtype=float)
    validity = np.ones(speed.size, dtype=bool)
    validity[4] = False
    speed[4] = heading[4] = np.nan
    arrays = (speed, heading, validity)
    originals = tuple(array.copy() for array in arrays)
    config = detection_config()
    original_config = replace(
        config,
        braking_thresholds_ms2=dict(config.braking_thresholds_ms2),
        acceleration_thresholds_ms2=dict(config.acceleration_thresholds_ms2),
        cornering_thresholds_ms2=dict(config.cornering_thresholds_ms2),
    )
    for array in arrays:
        array.flags.writeable = False

    detect_trace(speed, heading=heading, validity=validity, config=config)

    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)
    assert config == original_config


# --- regressions and boundaries -----------------------------------------------------


@pytest.mark.parametrize(
    "signal",
    [
        [0.0, 3.0, 3.0, 0.0, 7.0, 0.0],  # A later, rejected severe spike.
        [0.0, 3.0, 3.0, 0.0, 7.0],  # The rejected spike is the final sample.
        [0.0, 3.0, 3.0, 0.0, np.nan, 7.0, 0.0],
    ],
)
def test_rejected_later_spike_does_not_upgrade_an_earlier_event(signal: list[float]) -> None:
    """Severity uses the event's stop as well as its start, never the remaining trace."""

    assert count_banded_events(
        np.array(signal), TEST_BRAKING_THRESHOLDS_MS2, detection_config()
    ) == {"moderate": 1, "harsh": 0, "severe": 0}


def test_separated_events_keep_their_own_peak_severity() -> None:
    signal = np.array([3.0, 3.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 7.0, 7.0])

    assert count_banded_events(
        signal,
        TEST_BRAKING_THRESHOLDS_MS2,
        detection_config(minimum_event_separation_seconds=2),
    ) == {"moderate": 1, "harsh": 1, "severe": 1}


@pytest.mark.parametrize(
    ("magnitude", "band"), [(2.0, "moderate"), (4.0, "harsh"), (6.0, "severe")]
)
def test_event_thresholds_are_inclusive(magnitude: float, band: str) -> None:
    expected = {"moderate": 0, "harsh": 0, "severe": 0}
    expected[band] = 1

    assert (
        count_banded_events(
            np.array([magnitude, magnitude]), TEST_BRAKING_THRESHOLDS_MS2, detection_config()
        )
        == expected
    )


def test_short_runs_are_rejected_before_merging() -> None:
    assert count_banded_events(
        np.array([0.0, 7.0, 0.0, 7.0, 0.0]),
        TEST_BRAKING_THRESHOLDS_MS2,
        detection_config(),
    ) == {"moderate": 0, "harsh": 0, "severe": 0}


@pytest.mark.parametrize(
    ("acceleration", "expected"),
    [
        (4.0, replace(NO_EVENTS, accel_events_moderate=1)),
        (7.0, replace(NO_EVENTS, accel_events_harsh=1)),
        (10.0, replace(NO_EVENTS, accel_events_severe=1)),
    ],
)
def test_acceleration_uses_its_own_thresholds_and_sign(
    acceleration: float, expected: EventCounts
) -> None:
    speed = build_trace([(8, 0.0), (4, acceleration), (8, 0.0)], initial_speed_kmh=20.0)

    counts = detect_trace(speed)

    assert counts == expected
    assert counts.hard_acceleration_count == int(acceleration >= 6.0)


@pytest.mark.parametrize(
    ("yaw_degrees", "expected"),
    [
        (9.0, replace(NO_EVENTS, corner_events_moderate=1)),
        (15.0, replace(NO_EVENTS, corner_events_harsh=1)),
        (21.0, replace(NO_EVENTS, corner_events_severe=1)),
    ],
)
@pytest.mark.parametrize("direction", [-1, 1])
def test_cornering_uses_its_own_thresholds_in_both_directions(
    yaw_degrees: float, expected: EventCounts, direction: int
) -> None:
    # At 36 km/h (10 m/s), 9/15/21 degrees per second give 1.57/2.62/3.67 m/s².
    heading = np.arange(12) * yaw_degrees * direction % 360.0

    counts = detect_trace(np.full(heading.size, 36.0), heading=heading)

    assert counts == expected
    assert counts.harsh_cornering_count == int(yaw_degrees >= 15.0)


@pytest.mark.parametrize("length", [0, 1, 2])
def test_short_segments_count_speeding_without_derived_events(length: int) -> None:
    assert detect_trace(np.full(length, 60.0), speed_limit=50.0) == replace(
        NO_EVENTS, speeding_seconds=length
    )


def test_all_invalid_samples_return_zero_counts() -> None:
    assert (
        detect_trace(
            np.full(10, np.nan), heading=np.full(10, np.nan), validity=np.zeros(10, dtype=bool)
        )
        == NO_EVENTS
    )


def test_edge_trimming_applies_to_each_segment_but_not_speeding() -> None:
    # With no trim, each three-sample segment contains a sustained severe corner.
    speed = np.array([60.0, 60.0, 60.0, np.nan, 60.0, 60.0, 60.0])
    heading = np.array([0.0, 60.0, 120.0, np.nan, 0.0, 60.0, 120.0])
    validity = np.isfinite(speed)
    assert detect_trace(speed, heading=heading, validity=validity, speed_limit=50.0) == replace(
        NO_EVENTS, corner_events_severe=2, speeding_seconds=6
    )

    assert detect_trace(
        speed,
        heading=heading,
        validity=validity,
        speed_limit=50.0,
        config=detection_config(edge_trim_seconds=2),
    ) == replace(NO_EVENTS, speeding_seconds=6)


@pytest.mark.parametrize(
    ("speed", "heading", "validity", "limit", "message"),
    [
        (np.zeros((2, 2)), np.zeros(4), np.ones(4, dtype=bool), 50.0, "one-dimensional"),
        (np.zeros(3), np.zeros(2), np.ones(3, dtype=bool), 50.0, "same shape"),
        (np.zeros(3), np.zeros(3), np.ones(3), 50.0, "boolean"),
        (np.array([20.0, np.nan]), np.zeros(2), np.ones(2, dtype=bool), 50.0, "Valid speed"),
        (np.array([20.0, np.inf]), np.zeros(2), np.ones(2, dtype=bool), 50.0, "Valid speed"),
        (np.zeros(2), np.array([0.0, np.nan]), np.ones(2, dtype=bool), 50.0, "Valid heading"),
        (np.zeros(2), np.array([0.0, np.inf]), np.ones(2, dtype=bool), 50.0, "Valid heading"),
        (np.zeros(2), np.zeros(2), np.ones(2, dtype=bool), -1.0, "Speed limit"),
        (np.zeros(2), np.zeros(2), np.ones(2, dtype=bool), np.nan, "Speed limit"),
        (np.zeros(2), np.zeros(2), np.ones(2, dtype=bool), np.inf, "Speed limit"),
    ],
)
def test_invalid_detector_inputs_raise_clear_errors(
    speed: np.ndarray, heading: np.ndarray, validity: np.ndarray, limit: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_events(speed, heading, validity, limit, detection_config())


@pytest.mark.parametrize("infinite", [-np.inf, np.inf])
def test_counting_helpers_reject_infinity(infinite: float) -> None:
    with pytest.raises(ValueError, match="infinite"):
        count_banded_events(np.array([infinite]), TEST_BRAKING_THRESHOLDS_MS2, detection_config())
    with pytest.raises(ValueError, match="infinite"):
        count_speeding_seconds(np.array([infinite]), 50.0, detection_config())
