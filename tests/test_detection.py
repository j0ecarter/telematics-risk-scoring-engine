"""Specification for harsh-event detection.

These tests are the detector's specification, written before the implementation. Each one
names a property the detector must have; the body is left to be written alongside the code
it checks.

Remove the module-level skip below once you start, so the suite reports honest failures
rather than silently passing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.schema import KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND


# Baseline thresholds for tests. Deliberately round, so that a reader can see why an
# expected count is what it is without doing arithmetic, and deliberately DIFFERENT between
# the three signals: if the braking thresholds are ever passed where the acceleration ones
# belong, identical numbers would hide the mistake and these make a test fail immediately.
# Choose test accelerations that sit in the middle of a band, never on a boundary.
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

    # EventDetectionConfig only validates inside from_yaml, so a directly constructed one is
    # otherwise unchecked. Validating here means this helper cannot hand a test a config that
    # the real loader would reject, such as an even smoothing window.
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
    speed_ms = initial_speed_ms + np.concatenate(
        [[0,0], np.cumsum(per_second_acceleration_ms2)[:-1]]
    )

    speed_ms = np.maximum(speed_ms, 0.0)
    return speed_ms / KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND



# --- single events -----------------------------------------------------------------


def test_sustained_deceleration_is_one_event() -> None:
    """A clean two-second brake beyond the harsh threshold counts exactly once."""

    raise NotImplementedError


def test_long_deceleration_is_still_one_event() -> None:
    """A ten-second brake is one event, not ten. This is the debounce rule."""

    raise NotImplementedError


def test_deceleration_below_threshold_is_not_an_event() -> None:
    """Gentle braking produces no event at all."""

    raise NotImplementedError


def test_single_sample_spike_is_not_an_event() -> None:
    """A crossing shorter than the minimum event duration is rejected as noise."""

    raise NotImplementedError


# --- separation and banding --------------------------------------------------------


def test_closely_spaced_crossings_merge_into_one_event() -> None:
    """Two crossings a second apart are one event, per minimum_event_separation_seconds."""

    raise NotImplementedError


def test_widely_spaced_crossings_are_separate_events() -> None:
    """Two crossings ten seconds apart are two events."""

    raise NotImplementedError


def test_severity_bands_are_mutually_exclusive() -> None:
    """An event reaching the severe threshold appears in severe only, not also in harsh."""

    raise NotImplementedError


def test_hard_brake_count_sums_harsh_and_severe() -> None:
    """The legacy column keeps its meaning: harsh + severe, excluding moderate."""

    raise NotImplementedError


# --- robustness --------------------------------------------------------------------


def test_noise_at_constant_speed_produces_no_events() -> None:
    """The acceptance criterion for smoothing.

    A jittery trace at unchanging true speed must yield zero events. If this fails, every
    event rate the pipeline reports afterwards is noise, and no amount of calibration
    downstream will fix it.
    """

    raise NotImplementedError


def test_no_event_is_detected_across_a_dropout() -> None:
    """Differencing across an invalid stretch would invent a huge phantom event."""

    raise NotImplementedError


def test_segments_either_side_of_a_dropout_are_still_searched() -> None:
    """A dropout must not discard the valid data around it."""

    raise NotImplementedError


# --- cornering ---------------------------------------------------------------------


def test_heading_change_across_north_is_not_a_corner_event() -> None:
    """359 to 1 degree is a two-degree turn. Unwrapped arithmetic reads it as -358."""

    raise NotImplementedError


def test_sharp_turn_below_minimum_speed_is_not_a_corner_event() -> None:
    """A tight turn at walking pace is a parking manoeuvre, not harsh cornering."""

    raise NotImplementedError


def test_same_turn_above_minimum_speed_is_a_corner_event() -> None:
    """Counterpart to the previous test: the gate is speed, not the turn itself."""

    raise NotImplementedError


# --- speeding ----------------------------------------------------------------------


def test_speeding_is_counted_only_above_limit_plus_tolerance() -> None:
    """Sitting exactly on the limit with noise must not register as speeding."""

    raise NotImplementedError


def test_sustained_speeding_seconds_are_counted() -> None:
    """A clear overspeed stretch is counted second by second, not as one event."""

    raise NotImplementedError


# --- contract ----------------------------------------------------------------------


def test_detection_is_deterministic() -> None:
    """Identical input gives identical output. The detector holds no state and no RNG."""

    raise NotImplementedError


def test_detection_does_not_mutate_its_inputs() -> None:
    """Arrays passed in are unchanged afterwards; smoothing must not write in place."""

    raise NotImplementedError
