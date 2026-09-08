"""Harsh-event detection from 1 Hz telemetry traces.

The detector is the socket that a real telematics feed plugs into, so it deliberately
knows nothing about drivers, trips or DataFrames: plain arrays in, counts out. Adapting a
stored table into arrays belongs to the caller, not here.

Every function below is a stub. The docstrings state what each must do and the traps that
apply; the implementations are the work. See ``docs/telemetry_contract.md`` for the data
contract and ``configs/telemetry.yaml`` for the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexrisk.config import EventDetectionConfig


@dataclass(frozen=True)
class EventCounts:
    """Detected events for a single trip, banded by severity.

    Bands are mutually exclusive: an event is counted once, in its highest band only.
    """

    brake_events_moderate: int
    brake_events_harsh: int
    brake_events_severe: int
    accel_events_moderate: int
    accel_events_harsh: int
    accel_events_severe: int
    corner_events_moderate: int
    corner_events_harsh: int
    corner_events_severe: int
    speeding_seconds: int

    # The three properties below preserve the meaning the existing trip summary columns
    # already have, so that features.behavioural and simulation.claims need no changes.
    # Contract section 4 defines this mapping.

    @property
    def hard_brake_count(self) -> int:
        """Events at or above the harsh braking threshold."""

        return self.brake_events_harsh + self.brake_events_severe

    @property
    def hard_acceleration_count(self) -> int:
        """Events at or above the harsh acceleration threshold."""

        return self.accel_events_harsh + self.accel_events_severe

    @property
    def harsh_cornering_count(self) -> int:
        """Events at or above the harsh cornering threshold."""

        return self.corner_events_harsh + self.corner_events_severe


def valid_sample_segments(is_gps_valid: np.ndarray) -> list[tuple[int, int]]:
    """Split a trace into contiguous runs of valid samples, as (start, stop) index pairs.

    Detection runs inside each segment independently. Differencing across a dropout would
    compare speeds seconds or minutes apart and manufacture an enormous phantom event, so
    no derived quantity may ever span a gap.

    Approach: find where the validity mask changes by differencing it, and turn those
    boundaries into index pairs. No Python loop over samples.
    """

    raise NotImplementedError


def smooth_speed_kmh(speed_kmh: np.ndarray, config: EventDetectionConfig) -> np.ndarray:
    """Return the speed trace with GPS noise suppressed, ready for differentiation.

    Use a rolling median of ``config.smoothing_window_seconds``, not a rolling mean. GPS
    speed carries isolated spikes; a median removes a spike outright, while a mean spreads
    it across the whole window and creates two fake accelerations either side of it.

    Differentiating an unsmoothed trace is the single most common way to produce a detector
    that reports hundreds of events on a trace containing none.
    """

    raise NotImplementedError


def longitudinal_acceleration_ms2(smoothed_speed_kmh: np.ndarray) -> np.ndarray:
    """Return along-track acceleration in m/s^2, positive for speeding up.

    Points to watch:

    1. Convert km/h to m/s exactly once, using the constant in ``telemetry.schema``.
    2. Prefer a central difference over a backward one: it amplifies noticeably less noise
       for one extra line of arithmetic.
    3. A central difference is undefined at the first and last sample. Decide what the
       returned array's length and alignment are, and state it here, because every
       downstream index depends on that choice.
    """

    raise NotImplementedError


def lateral_acceleration_ms2(
    smoothed_speed_kmh: np.ndarray,
    heading_degrees: np.ndarray,
) -> np.ndarray:
    """Return cornering acceleration in m/s^2 as a positive magnitude.

    Lateral acceleration is approximately speed multiplied by yaw rate, with speed in m/s
    and yaw rate in radians per second.

    Points to watch:

    1. Difference headings with ``schema.wrapped_heading_change_degrees``. Naive subtraction
       reads a turn from 359 to 1 degree as -358 degrees and invents a severe event.
    2. Convert degrees per second to radians per second before multiplying.
    3. Direction of turn does not matter, only magnitude.
    4. Gating on ``config.cornering_minimum_speed_kmh`` belongs in the caller, not here, so
       that this stays a pure kinematic conversion.
    """

    raise NotImplementedError


def count_banded_events(
    signal_ms2: np.ndarray,
    thresholds: dict[str, float],
    config: EventDetectionConfig,
) -> dict[str, int]:
    """Count sustained, debounced threshold crossings, one count per severity band.

    This is the core of the detector and where the subtle bugs live. A four-second heavy
    brake is one event, not four.

    Suggested algorithm, all vectorised:

    1. Build a boolean mask of samples at or beyond the band's threshold.
    2. Find run boundaries by differencing the mask and taking the non-zero positions.
       That yields every run's start and end without iterating over samples.
    3. Discard runs shorter than ``config.minimum_event_duration_seconds``.
    4. Merge runs separated by less than ``config.minimum_event_separation_seconds``.
    5. Count what remains.

    Bands must come out mutually exclusive: an event reaching the severe threshold is
    counted in ``severe`` only, never additionally in ``harsh`` or ``moderate``. Decide
    whether to achieve that by classifying each run by its peak magnitude or by subtracting
    higher bands afterwards, and say which here — the two differ for a run that briefly
    touches a higher band.
    """

    raise NotImplementedError


def count_speeding_seconds(
    smoothed_speed_kmh: np.ndarray,
    speed_limit_kmh: float,
    config: EventDetectionConfig,
) -> int:
    """Count seconds spent above the speed limit plus the configured tolerance.

    The tolerance is not a detail. Observed speed is noisy, so counting every sample above
    the bare limit puts roughly half the samples of a driver travelling exactly at the limit
    over it, badly overstating speeding precisely where most drivers sit. Contract section 4
    requires the published figure to be described as a tolerance-adjusted estimate.
    """

    raise NotImplementedError


def detect_events(
    speed_kmh: np.ndarray,
    heading_degrees: np.ndarray,
    is_gps_valid: np.ndarray,
    speed_limit_kmh: float,
    config: EventDetectionConfig,
) -> EventCounts:
    """Detect all harsh events and speeding seconds in one trace.

    Composes the functions above: split into valid segments, smooth, derive the
    longitudinal and lateral signals, count banded events within each segment, and sum
    across segments.

    Must be pure and stateless. No random draws, no mutation of the inputs, and identical
    output for identical input — it will be called once per trip across roughly a million
    trips, so it must also stay free of Python-level loops over samples.

    Segments shorter than the smoothing window cannot produce a usable derivative. Decide
    what happens to them and record the decision here.
    """

    raise NotImplementedError
