"""Machine-readable form of the telemetry data contract.

This module is the executable counterpart of ``docs/telemetry_contract.md``. It defines
the published second-by-second telemetry schema and the invariants a generated trace must
satisfy for the trip summary derived from it to be truthful.

Nothing here generates traces or detects events. Keeping the contract in its own module
means the generator and the detector can be developed independently against a shared,
enforced definition rather than against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Sampling interval of the published telemetry. One hertz matches the GPS fix rate of a
# typical insurance dongle; see section 1 of the contract for the limitations it imposes.
TELEMETRY_SAMPLE_INTERVAL_SECONDS = 1

# Single definition of the km/h to m/s conversion, so the factor cannot drift between
# the generator, the detector and these checks.
KILOMETRES_PER_HOUR_TO_METRES_PER_SECOND = 1_000.0 / 3_600.0

# Published columns in order, with the dtype each must use. uint32 rather than uint16 for
# second_index because the current journey generator can in principle produce a trip of
# roughly 81,000 seconds, which overflows uint16.
TELEMETRY_COLUMN_DTYPES: dict[str, str] = {
    "trip_id": "string",
    "second_index": "uint32",
    "speed_kmh": "float32",
    "heading_degrees": "float32",
    "is_gps_valid": "bool",
}

TELEMETRY_COLUMNS: tuple[str, ...] = tuple(TELEMETRY_COLUMN_DTYPES)

# Relative tolerance when reconciling the integral of the speed trace against the trip's
# recorded distance. Loose enough to absorb float32 storage and per-sample rounding,
# tight enough to catch a genuine generator error.
DISTANCE_RECONCILIATION_TOLERANCE = 0.005

# Absolute tolerance, in km/h, when reconciling peak trace speed against the trip's
# recorded maximum speed.
PEAK_SPEED_RECONCILIATION_TOLERANCE_KMH = 0.5

# A vehicle cannot change heading arbitrarily fast. Lateral acceleration is approximately
# speed multiplied by yaw rate, so an unbounded heading change manufactures phantom severe
# cornering events. This is a deliberately generous ceiling intended to catch bugs such as
# unwrapped angle arithmetic, not to model vehicle dynamics precisely.
MAXIMUM_PLAUSIBLE_YAW_RATE_DEGREES_PER_SECOND = 30.0


@dataclass(frozen=True)
class TelemetryExposure:
    """Amount of a trip for which GPS-derived behaviour was observable.

    Event rates and speeding fractions must use these denominators. Using the complete
    trip distance or duration would silently classify GPS dropouts as uneventful driving.
    """

    gps_valid_seconds: int
    gps_valid_distance_km: float
    gps_coverage_fraction: float


def wrapped_heading_change_degrees(heading_degrees: np.ndarray) -> np.ndarray:
    """Return second-to-second heading changes, wrapped to the range (-180, 180].

    Compass headings are circular, so a turn from 359 degrees to 1 degree is a change of
    +2 degrees rather than -358. Naive subtraction is a reliable source of spurious
    cornering events, so every consumer should difference headings through this helper.
    """

    raw_change = np.diff(np.asarray(heading_degrees, dtype=np.float64))
    return (raw_change + 180.0) % 360.0 - 180.0


def validate_telemetry_frame(telemetry: pd.DataFrame) -> None:
    """Check a published telemetry table against the schema contract.

    Raises ``ValueError`` describing the first violated rule. Intended for use at the
    boundary where telemetry is written or read, not inside hot loops.
    """

    missing_columns = [column for column in TELEMETRY_COLUMNS if column not in telemetry.columns]
    if missing_columns:
        raise ValueError(f"Telemetry is missing required columns: {missing_columns}")
    if telemetry.empty:
        raise ValueError("Telemetry must not be empty")
    always_defined_columns = ["trip_id", "second_index", "is_gps_valid"]
    if telemetry[always_defined_columns].isna().to_numpy().any():
        raise ValueError(f"Telemetry must not contain null values in {always_defined_columns}")

    is_gps_valid = telemetry["is_gps_valid"].to_numpy(dtype=bool)
    speed = telemetry["speed_kmh"].to_numpy(dtype=np.float64)
    heading = telemetry["heading_degrees"].to_numpy(dtype=np.float64)

    # An invalid fix has no observed speed or heading. NaN is deliberate here: retaining a
    # plausible-looking zero or last-known value would allow callers to mistake a placeholder
    # for a measurement. The validity flag remains the authoritative segmentation mask.
    if not np.isfinite(speed[is_gps_valid]).all() or not np.isfinite(
        heading[is_gps_valid]
    ).all():
        raise ValueError("Valid GPS samples must have finite speed and heading values")
    if not np.isnan(speed[~is_gps_valid]).all() or not np.isnan(
        heading[~is_gps_valid]
    ).all():
        raise ValueError("Invalid GPS samples must store NaN for speed and heading")

    valid_speed = speed[is_gps_valid]
    if (valid_speed < 0.0).any():
        raise ValueError("speed_kmh must be non-negative, measurement noise included")

    valid_heading = heading[is_gps_valid]
    if ((valid_heading < 0.0) | (valid_heading >= 360.0)).any():
        raise ValueError("heading_degrees must lie in the half-open interval [0, 360)")

    # second_index must run 0, 1, 2, ... without gaps within each trip. A gap would mean a
    # dropped sample was omitted rather than flagged, which the contract forbids: dropouts
    # are represented by is_gps_valid, so that the time axis stays uniform.
    for trip_id, trip_rows in telemetry.groupby("trip_id", sort=False):
        observed_index = trip_rows["second_index"].to_numpy()
        expected_index = np.arange(len(trip_rows), dtype=observed_index.dtype)
        if not np.array_equal(observed_index, expected_index):
            raise ValueError(
                f"Trip {trip_id!r} has a non-contiguous second_index; dropped samples must "
                "be flagged with is_gps_valid rather than removed"
            )


def summarise_valid_exposure(
    speed_kmh: np.ndarray,
    is_gps_valid: np.ndarray,
) -> TelemetryExposure:
    """Return valid time, observed distance and coverage for one telemetry trace.

    ``gps_valid_distance_km`` is the integral of observed speed over valid samples only.
    It is the exposure denominator for GPS-derived event rates. Total trip distance stays
    separate because it also includes periods during which the detector could observe
    neither an event nor speeding.
    """

    speed = np.asarray(speed_kmh, dtype=np.float64)
    validity = np.asarray(is_gps_valid)
    if speed.ndim != 1 or validity.ndim != 1:
        raise ValueError("Speed and GPS validity must be one-dimensional arrays")
    if speed.shape != validity.shape:
        raise ValueError("Speed and GPS validity must have the same shape")
    if validity.dtype.kind != "b":
        raise ValueError("GPS validity must be a boolean array")

    valid_speed = speed[validity]
    if not np.isfinite(valid_speed).all():
        raise ValueError("Valid GPS samples must have finite speed values")
    if (valid_speed < 0.0).any():
        raise ValueError("Valid GPS speed must be non-negative")

    gps_valid_seconds = int(validity.sum())
    gps_valid_distance_km = float(
        valid_speed.sum() * TELEMETRY_SAMPLE_INTERVAL_SECONDS / 3_600.0
    )
    gps_coverage_fraction = gps_valid_seconds / len(validity) if len(validity) else 0.0
    return TelemetryExposure(
        gps_valid_seconds=gps_valid_seconds,
        gps_valid_distance_km=gps_valid_distance_km,
        gps_coverage_fraction=gps_coverage_fraction,
    )


def check_trace_invariants(
    speed_kmh: np.ndarray,
    heading_degrees: np.ndarray,
    *,
    distance_km: float,
    duration_seconds: int,
    maximum_speed_kmh: float,
) -> None:
    """Check an internal true trace against the trip summary row it must explain.

    The true trace contains no measurement noise or dropouts and is never persisted. Exact
    distance reconciliation belongs here; a noisy observed trace is instead validated by
    ``validate_telemetry_frame`` and measured with ``summarise_valid_exposure``.
    """

    speed = np.asarray(speed_kmh, dtype=np.float64)
    heading = np.asarray(heading_degrees, dtype=np.float64)

    if speed.shape != heading.shape:
        raise ValueError("speed and heading traces must have the same length")
    if not np.isfinite(speed).all() or not np.isfinite(heading).all():
        raise ValueError("True speed and heading traces must contain only finite values")
    if len(speed) != duration_seconds:
        raise ValueError(
            f"Trace has {len(speed)} samples but the trip summary records "
            f"{duration_seconds} seconds"
        )

    # At one sample per second the distance travelled is the integral of speed, which makes
    # this the strongest single check that the generator and the summary agree.
    implied_distance_km = float(speed.sum()) * TELEMETRY_SAMPLE_INTERVAL_SECONDS / 3_600.0
    if abs(implied_distance_km - distance_km) > DISTANCE_RECONCILIATION_TOLERANCE * distance_km:
        raise ValueError(
            f"Trace implies {implied_distance_km:.3f} km but the trip summary records "
            f"{distance_km:.3f} km"
        )

    if abs(float(speed.max()) - maximum_speed_kmh) > PEAK_SPEED_RECONCILIATION_TOLERANCE_KMH:
        raise ValueError(
            f"Trace peaks at {speed.max():.3f} km/h but the trip summary records "
            f"{maximum_speed_kmh:.3f} km/h"
        )

    if (speed < 0.0).any():
        raise ValueError("Trace contains negative speed")
    if ((heading < 0.0) | (heading >= 360.0)).any():
        raise ValueError("Trace contains a heading outside [0, 360)")

    yaw_rate = np.abs(wrapped_heading_change_degrees(heading))
    if yaw_rate.size and yaw_rate.max() > MAXIMUM_PLAUSIBLE_YAW_RATE_DEGREES_PER_SECOND:
        raise ValueError(
            f"Trace contains a yaw rate of {yaw_rate.max():.1f} deg/s, above the plausible "
            f"ceiling of {MAXIMUM_PLAUSIBLE_YAW_RATE_DEGREES_PER_SECOND} deg/s"
        )
