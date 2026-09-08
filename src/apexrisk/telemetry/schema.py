"""Machine-readable form of the telemetry data contract.

This module is the executable counterpart of ``docs/telemetry_contract.md``. It defines
the published second-by-second telemetry schema and the invariants a generated trace must
satisfy for the trip summary derived from it to be truthful.

Nothing here generates traces or detects events. Keeping the contract in its own module
means the generator and the detector can be developed independently against a shared,
enforced definition rather than against each other.
"""

from __future__ import annotations

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
    if telemetry[list(TELEMETRY_COLUMNS)].isna().to_numpy().any():
        raise ValueError("Telemetry must not contain null values")

    speed = telemetry["speed_kmh"].to_numpy()
    if (speed < 0.0).any():
        raise ValueError("speed_kmh must be non-negative, measurement noise included")

    heading = telemetry["heading_degrees"].to_numpy()
    if ((heading < 0.0) | (heading >= 360.0)).any():
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


def check_trace_invariants(
    speed_kmh: np.ndarray,
    heading_degrees: np.ndarray,
    *,
    distance_km: float,
    duration_seconds: int,
    maximum_speed_kmh: float,
) -> None:
    """Check one trace against the trip summary row it is supposed to explain.

    These are the acceptance criteria for the trace generator, listed in section 6 of the
    contract. Raises ``ValueError`` describing the first violated invariant.
    """

    speed = np.asarray(speed_kmh, dtype=np.float64)
    heading = np.asarray(heading_degrees, dtype=np.float64)

    if speed.shape != heading.shape:
        raise ValueError("speed and heading traces must have the same length")
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
