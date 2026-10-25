"""Small, deterministic 1 Hz scenarios for demonstrating the telemetry detector.

These are controlled examples, not a population simulator or a definition of safe
driving. Each call builds fresh three-minute traces without random noise. Scenario
profiles are fixed independently of detector settings so changing a threshold does
not silently change the input being demonstrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apexrisk.telemetry.schema import TELEMETRY_COLUMN_DTYPES, validate_telemetry_frame

DEMO_DURATION_SECONDS = 180


@dataclass
class DemoScenario:
    """Observed telemetry and the context needed to interpret a single scenario."""

    name: str
    description: str
    speed_limit_kmh: float
    telemetry: pd.DataFrame


def _make_scenario(
    name: str,
    description: str,
    speed_limit_kmh: float,
    speed_kmh: np.ndarray,
    heading_degrees: np.ndarray,
    is_gps_valid: np.ndarray,
) -> DemoScenario:
    frame = pd.DataFrame(
        {
            "trip_id": f"demo-{name}",
            "second_index": np.arange(DEMO_DURATION_SECONDS),
            "speed_kmh": np.where(is_gps_valid, speed_kmh, np.nan),
            "heading_degrees": np.where(is_gps_valid, heading_degrees, np.nan),
            "is_gps_valid": is_gps_valid,
        }
    ).astype(TELEMETRY_COLUMN_DTYPES)
    validate_telemetry_frame(frame)
    return DemoScenario(name, description, speed_limit_kmh, frame)


def build_demo_scenarios() -> list[DemoScenario]:
    """Build six independent traces, with manoeuvres away from segment edges.

    Braking and acceleration change speed by 43.2 km/h over four seconds
    (3 m/s²). Cornering changes heading by 15 degrees/s at 60 km/h
    (approximately 4.36 m/s² lateral acceleration). Speeding uses gentle
    1 km/h-per-second ramps to avoid introducing harsh acceleration or braking.

    GPS samples at seconds 60 through 64 are missing. Speed and heading differ
    across that gap; the missing movement must not be inferred from its endpoints.
    """
    seconds = np.arange(DEMO_DURATION_SECONDS)
    straight = np.full(DEMO_DURATION_SECONDS, 90.0)
    valid = np.ones(DEMO_DURATION_SECONDS, dtype=bool)
    dropout_valid = valid.copy()
    dropout_valid[60:65] = False

    return [
        _make_scenario(
            "steady",
            "Constant 50 km/h and heading under a 60 km/h limit.",
            60.0,
            np.full(DEMO_DURATION_SECONDS, 50.0),
            straight,
            valid,
        ),
        _make_scenario(
            "braking",
            "Speed falls from 72 to 28.8 km/h between seconds 60 and 64.",
            80.0,
            np.interp(seconds, [60, 64], [72.0, 28.8]),
            straight,
            valid,
        ),
        _make_scenario(
            "acceleration",
            "Speed rises from 28.8 to 72 km/h between seconds 60 and 64.",
            80.0,
            np.interp(seconds, [60, 64], [28.8, 72.0]),
            straight,
            valid,
        ),
        _make_scenario(
            "cornering",
            "At 60 km/h, heading turns from 90 to 150 degrees over four seconds.",
            80.0,
            np.full(DEMO_DURATION_SECONDS, 60.0),
            np.interp(seconds, [60, 64], [90.0, 150.0]),
            valid,
        ),
        _make_scenario(
            "speeding",
            "Gentle ramps from 50 to 70 km/h and back, with a 60 km/h limit.",
            60.0,
            np.interp(seconds, [50, 70, 90, 110], [50.0, 70.0, 70.0, 50.0]),
            straight,
            valid,
        ),
        _make_scenario(
            "gps_dropout",
            "Five missing fixes separate 72 km/h eastbound and 36 km/h southbound.",
            80.0,
            np.where(seconds < 65, 72.0, 36.0),
            np.where(seconds < 65, 90.0, 180.0),
            dropout_valid,
        ),
    ]
