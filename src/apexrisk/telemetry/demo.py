"""Small, deterministic 1 Hz scenarios for demonstrating the telemetry detector.

These are controlled examples, not a population simulator or a definition of safe
driving. Each call builds fresh three-minute traces without random noise. Scenario
profiles are fixed independently of detector settings so changing a threshold does
not silently change the input being demonstrated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.detection import EventCounts, detect_events
from apexrisk.telemetry.schema import (
    TELEMETRY_COLUMN_DTYPES,
    TELEMETRY_SAMPLE_INTERVAL_SECONDS,
    summarise_valid_exposure,
    validate_telemetry_frame,
)

DEMO_DURATION_SECONDS = 180

# Fixed expectations for the documented baseline settings, not detector-derived
# answers. Deliberately changed settings may fail these comparisons. The tests
# retain an independent specification to catch mistakes in this output mapping.
_NO_EVENTS = EventCounts(
    brake_events_moderate=0,
    brake_events_harsh=0,
    brake_events_severe=0,
    accel_events_moderate=0,
    accel_events_harsh=0,
    accel_events_severe=0,
    corner_events_moderate=0,
    corner_events_harsh=0,
    corner_events_severe=0,
    speeding_seconds=0,
)
_EXPECTED_RESULTS = {
    "steady": _NO_EVENTS,
    "braking": replace(_NO_EVENTS, brake_events_harsh=1),
    "acceleration": replace(_NO_EVENTS, accel_events_harsh=1),
    "cornering": replace(_NO_EVENTS, corner_events_harsh=1),
    "speeding": replace(_NO_EVENTS, speeding_seconds=29),
    "gps_dropout": _NO_EVENTS,
}


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


def run_telemetry_demo(
    config: EventDetectionConfig, output_dir: str | Path
) -> dict[str, Any]:
    """Run all scenarios and save observed traces plus a deterministic summary.

    Re-running replaces the six named Parquet files and summary.json in output_dir.
    Other files are left alone. Comparisons use fixed baseline expectations even
    when settings change; failures are saved and returned with all_passed=False.
    Invalid configuration or file-writing errors raise normally.

    Relative filenames and no timestamp keep the JSON independent of run time and
    output directory. Only observed telemetry is persisted, including NaN readings
    for missing GPS fixes; no hidden truth channels are written.
    """
    config.validate()
    scenarios = build_demo_scenarios()
    results = []
    for scenario in scenarios:
        frame = scenario.telemetry
        speed = frame["speed_kmh"].to_numpy()
        heading = frame["heading_degrees"].to_numpy()
        validity = frame["is_gps_valid"].to_numpy()
        actual = asdict(detect_events(speed, heading, validity, scenario.speed_limit_kmh, config))
        expected = asdict(_EXPECTED_RESULTS[scenario.name])
        mismatches = {
            field: {"expected": expected[field], "actual": value}
            for field, value in actual.items()
            if value != expected[field]
        }
        results.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "speed_limit_kmh": scenario.speed_limit_kmh,
                "telemetry_file": f"{scenario.name}.parquet",
                "sample_count": len(frame),
                "gps_exposure": asdict(summarise_valid_exposure(speed, validity)),
                "expected": expected,
                "actual": actual,
                "passed": not mismatches,
                "mismatches": mismatches,
            }
        )

    summary = {
        "schema_version": 1,
        "data_classification": "fully_synthetic",
        "raw_truth_channels_published": False,
        "sample_interval_seconds": TELEMETRY_SAMPLE_INTERVAL_SECONDS,
        "detector_config": asdict(config),
        "expectation_basis": "Fixed baseline demo expectations; changed settings may fail.",
        "all_passed": all(result["passed"] for result in results),
        "scenarios": results,
    }
    summary_json = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        scenario.telemetry.to_parquet(
            destination / f"{scenario.name}.parquet", engine="pyarrow", index=False
        )
    (destination / "summary.json").write_text(summary_json, encoding="utf-8")
    return summary
