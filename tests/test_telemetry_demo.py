"""Check demo input profiles and expected detection with the project configuration."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.demo import build_demo_scenarios, run_telemetry_demo
from apexrisk.telemetry.detection import EventCounts, detect_events
from apexrisk.telemetry.schema import TELEMETRY_COLUMN_DTYPES, validate_telemetry_frame

NO_EVENTS = EventCounts(
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

# These integration expectations intentionally use configs/telemetry.yaml; the
# detector unit tests retain their independent settings. Review these expectations
# when project settings change, rather than deriving them from detector output.
# Braking/acceleration peak at 3 m/s², inside the harsh band, with enough consecutive
# qualifying samples after smoothing and differencing. Cornering gives
# (60 / 3.6) * radians(15) ≈ 4.36 m/s² for four seconds, also in the harsh band.
# Speed exceeds 60 + 5 km/h at seconds 66..94 inclusive: 94 - 66 + 1 = 29 samples.
# The median filter preserves these ramps and plateau. All unlisted fields must
# remain zero, including events that would incorrectly span the GPS dropout.
EXPECTED_RESULTS = {
    "steady": NO_EVENTS,
    "braking": replace(NO_EVENTS, brake_events_harsh=1),
    "acceleration": replace(NO_EVENTS, accel_events_harsh=1),
    "cornering": replace(NO_EVENTS, corner_events_harsh=1),
    "speeding": replace(NO_EVENTS, speeding_seconds=29),
    "gps_dropout": NO_EVENTS,
}


@pytest.mark.parametrize("name,expected", EXPECTED_RESULTS.items(), ids=EXPECTED_RESULTS)
def test_demo_detection_matches_expected_results(name: str, expected: EventCounts) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "telemetry.yaml"
    config = EventDetectionConfig.from_yaml(config_path)
    scenario = next(s for s in build_demo_scenarios() if s.name == name)
    frame = scenario.telemetry

    actual = detect_events(
        speed_kmh=frame["speed_kmh"].to_numpy(),
        heading_degrees=frame["heading_degrees"].to_numpy(),
        is_gps_valid=frame["is_gps_valid"].to_numpy(),
        speed_limit_kmh=scenario.speed_limit_kmh,
        config=config,
    )

    assert actual == expected, f"{name}: expected {expected}, got {actual}"


@pytest.mark.parametrize(
    "name", ["steady", "braking", "acceleration", "cornering", "speeding", "gps_dropout"]
)
def test_scenario_has_three_minutes_of_validly_typed_telemetry(name: str) -> None:
    scenarios = {scenario.name: scenario for scenario in build_demo_scenarios()}
    frame = scenarios[name].telemetry
    validate_telemetry_frame(frame)
    assert len(frame) == 180
    assert frame["second_index"].tolist() == list(range(180))
    assert frame["trip_id"].unique().tolist() == [f"demo-{name}"]
    assert frame.dtypes.astype(str).to_dict() == TELEMETRY_COLUMN_DTYPES


def test_builds_are_repeatable_and_do_not_share_mutable_traces() -> None:
    first = build_demo_scenarios()
    second = build_demo_scenarios()
    assert len(first) == len(second) == 6
    for left, right in zip(first, second, strict=True):
        assert (left.name, left.description, left.speed_limit_kmh) == (
            right.name, right.description, right.speed_limit_kmh
        )
        pd.testing.assert_frame_equal(left.telemetry, right.telemetry)

    first[0].telemetry.loc[0, "heading_degrees"] = 0.0
    assert second[0].telemetry.loc[0, "heading_degrees"] == 90.0
    assert first[1].telemetry.loc[0, "heading_degrees"] == 90.0


def test_manoeuvre_profiles_have_the_documented_rates() -> None:
    scenarios = {scenario.name: scenario for scenario in build_demo_scenarios()}
    for name, acceleration in [("braking", -3.0), ("acceleration", 3.0)]:
        speed = scenarios[name].telemetry["speed_kmh"].to_numpy(dtype=float)
        np.testing.assert_allclose(np.diff(speed[60:65]) / 3.6, acceleration, atol=1e-6)
        np.testing.assert_allclose(speed[:61], speed[60])
        np.testing.assert_allclose(speed[64:], speed[64])

    corner = scenarios["cornering"].telemetry
    np.testing.assert_allclose(corner["speed_kmh"], 60.0)
    np.testing.assert_allclose(np.diff(corner.loc[60:64, "heading_degrees"]), 15.0)

    steady = scenarios["steady"].telemetry
    np.testing.assert_allclose(steady["speed_kmh"], 50.0)
    np.testing.assert_allclose(steady["heading_degrees"], 90.0)


def test_speeding_has_gentle_ramps_and_a_sustained_plateau() -> None:
    scenario = next(s for s in build_demo_scenarios() if s.name == "speeding")
    speed = scenario.telemetry["speed_kmh"].to_numpy()
    assert scenario.speed_limit_kmh == 60.0
    assert np.max(np.abs(np.diff(speed))) == 1.0
    np.testing.assert_allclose(speed[70:91], 70.0)
    assert speed[0] == speed[-1] == 50.0


def test_dropout_preserves_time_and_hides_both_measurement_channels() -> None:
    scenario = next(s for s in build_demo_scenarios() if s.name == "gps_dropout")
    frame = scenario.telemetry
    missing = ~frame["is_gps_valid"]
    assert frame.loc[missing, "second_index"].tolist() == [60, 61, 62, 63, 64]
    assert frame.loc[missing, ["speed_kmh", "heading_degrees"]].isna().all().all()
    assert frame.loc[59, "speed_kmh"] == 72.0
    assert frame.loc[65, "speed_kmh"] == 36.0
    assert frame.loc[59, "heading_degrees"] == 90.0
    assert frame.loc[65, "heading_degrees"] == 180.0


def test_saved_demo_round_trips_traces_and_comparison_evidence(tmp_path: Path) -> None:
    config = EventDetectionConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "telemetry.yaml"
    )
    output_dir = tmp_path / "demo"
    summary = run_telemetry_demo(config, output_dir)
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved == summary
    assert saved["all_passed"] is True
    assert saved["detector_config"] == asdict(config)
    assert saved["raw_truth_channels_published"] is False
    assert saved["sample_interval_seconds"] == 1
    assert {p.name for p in output_dir.iterdir()} == {
        "summary.json", *(f"{name}.parquet" for name in EXPECTED_RESULTS)
    }
    results = {result["name"]: result for result in saved["scenarios"]}
    assert set(results) == set(EXPECTED_RESULTS)
    for scenario in build_demo_scenarios():
        result = results[scenario.name]
        frame = pd.read_parquet(output_dir / result["telemetry_file"])
        pd.testing.assert_frame_equal(frame, scenario.telemetry)
        validate_telemetry_frame(frame)
        assert frame.dtypes.astype(str).to_dict() == TELEMETRY_COLUMN_DTYPES
        assert result["sample_count"] == 180
        assert result["speed_limit_kmh"] == scenario.speed_limit_kmh
        assert result["expected"] == result["actual"] == asdict(EXPECTED_RESULTS[scenario.name])
        assert result["passed"] is True
        assert result["mismatches"] == {}
        exposure = result["gps_exposure"]
        valid_seconds = 175 if scenario.name == "gps_dropout" else 180
        assert exposure["gps_valid_seconds"] == valid_seconds
        assert exposure["gps_coverage_fraction"] == pytest.approx(valid_seconds / 180)
    assert results["gps_dropout"]["gps_exposure"]["gps_valid_distance_km"] == pytest.approx(2.35)


def test_saved_demo_is_repeatable_across_runs_and_directories(tmp_path: Path) -> None:
    config = EventDetectionConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "telemetry.yaml"
    )
    first_dir = tmp_path / "first"
    run_telemetry_demo(config, first_dir)
    first_json = (first_dir / "summary.json").read_bytes()
    first_frames = {p.name: pd.read_parquet(p) for p in first_dir.glob("*.parquet")}
    unrelated = first_dir / "notes.txt"
    unrelated.write_text("Keep this file.")

    for directory in (first_dir, tmp_path / "second"):
        run_telemetry_demo(config, directory)
        assert (directory / "summary.json").read_bytes() == first_json
        for name, frame in first_frames.items():
            pd.testing.assert_frame_equal(pd.read_parquet(directory / name), frame)
    assert unrelated.read_text() == "Keep this file."


def test_saved_demo_reports_mismatch_without_changing_expectations(tmp_path: Path) -> None:
    config = EventDetectionConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "telemetry.yaml"
    )
    # At a 10 km/h tolerance the trace never strictly exceeds 60 + 10 km/h.
    changed = replace(config, speeding_tolerance_kmh=10.0)
    summary = run_telemetry_demo(changed, tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved == summary
    assert saved["all_passed"] is False
    assert saved["detector_config"]["speeding_tolerance_kmh"] == 10.0
    failures = [result for result in saved["scenarios"] if not result["passed"]]
    assert len(failures) == 1
    result = failures[0]
    assert result["name"] == "speeding"
    assert result["expected"] == asdict(EXPECTED_RESULTS["speeding"])
    assert result["actual"]["speeding_seconds"] == 0
    assert result["mismatches"] == {"speeding_seconds": {"expected": 29, "actual": 0}}
    assert len(pd.read_parquet(tmp_path / result["telemetry_file"])) == 180
