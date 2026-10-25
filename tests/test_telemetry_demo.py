"""Check the demo's input profiles independently of event detection."""

import numpy as np
import pandas as pd
import pytest

from apexrisk.telemetry.demo import build_demo_scenarios
from apexrisk.telemetry.schema import TELEMETRY_COLUMN_DTYPES, validate_telemetry_frame


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
