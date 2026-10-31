"""Validate the request boundary before exposing telemetry analysis over HTTP."""

import pytest
from pydantic import ValidationError

from apexrisk.api.main import TraceRequest

VALID_TRACE = {
    "speed_limit_kmh": 60,
    "speed_kmh": [50, 50, 50],
    "heading_degrees": [90, 90, 90],
    "is_gps_valid": [True, True, True],
}


def test_accepts_missing_gps_without_removing_the_sample() -> None:
    trace = TraceRequest.model_validate(
        VALID_TRACE
        | {
            "speed_kmh": [50, None, 50],
            "heading_degrees": [90, None, 90],
            "is_gps_valid": [True, False, True],
        }
    )

    assert trace.speed_kmh == [50, None, 50]
    assert trace.heading_degrees == [90, None, 90]
    assert trace.is_gps_valid == [True, False, True]


@pytest.mark.parametrize("sample_count", [1, 600])
def test_accepts_trace_length_and_measurement_boundaries(sample_count: int) -> None:
    trace = TraceRequest(
        speed_limit_kmh=0,
        speed_kmh=[0] * sample_count,
        heading_degrees=[359.999] * sample_count,
        is_gps_valid=[True] * sample_count,
    )

    assert len(trace.speed_kmh) == sample_count
    assert trace.speed_limit_kmh == 0
    assert TraceRequest.model_validate(
        VALID_TRACE | {"heading_degrees": [0, 0, 0]}
    ).heading_degrees == [0, 0, 0]


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({"heading_degrees": [90]}, id="unequal-headings"),
        pytest.param({"is_gps_valid": [True]}, id="unequal-gps-flags"),
        pytest.param({"speed_kmh": []}, id="empty-trace"),
        pytest.param({"speed_kmh": [50] * 601}, id="over-length-limit"),
        pytest.param({"speed_limit_kmh": -1}, id="negative-speed-limit"),
        pytest.param({"speed_kmh": [50, -1, 50]}, id="negative-speed"),
        pytest.param({"heading_degrees": [90, -1, 90]}, id="negative-heading"),
        pytest.param({"heading_degrees": [90, 360, 90]}, id="heading-360"),
        pytest.param({"speed_kmh": [50, None, 50]}, id="valid-gps-missing-speed"),
        pytest.param(
            {"heading_degrees": [90, None, 90]}, id="valid-gps-missing-heading"
        ),
        pytest.param({"is_gps_valid": [True, False, True]}, id="invalid-gps-with-readings"),
        pytest.param(
            {"is_gps_valid": [True, False, True], "speed_kmh": [50, None, 50]},
            id="invalid-gps-with-heading",
        ),
        pytest.param(
            {"is_gps_valid": [True, False, True], "heading_degrees": [90, None, 90]},
            id="invalid-gps-with-speed",
        ),
        pytest.param({"speed_limit_kmh": "60"}, id="string-number"),
        pytest.param({"speed_kmh": [50, True, 50]}, id="boolean-speed"),
        pytest.param({"is_gps_valid": [True, 1, True]}, id="numeric-gps-flag"),
        pytest.param({"unexpected": True}, id="unknown-field"),
    ],
)
def test_rejects_invalid_trace_requests(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TraceRequest.model_validate(VALID_TRACE | updates)


@pytest.mark.parametrize("field", ["speed_limit_kmh", "speed_kmh", "heading_degrees"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_measurements(field: str, value: float) -> None:
    update = value if field == "speed_limit_kmh" else [50, value, 50]
    with pytest.raises(ValidationError):
        TraceRequest.model_validate(VALID_TRACE | {field: update})


def test_reports_the_invalid_sample_position() -> None:
    with pytest.raises(ValidationError, match="Sample 1: speed must be non-negative"):
        TraceRequest.model_validate(VALID_TRACE | {"speed_kmh": [50, -1, 50]})
