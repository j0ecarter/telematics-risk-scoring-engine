"""Exercise telemetry validation and detector parity through the HTTP boundary."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import numpy as np
import pytest

from apexrisk.api.main import app
from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.demo import build_demo_scenarios
from apexrisk.telemetry.detection import detect_events

VALID_TRACE = {
    "speed_limit_kmh": 60,
    "speed_kmh": [50] * 10,
    "heading_degrees": [90] * 10,
    "is_gps_valid": [True] * 10,
}


def post_body(body: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(
                "/journeys/analyse", content=body, headers={"content-type": "application/json"}
            )

    return asyncio.run(request())


def post_trace(payload: object) -> httpx.Response:
    return post_body(json.dumps(payload))


@pytest.mark.parametrize(
    "name", ["steady", "braking", "acceleration", "cornering", "speeding", "gps_dropout"]
)
def test_analysis_response_matches_the_detector(name: str) -> None:
    scenario = next(scenario for scenario in build_demo_scenarios() if scenario.name == name)
    frame = scenario.telemetry
    valid = frame["is_gps_valid"].to_numpy()
    payload = {
        "speed_limit_kmh": scenario.speed_limit_kmh,
        "speed_kmh": [
            float(value) if observed else None
            for value, observed in zip(frame["speed_kmh"], valid, strict=True)
        ],
        "heading_degrees": [
            float(value) if observed else None
            for value, observed in zip(frame["heading_degrees"], valid, strict=True)
        ],
        "is_gps_valid": valid.tolist(),
    }
    config = EventDetectionConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs/telemetry.yaml"
    )
    expected = detect_events(
        frame["speed_kmh"].to_numpy(),
        frame["heading_degrees"].to_numpy(),
        valid,
        scenario.speed_limit_kmh,
        config,
    )

    response = post_trace(payload)

    assert response.status_code == 200
    assert response.json() == asdict(expected)


@pytest.mark.parametrize("sample_count", [1, 600])
def test_analysis_accepts_trace_length_boundaries(sample_count: int) -> None:
    response = post_trace(
        {
            "speed_limit_kmh": 60,
            "speed_kmh": [70] * sample_count,
            "heading_degrees": [90] * sample_count,
            "is_gps_valid": [True] * sample_count,
        }
    )

    assert response.status_code == 200
    result = response.json()
    assert result.pop("speeding_seconds") == sample_count
    assert all(value == 0 for value in result.values())


def test_analysis_accepts_a_trace_with_no_valid_gps() -> None:
    response = post_trace(
        VALID_TRACE
        | {
            "speed_kmh": [None] * 10,
            "heading_degrees": [None] * 10,
            "is_gps_valid": [False] * 10,
        }
    )

    assert response.status_code == 200
    assert all(value == 0 for value in response.json().values())


@pytest.mark.parametrize(
    "updates",
    [
        pytest.param({"heading_degrees": [90]}, id="mismatched-lengths"),
        pytest.param({"speed_kmh": []}, id="empty-trace"),
        pytest.param({"speed_kmh": [50] * 601}, id="too-long"),
        pytest.param({"speed_limit_kmh": -1}, id="negative-speed-limit"),
        pytest.param({"speed_kmh": [-1] * 10}, id="negative-speed"),
        pytest.param({"heading_degrees": [360] * 10}, id="invalid-heading"),
        pytest.param({"speed_kmh": [None] * 10}, id="valid-gps-missing-readings"),
        pytest.param({"is_gps_valid": [False] * 10}, id="invalid-gps-with-readings"),
        pytest.param({"is_gps_valid": [1] * 10}, id="numeric-gps-flags"),
        pytest.param({"speed_limit_kmh": "60"}, id="string-number"),
        pytest.param({"unexpected": True}, id="unknown-field"),
    ],
)
def test_invalid_analysis_requests_return_validation_errors(updates: dict[str, object]) -> None:
    response = post_trace(VALID_TRACE | updates)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors
    assert all({"loc", "msg", "type"}.issubset(error) for error in errors)


@pytest.mark.parametrize("field", ["speed_limit_kmh", "speed_kmh", "heading_degrees"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_return_json_errors_instead_of_server_errors(
    field: str, value: float
) -> None:
    # Python's JSON decoder accepts these non-standard numeric tokens. They must
    # be rejected without echoing them into a JSON error response.
    update = value if field == "speed_limit_kmh" else [value] * 10
    response = post_trace(VALID_TRACE | {field: update})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"
    errors = response.json()["detail"]
    assert any(error["loc"][:2] == ["body", field] for error in errors)


def test_invalid_json_returns_a_validation_error() -> None:
    response = post_body('{"speed_limit_kmh":')

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


def test_analysis_handles_large_finite_speeds_without_numeric_overflow() -> None:
    with np.errstate(over="raise", invalid="raise"):
        response = post_trace(VALID_TRACE | {"speed_kmh": [1e308] * 10})

    assert response.status_code == 200
    result = response.json()
    assert result.pop("speeding_seconds") == 10
    assert all(value == 0 for value in result.values())



def test_documented_example_can_be_executed() -> None:
    schema = app.openapi()
    body_schema = schema["paths"]["/journeys/analyse"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert body_schema["$ref"] == "#/components/schemas/TraceRequest"
    example = schema["components"]["schemas"]["TraceRequest"]["examples"][0]

    response = post_trace(example)

    assert response.status_code == 200
    result = response.json()
    assert result.pop("speeding_seconds") == 10
    assert all(value == 0 for value in result.values())
