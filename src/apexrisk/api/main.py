"""Health checks and analysis of one small, validated 1 Hz telemetry trace."""

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apexrisk import __version__
from apexrisk.config import EventDetectionConfig
from apexrisk.telemetry.detection import EventCounts, detect_events


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    data_mode: str
    production_pricing: bool


class TraceRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "speed_limit_kmh": 60,
                    "speed_kmh": [70] * 10,
                    "heading_degrees": [90] * 10,
                    "is_gps_valid": [True] * 10,
                }
            ]
        },
    )

    speed_limit_kmh: float = Field(ge=0)

    speed_kmh: list[float | None] = Field(
        min_length=1,
        max_length=600,
    )

    heading_degrees: list[float | None] = Field(
        min_length=1,
        max_length=600,
    )

    is_gps_valid: list[bool] = Field(
        min_length=1,
        max_length=600,
    )

    @model_validator(mode="after")
    def validate_trace(self) -> "TraceRequest":
        if not (
            len(self.speed_kmh)
            == len(self.heading_degrees)
            == len(self.is_gps_valid)
        ):
            raise ValueError("Telemetry arrays must have matching lengths")

        for index, (speed, heading, valid) in enumerate(
            zip(
                self.speed_kmh,
                self.heading_degrees,
                self.is_gps_valid,
                strict=True,
            )
        ):
            if valid:
                if speed is None or heading is None:
                    raise ValueError(
                        f"Sample {index}: valid GPS requires speed and heading"
                    )

                if speed < 0:
                    raise ValueError(
                        f"Sample {index}: speed must be non-negative"
                    )

                if not 0 <= heading < 360:
                    raise ValueError(
                        f"Sample {index}: heading must be between 0 and 360, "
                        "excluding 360"
                    )

            elif speed is not None or heading is not None:
                raise ValueError(
                    f"Sample {index}: invalid GPS requires null speed and heading"
                )

        return self


DETECTOR_CONFIG = EventDetectionConfig.from_yaml("configs/telemetry.yaml")
app = FastAPI(
    title="ApexRisk API",
    version=__version__,
    description=(
        "Synthetic telematics event detection and claim-frequency demonstration. "
        "Analyses a single 1 Hz telemetry trace and returns detected event counts. "
        "Not a production insurance pricing service."
    ),
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Invalid input can contain NaN/infinity, which cannot be encoded in JSON.
    # Return useful error locations and messages without re-encoding that input.
    details = [
        {key: error[key] for key in ("loc", "msg", "type")}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="apexrisk",
        version=__version__,
        data_mode="synthetic",
        production_pricing=False,
    )


@app.post(
    "/journeys/analyse",
    response_model=EventCounts,
    tags=["telemetry"],
)
def analyse_journey(trace: TraceRequest) -> EventCounts:
    return detect_events(
        speed_kmh=np.asarray(trace.speed_kmh, dtype=float),
        heading_degrees=np.asarray(
            trace.heading_degrees,
            dtype=float,
        ),
        is_gps_valid=np.asarray(
            trace.is_gps_valid,
            dtype=bool,
        ),
        speed_limit_kmh=trace.speed_limit_kmh,
        config=DETECTOR_CONFIG,
    )
