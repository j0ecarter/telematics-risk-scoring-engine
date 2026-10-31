"""Minimal API surface for container and deployment verification."""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from apexrisk import __version__


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


app = FastAPI(
    title="ApexRisk API",
    version=__version__,
    description=(
        "Synthetic telematics event detection and claim-frequency demonstration. "
        "This API currently provides a health check only. "
        "Not a production insurance pricing service."
    ),
)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="apexrisk",
        version=__version__,
        data_mode="synthetic",
        production_pricing=False,
    )
