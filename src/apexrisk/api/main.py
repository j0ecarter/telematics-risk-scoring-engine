"""Minimal API surface for container and deployment verification."""

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

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
