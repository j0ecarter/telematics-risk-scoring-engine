"""Minimal API surface for container and deployment verification."""

from fastapi import FastAPI
from pydantic import BaseModel

from apexrisk import __version__


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    data_mode: str
    production_pricing: bool


app = FastAPI(
    title="ApexRisk API",
    version=__version__,
    description=(
        "Synthetic-first telematics risk-scoring methodology demonstrator. "
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
