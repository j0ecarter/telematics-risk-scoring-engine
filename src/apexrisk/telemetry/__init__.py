"""Second-by-second synthetic telemetry: data contract, generation and event detection."""

from apexrisk.telemetry.detection import EventCounts, detect_events
from apexrisk.telemetry.schema import (
    TELEMETRY_COLUMN_DTYPES,
    TELEMETRY_COLUMNS,
    TELEMETRY_SAMPLE_INTERVAL_SECONDS,
    TelemetryExposure,
    check_trace_invariants,
    summarise_valid_exposure,
    validate_telemetry_frame,
    wrapped_heading_change_degrees,
)

__all__ = [
    "EventCounts",
    "TELEMETRY_COLUMNS",
    "TELEMETRY_COLUMN_DTYPES",
    "TELEMETRY_SAMPLE_INTERVAL_SECONDS",
    "TelemetryExposure",
    "check_trace_invariants",
    "detect_events",
    "summarise_valid_exposure",
    "validate_telemetry_frame",
    "wrapped_heading_change_degrees",
]
