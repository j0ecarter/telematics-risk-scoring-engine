"""Second-by-second synthetic telemetry: data contract, generation and event detection."""

from apexrisk.telemetry.schema import (
    TELEMETRY_COLUMN_DTYPES,
    TELEMETRY_COLUMNS,
    TELEMETRY_SAMPLE_INTERVAL_SECONDS,
    check_trace_invariants,
    validate_telemetry_frame,
    wrapped_heading_change_degrees,
)

__all__ = [
    "TELEMETRY_COLUMNS",
    "TELEMETRY_COLUMN_DTYPES",
    "TELEMETRY_SAMPLE_INTERVAL_SECONDS",
    "check_trace_invariants",
    "validate_telemetry_frame",
    "wrapped_heading_change_degrees",
]
