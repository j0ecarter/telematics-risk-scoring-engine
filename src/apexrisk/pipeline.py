"""Run the initial synthetic-data-to-features vertical slice."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apexrisk.config import ClaimsConfig, FeatureConfig, SimulationConfig
from apexrisk.features import build_driver_month_features, build_trip_features
from apexrisk.simulation import (
    generate_drivers,
    generate_journeys,
    generate_policy_exposure_and_claims,
    generate_vehicles,
)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


def run_synthetic_pipeline(
    simulation_config: SimulationConfig,
    feature_config: FeatureConfig,
    output_dir: str | Path,
    claims_config: ClaimsConfig | None = None,
) -> dict[str, Any]:
    """Generate synthetic entities, trips, features, exposure, and claims."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(simulation_config.seed)
    claims_config = claims_config or ClaimsConfig()

    population = generate_drivers(simulation_config.n_drivers, rng)
    vehicles = generate_vehicles(population.observed, rng)
    trips = generate_journeys(
        population.observed,
        vehicles,
        population.latent,
        start_date=simulation_config.start_date,
        months=simulation_config.months,
        mean_trips_per_driver_per_month=(simulation_config.mean_trips_per_driver_per_month),
        rng=rng,
    )
    trip_features = build_trip_features(trips)
    driver_month_features = build_driver_month_features(
        trips,
        minimum_distance_km=feature_config.minimum_monthly_distance_km,
    )
    outcomes = generate_policy_exposure_and_claims(
        population.observed,
        vehicles,
        population.latent,
        driver_month_features,
        start_date=simulation_config.start_date,
        months=simulation_config.months,
        config=claims_config,
        rng=rng,
    )

    outputs = {
        "drivers": destination / "drivers.parquet",
        "vehicles": destination / "vehicles.parquet",
        "trips": destination / "trips.parquet",
        "trip_features": destination / "trip_features.parquet",
        "driver_month_features": destination / "driver_month_features.parquet",
        "policy_exposure": destination / "policy_exposure.parquet",
        "claims": destination / "claims.parquet",
    }
    for name, frame in {
        "drivers": population.observed,
        "vehicles": vehicles,
        "trips": trips,
        "trip_features": trip_features,
        "driver_month_features": driver_month_features,
        "policy_exposure": outcomes.policy_exposure,
        "claims": outcomes.claims,
    }.items():
        _write_parquet(frame, outputs[name])

    manifest: dict[str, Any] = {
        "data_classification": "fully_synthetic",
        "intended_use": "methodological_demonstration_only",
        "production_pricing_approved": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "simulation_config": {
            **asdict(simulation_config),
            "start_date": simulation_config.start_date.isoformat(),
        },
        "claims_config": asdict(claims_config),
        "row_counts": {
            "drivers": len(population.observed),
            "vehicles": len(vehicles),
            "trips": len(trips),
            "trip_features": len(trip_features),
            "driver_month_features": len(driver_month_features),
            "policy_exposure": len(outcomes.policy_exposure),
            "claims": len(outcomes.claims),
        },
        "files": {name: path.name for name, path in outputs.items()},
        "private_latent_traits_published": False,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
