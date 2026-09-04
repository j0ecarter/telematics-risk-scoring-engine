import json
from datetime import date

import pandas as pd

from apexrisk.config import FeatureConfig, SimulationConfig
from apexrisk.pipeline import run_synthetic_pipeline


def test_pipeline_writes_documented_parquet_outputs(tmp_path) -> None:
    manifest = run_synthetic_pipeline(
        SimulationConfig(
            seed=123,
            n_drivers=10,
            start_date=date(2025, 1, 1),
            months=1,
            mean_trips_per_driver_per_month=3,
        ),
        FeatureConfig(minimum_monthly_distance_km=1.0),
        tmp_path,
    )

    expected_files = {
        "drivers.parquet",
        "vehicles.parquet",
        "trips.parquet",
        "trip_features.parquet",
        "driver_month_features.parquet",
        "manifest.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    assert manifest["data_classification"] == "fully_synthetic"
    assert manifest["private_latent_traits_published"] is False

    drivers = pd.read_parquet(tmp_path / "drivers.parquet")
    assert "aggressiveness" not in drivers.columns
    stored_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert stored_manifest["row_counts"]["drivers"] == 10
