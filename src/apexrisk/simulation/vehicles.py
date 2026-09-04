"""Generate one synthetic insured vehicle per driver."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_vehicles(drivers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate vehicles with plausible, correlated characteristics."""

    if drivers.empty:
        raise ValueError("drivers must not be empty")
    if "driver_id" not in drivers:
        raise ValueError("drivers must contain driver_id")

    n_vehicles = len(drivers)
    area = drivers["home_area_type"].to_numpy()
    vehicle_type = np.array(
        [
            rng.choice(
                ["hatchback", "saloon", "suv", "estate"],
                p=([0.48, 0.25, 0.17, 0.10] if area_type == "urban" else [0.30, 0.25, 0.28, 0.17]),
            )
            for area_type in area
        ]
    )
    power_baseline = {
        "hatchback": 82,
        "saloon": 112,
        "suv": 128,
        "estate": 116,
    }
    engine_power = np.array(
        [max(45, rng.normal(power_baseline[kind], 24)) for kind in vehicle_type]
    ).round(1)
    vehicle_age = np.rint(rng.triangular(0, 5, 17, size=n_vehicles)).astype(int)
    safety_rating = np.clip(
        np.rint(5.0 - vehicle_age / 10.0 + rng.normal(0, 0.65, size=n_vehicles)),
        2,
        5,
    ).astype(int)
    value = np.clip(
        9_000 + engine_power * 105 - vehicle_age * 900 + rng.normal(0, 2_500, n_vehicles),
        1_500,
        65_000,
    ).round(2)

    return pd.DataFrame(
        {
            "vehicle_id": [f"VEH-{index:06d}" for index in range(1, n_vehicles + 1)],
            "driver_id": drivers["driver_id"].to_numpy(),
            "vehicle_type": vehicle_type,
            "vehicle_age_years": vehicle_age,
            "engine_power_kw": engine_power,
            "vehicle_value_gbp": value,
            "safety_rating": safety_rating,
        }
    )
