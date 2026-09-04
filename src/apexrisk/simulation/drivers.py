"""Generate observed driver profiles and private simulation traits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriverPopulation:
    """Observed driver data plus private ground-truth simulation traits.

    The latent table drives the simulation but is deliberately not written to the
    modelling outputs. This prevents downstream models from learning directly from
    variables that would be unobservable in a real insurance setting.
    """

    observed: pd.DataFrame
    latent: pd.DataFrame


def generate_drivers(n_drivers: int, rng: np.random.Generator) -> DriverPopulation:
    """Generate a reproducible synthetic driver population."""

    if n_drivers <= 0:
        raise ValueError("n_drivers must be positive")

    driver_ids = np.array([f"DRV-{index:06d}" for index in range(1, n_drivers + 1)])
    ages = np.rint(rng.triangular(18, 41, 80, size=n_drivers)).astype(int)
    max_experience = np.maximum(0, ages - 17)
    years_licensed = np.array([rng.integers(0, upper + 1) for upper in max_experience], dtype=int)
    area_types = rng.choice(["urban", "suburban", "rural"], size=n_drivers, p=[0.46, 0.34, 0.20])
    driving_patterns = rng.choice(
        ["commuter", "shift_worker", "flexible"],
        size=n_drivers,
        p=[0.58, 0.14, 0.28],
    )
    annual_mileage = (
        np.clip(
            rng.lognormal(mean=np.log(12_000), sigma=0.45, size=n_drivers),
            2_000,
            45_000,
        )
        .round()
        .astype(int)
    )

    observed = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "age": ages,
            "years_licensed": years_licensed,
            "annual_mileage_km": annual_mileage,
            "home_area_type": area_types,
            "driving_pattern": driving_patterns,
        }
    )

    aggressiveness = rng.beta(2.2, 4.8, size=n_drivers)
    attentiveness = rng.beta(5.0, 2.0, size=n_drivers)
    base_night_tendency = rng.beta(1.3, 12.0, size=n_drivers)
    shift_adjustment = np.where(driving_patterns == "shift_worker", 0.22, 0.0)

    latent = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "aggressiveness": aggressiveness,
            "attentiveness": attentiveness,
            "night_driving_tendency": np.clip(base_night_tendency + shift_adjustment, 0.01, 0.65),
        }
    )
    return DriverPopulation(observed=observed, latent=latent)
