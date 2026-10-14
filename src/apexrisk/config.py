"""Typed configuration loading for the synthetic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import isclose
from pathlib import Path
from typing import Any

import yaml


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


@dataclass(frozen=True)
class SimulationConfig:
    seed: int
    n_drivers: int
    start_date: date
    months: int
    mean_trips_per_driver_per_month: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimulationConfig:
        payload = _read_yaml(Path(path)).get("simulation", {})
        config = cls(
            seed=int(payload["seed"]),
            n_drivers=int(payload["n_drivers"]),
            start_date=date.fromisoformat(str(payload["start_date"])),
            months=int(payload["months"]),
            mean_trips_per_driver_per_month=float(payload["mean_trips_per_driver_per_month"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.n_drivers <= 0:
            raise ValueError("n_drivers must be positive")
        if self.months <= 0:
            raise ValueError("months must be positive")
        if self.mean_trips_per_driver_per_month <= 0:
            raise ValueError("mean_trips_per_driver_per_month must be positive")


@dataclass(frozen=True)
class FeatureConfig:
    minimum_monthly_distance_km: float = 1.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureConfig:
        payload = _read_yaml(Path(path)).get("features", {})
        config = cls(
            minimum_monthly_distance_km=float(payload.get("minimum_monthly_distance_km", 1.0))
        )
        if config.minimum_monthly_distance_km <= 0:
            raise ValueError("minimum_monthly_distance_km must be positive")
        return config


@dataclass(frozen=True)
class ClaimsConfig:
    """Versioned assumptions for synthetic claim frequency and severity."""

    base_annual_claim_frequency: float = 0.22
    aggressiveness_log_weight: float = 1.10
    inattention_log_weight: float = 1.00
    speeding_log_weight: float = 1.50
    night_driving_log_weight: float = 0.70
    hard_braking_log_weight: float = 0.18
    young_driver_log_weight: float = 0.45
    low_safety_log_weight: float = 0.25
    distance_log_weight: float = 0.20
    minimum_frequency_multiplier: float = 0.25
    maximum_frequency_multiplier: float = 4.00
    severity_median_gbp: float = 3_200.0
    severity_log_sigma: float = 0.85
    vehicle_value_elasticity: float = 0.20
    large_loss_probability: float = 0.03
    large_loss_minimum_gbp: float = 25_000.0
    large_loss_pareto_shape: float = 1.80
    maximum_claim_cost_gbp: float = 500_000.0
    claim_type_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "collision": 0.62,
            "single_vehicle": 0.20,
            "weather": 0.10,
            "bodily_injury": 0.08,
        }
    )
    claim_type_severity_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "collision": 1.00,
            "single_vehicle": 0.85,
            "weather": 0.70,
            "bodily_injury": 2.50,
        }
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ClaimsConfig:
        payload = _read_yaml(Path(path)).get("claims", {})
        defaults = cls()
        config = cls(
            base_annual_claim_frequency=float(
                payload.get("base_annual_claim_frequency", defaults.base_annual_claim_frequency)
            ),
            aggressiveness_log_weight=float(
                payload.get("aggressiveness_log_weight", defaults.aggressiveness_log_weight)
            ),
            inattention_log_weight=float(
                payload.get("inattention_log_weight", defaults.inattention_log_weight)
            ),
            speeding_log_weight=float(
                payload.get("speeding_log_weight", defaults.speeding_log_weight)
            ),
            night_driving_log_weight=float(
                payload.get("night_driving_log_weight", defaults.night_driving_log_weight)
            ),
            hard_braking_log_weight=float(
                payload.get("hard_braking_log_weight", defaults.hard_braking_log_weight)
            ),
            young_driver_log_weight=float(
                payload.get("young_driver_log_weight", defaults.young_driver_log_weight)
            ),
            low_safety_log_weight=float(
                payload.get("low_safety_log_weight", defaults.low_safety_log_weight)
            ),
            distance_log_weight=float(
                payload.get("distance_log_weight", defaults.distance_log_weight)
            ),
            minimum_frequency_multiplier=float(
                payload.get(
                    "minimum_frequency_multiplier", defaults.minimum_frequency_multiplier
                )
            ),
            maximum_frequency_multiplier=float(
                payload.get(
                    "maximum_frequency_multiplier", defaults.maximum_frequency_multiplier
                )
            ),
            severity_median_gbp=float(
                payload.get("severity_median_gbp", defaults.severity_median_gbp)
            ),
            severity_log_sigma=float(
                payload.get("severity_log_sigma", defaults.severity_log_sigma)
            ),
            vehicle_value_elasticity=float(
                payload.get("vehicle_value_elasticity", defaults.vehicle_value_elasticity)
            ),
            large_loss_probability=float(
                payload.get("large_loss_probability", defaults.large_loss_probability)
            ),
            large_loss_minimum_gbp=float(
                payload.get("large_loss_minimum_gbp", defaults.large_loss_minimum_gbp)
            ),
            large_loss_pareto_shape=float(
                payload.get("large_loss_pareto_shape", defaults.large_loss_pareto_shape)
            ),
            maximum_claim_cost_gbp=float(
                payload.get("maximum_claim_cost_gbp", defaults.maximum_claim_cost_gbp)
            ),
            claim_type_probabilities={
                str(name): float(probability)
                for name, probability in payload.get(
                    "claim_type_probabilities", defaults.claim_type_probabilities
                ).items()
            },
            claim_type_severity_multipliers={
                str(name): float(multiplier)
                for name, multiplier in payload.get(
                    "claim_type_severity_multipliers",
                    defaults.claim_type_severity_multipliers,
                ).items()
            },
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_fields = {
            "base_annual_claim_frequency": self.base_annual_claim_frequency,
            "minimum_frequency_multiplier": self.minimum_frequency_multiplier,
            "maximum_frequency_multiplier": self.maximum_frequency_multiplier,
            "severity_median_gbp": self.severity_median_gbp,
            "severity_log_sigma": self.severity_log_sigma,
            "large_loss_minimum_gbp": self.large_loss_minimum_gbp,
            "large_loss_pareto_shape": self.large_loss_pareto_shape,
            "maximum_claim_cost_gbp": self.maximum_claim_cost_gbp,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.maximum_frequency_multiplier < self.minimum_frequency_multiplier:
            raise ValueError(
                "maximum_frequency_multiplier must be at least minimum_frequency_multiplier"
            )
        if not 0 <= self.large_loss_probability <= 1:
            raise ValueError("large_loss_probability must be between zero and one")
        if self.vehicle_value_elasticity < 0:
            raise ValueError("vehicle_value_elasticity must not be negative")
        if not self.claim_type_probabilities:
            raise ValueError("claim_type_probabilities must not be empty")
        if set(self.claim_type_probabilities) != set(self.claim_type_severity_multipliers):
            raise ValueError("claim type probability and severity keys must match")
        if any(probability < 0 for probability in self.claim_type_probabilities.values()):
            raise ValueError("claim type probabilities must not be negative")
        if not isclose(sum(self.claim_type_probabilities.values()), 1.0, abs_tol=1e-9):
            raise ValueError("claim type probabilities must sum to one")
        if any(multiplier <= 0 for multiplier in self.claim_type_severity_multipliers.values()):
            raise ValueError("claim type severity multipliers must be positive")
