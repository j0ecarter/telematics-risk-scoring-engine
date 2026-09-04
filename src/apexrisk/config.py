"""Typed configuration loading for the synthetic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
