"""Generate synthetic policy-month exposure and individual insurance claims."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from apexrisk.config import ClaimsConfig

EXPOSURE_COLUMNS = [
    "policy_month_id",
    "driver_id",
    "vehicle_id",
    "year_month",
    "period_start_utc",
    "period_end_utc",
    "exposure_days",
    "exposure_years",
    "trip_count",
    "distance_km",
    "claim_count",
    "incurred_claim_cost_gbp",
]

CLAIM_COLUMNS = [
    "claim_id",
    "policy_month_id",
    "driver_id",
    "vehicle_id",
    "claim_timestamp_utc",
    "claim_type",
    "incurred_cost_gbp",
    "is_large_loss",
]

FEATURE_COLUMNS = [
    "driver_id",
    "year_month",
    "trip_count",
    "distance_km",
    "night_driving_fraction",
    "speeding_time_fraction",
    "hard_brakes_per_100km",
]


@dataclass(frozen=True)
class ClaimsSimulation:
    """Published synthetic outcomes; private risk scores are never retained."""

    policy_exposure: pd.DataFrame
    claims: pd.DataFrame


def _validate_inputs(
    drivers: pd.DataFrame,
    vehicles: pd.DataFrame,
    latent_traits: pd.DataFrame,
    driver_month_features: pd.DataFrame,
) -> None:
    required_driver_columns = {"driver_id", "age", "annual_mileage_km"}
    required_vehicle_columns = {
        "driver_id",
        "vehicle_id",
        "vehicle_value_gbp",
        "safety_rating",
    }
    required_latent_columns = {"driver_id", "aggressiveness", "attentiveness"}
    requirements = [
        ("drivers", drivers, required_driver_columns),
        ("vehicles", vehicles, required_vehicle_columns),
        ("latent_traits", latent_traits, required_latent_columns),
        ("driver_month_features", driver_month_features, set(FEATURE_COLUMNS)),
    ]
    for name, frame, required in requirements:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    if drivers.empty:
        raise ValueError("drivers must not be empty")
    if not drivers["driver_id"].is_unique:
        raise ValueError("drivers must contain one row per driver_id")
    if not vehicles["driver_id"].is_unique:
        raise ValueError("vehicles must contain one row per driver_id")
    if not latent_traits["driver_id"].is_unique:
        raise ValueError("latent_traits must contain one row per driver_id")
    if driver_month_features.duplicated(["driver_id", "year_month"]).any():
        raise ValueError("driver_month_features must contain one row per driver and month")
    driver_ids = set(drivers["driver_id"])
    if not driver_ids.issubset(set(vehicles["driver_id"])):
        raise ValueError("every driver must have a vehicle")
    if not driver_ids.issubset(set(latent_traits["driver_id"])):
        raise ValueError("every driver must have latent simulation traits")


def _policy_month_calendar(
    driver_ids: pd.Series, *, start_date: date, months: int
) -> pd.DataFrame:
    starts = [
        pd.Timestamp(start_date, tz="UTC") + pd.DateOffset(months=index)
        for index in range(months)
    ]
    records = [
        {
            "driver_id": driver_id,
            "year_month": start.strftime("%Y-%m"),
            "period_start_utc": start,
            "period_end_utc": start + pd.DateOffset(months=1),
        }
        for driver_id in driver_ids
        for start in starts
    ]
    calendar = pd.DataFrame.from_records(records)
    calendar["policy_month_id"] = (
        "POLM-"
        + calendar["driver_id"].astype(str)
        + "-"
        + calendar["year_month"].str.replace("-", "", regex=False)
    )
    calendar["exposure_days"] = (
        calendar["period_end_utc"] - calendar["period_start_utc"]
    ).dt.days
    calendar["exposure_years"] = calendar["exposure_days"] / 365.25
    return calendar


def _frequency_multiplier(table: pd.DataFrame, config: ClaimsConfig) -> np.ndarray:
    young_driver = np.clip((25.0 - table["age"].to_numpy()) / 7.0, 0.0, 1.0)
    low_safety = np.clip((5.0 - table["safety_rating"].to_numpy()) / 3.0, 0.0, 1.0)
    inattention = 1.0 - table["attentiveness"].to_numpy()
    hard_braking = np.log1p(np.clip(table["hard_brakes_per_100km"].to_numpy(), 0.0, 20.0))
    monthly_expected_distance = table["annual_mileage_km"].to_numpy() / 12.0
    distance_ratio = np.clip(
        (table["distance_km"].to_numpy() + 50.0) / (monthly_expected_distance + 50.0),
        0.10,
        3.00,
    )

    log_multiplier = (
        config.aggressiveness_log_weight * (table["aggressiveness"].to_numpy() - 0.30)
        + config.inattention_log_weight * (inattention - 0.30)
        + config.speeding_log_weight
        * (table["speeding_time_fraction"].to_numpy() - 0.08)
        + config.night_driving_log_weight
        * (table["night_driving_fraction"].to_numpy() - 0.10)
        + config.hard_braking_log_weight * (hard_braking - np.log1p(1.50))
        + config.young_driver_log_weight * young_driver
        + config.low_safety_log_weight * low_safety
        + config.distance_log_weight * np.log(distance_ratio)
    )
    return np.clip(
        np.exp(log_multiplier),
        config.minimum_frequency_multiplier,
        config.maximum_frequency_multiplier,
    )


def _claim_cost(
    row: object,
    claim_type: str,
    *,
    config: ClaimsConfig,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    is_large_loss = bool(rng.random() < config.large_loss_probability)
    if is_large_loss:
        base_cost = config.large_loss_minimum_gbp * (
            1.0 + rng.pareto(config.large_loss_pareto_shape)
        )
    else:
        base_cost = rng.lognormal(
            mean=np.log(config.severity_median_gbp),
            sigma=config.severity_log_sigma,
        )
    vehicle_value_factor = (max(float(row.vehicle_value_gbp), 1_000.0) / 20_000.0) ** (
        config.vehicle_value_elasticity
    )
    cost = (
        base_cost
        * config.claim_type_severity_multipliers[claim_type]
        * vehicle_value_factor
    )
    return round(float(np.clip(cost, 100.0, config.maximum_claim_cost_gbp)), 2), is_large_loss


def generate_policy_exposure_and_claims(
    drivers: pd.DataFrame,
    vehicles: pd.DataFrame,
    latent_traits: pd.DataFrame,
    driver_month_features: pd.DataFrame,
    *,
    start_date: date,
    months: int,
    config: ClaimsConfig,
    rng: np.random.Generator,
) -> ClaimsSimulation:
    """Create one exposure row per driver-month and zero or more claim rows."""

    if months <= 0:
        raise ValueError("months must be positive")
    config.validate()
    _validate_inputs(drivers, vehicles, latent_traits, driver_month_features)

    calendar = _policy_month_calendar(drivers["driver_id"], start_date=start_date, months=months)
    feature_table = driver_month_features[FEATURE_COLUMNS].copy()
    table = calendar.merge(feature_table, on=["driver_id", "year_month"], how="left")
    numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in {"driver_id", "year_month"}
    ]
    table[numeric_features] = table[numeric_features].fillna(0.0)
    table = table.merge(
        drivers[["driver_id", "age", "annual_mileage_km"]],
        on="driver_id",
        validate="many_to_one",
    )
    table = table.merge(
        vehicles[["driver_id", "vehicle_id", "vehicle_value_gbp", "safety_rating"]],
        on="driver_id",
        validate="many_to_one",
    )
    table = table.merge(
        latent_traits[["driver_id", "aggressiveness", "attentiveness"]],
        on="driver_id",
        validate="many_to_one",
    ).sort_values(["driver_id", "period_start_utc"])

    multiplier = _frequency_multiplier(table, config)
    expected_claim_count = config.base_annual_claim_frequency * table["exposure_years"] * multiplier
    table["claim_count"] = rng.poisson(expected_claim_count).astype(int)

    claim_types = list(config.claim_type_probabilities)
    claim_probabilities = [config.claim_type_probabilities[name] for name in claim_types]
    claim_records: list[dict[str, object]] = []
    claim_number = 1
    for row in table.itertuples(index=False):
        for _ in range(row.claim_count):
            claim_type = str(rng.choice(claim_types, p=claim_probabilities))
            cost, is_large_loss = _claim_cost(row, claim_type, config=config, rng=rng)
            elapsed_seconds = int(rng.integers(0, int(row.exposure_days) * 24 * 60 * 60))
            claim_records.append(
                {
                    "claim_id": f"CLM-{claim_number:09d}",
                    "policy_month_id": row.policy_month_id,
                    "driver_id": row.driver_id,
                    "vehicle_id": row.vehicle_id,
                    "claim_timestamp_utc": row.period_start_utc
                    + pd.Timedelta(seconds=elapsed_seconds),
                    "claim_type": claim_type,
                    "incurred_cost_gbp": cost,
                    "is_large_loss": is_large_loss,
                }
            )
            claim_number += 1

    claims = pd.DataFrame.from_records(claim_records, columns=CLAIM_COLUMNS)
    if claims.empty:
        incurred_by_month = pd.Series(dtype=float)
    else:
        incurred_by_month = claims.groupby("policy_month_id")["incurred_cost_gbp"].sum()
    table["incurred_claim_cost_gbp"] = (
        table["policy_month_id"].map(incurred_by_month).fillna(0.0).round(2)
    )
    table["trip_count"] = table["trip_count"].astype(int)

    exposure = table[EXPOSURE_COLUMNS].reset_index(drop=True)
    return ClaimsSimulation(policy_exposure=exposure, claims=claims)
