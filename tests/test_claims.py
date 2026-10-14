from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from apexrisk.config import ClaimsConfig
from apexrisk.features import build_driver_month_features
from apexrisk.simulation import (
    generate_drivers,
    generate_journeys,
    generate_policy_exposure_and_claims,
    generate_vehicles,
)


def _example_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(41)
    population = generate_drivers(40, rng)
    vehicles = generate_vehicles(population.observed, rng)
    trips = generate_journeys(
        population.observed,
        vehicles,
        population.latent,
        start_date=date(2025, 1, 1),
        months=3,
        mean_trips_per_driver_per_month=5,
        rng=rng,
    )
    features = build_driver_month_features(trips)
    return population.observed, vehicles, population.latent, features


def test_claim_generation_is_reproducible_and_reconciles_to_exposure() -> None:
    drivers, vehicles, latent, features = _example_inputs()
    config = replace(
        ClaimsConfig(),
        base_annual_claim_frequency=2.0,
        large_loss_probability=0.20,
    )

    first = generate_policy_exposure_and_claims(
        drivers,
        vehicles,
        latent,
        features,
        start_date=date(2025, 1, 1),
        months=3,
        config=config,
        rng=np.random.default_rng(99),
    )
    second = generate_policy_exposure_and_claims(
        drivers,
        vehicles,
        latent,
        features,
        start_date=date(2025, 1, 1),
        months=3,
        config=config,
        rng=np.random.default_rng(99),
    )

    pd.testing.assert_frame_equal(first.policy_exposure, second.policy_exposure)
    pd.testing.assert_frame_equal(first.claims, second.claims)
    assert len(first.policy_exposure) == len(drivers) * 3
    assert first.policy_exposure["policy_month_id"].is_unique
    assert not first.claims.empty

    counts = first.claims.groupby("policy_month_id").size()
    expected_counts = first.policy_exposure.set_index("policy_month_id")["claim_count"]
    pd.testing.assert_series_equal(
        counts.reindex(expected_counts.index, fill_value=0).astype(int),
        expected_counts,
        check_names=False,
    )
    costs = first.claims.groupby("policy_month_id")["incurred_cost_gbp"].sum()
    expected_costs = first.policy_exposure.set_index("policy_month_id")[
        "incurred_claim_cost_gbp"
    ]
    pd.testing.assert_series_equal(
        costs.reindex(expected_costs.index, fill_value=0.0).round(2),
        expected_costs,
        check_names=False,
    )

    private_columns = {"aggressiveness", "attentiveness", "frequency_multiplier"}
    assert private_columns.isdisjoint(first.policy_exposure.columns)
    assert private_columns.isdisjoint(first.claims.columns)

    dated_claims = first.claims.merge(
        first.policy_exposure[
            ["policy_month_id", "period_start_utc", "period_end_utc"]
        ],
        on="policy_month_id",
        validate="many_to_one",
    )
    assert (dated_claims["claim_timestamp_utc"] >= dated_claims["period_start_utc"]).all()
    assert (dated_claims["claim_timestamp_utc"] < dated_claims["period_end_utc"]).all()


def test_riskier_synthetic_profiles_have_more_claims_on_average() -> None:
    n_per_group = 100
    driver_ids = [f"D-{index:04d}" for index in range(n_per_group * 2)]
    risk_group = np.repeat(["lower", "higher"], n_per_group)
    drivers = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "age": 35,
            "annual_mileage_km": 12_000,
            "risk_group": risk_group,
        }
    )
    vehicles = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "vehicle_id": [f"V-{index:04d}" for index in range(n_per_group * 2)],
            "vehicle_value_gbp": 20_000.0,
            "safety_rating": 5,
        }
    )
    higher_risk = risk_group == "higher"
    latent = pd.DataFrame(
        {
            "driver_id": driver_ids,
            "aggressiveness": np.where(higher_risk, 0.90, 0.05),
            "attentiveness": np.where(higher_risk, 0.10, 0.95),
        }
    )
    feature_rows = []
    for driver_id, is_higher in zip(driver_ids, higher_risk, strict=True):
        for month in range(1, 13):
            feature_rows.append(
                {
                    "driver_id": driver_id,
                    "year_month": f"2025-{month:02d}",
                    "trip_count": 20,
                    "distance_km": 1_000.0,
                    "night_driving_fraction": 0.45 if is_higher else 0.02,
                    "speeding_time_fraction": 0.40 if is_higher else 0.01,
                    "hard_brakes_per_100km": 9.0 if is_higher else 0.5,
                }
            )

    result = generate_policy_exposure_and_claims(
        drivers,
        vehicles,
        latent,
        pd.DataFrame.from_records(feature_rows),
        start_date=date(2025, 1, 1),
        months=12,
        config=replace(ClaimsConfig(), base_annual_claim_frequency=1.0),
        rng=np.random.default_rng(123),
    )
    totals = (
        result.policy_exposure.merge(
            drivers[["driver_id", "risk_group"]], on="driver_id", validate="many_to_one"
        )
        .groupby("risk_group")["claim_count"]
        .sum()
    )

    assert totals["higher"] > totals["lower"] * 3


def test_large_loss_component_produces_bounded_heavy_tail_claims() -> None:
    drivers, vehicles, latent, features = _example_inputs()
    config = replace(
        ClaimsConfig(),
        base_annual_claim_frequency=5.0,
        vehicle_value_elasticity=0.0,
        large_loss_probability=1.0,
        claim_type_probabilities={"collision": 1.0},
        claim_type_severity_multipliers={"collision": 1.0},
    )

    result = generate_policy_exposure_and_claims(
        drivers,
        vehicles,
        latent,
        features,
        start_date=date(2025, 1, 1),
        months=3,
        config=config,
        rng=np.random.default_rng(7),
    )

    assert not result.claims.empty
    assert result.claims["is_large_loss"].all()
    assert (result.claims["incurred_cost_gbp"] >= config.large_loss_minimum_gbp).all()
    assert (result.claims["incurred_cost_gbp"] <= config.maximum_claim_cost_gbp).all()
