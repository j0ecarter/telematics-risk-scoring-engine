from datetime import date

import numpy as np
import pytest

from apexrisk.simulation import generate_drivers, generate_journeys, generate_vehicles
from apexrisk.simulation.journeys import (
    MAXIMUM_TRIP_DURATION_SECONDS,
    MINIMUM_TRIP_DURATION_SECONDS,
    _bounded_duration_and_average_speed,
)


def test_journeys_have_valid_references_and_ranges() -> None:
    rng = np.random.default_rng(12)
    population = generate_drivers(8, rng)
    vehicles = generate_vehicles(population.observed, rng)
    trips = generate_journeys(
        population.observed,
        vehicles,
        population.latent,
        start_date=date(2025, 1, 1),
        months=2,
        mean_trips_per_driver_per_month=4,
        rng=rng,
    )

    assert trips["trip_id"].is_unique
    assert set(trips["driver_id"]).issubset(set(population.observed["driver_id"]))
    assert set(trips["vehicle_id"]).issubset(set(vehicles["vehicle_id"]))
    assert (trips["distance_km"] > 0).all()
    assert (trips["duration_seconds"] >= MINIMUM_TRIP_DURATION_SECONDS).all()
    assert (trips["duration_seconds"] <= MAXIMUM_TRIP_DURATION_SECONDS).all()
    implied_average_speed = trips["distance_km"] / trips["duration_seconds"] * 3_600
    assert np.allclose(trips["average_speed_kmh"], implied_average_speed, atol=0.01)
    assert (trips["speeding_duration_seconds"] <= trips["duration_seconds"]).all()
    assert trips["start_timestamp_utc"].dt.tz is not None
    assert trips["start_timestamp_utc"].max().date() < date(2025, 3, 1)


def test_extreme_journey_is_bounded_to_four_hours_without_breaking_average_speed() -> None:
    duration_seconds, average_speed_kmh = _bounded_duration_and_average_speed(
        distance_km=180.0,
        target_average_speed_kmh=8.0,
    )

    assert duration_seconds == MAXIMUM_TRIP_DURATION_SECONDS
    assert average_speed_kmh == pytest.approx(45.0)


def test_very_short_journey_keeps_minimum_duration_and_coherent_average_speed() -> None:
    duration_seconds, average_speed_kmh = _bounded_duration_and_average_speed(
        distance_km=0.8,
        target_average_speed_kmh=120.0,
    )

    assert duration_seconds == MINIMUM_TRIP_DURATION_SECONDS
    assert average_speed_kmh == pytest.approx(16.0)
