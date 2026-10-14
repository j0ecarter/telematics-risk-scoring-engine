from datetime import date

import numpy as np

from apexrisk.simulation import generate_drivers, generate_journeys, generate_vehicles


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
    assert (trips["duration_seconds"] > 0).all()
    assert (trips["speeding_duration_seconds"] <= trips["duration_seconds"]).all()
    assert trips["start_timestamp_utc"].dt.tz is not None
    assert trips["start_timestamp_utc"].max().date() < date(2025, 3, 1)
