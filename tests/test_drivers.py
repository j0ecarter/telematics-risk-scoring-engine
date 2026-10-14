import numpy as np
import pandas as pd

from apexrisk.simulation.drivers import generate_drivers


def test_driver_generation_is_reproducible() -> None:
    first = generate_drivers(20, np.random.default_rng(42))
    second = generate_drivers(20, np.random.default_rng(42))

    pd.testing.assert_frame_equal(first.observed, second.observed)
    pd.testing.assert_frame_equal(first.latent, second.latent)


def test_private_traits_are_not_in_observed_driver_table() -> None:
    population = generate_drivers(25, np.random.default_rng(7))

    assert population.observed["driver_id"].is_unique
    assert population.observed["age"].between(18, 80).all()
    assert "aggressiveness" not in population.observed.columns
    assert "attentiveness" not in population.observed.columns
    assert set(population.observed["driver_id"]) == set(population.latent["driver_id"])
