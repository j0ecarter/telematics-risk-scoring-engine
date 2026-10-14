"""Generate context-aware synthetic trip summaries."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

SPEED_LIMIT_KMH = {"urban": 48.3, "rural": 96.6, "motorway": 112.7}
BASE_SPEED_RATIO = {"urban": 0.55, "rural": 0.72, "motorway": 0.82}
MINIMUM_TRIP_DURATION_SECONDS = 180
MAXIMUM_TRIP_DURATION_SECONDS = 4 * 60 * 60


def _bounded_duration_and_average_speed(
    distance_km: float,
    target_average_speed_kmh: float,
) -> tuple[int, float]:
    """Return a coherent duration and average speed for one synthetic trip.

    A trip is one uninterrupted driving leg. Durations are bounded to three minutes through
    four hours, then average speed is recomputed so distance, duration and average speed
    continue to describe the same journey rather than three independently clipped values.
    """

    unbounded_duration_seconds = round(distance_km / target_average_speed_kmh * 3_600)
    duration_seconds = int(
        np.clip(
            unbounded_duration_seconds,
            MINIMUM_TRIP_DURATION_SECONDS,
            MAXIMUM_TRIP_DURATION_SECONDS,
        )
    )
    average_speed_kmh = distance_km / duration_seconds * 3_600
    return duration_seconds, average_speed_kmh


def _road_type(area_type: str, rng: np.random.Generator) -> str:
    probabilities = {
        "urban": [0.68, 0.18, 0.14],
        "suburban": [0.44, 0.31, 0.25],
        "rural": [0.18, 0.55, 0.27],
    }
    return str(rng.choice(["urban", "rural", "motorway"], p=probabilities[area_type]))


def _departure_hour(driving_pattern: str, night_tendency: float, rng: np.random.Generator) -> int:
    if rng.random() < night_tendency:
        return int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))
    if driving_pattern == "commuter" and rng.random() < 0.62:
        return int(rng.choice([7, 8, 9, 16, 17, 18]))
    return int(rng.integers(6, 22))


def generate_journeys(
    drivers: pd.DataFrame,
    vehicles: pd.DataFrame,
    latent_traits: pd.DataFrame,
    *,
    start_date: date,
    months: int,
    mean_trips_per_driver_per_month: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate trip-level observations influenced by driver and road context."""

    if months <= 0 or mean_trips_per_driver_per_month <= 0:
        raise ValueError("months and mean trip frequency must be positive")
    if drivers.empty:
        raise ValueError("drivers must not be empty")

    driver_table = drivers.merge(latent_traits, on="driver_id", validate="one_to_one")
    driver_table = driver_table.merge(
        vehicles[["vehicle_id", "driver_id", "engine_power_kw"]],
        on="driver_id",
        validate="one_to_one",
    )

    start = pd.Timestamp(start_date, tz="UTC")
    end = start + pd.DateOffset(months=months)
    n_days = (end - start).days
    records: list[dict[str, object]] = []
    trip_number = 1

    for driver in driver_table.itertuples(index=False):
        n_trips = max(
            1,
            int(rng.poisson(mean_trips_per_driver_per_month * months)),
        )
        for _ in range(n_trips):
            road_type = _road_type(driver.home_area_type, rng)
            speed_limit = SPEED_LIMIT_KMH[road_type]
            hour = _departure_hour(driver.driving_pattern, driver.night_driving_tendency, rng)
            departure = start + pd.Timedelta(
                days=int(rng.integers(0, n_days)),
                hours=hour,
                minutes=int(rng.integers(0, 60)),
            )
            is_night = hour >= 22 or hour < 6
            weather = str(rng.choice(["clear", "rain", "fog", "ice"], p=[0.67, 0.25, 0.06, 0.02]))
            traffic_prob = 0.68 if road_type == "urban" else 0.38
            if hour in {7, 8, 9, 16, 17, 18}:
                traffic_prob += 0.18
            traffic_level = str(
                rng.choice(
                    ["light", "moderate", "heavy"],
                    p=([0.16, 0.43, 0.41] if traffic_prob > 0.65 else [0.48, 0.38, 0.14]),
                )
            )

            distance_scale = {"urban": 7.0, "rural": 14.0, "motorway": 28.0}[road_type]
            distance_km = float(np.clip(rng.lognormal(np.log(distance_scale), 0.58), 0.8, 180.0))
            traffic_factor = {"light": 1.0, "moderate": 0.85, "heavy": 0.65}[traffic_level]
            weather_factor = {"clear": 1.0, "rain": 0.90, "fog": 0.78, "ice": 0.65}[weather]
            target_average_speed = float(
                np.clip(
                    speed_limit
                    * BASE_SPEED_RATIO[road_type]
                    * traffic_factor
                    * weather_factor
                    * rng.normal(1.0, 0.07),
                    8.0,
                    speed_limit * 1.05,
                )
            )
            duration_seconds, average_speed = _bounded_duration_and_average_speed(
                distance_km,
                target_average_speed,
            )

            speed_propensity = 0.17 + 0.72 * driver.aggressiveness
            has_speeding = rng.random() < speed_propensity
            speeding_fraction = (
                float(
                    np.clip(
                        rng.beta(1.0 + 4.0 * driver.aggressiveness, 7.0),
                        0.01,
                        0.75,
                    )
                )
                if has_speeding
                else 0.0
            )
            speeding_seconds = int(round(duration_seconds * speeding_fraction))
            mean_speeding_excess_pct = (
                float(np.clip(rng.normal(6.0 + 20.0 * driver.aggressiveness, 4.0), 1, 45))
                if has_speeding
                else 0.0
            )
            max_speed = float(
                min(
                    190.0,
                    max(
                        average_speed,
                        speed_limit * (1.0 + mean_speeding_excess_pct / 100.0)
                        if has_speeding
                        else average_speed * rng.uniform(1.02, 1.18),
                    ),
                )
            )

            heavy_traffic = 1.0 if traffic_level == "heavy" else 0.0
            adverse_weather = 1.0 if weather != "clear" else 0.0
            exposure_units = distance_km / 100.0
            hard_brake_rate = np.exp(
                -0.10
                + 1.75 * driver.aggressiveness
                + 0.60 * heavy_traffic
                + 0.35 * adverse_weather
                + 0.20 * speeding_fraction
            )
            hard_accel_rate = np.exp(
                -0.35 + 2.10 * driver.aggressiveness + 0.25 * (driver.engine_power_kw / 100.0)
            )
            cornering_rate = np.exp(
                -0.55 + 1.55 * driver.aggressiveness + 0.45 * (road_type != "motorway")
            )

            records.append(
                {
                    "trip_id": f"TRIP-{trip_number:09d}",
                    "driver_id": driver.driver_id,
                    "vehicle_id": driver.vehicle_id,
                    "start_timestamp_utc": departure,
                    "end_timestamp_utc": departure + pd.Timedelta(seconds=duration_seconds),
                    "distance_km": round(distance_km, 3),
                    "duration_seconds": duration_seconds,
                    "road_type": road_type,
                    "speed_limit_kmh": speed_limit,
                    "average_speed_kmh": round(average_speed, 3),
                    "maximum_speed_kmh": round(max_speed, 3),
                    "weather": weather,
                    "traffic_level": traffic_level,
                    "is_night_trip": is_night,
                    "hard_brake_count": int(rng.poisson(hard_brake_rate * exposure_units)),
                    "hard_acceleration_count": int(rng.poisson(hard_accel_rate * exposure_units)),
                    "harsh_cornering_count": int(rng.poisson(cornering_rate * exposure_units)),
                    "speeding_duration_seconds": speeding_seconds,
                    "mean_speeding_excess_pct": round(mean_speeding_excess_pct, 3),
                }
            )
            trip_number += 1

    trips = pd.DataFrame.from_records(records)
    return trips.sort_values(["driver_id", "start_timestamp_utc"]).reset_index(drop=True)
