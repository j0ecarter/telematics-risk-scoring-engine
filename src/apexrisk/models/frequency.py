"""Learning scaffold for a Poisson claim-frequency model; no model is fitted yet."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "generated"

FEATURE_COLUMNS = [
    "speeding_time_fraction",
    "hard_brakes_per_100km",
    "night_driving_fraction",
]
TARGET_COLUMN = "claim_count"
EXPOSURE_COLUMN = "exposure_years"


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load driving features and outcomes, including months with zero claims."""
    features = pd.read_parquet(data_dir / "driver_month_features.parquet")
    exposure = pd.read_parquet(data_dir / "policy_exposure.parquet")
    return features, exposure


def build_training_table(features: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    """Match previous calendar month's behaviour to the target month's outcomes."""
    # TODO 1: Select driver_id, year_month and FEATURE_COLUMNS from features.
    # TODO 2: Convert year_month to monthly periods and add one calendar month.
    #         This labels January's behaviour as an input for February's claims.
    # TODO 3: Join to exposure on driver_id and target month, validating one-to-one.
    #         Keep zero-claim rows. Do not use target-month driving as inputs.
    # TODO 4: Explicitly handle missing history and check exposure_years > 0.
    raise NotImplementedError("Next exercise: build the training table.")


def fit_frequency_model(training_data: pd.DataFrame):
    """Fit a Poisson model using only the training portion of the data."""
    # TODO 5: Calculate a benchmark: total claims / total exposure_years.
    # TODO 6: Choose a modelling library and fit using FEATURE_COLUMNS.
    #         Account for exposure; it is not an ordinary behavioural feature.
    raise NotImplementedError("Later exercise: fit the Poisson model.")


def main() -> None:
    """Start by inspecting the inputs before implementing the TODOs."""
    features, exposure = load_data()
    print("Driving features:")
    print(features.head().to_string(index=False))
    print("\nExposure and claims:")
    print(exposure.head().to_string(index=False))
    # Next: call build_training_table(features, exposure) after completing it.
    # Before fitting: reserve later months / held-out drivers for evaluation.


if __name__ == "__main__":
    main()
