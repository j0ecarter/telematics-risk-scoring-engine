"""Exposure-aware Poisson claim-frequency baseline for synthetic data."""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

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
    # Keeping only the driving information that we want the model to be using:
    previous_features = features[
        ["driver_id", "year_month", *FEATURE_COLUMNS]
    ].copy()

    # January's driving becomes an input for predicting February.
    previous_features["year_month"] = (
        pd.PeriodIndex(previous_features["year_month"], freq="M") + 1
    )

    # These are the outcomes and exposure for the month that is being predicted:
    outcomes = exposure[
        ["driver_id", "year_month", TARGET_COLUMN, EXPOSURE_COLUMN]
    ].copy()

    outcomes["year_month"] = pd.PeriodIndex(
        outcomes["year_month"], freq="M"
    )

    table = outcomes.merge(
        previous_features,
        on=["driver_id", "year_month"],
        how="left",
        validate="one_to_one"
    )

    # For this first version, use only rows with complete driving history:
    # Zero claims are valid and are retained:
    original_rows = len(table)
    table = table.dropna(subset=FEATURE_COLUMNS).copy()
    print(f"Excluded {original_rows - len(table)} rows with missing history.")

    if table.empty:
        raise ValueError("No rows have complete previous-month features.")

    if table[[TARGET_COLUMN, EXPOSURE_COLUMN]].isna().any().any():
        raise ValueError("Claim counts and exposure must not be missing.")

    if (table[EXPOSURE_COLUMN] <= 0).any():
        raise ValueError("Exposure must be greater than zero.")

    return table.sort_values(
        ["year_month", "driver_id"]
    ).reset_index(drop=True)


def fit_frequency_model(training_data: pd.DataFrame):
    """Fit a Poisson model using only the training portion of the data."""
    if training_data.empty:
        raise ValueError("The training data is empty.")

    # what we want to predict i.e. number of claims
    y = training_data[TARGET_COLUMN]

    # X contains the behavioural inputs
    X = sm.add_constant(
        training_data[FEATURE_COLUMNS],
        has_constant="add",
    )

    exposure = training_data[EXPOSURE_COLUMN]

    if y.sum() == 0:
        raise ValueError("The training data needs at least one claim in it")

    # Simple comparison giving every driver the same annual claims rate:
    baseline_rate = y.sum() / exposure.sum()

    model = sm.GLM(
        endog=y,
        exog=X,
        family=sm.families.Poisson(),
        exposure=exposure,
        missing="raise",
    ).fit()

    if not model.converged:
        raise RuntimeError("The Poisson model did not converge")

    return model, baseline_rate


def main() -> None:
    """Fit on earlier months and compare latest-month predictions with a constant-rate baseline."""
    features, exposure = load_data()
    table = build_training_table(features, exposure)

    # Reserve the latest month for a basic forward-looking check.
    latest_month = table["year_month"].max()
    training_data = table[table["year_month"] < latest_month].copy()
    test_data = table[table["year_month"] == latest_month].copy()

    model, baseline_rate = fit_frequency_model(training_data)

    X_test = sm.add_constant(
        test_data[FEATURE_COLUMNS],
        has_constant="add",
    )

    predicted_claims = model.predict(
        X_test,
        exposure=test_data[EXPOSURE_COLUMN],
    )

    baseline_claims = baseline_rate * test_data[EXPOSURE_COLUMN]

    print(f"Training rows: {len(training_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Baseline annual claim rate: {baseline_rate:.4f}")
    print(f"Actual test claims: {test_data[TARGET_COLUMN].sum()}")
    print(f"Model predicted claims: {predicted_claims.sum():.2f}")
    print(f"Baseline predicted claims: {baseline_claims.sum():.2f}")


if __name__ == "__main__":
    main()
