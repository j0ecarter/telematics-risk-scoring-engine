"""Evaluate synthetic claim-frequency predictions on a held-out later month.

Run from the project folder: .venv/bin/python -m apexrisk.models.evaluation
Report portfolio totals, Poisson deviance, and calibration by predicted risk group.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from apexrisk.models.frequency import (
    EXPOSURE_COLUMN,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_training_table,
    fit_frequency_model,
    load_data,
)


def build_prediction_table() -> pd.DataFrame:
    """Fit on earlier months and return predictions for the latest month."""
    features, exposure = load_data()
    table = build_training_table(features, exposure)
    latest_month = table["year_month"].max()
    training = table[table["year_month"] < latest_month].copy()
    test = table[table["year_month"] == latest_month].copy()
    if training.empty or test.empty:
        raise ValueError("Need at least two usable target months for this split.")

    model, baseline_rate = fit_frequency_model(training)
    inputs = sm.add_constant(test[FEATURE_COLUMNS], has_constant="add")
    predictions = test[
        ["driver_id", "year_month", TARGET_COLUMN, EXPOSURE_COLUMN]
    ].copy()
    predictions["model_claims"] = model.predict(inputs, exposure=test[EXPOSURE_COLUMN])
    predictions["baseline_claims"] = baseline_rate * test[EXPOSURE_COLUMN]
    predictions["model_annual_rate"] = (
        predictions["model_claims"] / predictions[EXPOSURE_COLUMN]
    )
    return predictions.reset_index(drop=True)


def summarise_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """ Comparing the actual claims with the model and the benchmarks predictions"""

    actual_total = predictions["claim_count"].sum()
    total_exposure = predictions["exposure_years"].sum()

    if predictions.empty or total_exposure <= 0:
        raise ValueError("Need prediction rows with a positive total exposure")

    results = []

    for name, column in [
        ("Poisson model", "model_claims"),
        ("Benchmark", "baseline_claims"),
    ]:
        predicted_total = predictions[column].sum()

        results.append({
            "method": name,
            "actual_claims": actual_total,
            "predicted_claims": predicted_total,
            "actual_annual_rate": actual_total / total_exposure,
            "predicted_annual_rate": predicted_total / total_exposure,
            "observed_to_expected": (
                actual_total / predicted_total
                if predicted_total > 0
                else np.nan
            ),
        })

    return pd.DataFrame(results)


def mean_poisson_deviance(actual: pd.Series, predicted: pd.Series) -> float:
    # Use values in row order, without pandas matching index labels
    y = actual.to_numpy(dtype=float)
    mu = predicted.to_numpy(dtype=float)

    if len(y) == 0 or y.shape != mu.shape:
        raise ValueError("Inputs must have the same non-zero length.")

    if not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError("Inputs must contain only finite numbers.")

    if (y < 0).any():
        raise ValueError("Actual claims count cannot be negative.")

    if (mu <= 0).any():
        raise ValueError("Predicted claim counts must be positive")

    # for the rows with no claims, the error is simply x2 the prediction:
    errors = 2 * mu.copy()

    # Only calculate the logarithm where actual claims are positive:
    positive = y > 0

    errors[positive] = 2 * (
        y[positive] * np.log(y[positive] / mu[positive])
        - y[positive]
        + mu[positive]
    )

    return float(errors.mean())

def calibration_by_risk(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare predicted and actual annual claim rates by risk group."""

    if predictions.empty:
        raise ValueError("The prediction table is empty.")

    table = predictions.copy()

    # If every prediction is identical, there is only one risk group.
    if table["model_annual_rate"].nunique() == 1:
        table["risk_group"] = 1
    else:
        # Divide drivers into up to three groups, ordered by predicted risk.
        table["risk_group"] = pd.qcut(
            table["model_annual_rate"],
            q=3,
            labels=False,
            duplicates="drop",
        ) + 1

    # Add up outcomes and exposure within each group.
    summary = (
        table.groupby("risk_group", observed=True)
        .agg(
            driver_months=("driver_id", "size"),
            exposure_years=("exposure_years", "sum"),
            actual_claims=("claim_count", "sum"),
            predicted_claims=("model_claims", "sum"),
        )
        .reset_index()
    )

    # Put actual and predicted claims on the same annual-rate scale.
    summary["actual_annual_rate"] = (
        summary["actual_claims"] / summary["exposure_years"]
    )

    summary["predicted_annual_rate"] = (
        summary["predicted_claims"] / summary["exposure_years"]
    )

    return summary

def main() -> None:
    predictions = build_prediction_table()
    print("Held-out month:", predictions["year_month"].iloc[0])
    print(predictions.head().to_string(index=False))
    print("\nPrediction summary:")
    print(summarise_predictions(predictions).to_string(index=False))

    model_error = mean_poisson_deviance(
        predictions["claim_count"],
        predictions["model_claims"],
    )

    baseline_error = mean_poisson_deviance(
        predictions["claim_count"],
        predictions["baseline_claims"],
    )

    print(f"\nModel Poisson deviance: {model_error:.4f}")
    print(f"Benchmark Poisson deviance: {baseline_error:.4f}")
    print("\nClaims by predicted risk group:")
    print(calibration_by_risk(predictions).to_string(index=False))


if __name__ == "__main__":
    main()
