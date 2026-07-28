"""Leakage-aware classical forecasting on a synthetic monthly series."""

import numpy as np
import pandas as pd

from asset_management_toolkit.time_series import (
    chronological_train_test_split,
    exponential_smoothing_forecast,
    forecast_metrics,
    seasonal_naive_forecast,
)


def main() -> None:
    """Compare a seasonal baseline with additive Holt-Winters."""
    generator = np.random.default_rng(17)
    index = pd.period_range("2018-01", periods=84, freq="M")
    trend = np.linspace(100.0, 125.0, len(index))
    seasonal_pattern = np.tile(
        [0.0, 2.0, 4.0, 3.0, 1.0, -1.0, -3.0, -2.0, 0.0, 1.0, 3.0, 2.0],
        7,
    )
    series = pd.Series(
        trend + seasonal_pattern + generator.normal(0.0, 0.5, len(index)),
        index=index,
        name="synthetic_level",
    )
    train, test = chronological_train_test_split(series, test_size=12)

    baseline = seasonal_naive_forecast(
        train,
        horizon=len(test),
        seasonal_period=12,
    )
    holt_winters = exponential_smoothing_forecast(
        train,
        horizon=len(test),
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    )

    comparison = pd.DataFrame(
        {
            "seasonal_naive": forecast_metrics(test, baseline),
            "holt_winters": forecast_metrics(test, holt_winters.forecast),
        }
    )
    print(comparison.round(4))


if __name__ == "__main__":
    main()
