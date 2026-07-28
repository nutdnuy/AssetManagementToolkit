"""Forecast evaluation with explicit label alignment and chronology."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import pandas as pd

from asset_management_toolkit.time_series._validation import (
    coerce_time_series,
    validate_positive_integer,
)
from asset_management_toolkit.time_series.preparation import rolling_origin_splits

ForecastOutput = Union[pd.Series, np.ndarray, list[float], tuple[float, ...]]
Forecaster = Callable[[pd.Series, int], ForecastOutput]


def forecast_metrics(
    actual: pd.Series,
    predicted: pd.Series,
) -> pd.Series:
    """Compute MAE, RMSE, bias, MAPE, and symmetric MAPE.

    MAPE excludes observations whose actual value is zero. Symmetric MAPE
    excludes observations where both actual and predicted values are zero.
    """
    observed = coerce_time_series(actual, name="actual")
    forecast = coerce_time_series(predicted, name="predicted")
    if not observed.index.equals(forecast.index):
        raise ValueError("actual and predicted indexes must match exactly")

    errors = forecast.to_numpy() - observed.to_numpy()
    absolute_errors = np.abs(errors)
    actual_values = observed.to_numpy()
    forecast_values = forecast.to_numpy()

    nonzero_actual = ~np.isclose(actual_values, 0.0)
    mape = (
        float(
            np.mean(
                absolute_errors[nonzero_actual] / np.abs(actual_values[nonzero_actual])
            )
        )
        if nonzero_actual.any()
        else float("nan")
    )
    symmetric_denominator = np.abs(actual_values) + np.abs(forecast_values)
    nonzero_symmetric = ~np.isclose(symmetric_denominator, 0.0)
    smape = (
        float(
            np.mean(
                2.0
                * absolute_errors[nonzero_symmetric]
                / symmetric_denominator[nonzero_symmetric]
            )
        )
        if nonzero_symmetric.any()
        else float("nan")
    )
    return pd.Series(
        {
            "mae": float(np.mean(absolute_errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "bias": float(np.mean(errors)),
            "mape": mape,
            "smape": smape,
        },
        name="forecast_metrics",
    )


def walk_forward_forecast(
    series: pd.Series,
    forecaster: Forecaster,
    initial_train_size: int,
    test_size: int = 1,
    *,
    step: Optional[int] = None,
    window: str = "expanding",
) -> pd.DataFrame:
    """Evaluate a forecasting callable on chronological out-of-sample folds."""
    if not callable(forecaster):
        raise TypeError("forecaster must be callable")
    training = validate_positive_integer(initial_train_size, "initial_train_size")
    testing = validate_positive_integer(test_size, "test_size")
    folds = rolling_origin_splits(
        series,
        training,
        testing,
        step=step,
        window=window,
    )

    rows: list[pd.DataFrame] = []
    for fold in folds:
        raw_forecast = forecaster(fold.train.copy(deep=True), len(fold.test))
        if isinstance(raw_forecast, pd.Series):
            forecast = coerce_time_series(raw_forecast, name="forecast")
            if not forecast.index.equals(fold.test.index):
                raise ValueError(
                    "forecaster Series index must match the fold test index"
                )
        else:
            array = np.asarray(raw_forecast, dtype=float)
            if array.ndim != 1:
                raise ValueError("forecaster output must be one-dimensional")
            if not np.isfinite(array).all():
                raise ValueError("forecaster output must contain only finite values")
            forecast = pd.Series(
                array,
                index=fold.test.index,
                name="forecast",
            )
        if len(forecast) != len(fold.test):
            raise ValueError("forecaster output length must equal the test horizon")

        block = pd.DataFrame(
            {
                "actual": fold.test,
                "forecast": forecast,
            }
        )
        block["error"] = block["forecast"] - block["actual"]
        block["fold"] = fold.fold
        block["train_start"] = fold.train_start
        block["train_end"] = fold.train_end
        rows.append(block)

    result = pd.concat(rows)
    result.index.name = series.index.name
    return result
