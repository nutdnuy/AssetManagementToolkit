"""Classical forecasting models with labelled, out-of-sample results."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from asset_management_toolkit.time_series._validation import (
    coerce_time_series,
    future_index,
    validate_positive_integer,
)
from asset_management_toolkit.time_series.result import ForecastResult


def seasonal_naive_forecast(
    series: pd.Series,
    horizon: int,
    seasonal_period: int = 1,
) -> pd.Series:
    """Repeat the most recent seasonal cycle as a transparent baseline."""
    values = coerce_time_series(series)
    steps = validate_positive_integer(horizon, "horizon")
    period = validate_positive_integer(seasonal_period, "seasonal_period")
    if period > len(values):
        raise ValueError("seasonal_period must not exceed the series length")

    season = values.iloc[-period:].to_numpy()
    forecast = np.resize(season, steps)
    return pd.Series(
        forecast,
        index=future_index(values.index, steps),
        name=values.name,
    )


def exponential_smoothing_forecast(
    series: pd.Series,
    horizon: int,
    *,
    trend: Optional[str] = None,
    damped_trend: bool = False,
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
) -> ForecastResult:
    """Fit ETS/Holt-Winters exponential smoothing and forecast future periods."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    values = coerce_time_series(series, min_observations=3)
    steps = validate_positive_integer(horizon, "horizon")
    if trend not in {None, "add", "mul"}:
        raise ValueError("trend must be None, 'add', or 'mul'")
    if seasonal not in {None, "add", "mul"}:
        raise ValueError("seasonal must be None, 'add', or 'mul'")
    if damped_trend and trend is None:
        raise ValueError("damped_trend requires a trend component")
    if (trend == "mul" or seasonal == "mul") and (values <= 0.0).any():
        raise ValueError("multiplicative components require strictly positive data")
    if seasonal is None:
        if seasonal_periods is not None:
            raise ValueError("seasonal_periods requires a seasonal component")
        period = None
    else:
        if seasonal_periods is None:
            raise ValueError("seasonal_periods is required for a seasonal component")
        period = validate_positive_integer(seasonal_periods, "seasonal_periods")
        if period < 2:
            raise ValueError("seasonal_periods must be at least two")
        if len(values) < 2 * period:
            raise ValueError("seasonal models require at least two complete cycles")

    fitted_model = ExponentialSmoothing(
        values,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=period,
        initialization_method="estimated",
    ).fit(optimized=True, remove_bias=False)
    fitted_values = pd.Series(
        np.asarray(fitted_model.fittedvalues, dtype=float),
        index=values.index,
        name="fitted",
    )
    forecast = pd.Series(
        np.asarray(fitted_model.forecast(steps), dtype=float),
        index=future_index(values.index, steps),
        name="forecast",
    )
    residuals = (values - fitted_values).rename("residual")
    return ForecastResult(
        model="exponential_smoothing",
        fitted_values=fitted_values,
        residuals=residuals,
        forecast=forecast,
        parameters={
            "trend": trend,
            "damped_trend": damped_trend,
            "seasonal": seasonal,
            "seasonal_periods": period,
        },
        aic=float(fitted_model.aic),
        bic=float(fitted_model.bic),
    )


def sarima_forecast(
    series: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 0, 0),
    *,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: Optional[str] = None,
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> ForecastResult:
    """Fit a specified SARIMA model without automated in-sample order search."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    values = coerce_time_series(series, min_observations=4)
    steps = validate_positive_integer(horizon, "horizon")
    arima_order = _validate_order(order)
    seasonal = _validate_seasonal_order(seasonal_order)
    if trend not in {None, "n", "c", "t", "ct"}:
        raise ValueError("trend must be None, 'n', 'c', 't', or 'ct'")

    fitted_model = SARIMAX(
        values,
        order=arima_order,
        seasonal_order=seasonal,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    ).fit(disp=False)
    fitted_values = pd.Series(
        np.asarray(fitted_model.fittedvalues, dtype=float),
        index=values.index,
        name="fitted",
    )
    forecast = pd.Series(
        np.asarray(fitted_model.forecast(steps), dtype=float),
        index=future_index(values.index, steps),
        name="forecast",
    )
    residuals = (values - fitted_values).rename("residual")
    return ForecastResult(
        model="sarima",
        fitted_values=fitted_values,
        residuals=residuals,
        forecast=forecast,
        parameters={
            "order": arima_order,
            "seasonal_order": seasonal,
            "trend": trend,
            "enforce_stationarity": enforce_stationarity,
            "enforce_invertibility": enforce_invertibility,
        },
        aic=float(fitted_model.aic),
        bic=float(fitted_model.bic),
    )


def _validate_order(order: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(order, tuple) or len(order) != 3:
        raise TypeError("order must be a three-integer tuple")
    values = tuple(_validate_nonnegative_integer(value, "order") for value in order)
    return values  # type: ignore[return-value]


def _validate_seasonal_order(
    order: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if not isinstance(order, tuple) or len(order) != 4:
        raise TypeError("seasonal_order must be a four-integer tuple")
    values = tuple(
        _validate_nonnegative_integer(value, "seasonal_order") for value in order
    )
    if any(values[:3]) and values[3] < 2:
        raise ValueError(
            "seasonal_order period must be at least two when P, D, or Q is nonzero"
        )
    return values  # type: ignore[return-value]


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} values must be integers")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} values must be nonnegative")
    return result
