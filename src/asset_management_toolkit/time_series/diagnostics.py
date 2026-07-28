"""Classical time-series decomposition diagnostics."""

from __future__ import annotations

import pandas as pd

from asset_management_toolkit.time_series._validation import (
    coerce_time_series,
    validate_positive_integer,
)
from asset_management_toolkit.time_series.result import DecompositionResult


def decompose_time_series(
    series: pd.Series,
    period: int,
    *,
    model: str = "additive",
    extrapolate_trend: int = 0,
) -> DecompositionResult:
    """Decompose a regular series into trend, seasonal, and residual components."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    seasonal_period = validate_positive_integer(period, "period")
    values = coerce_time_series(
        series,
        min_observations=2 * seasonal_period,
    )
    if model not in {"additive", "multiplicative"}:
        raise ValueError("model must be 'additive' or 'multiplicative'")
    if model == "multiplicative" and (values <= 0.0).any():
        raise ValueError("multiplicative decomposition requires positive data")
    if isinstance(extrapolate_trend, bool) or not isinstance(extrapolate_trend, int):
        raise TypeError("extrapolate_trend must be an integer")
    if extrapolate_trend < 0:
        raise ValueError("extrapolate_trend must be nonnegative")

    decomposition = seasonal_decompose(
        values,
        model=model,
        period=seasonal_period,
        extrapolate_trend=extrapolate_trend,
    )
    return DecompositionResult(
        observed=decomposition.observed.rename("observed"),
        trend=decomposition.trend.rename("trend"),
        seasonal=decomposition.seasonal.rename("seasonal"),
        residual=decomposition.resid.rename("residual"),
        model=model,
        period=seasonal_period,
    )
