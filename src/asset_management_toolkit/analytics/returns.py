"""Return calculations for periodic simple-return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import (
    MetricResult,
    ReturnInput,
    align_pair,
    coerce_benchmark,
    coerce_returns,
    restore_metric,
    validate_periods_per_year,
)


def total_return(returns: ReturnInput) -> MetricResult:
    """Calculate compounded total return for each input column."""
    frame, was_series = coerce_returns(returns)
    result = frame.apply(_total_return_series)
    return restore_metric(result, was_series)


def annualized_return(
    returns: ReturnInput,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate geometric annualized return for each input column."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    result = frame.apply(
        lambda series: _annualized_return_series(series, periods_per_year)
    )
    return restore_metric(result, was_series)


def annualized_mean_return(
    returns: ReturnInput,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate arithmetic mean return annualized by multiplication."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    result = frame.apply(lambda series: series.dropna().mean() * periods_per_year)
    return restore_metric(result.astype(float), was_series)


def active_return(
    returns: ReturnInput,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized arithmetic return relative to a benchmark."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    benchmark_series = coerce_benchmark(benchmark)

    values = {}
    for column in frame:
        aligned = align_pair(frame[column], benchmark_series)
        values[column] = (
            aligned["portfolio"] - aligned["benchmark"]
        ).mean() * periods_per_year

    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def aggregate_returns(
    returns: ReturnInput,
    frequency: str,
) -> ReturnInput:
    """Compound simple returns into calendar periods.

    ``frequency`` follows pandas offset aliases, for example ``"ME"`` for
    month-end or ``"YE"`` for year-end. Periods containing no observations
    for an asset remain missing.
    """
    if not isinstance(frequency, str):
        raise TypeError("frequency must be a string pandas offset alias")
    if not frequency.strip():
        raise ValueError("frequency must not be empty")

    frame, was_series = coerce_returns(returns)
    if not isinstance(frame.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise TypeError("returns index must be a DatetimeIndex or PeriodIndex")
    if not frame.index.is_unique:
        raise ValueError("returns index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order")

    try:
        result = (1.0 + frame).resample(frequency).prod(min_count=1) - 1.0
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid aggregation frequency: {frequency}") from error

    if was_series:
        series = result.iloc[:, 0]
        series.name = returns.name
        return series
    return result


def rolling_returns(
    returns: ReturnInput,
    window: int,
) -> ReturnInput:
    """Compound simple returns over trailing fixed-observation windows.

    A result is reported only when all ``window`` observations for an asset are
    present. The calculation follows the input order, which must be uniquely
    labelled and increasing.
    """
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an integer")
    if window <= 0:
        raise ValueError("window must be greater than zero")
    frame, was_series = coerce_returns(returns)
    if not frame.index.is_unique:
        raise ValueError("returns index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order")

    result = (1.0 + frame).rolling(window=window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1.0
    if was_series:
        series = result.iloc[:, 0]
        series.name = returns.name
        return series
    return result


def _total_return_series(series: pd.Series) -> float:
    clean = series.dropna()
    return float(np.prod(1.0 + clean.to_numpy()) - 1.0)


def _annualized_return_series(
    series: pd.Series,
    periods_per_year: int,
) -> float:
    clean = series.dropna()
    compounded_growth = 1.0 + _total_return_series(clean)
    if np.isclose(compounded_growth, 0.0):
        return -1.0
    return float(compounded_growth ** (periods_per_year / len(clean)) - 1.0)
