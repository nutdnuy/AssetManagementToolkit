"""Validation helpers for labelled univariate time series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def validate_positive_integer(value: int, name: str) -> int:
    """Return a validated strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def coerce_time_series(
    values: pd.Series,
    *,
    name: str = "series",
    min_observations: int = 1,
) -> pd.Series:
    """Return a finite, ordered, numeric defensive copy."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if values.empty:
        raise ValueError(f"{name} must contain at least one observation")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(f"{name} must contain numeric values")
    if not values.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not values.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order")

    result = values.astype(float).copy(deep=True)
    if result.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    if len(result) < min_observations:
        raise ValueError(
            f"{name} must contain at least {min_observations} observations"
        )
    if result.name is None:
        result.name = name
    return result


def validate_windows(windows: Sequence[int], name: str) -> tuple[int, ...]:
    """Validate a non-empty sequence of unique positive window lengths."""
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(validate_positive_integer(value, name) for value in windows)
    if not result:
        raise ValueError(f"{name} must contain at least one window")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate windows")
    return result


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    """Construct labelled future periods without guessing an irregular frequency."""
    steps = validate_positive_integer(horizon, "horizon")
    if isinstance(index, pd.PeriodIndex):
        if index.freq is None:
            raise ValueError("series PeriodIndex must have a frequency")
        return pd.period_range(index[-1] + 1, periods=steps, freq=index.freq)
    if isinstance(index, pd.DatetimeIndex):
        frequency = index.freq or pd.infer_freq(index)
        if frequency is None:
            raise ValueError(
                "series DatetimeIndex must have a fixed or inferable frequency"
            )
        return pd.date_range(index[-1], periods=steps + 1, freq=frequency)[1:]
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(
            start=index[-1] + index.step,
            stop=index[-1] + index.step * (steps + 1),
            step=index.step,
        )
    raise TypeError("forecasting requires a DatetimeIndex, PeriodIndex, or RangeIndex")
