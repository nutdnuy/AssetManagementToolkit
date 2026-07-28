"""Shared validation helpers for analytics modules."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

ReturnInput = Union[pd.Series, pd.DataFrame]
MetricResult = Union[float, pd.Series]


def validate_periods_per_year(periods_per_year: int) -> None:
    """Validate an annualization frequency."""
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise TypeError("periods_per_year must be an integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be greater than zero")


def validate_annual_rate(value: float, name: str) -> None:
    """Validate an annual rate that will be converted to a periodic rate."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= -1.0:
        raise ValueError(f"{name} must be greater than -1.0")


def validate_probability(value: float, name: str) -> None:
    """Validate a probability strictly between zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be between zero and one")


def annual_to_periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    """Convert an effective annual rate to an effective periodic rate."""
    validate_annual_rate(annual_rate, "annual_rate")
    validate_periods_per_year(periods_per_year)
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def coerce_returns(
    returns: ReturnInput,
    *,
    default_name: str = "portfolio",
) -> tuple[pd.DataFrame, bool]:
    """Return a validated defensive DataFrame copy and whether input was a Series."""
    if isinstance(returns, pd.Series):
        column_name = returns.name if returns.name is not None else default_name
        frame = returns.to_frame(name=column_name)
        was_series = True
    elif isinstance(returns, pd.DataFrame):
        frame = returns.copy(deep=True)
        was_series = False
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("returns must contain at least one column and one row")
    if not frame.columns.is_unique:
        raise ValueError("returns columns must be unique")

    non_numeric = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        joined = ", ".join(non_numeric)
        raise TypeError(f"returns columns must be numeric: {joined}")

    frame = frame.astype(float)
    finite_or_missing = np.isfinite(frame.to_numpy()) | frame.isna().to_numpy()
    if not finite_or_missing.all():
        raise ValueError("returns must not contain infinite values")
    if (frame < -1.0).any().any():
        raise ValueError("simple returns cannot be below -1.0")

    empty_columns = [str(column) for column in frame if frame[column].dropna().empty]
    if empty_columns:
        joined = ", ".join(empty_columns)
        raise ValueError(f"returns columns contain no observations: {joined}")

    return frame, was_series


def coerce_benchmark(benchmark: pd.Series) -> pd.Series:
    """Return a validated defensive benchmark copy."""
    if not isinstance(benchmark, pd.Series):
        raise TypeError("benchmark must be a pandas Series")
    frame, _ = coerce_returns(benchmark, default_name="benchmark")
    result = frame.iloc[:, 0]
    result.name = benchmark.name if benchmark.name is not None else "benchmark"
    return result


def restore_metric(result: pd.Series, was_series: bool) -> MetricResult:
    """Restore a scalar for Series input or retain a Series for DataFrame input."""
    if was_series:
        return float(result.iloc[0])
    return result


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide finite values and return NaN for zero or unavailable denominators."""
    if not np.isfinite(numerator):
        return float("nan")
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return float("nan")
    return float(numerator / denominator)


def align_pair(
    returns: pd.Series,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Pairwise-align portfolio and benchmark returns and drop missing rows."""
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("returns and benchmark have no overlapping observations")
    return aligned
