"""Validation helpers for dynamic allocation strategies."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

ReturnPaths = Union[pd.Series, pd.DataFrame]


def validate_return_paths(risky_returns: ReturnPaths) -> pd.DataFrame:
    """Return finite labelled simple-return paths."""
    if isinstance(risky_returns, pd.Series):
        name = risky_returns.name if risky_returns.name is not None else "strategy"
        frame = risky_returns.to_frame(name=name)
    elif isinstance(risky_returns, pd.DataFrame):
        frame = risky_returns.copy()
    else:
        raise TypeError("risky_returns must be a pandas Series or DataFrame")

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("risky_returns must contain at least one observation and path")
    if not frame.index.is_unique:
        raise ValueError("risky_returns index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("risky_returns index must be monotonic increasing")
    if not frame.columns.is_unique:
        raise ValueError("risky_returns columns must be unique")

    try:
        clean = frame.astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError("risky_returns must contain only numeric values") from error
    values = clean.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("risky_returns must contain only finite values")
    if np.any(values < -1.0):
        raise ValueError("risky_returns simple returns cannot be below -1.0")
    return clean


def validate_reserve_return_paths(
    reserve_returns: ReturnPaths,
    risky_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Return reserve paths aligned to already validated risky-return paths.

    A Series represents one common reserve asset and is broadcast across all
    risky paths. A DataFrame must have exactly the same index and columns as
    the risky-return DataFrame.
    """
    if isinstance(reserve_returns, pd.Series):
        if not reserve_returns.index.equals(risky_returns.index):
            raise ValueError("reserve_returns index must match risky_returns exactly")
        try:
            values = reserve_returns.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "reserve_returns must contain only numeric values"
            ) from error
        frame = pd.concat(
            [values.rename(column) for column in risky_returns.columns],
            axis=1,
        )
    elif isinstance(reserve_returns, pd.DataFrame):
        if not reserve_returns.index.equals(risky_returns.index):
            raise ValueError("reserve_returns index must match risky_returns exactly")
        if not reserve_returns.columns.equals(risky_returns.columns):
            raise ValueError("reserve_returns columns must match risky_returns exactly")
        try:
            frame = reserve_returns.astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "reserve_returns must contain only numeric values"
            ) from error
    else:
        raise TypeError("reserve_returns must be a pandas Series or DataFrame")

    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("reserve_returns must contain only finite values")
    if np.any(values < -1.0):
        raise ValueError("reserve_returns simple returns cannot be below -1.0")
    return frame


def validate_real(value: float, name: str) -> float:
    """Return a finite real number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_positive(value: float, name: str) -> float:
    """Return a strictly positive finite real number."""
    result = validate_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def validate_fraction(value: float, name: str) -> float:
    """Return a finite fraction in the closed unit interval."""
    result = validate_real(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def validate_non_negative(value: float, name: str) -> float:
    """Return a finite non-negative real number."""
    result = validate_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return result


def validate_positive_integer(value: int, name: str) -> int:
    """Return a positive integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def validate_optional_positive_integer(
    value: Optional[int],
    name: str,
) -> Optional[int]:
    """Return ``None`` or a positive integer."""
    if value is None:
        return None
    return validate_positive_integer(value, name)


def validate_allocation_bounds(
    minimum_risky_weight: float,
    maximum_risky_weight: float,
) -> tuple[float, float]:
    """Return valid risky-allocation bounds."""
    minimum = validate_non_negative(minimum_risky_weight, "minimum_risky_weight")
    maximum = validate_non_negative(maximum_risky_weight, "maximum_risky_weight")
    if minimum > maximum:
        raise ValueError("minimum_risky_weight must not exceed maximum_risky_weight")
    return minimum, maximum


def periodic_safe_return(risk_free_rate: float, periods_per_year: int) -> float:
    """Convert an annual effective safe rate to a periodic effective return."""
    annual_rate = validate_real(risk_free_rate, "risk_free_rate")
    if annual_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.0")
    periods = validate_positive_integer(periods_per_year, "periods_per_year")
    return float((1.0 + annual_rate) ** (1.0 / periods) - 1.0)


def validate_transaction_cost_rate(value: float) -> float:
    """Return a transaction-cost rate in ``[0, 1)``."""
    result = validate_non_negative(value, "transaction_cost_rate")
    if result >= 1.0:
        raise ValueError("transaction_cost_rate must be less than 1")
    return result
