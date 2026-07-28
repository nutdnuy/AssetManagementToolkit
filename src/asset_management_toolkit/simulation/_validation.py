"""Validation helpers for simulation modules."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import validate_periods_per_year


def validate_real(value: float, name: str) -> float:
    """Return a finite real value while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_positive_real(value: float, name: str) -> float:
    """Return a finite real value greater than zero."""
    result = validate_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def validate_non_negative_real(value: float, name: str) -> float:
    """Return a finite real value greater than or equal to zero."""
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


def validate_seed(seed: Optional[int]) -> Optional[int]:
    """Return a valid non-negative NumPy random seed."""
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer or None")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be greater than or equal to zero")
    return result


def simulation_steps(n_years: float, periods_per_year: int) -> int:
    """Convert a horizon to an exact positive number of simulation steps."""
    years = validate_positive_real(n_years, "n_years")
    validate_periods_per_year(periods_per_year)
    raw_steps = years * periods_per_year
    rounded_steps = round(raw_steps)
    if not np.isclose(raw_steps, rounded_steps, rtol=0.0, atol=1e-12):
        raise ValueError("n_years * periods_per_year must be a whole number of periods")
    return int(rounded_steps)


def scenario_labels(n_scenarios: int) -> pd.Index:
    """Return deterministic, zero-padded scenario labels."""
    width = max(4, len(str(n_scenarios - 1)))
    return pd.Index(
        [f"scenario_{index:0{width}d}" for index in range(n_scenarios)],
        name="scenario",
    )
