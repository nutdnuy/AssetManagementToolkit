"""Observed-return validation for simulation diagnostics and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import (
    ReturnInput,
    coerce_returns,
)


def flatten_simple_returns(
    returns: ReturnInput,
    *,
    min_observations: int,
) -> np.ndarray:
    """Return finite non-missing simple returns as one observation vector."""
    frame, _ = coerce_returns(returns)
    values = frame.to_numpy().ravel()
    observations = values[np.isfinite(values)]
    if observations.size < min_observations:
        raise ValueError(
            f"returns must contain at least {min_observations} observations"
        )
    return observations.astype(float, copy=False)


def calibration_series(
    returns: pd.Series,
    *,
    min_observations: int,
) -> pd.Series:
    """Return one validated asset series for calibration."""
    if not isinstance(returns, pd.Series):
        raise TypeError("calibration returns must be a pandas Series")
    frame, _ = coerce_returns(returns)
    series = frame.iloc[:, 0]
    if series.isna().any():
        raise ValueError("calibration returns must not contain missing values")
    if len(series) < min_observations:
        raise ValueError(
            f"returns must contain at least {min_observations} observations"
        )
    if (series <= -1.0).any():
        raise ValueError("calibration returns must be greater than -1.0")
    return series.astype(float)


def log_return_observations(
    returns: pd.Series,
    *,
    min_observations: int,
) -> np.ndarray:
    """Return validated log-return observations for one asset."""
    series = calibration_series(
        returns,
        min_observations=min_observations,
    )
    return np.log1p(series.to_numpy())
