"""Validation helpers for probability-free stress scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd


def validate_return_frame(
    returns: pd.DataFrame,
    name: str,
    *,
    require_datetime_index: bool = False,
) -> pd.DataFrame:
    """Return a finite labelled simple-return frame."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one asset")
    if not returns.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not returns.columns.is_unique:
        raise ValueError(f"{name} columns must be unique")
    if require_datetime_index and not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a pandas DatetimeIndex")
    if require_datetime_index and not returns.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")

    try:
        clean = returns.astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain only numeric values") from error
    values = clean.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < -1.0):
        raise ValueError(f"{name} simple returns cannot be below -1.0")
    return clean


def validate_weights(weights: pd.Series, assets: pd.Index) -> pd.Series:
    """Return finite, fully invested weights aligned to scenario assets."""
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")
    if weights.empty or not weights.index.is_unique:
        raise ValueError("weights must have a non-empty unique index")
    try:
        clean = weights.astype(float)
    except (TypeError, ValueError) as error:
        raise TypeError("weights must contain only numeric values") from error
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("weights must contain only finite values")

    missing = assets.difference(clean.index)
    extra = clean.index.difference(assets)
    if not missing.empty or not extra.empty:
        raise ValueError(
            "weights and scenario assets must match exactly; "
            f"missing={missing.tolist()}, extra={extra.tolist()}"
        )
    if not np.isclose(float(clean.sum()), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("weights must sum to 1.0")
    return clean.reindex(assets)


def validate_thresholds(
    loss_thresholds: Optional[Mapping[str, float]],
) -> dict[str, float]:
    """Return ordered non-negative portfolio-loss thresholds."""
    if loss_thresholds is None:
        return {}
    if not isinstance(loss_thresholds, Mapping):
        raise TypeError("loss_thresholds must be a mapping or None")

    result: dict[str, float] = {}
    for label, value in loss_thresholds.items():
        if not isinstance(label, str) or not label.strip():
            raise TypeError("loss_threshold labels must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError("loss_threshold values must be real numbers")
        threshold = float(value)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("loss_threshold values must be finite and non-negative")
        result[label] = threshold
    return result
