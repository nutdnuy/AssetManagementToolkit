"""Reusable labelled portfolio-weighting policies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def equal_weights(asset_names: Iterable[Any]) -> pd.Series:
    """Return fully invested equal weights for unique asset labels."""
    if isinstance(asset_names, (str, bytes)):
        raise TypeError("asset_names must be an iterable of labels")
    labels = pd.Index(list(asset_names))
    if labels.empty:
        raise ValueError("asset_names must contain at least one label")
    if not labels.is_unique:
        raise ValueError("asset_names must be unique")
    return pd.Series(
        np.repeat(1.0 / len(labels), len(labels)),
        index=labels,
        name="weight",
        dtype=float,
    )


def capitalization_weights(market_capitalizations: pd.Series) -> pd.Series:
    """Normalize non-negative labelled market capitalizations into weights."""
    values = _validated_capitalizations(market_capitalizations)
    weights = values / values.sum()
    weights.name = "weight"
    return weights


def capped_equal_weights(
    market_capitalizations: pd.Series,
    *,
    minimum_capitalization_weight: float = 0.0,
    maximum_multiple_of_cap_weight: float | None = None,
) -> pd.Series:
    """Return equal-oriented weights subject to size screening and caps.

    Assets below ``minimum_capitalization_weight`` receive zero weight. When a
    maximum multiple is supplied, each remaining weight is capped at that
    multiple of its capitalization weight. A water-filling projection keeps
    the portfolio fully invested without renormalizing any weight past its cap.
    """
    cap_weights = capitalization_weights(market_capitalizations)
    threshold = _unit_interval(
        minimum_capitalization_weight,
        "minimum_capitalization_weight",
        upper_inclusive=False,
    )
    multiple = _optional_positive_real(
        maximum_multiple_of_cap_weight,
        "maximum_multiple_of_cap_weight",
    )
    eligible = cap_weights >= threshold
    if not eligible.any():
        raise ValueError("capitalization screen excludes every asset")
    if multiple is None:
        result = pd.Series(0.0, index=cap_weights.index, name="weight")
        result.loc[eligible] = 1.0 / int(eligible.sum())
        return result

    caps = multiple * cap_weights
    caps.loc[~eligible] = 0.0
    if caps.sum() < 1.0 - 1e-12:
        raise ValueError(
            "weight caps are infeasible because their eligible sum is below one"
        )
    values = _water_fill(caps.to_numpy(dtype=float))
    return pd.Series(values, index=cap_weights.index, name="weight", dtype=float)


def _validated_capitalizations(values: pd.Series) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError("market_capitalizations must be a pandas Series")
    if values.empty:
        raise ValueError("market_capitalizations must not be empty")
    if not values.index.is_unique:
        raise ValueError("market_capitalizations index must be unique")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError("market_capitalizations must be numeric")
    result = values.astype(float).copy(deep=True)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("market_capitalizations must contain only finite values")
    if (result < 0.0).any() or result.sum() <= 0.0:
        raise ValueError(
            "market_capitalizations must be non-negative with a positive total"
        )
    return result


def _water_fill(caps: np.ndarray) -> np.ndarray:
    lower = 0.0
    upper = float(caps.max())
    for _ in range(100):
        level = (lower + upper) / 2.0
        total = float(np.minimum(caps, level).sum())
        if total < 1.0:
            lower = level
        else:
            upper = level
    weights = np.minimum(caps, upper)
    weights = weights / weights.sum()
    if np.any(weights > caps + 1e-10):
        raise RuntimeError("capped equal-weight projection exceeded a weight cap")
    return weights


def _unit_interval(
    value: float,
    name: str,
    *,
    upper_inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    valid_upper = numeric <= 1.0 if upper_inclusive else numeric < 1.0
    if not np.isfinite(numeric) or numeric < 0.0 or not valid_upper:
        bound = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be finite and in {bound}")
    return numeric


def _optional_positive_real(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number or None")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric
