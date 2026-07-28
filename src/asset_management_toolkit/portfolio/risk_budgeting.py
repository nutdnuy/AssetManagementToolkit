"""Long-only target-risk and equal-risk-contribution portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from asset_management_toolkit.portfolio._validation import (
    MatrixInput,
    VectorInput,
    as_covariance,
    as_vector,
)


def target_risk_contribution_weights(
    target_contributions: VectorInput,
    covariance: MatrixInput,
) -> pd.Series:
    """Find long-only weights whose normalized risk contributions match a target.

    Target contributions must be strictly positive and sum to one. The
    covariance matrix and target vector must use the same labels and order when
    both are labelled pandas objects.
    """
    target = as_vector(target_contributions, "target_contributions")
    matrix = as_covariance(covariance, len(target))
    labels = _asset_labels(target_contributions, covariance, len(target))
    _validate_alignment(target_contributions, covariance)
    if np.any(target <= 0.0) or not np.isclose(target.sum(), 1.0, atol=1e-10):
        raise ValueError(
            "target_contributions must be strictly positive and sum to one"
        )
    variances = np.diag(matrix)
    if np.any(variances <= 0.0):
        raise ValueError(
            "covariance must have strictly positive variance for every asset"
        )

    initial = np.sqrt(target) / np.sqrt(variances)
    initial = initial / initial.sum()

    def objective(weights: np.ndarray) -> float:
        contributions = _normalized_contributions(weights, matrix)
        difference = contributions - target
        return float(difference @ difference)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=((1e-12, 1.0),) * len(target),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-14, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"target_risk_contribution_weights failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights = weights / weights.sum()
    achieved = _normalized_contributions(weights, matrix)
    if not np.allclose(achieved, target, rtol=1e-5, atol=1e-6):
        maximum_error = float(np.max(np.abs(achieved - target)))
        raise RuntimeError(
            "target_risk_contribution_weights did not achieve the requested "
            f"risk budget; maximum contribution error was {maximum_error:.3g}"
        )
    return pd.Series(weights, index=labels, name="weight", dtype=float)


def equal_risk_contribution_weights(covariance: MatrixInput) -> pd.Series:
    """Find a long-only portfolio with equal normalized risk contributions."""
    raw = np.asarray(covariance, dtype=float)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
        raise ValueError("covariance must be a non-empty square matrix")
    labels = _covariance_labels(covariance, raw.shape[0])
    target = pd.Series(
        np.repeat(1.0 / raw.shape[0], raw.shape[0]),
        index=labels,
        name="target_contribution",
        dtype=float,
    )
    if isinstance(covariance, pd.DataFrame):
        return target_risk_contribution_weights(target, covariance)
    return target_risk_contribution_weights(target.to_numpy(), covariance)


def _normalized_contributions(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    component_variance = weights * (covariance @ weights)
    portfolio_variance = float(component_variance.sum())
    if portfolio_variance <= 1e-20:
        return np.full_like(weights, np.inf)
    return component_variance / portfolio_variance


def _asset_labels(
    target: VectorInput,
    covariance: MatrixInput,
    n_assets: int,
) -> pd.Index:
    if isinstance(target, pd.Series):
        if not target.index.is_unique:
            raise ValueError("target_contributions index must be unique")
        return target.index.copy()
    return _covariance_labels(covariance, n_assets)


def _covariance_labels(covariance: MatrixInput, n_assets: int) -> pd.Index:
    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.is_unique or not covariance.columns.is_unique:
            raise ValueError("covariance labels must be unique")
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance DataFrame index and columns must match")
        return covariance.index.copy()
    return pd.Index([f"asset_{index}" for index in range(n_assets)])


def _validate_alignment(target: VectorInput, covariance: MatrixInput) -> None:
    if isinstance(target, pd.Series) and isinstance(covariance, pd.DataFrame):
        if not target.index.equals(covariance.index) or not target.index.equals(
            covariance.columns
        ):
            raise ValueError(
                "target_contributions and covariance labels and order must match"
            )
