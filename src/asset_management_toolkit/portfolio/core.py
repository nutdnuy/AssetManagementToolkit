"""Core portfolio return and volatility calculations."""

from __future__ import annotations

import numpy as np

from asset_management_toolkit.portfolio._validation import (
    MatrixInput,
    VectorInput,
    as_covariance,
    as_vector,
    validate_weight_vector,
)


def portfolio_return(
    weights: VectorInput,
    expected_returns: VectorInput,
) -> float:
    """Calculate expected portfolio return as ``weights @ expected_returns``."""
    returns_vector = as_vector(expected_returns, "expected_returns")
    weight_vector = validate_weight_vector(weights, len(returns_vector))
    return float(weight_vector @ returns_vector)


def portfolio_volatility(
    weights: VectorInput,
    covariance: MatrixInput,
) -> float:
    """Calculate portfolio volatility from weights and a covariance matrix."""
    weight_vector = as_vector(weights, "weights")
    covariance_matrix = as_covariance(covariance, len(weight_vector))
    variance = float(weight_vector @ covariance_matrix @ weight_vector)
    return float(np.sqrt(max(variance, 0.0)))
