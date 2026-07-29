"""Long-only, fully invested Markowitz portfolio construction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from asset_management_toolkit.portfolio._validation import (
    MatrixInput,
    VectorInput,
    as_covariance,
    as_vector,
    asset_labels,
)
from asset_management_toolkit.portfolio.core import (
    portfolio_return,
    portfolio_volatility,
)


def minimum_volatility(
    target_return: float,
    expected_returns: VectorInput,
    covariance: MatrixInput,
) -> pd.Series:
    """Find long-only weights with minimum volatility at a target return."""
    returns_vector, covariance_matrix, labels = _validated_inputs(
        expected_returns,
        covariance,
    )
    if not np.isfinite(target_return):
        raise ValueError("target_return must be finite")
    if target_return < returns_vector.min() - 1e-12 or target_return > (
        returns_vector.max() + 1e-12
    ):
        raise ValueError("target_return is outside the long-only feasible range")

    constraints = (
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "eq",
            "fun": lambda weights: (
                portfolio_return(weights, returns_vector) - target_return
            ),
        },
    )
    result = minimize(
        portfolio_volatility,
        _initial_weights(len(returns_vector)),
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=_long_only_bounds(len(returns_vector)),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return _weights_from_result(result, labels, "minimum_volatility")


def maximum_sharpe_ratio(
    risk_free_rate: float,
    expected_returns: VectorInput,
    covariance: MatrixInput,
) -> pd.Series:
    """Find long-only weights that maximize the ex-ante Sharpe ratio."""
    returns_vector, covariance_matrix, labels = _validated_inputs(
        expected_returns,
        covariance,
    )
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    def negative_sharpe(weights: np.ndarray) -> float:
        volatility = portfolio_volatility(weights, covariance_matrix)
        if np.isclose(volatility, 0.0):
            return float("inf")
        expected = portfolio_return(weights, returns_vector)
        return -float((expected - risk_free_rate) / volatility)

    result = minimize(
        negative_sharpe,
        _initial_weights(len(returns_vector)),
        method="SLSQP",
        bounds=_long_only_bounds(len(returns_vector)),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return _weights_from_result(result, labels, "maximum_sharpe_ratio")


def global_minimum_variance(
    covariance: MatrixInput,
    *,
    asset_names: Optional[list[str]] = None,
) -> pd.Series:
    """Find long-only weights for the global minimum-variance portfolio."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    covariance_matrix = as_covariance(matrix, matrix.shape[0])
    if asset_names is None and isinstance(covariance, pd.DataFrame):
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance DataFrame index and columns must match")
        labels = covariance.index.copy()
    elif asset_names is None:
        labels = pd.Index(
            [f"asset_{index}" for index in range(covariance_matrix.shape[0])]
        )
    else:
        if len(asset_names) != covariance_matrix.shape[0]:
            raise ValueError("asset_names length must match covariance dimensions")
        labels = pd.Index(asset_names)

    result = minimize(
        portfolio_volatility,
        _initial_weights(len(labels)),
        args=(covariance_matrix,),
        method="SLSQP",
        bounds=_long_only_bounds(len(labels)),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    return _weights_from_result(result, labels, "global_minimum_variance")


def efficient_frontier_weights(
    n_points: int,
    expected_returns: VectorInput,
    covariance: MatrixInput,
) -> pd.DataFrame:
    """Return long-only minimum-volatility weights across target returns."""
    if isinstance(n_points, bool) or not isinstance(n_points, int):
        raise TypeError("n_points must be an integer")
    if n_points < 2:
        raise ValueError("n_points must be at least two")
    returns_vector = as_vector(expected_returns, "expected_returns")
    targets = np.linspace(returns_vector.min(), returns_vector.max(), n_points)
    rows = [
        minimum_volatility(target, expected_returns, covariance) for target in targets
    ]
    result = pd.DataFrame(rows, index=targets)
    result.index.name = "target_return"
    return result


def efficient_frontier(
    n_points: int,
    expected_returns: VectorInput,
    covariance: MatrixInput,
) -> pd.DataFrame:
    """Return expected return and volatility along the long-only frontier."""
    weights = efficient_frontier_weights(
        n_points,
        expected_returns,
        covariance,
    )
    returns_vector, covariance_matrix, _ = _validated_inputs(
        expected_returns,
        covariance,
    )
    expected = weights.apply(
        lambda row: portfolio_return(row.to_numpy(), returns_vector),
        axis=1,
    )
    volatility = weights.apply(
        lambda row: portfolio_volatility(row.to_numpy(), covariance_matrix),
        axis=1,
    )
    return pd.DataFrame(
        {
            "expected_return": expected,
            "volatility": volatility,
        },
        index=weights.index,
    )


def _validated_inputs(
    expected_returns: VectorInput,
    covariance: MatrixInput,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    returns_vector = as_vector(expected_returns, "expected_returns")
    covariance_matrix = as_covariance(covariance, len(returns_vector))
    labels = asset_labels(expected_returns)
    if isinstance(covariance, pd.DataFrame) and isinstance(expected_returns, pd.Series):
        if not covariance.index.equals(expected_returns.index) or not (
            covariance.columns.equals(expected_returns.index)
        ):
            raise ValueError(
                "covariance labels must match expected_returns labels and order"
            )
    return returns_vector, covariance_matrix, labels


def _initial_weights(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def _long_only_bounds(n_assets: int) -> tuple[tuple[float, float], ...]:
    return ((0.0, 1.0),) * n_assets


def _weights_from_result(
    result: OptimizeResult,
    labels: pd.Index,
    operation: str,
) -> pd.Series:
    if not result.success:
        raise RuntimeError(f"{operation} failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights[np.isclose(weights, 0.0, atol=1e-12)] = 0.0
    weights[np.isclose(weights, 1.0, atol=1e-12)] = 1.0
    return pd.Series(weights, index=labels, name="weight")
