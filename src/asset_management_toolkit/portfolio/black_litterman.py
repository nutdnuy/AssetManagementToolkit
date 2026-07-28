"""Labelled Black–Litterman posterior estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from asset_management_toolkit.portfolio._validation import (
    MatrixInput,
    VectorInput,
    as_covariance,
    as_vector,
)

ViewMatrixInput = Union[np.ndarray, pd.DataFrame, list[list[float]]]


@dataclass(frozen=True)
class BlackLittermanResult:
    """Auditable prior, views, and posterior estimates."""

    prior_returns: pd.Series
    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame
    view_uncertainty: pd.DataFrame


def implied_equilibrium_returns(
    market_weights: VectorInput,
    covariance: MatrixInput,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Reverse-optimize market weights into equilibrium excess returns.

    The result follows ``pi = risk_aversion * covariance @ market_weights``.
    Market weights must be non-negative and sum to one.
    """
    weights, covariance_matrix, labels = _validated_market_inputs(
        market_weights,
        covariance,
    )
    delta = _positive_scalar(risk_aversion, "risk_aversion")
    return pd.Series(
        delta * covariance_matrix @ weights,
        index=labels,
        name="implied_equilibrium_return",
        dtype=float,
    )


def proportional_view_uncertainty(
    covariance: MatrixInput,
    pick_matrix: ViewMatrixInput,
    tau: float = 0.05,
) -> pd.DataFrame:
    """Return diagonal He–Litterman view uncertainty.

    Each diagonal element is the corresponding view variance from
    ``tau * P @ covariance @ P.T``. Off-diagonal elements are set to zero.
    """
    covariance_matrix, asset_labels = _validated_covariance(covariance)
    picks, view_labels = _validated_pick_matrix(
        pick_matrix,
        asset_labels,
    )
    scaling = _tau(tau)
    full_view_covariance = scaling * picks @ covariance_matrix @ picks.T
    diagonal = np.diag(full_view_covariance)
    if np.any(diagonal <= np.finfo(float).eps):
        raise ValueError(
            "proportional view uncertainty requires every view to have "
            "positive variance"
        )
    return pd.DataFrame(
        np.diag(diagonal),
        index=view_labels,
        columns=view_labels,
        dtype=float,
    )


def black_litterman_posterior(
    market_weights: VectorInput,
    covariance: MatrixInput,
    pick_matrix: ViewMatrixInput,
    views: VectorInput,
    *,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_uncertainty: Optional[MatrixInput] = None,
) -> BlackLittermanResult:
    """Combine equilibrium excess returns with absolute or relative views.

    ``pick_matrix`` contains one row per view and one column per asset.
    ``views`` and all covariance inputs must use the same return units and
    horizon. When ``view_uncertainty`` is omitted, the diagonal
    He–Litterman proportional uncertainty is used.
    """
    weights, covariance_matrix, asset_labels = _validated_market_inputs(
        market_weights,
        covariance,
    )
    delta = _positive_scalar(risk_aversion, "risk_aversion")
    scaling = _tau(tau)
    picks, initial_view_labels = _validated_pick_matrix(
        pick_matrix,
        asset_labels,
    )
    view_vector, view_labels = _validated_views(
        views,
        picks.shape[0],
        initial_view_labels,
        pick_matrix_is_labelled=isinstance(pick_matrix, pd.DataFrame),
    )
    prior_values = delta * covariance_matrix @ weights

    if view_uncertainty is None:
        full_view_covariance = scaling * picks @ covariance_matrix @ picks.T
        diagonal = np.diag(full_view_covariance)
        if np.any(diagonal <= np.finfo(float).eps):
            raise ValueError(
                "default view uncertainty requires every view to have positive variance"
            )
        omega = np.diag(diagonal)
    else:
        omega = _validated_view_uncertainty(
            view_uncertainty,
            view_labels,
        )

    scaled_covariance = scaling * covariance_matrix
    scaled_covariance_p = scaled_covariance @ picks.T
    system = picks @ scaled_covariance_p + omega
    innovation = view_vector - picks @ prior_values
    posterior_adjustment = scaled_covariance_p @ _solve_view_system(
        system,
        innovation,
    )
    mean_uncertainty = scaled_covariance - scaled_covariance_p @ _solve_view_system(
        system,
        picks @ scaled_covariance,
    )
    posterior_covariance = covariance_matrix + mean_uncertainty
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2.0
    if np.linalg.eigvalsh(posterior_covariance).min() < -1e-10:
        raise RuntimeError(
            "Black–Litterman posterior covariance is not positive semidefinite"
        )

    prior = pd.Series(
        prior_values,
        index=asset_labels,
        name="implied_equilibrium_return",
        dtype=float,
    )
    posterior_returns = pd.Series(
        prior_values + posterior_adjustment,
        index=asset_labels,
        name="black_litterman_return",
        dtype=float,
    )
    posterior_covariance_frame = pd.DataFrame(
        posterior_covariance,
        index=asset_labels,
        columns=asset_labels,
        dtype=float,
    )
    omega_frame = pd.DataFrame(
        omega,
        index=view_labels,
        columns=view_labels,
        dtype=float,
    )
    return BlackLittermanResult(
        prior_returns=prior,
        posterior_returns=posterior_returns,
        posterior_covariance=posterior_covariance_frame,
        view_uncertainty=omega_frame,
    )


def _validated_market_inputs(
    market_weights: VectorInput,
    covariance: MatrixInput,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    weights = as_vector(market_weights, "market_weights")
    covariance_matrix = as_covariance(covariance, len(weights))
    labels = _asset_labels(market_weights, covariance, len(weights))
    if np.any(weights < 0.0):
        raise ValueError("market_weights must be non-negative")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("market_weights must sum to one")
    return weights, covariance_matrix, labels


def _validated_covariance(
    covariance: MatrixInput,
) -> tuple[np.ndarray, pd.Index]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    covariance_matrix = as_covariance(covariance, matrix.shape[0])
    labels = _asset_labels(
        np.repeat(1.0 / matrix.shape[0], matrix.shape[0]),
        covariance,
        matrix.shape[0],
    )
    return covariance_matrix, labels


def _asset_labels(
    market_weights: VectorInput,
    covariance: MatrixInput,
    n_assets: int,
) -> pd.Index:
    if isinstance(market_weights, pd.Series):
        if not market_weights.index.is_unique:
            raise ValueError("market_weights index must be unique")
        labels = market_weights.index.copy()
    elif isinstance(covariance, pd.DataFrame):
        labels = covariance.index.copy()
    else:
        labels = pd.Index([f"asset_{index}" for index in range(n_assets)])

    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.is_unique or not covariance.columns.is_unique:
            raise ValueError("covariance labels must be unique")
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance DataFrame index and columns must match")
        if not covariance.index.equals(labels):
            raise ValueError(
                "covariance labels must match market_weights labels and order"
            )
    return labels


def _validated_pick_matrix(
    pick_matrix: ViewMatrixInput,
    asset_labels: pd.Index,
) -> tuple[np.ndarray, pd.Index]:
    picks = np.asarray(pick_matrix, dtype=float)
    expected_columns = len(asset_labels)
    if picks.ndim != 2 or picks.shape[0] == 0 or picks.shape[1] != expected_columns:
        raise ValueError(
            "pick_matrix must have one or more rows and "
            f"{expected_columns} asset columns"
        )
    if not np.isfinite(picks).all():
        raise ValueError("pick_matrix must contain only finite values")
    if np.any(np.all(np.isclose(picks, 0.0), axis=1)):
        raise ValueError("pick_matrix rows must define non-zero views")

    if isinstance(pick_matrix, pd.DataFrame):
        if not pick_matrix.index.is_unique or not pick_matrix.columns.is_unique:
            raise ValueError("pick_matrix labels must be unique")
        if not pick_matrix.columns.equals(asset_labels):
            raise ValueError("pick_matrix columns must match asset labels and order")
        view_labels = pick_matrix.index.copy()
    else:
        view_labels = pd.Index(
            [f"view_{index}" for index in range(picks.shape[0])],
            name="view",
        )
    return picks, view_labels


def _validated_views(
    views: VectorInput,
    n_views: int,
    initial_labels: pd.Index,
    *,
    pick_matrix_is_labelled: bool,
) -> tuple[np.ndarray, pd.Index]:
    vector = as_vector(views, "views")
    if len(vector) != n_views:
        raise ValueError(f"views must contain {n_views} values, received {len(vector)}")
    if isinstance(views, pd.Series):
        if not views.index.is_unique:
            raise ValueError("views index must be unique")
        if pick_matrix_is_labelled and not views.index.equals(initial_labels):
            raise ValueError("views index must match pick_matrix index and order")
        return vector, views.index.copy()
    return vector, initial_labels


def _validated_view_uncertainty(
    view_uncertainty: MatrixInput,
    view_labels: pd.Index,
) -> np.ndarray:
    matrix = as_covariance(view_uncertainty, len(view_labels))
    if isinstance(view_uncertainty, pd.DataFrame):
        if not view_uncertainty.index.equals(view_labels) or not (
            view_uncertainty.columns.equals(view_labels)
        ):
            raise ValueError("view_uncertainty labels must match view labels and order")
    return matrix


def _positive_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _tau(value: float) -> float:
    result = _positive_scalar(value, "tau")
    if result > 1.0:
        raise ValueError("tau must be less than or equal to one")
    return result


def _solve_view_system(system: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "view system is singular; revise pick_matrix or view_uncertainty"
        ) from error
