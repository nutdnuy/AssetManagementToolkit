"""Validation helpers for portfolio calculations."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

VectorInput = Union[np.ndarray, pd.Series, list[float], tuple[float, ...]]
MatrixInput = Union[np.ndarray, pd.DataFrame, list[list[float]]]


def as_vector(values: VectorInput, name: str) -> np.ndarray:
    """Convert a one-dimensional numeric input to a finite float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def as_covariance(covariance: MatrixInput, n_assets: int) -> np.ndarray:
    """Convert and validate a symmetric covariance matrix."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"covariance must have shape ({n_assets}, {n_assets}), "
            f"received {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def asset_labels(expected_returns: VectorInput) -> pd.Index:
    """Return stable asset labels for optimization output."""
    if isinstance(expected_returns, pd.Series):
        if not expected_returns.index.is_unique:
            raise ValueError("expected_returns index must be unique")
        return expected_returns.index.copy()
    return pd.Index([f"asset_{index}" for index in range(len(expected_returns))])


def validate_weight_vector(weights: VectorInput, n_assets: int) -> np.ndarray:
    """Validate a weight vector against the asset dimension."""
    vector = as_vector(weights, "weights")
    if len(vector) != n_assets:
        raise ValueError(
            f"weights must contain {n_assets} values, received {len(vector)}"
        )
    return vector
