"""Explicit positive-semidefinite matrix handling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_psd_policy(
    matrix: pd.DataFrame,
    *,
    policy: str = "raise",
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Validate or repair a labelled symmetric matrix.

    ``policy="raise"`` rejects eigenvalues below ``-tolerance``.
    ``policy="clip"`` projects all negative eigenvalues to zero.
    """
    validated = _validated_symmetric_matrix(matrix)
    if policy not in {"raise", "clip"}:
        raise ValueError("policy must be 'raise' or 'clip'")
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float, np.number)
    ):
        raise TypeError("tolerance must be a real number")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    values = validated.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    if policy == "raise":
        if float(eigenvalues.min()) < -tolerance_value:
            raise ValueError("matrix is not positive semidefinite")
        return validated

    repaired = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0
    minimum = float(np.linalg.eigvalsh(repaired).min())
    if minimum < 0.0:
        repaired += np.eye(len(repaired)) * np.nextafter(-minimum, np.inf)
    return pd.DataFrame(
        repaired,
        index=validated.index.copy(),
        columns=validated.columns.copy(),
        dtype=float,
    )


def _validated_symmetric_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame")
    if matrix.empty or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a non-empty square matrix")
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("matrix labels must be unique")
    if not matrix.index.equals(matrix.columns):
        raise ValueError("matrix row and column labels must match exactly")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise TypeError("matrix must contain only numeric values")
    result = matrix.astype(float).copy(deep=True)
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("matrix must be symmetric")
    return result
