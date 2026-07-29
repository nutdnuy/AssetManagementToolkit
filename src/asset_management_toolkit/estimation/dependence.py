"""Labelled dependence-matrix transformations and factor covariance models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    """Convert a labelled covariance matrix to a correlation matrix.

    Every asset must have strictly positive variance. The input must be finite,
    symmetric, positive semidefinite, and use identical row and column labels.
    """
    matrix = _validated_square_matrix(
        covariance,
        name="covariance",
        require_unit_diagonal=False,
    )
    variances = np.diag(matrix.to_numpy(dtype=float))
    if np.any(variances <= 0.0):
        raise ValueError("covariance must have strictly positive variances")

    standard_deviations = np.sqrt(variances)
    values = matrix.to_numpy(dtype=float) / np.outer(
        standard_deviations,
        standard_deviations,
    )
    values = np.clip(values, -1.0, 1.0)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(
        values,
        index=matrix.index.copy(),
        columns=matrix.columns.copy(),
        dtype=float,
    )


def correlation_to_covariance(
    correlation: pd.DataFrame,
    volatilities: pd.Series,
) -> pd.DataFrame:
    """Convert labelled correlations and volatilities to covariance."""
    matrix = _validated_square_matrix(
        correlation,
        name="correlation",
        require_unit_diagonal=True,
    )
    sigma = _validated_labelled_vector(
        volatilities,
        labels=matrix.index,
        name="volatilities",
        non_negative=True,
    )
    values = matrix.to_numpy(dtype=float) * np.outer(sigma, sigma)
    return pd.DataFrame(
        values,
        index=matrix.index.copy(),
        columns=matrix.columns.copy(),
        dtype=float,
    )


def factor_model_covariance(
    factor_loadings: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_volatilities: pd.Series,
) -> pd.DataFrame:
    """Construct asset covariance from factor and specific risk.

    The calculation is ``B F B.T + diag(sigma_specific ** 2)``. Factor
    covariance and specific volatility must use the same time and
    annualization units intended for the resulting asset covariance.
    """
    loadings = _validated_loadings(factor_loadings)
    factors = _validated_square_matrix(
        factor_covariance,
        name="factor_covariance",
        require_unit_diagonal=False,
    )
    if not loadings.columns.equals(factors.index):
        raise ValueError(
            "factor_loadings columns must exactly match factor_covariance labels"
        )
    specific = _validated_labelled_vector(
        specific_volatilities,
        labels=loadings.index,
        name="specific_volatilities",
        non_negative=True,
    )

    loading_values = loadings.to_numpy(dtype=float)
    values = loading_values @ factors.to_numpy(
        dtype=float
    ) @ loading_values.T + np.diag(np.square(specific))
    return pd.DataFrame(
        values,
        index=loadings.index.copy(),
        columns=loadings.index.copy(),
        dtype=float,
    )


def _validated_square_matrix(
    matrix: pd.DataFrame,
    *,
    name: str,
    require_unit_diagonal: bool,
) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if matrix.empty or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError(f"{name} labels must be unique")
    if not matrix.index.equals(matrix.columns):
        raise ValueError(f"{name} row and column labels must match exactly")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
        raise TypeError(f"{name} must contain only numeric values")

    result = matrix.astype(float).copy(deep=True)
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if require_unit_diagonal:
        if not np.allclose(np.diag(values), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("correlation diagonal must contain ones")
        if np.any(np.abs(values) > 1.0 + 1e-12):
            raise ValueError("correlation values must be between -1 and 1")
    elif np.any(np.diag(values) < 0.0):
        raise ValueError(f"{name} diagonal must be non-negative")

    tolerance = max(1.0, float(np.linalg.norm(values, ord=2))) * 1e-10
    if float(np.linalg.eigvalsh(values).min()) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


def _validated_labelled_vector(
    values: pd.Series,
    *,
    labels: pd.Index,
    name: str,
    non_negative: bool,
) -> np.ndarray:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not values.index.is_unique:
        raise ValueError(f"{name} labels must be unique")
    if not values.index.equals(labels):
        raise ValueError(f"{name} labels must exactly match the matrix labels")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(f"{name} must contain numeric values")
    result = values.to_numpy(dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if non_negative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


def _validated_loadings(loadings: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(loadings, pd.DataFrame):
        raise TypeError("factor_loadings must be a pandas DataFrame")
    if loadings.empty or loadings.shape[1] == 0:
        raise ValueError("factor_loadings must contain assets and factors")
    if not loadings.index.is_unique or not loadings.columns.is_unique:
        raise ValueError("factor_loadings labels must be unique")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in loadings.dtypes):
        raise TypeError("factor_loadings must contain only numeric values")
    result = loadings.astype(float).copy(deep=True)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("factor_loadings must contain only finite values")
    return result
