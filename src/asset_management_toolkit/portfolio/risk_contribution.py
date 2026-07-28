"""Portfolio risk-contribution analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.portfolio._validation import (
    MatrixInput,
    VectorInput,
    as_covariance,
    as_vector,
)


def risk_contributions(
    weights: VectorInput,
    covariance: MatrixInput,
    *,
    normalize: bool = True,
) -> pd.Series:
    """Calculate asset contributions to portfolio volatility.

    With ``normalize=True``, contributions sum to one. With
    ``normalize=False``, contributions are expressed in volatility units and
    sum to the portfolio volatility.
    """
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")

    weight_vector = as_vector(weights, "weights")
    covariance_matrix = as_covariance(covariance, len(weight_vector))
    labels = _risk_labels(weights, covariance)
    _validate_risk_alignment(weights, covariance)

    portfolio_variance = float(weight_vector @ covariance_matrix @ weight_vector)
    if portfolio_variance <= 1e-20:
        raise ValueError("portfolio variance must be greater than zero")
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    contribution = (
        weight_vector * (covariance_matrix @ weight_vector) / portfolio_volatility
    )
    if normalize:
        contribution = contribution / portfolio_volatility
        name = "risk_contribution"
    else:
        name = "volatility_contribution"
    return pd.Series(contribution, index=labels, name=name, dtype=float)


def _risk_labels(
    weights: VectorInput,
    covariance: MatrixInput,
) -> pd.Index:
    if isinstance(weights, pd.Series):
        if not weights.index.is_unique:
            raise ValueError("weights index must be unique")
        return weights.index.copy()
    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.is_unique:
            raise ValueError("covariance labels must be unique")
        return covariance.index.copy()
    return pd.Index([f"asset_{index}" for index in range(len(weights))])


def _validate_risk_alignment(
    weights: VectorInput,
    covariance: MatrixInput,
) -> None:
    if isinstance(covariance, pd.DataFrame):
        if not covariance.index.equals(covariance.columns):
            raise ValueError("covariance DataFrame index and columns must match")
        if isinstance(weights, pd.Series) and not weights.index.equals(
            covariance.index
        ):
            raise ValueError("weights and covariance labels and order must match")
