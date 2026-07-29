"""Hierarchical allocations derived from correlation distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from asset_management_toolkit.estimation import covariance_to_correlation
from asset_management_toolkit.portfolio._validation import as_covariance

_SUPPORTED_LINKAGE_METHODS = {"single", "complete", "average"}


def condensed_correlation_distance(correlation: pd.DataFrame) -> np.ndarray:
    """Return SciPy's condensed distance vector from labelled correlation."""
    if not isinstance(correlation, pd.DataFrame):
        raise TypeError("correlation must be a pandas DataFrame")
    if correlation.empty or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be a non-empty square matrix")
    if not correlation.index.equals(correlation.columns):
        raise ValueError("correlation row and column labels must match exactly")
    values = correlation.to_numpy(dtype=float, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("correlation must contain only finite values")
    if not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(values), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("correlation diagonal must contain ones")
    if np.any(np.abs(values) > 1.0 + 1e-12):
        raise ValueError("correlation values must be between -1 and 1")
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    return np.asarray(squareform(distance, checks=False), dtype=float)


def hrp_weights(
    covariance: pd.DataFrame,
    *,
    linkage_method: str = "single",
) -> pd.Series:
    """Return hierarchical risk parity weights."""
    return _hierarchical_weights(
        covariance,
        linkage_method=linkage_method,
        variance_exponent=1.0,
    )


def herc_weights(
    covariance: pd.DataFrame,
    *,
    linkage_method: str = "single",
) -> pd.Series:
    """Return hierarchical equal-risk-contribution cluster weights."""
    return _hierarchical_weights(
        covariance,
        linkage_method=linkage_method,
        variance_exponent=0.5,
    )


def _hierarchical_weights(
    covariance: pd.DataFrame,
    *,
    linkage_method: str,
    variance_exponent: float,
) -> pd.Series:
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("covariance row and column labels must match exactly")
    matrix = as_covariance(covariance, len(covariance))
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError("covariance must have strictly positive variances")
    if linkage_method not in _SUPPORTED_LINKAGE_METHODS:
        raise ValueError("linkage_method must be 'single', 'complete', or 'average'")
    if len(covariance) == 1:
        return pd.Series([1.0], index=covariance.index, name="weight")

    correlation = covariance_to_correlation(covariance)
    tree = linkage(
        condensed_correlation_distance(correlation),
        method=linkage_method,
    )
    leaf_count = len(covariance)
    members: dict[int, list[int]] = {index: [index] for index in range(leaf_count)}
    children: list[tuple[int, int]] = []
    for row_index, row in enumerate(tree):
        left, right = int(row[0]), int(row[1])
        children.append((left, right))
        members[leaf_count + row_index] = members[left] + members[right]

    weights = np.ones(leaf_count, dtype=float)
    pending = [leaf_count + len(children) - 1]
    while pending:
        node = pending.pop()
        if node < leaf_count:
            continue
        left, right = children[node - leaf_count]
        left_members, right_members = members[left], members[right]
        left_risk = _cluster_variance(matrix, left_members) ** variance_exponent
        right_risk = _cluster_variance(matrix, right_members) ** variance_exponent
        denominator = left_risk + right_risk
        weights[left_members] *= right_risk / denominator
        weights[right_members] *= left_risk / denominator
        pending.extend((left, right))
    weights /= weights.sum()
    return pd.Series(weights, index=covariance.index.copy(), name="weight")


def _cluster_variance(covariance: np.ndarray, members: list[int]) -> float:
    submatrix = covariance[np.ix_(members, members)]
    inverse_variance = 1.0 / np.diag(submatrix)
    weights = inverse_variance / inverse_variance.sum()
    variance = float(weights @ submatrix @ weights)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("cluster variance must be finite and strictly positive")
    return variance
