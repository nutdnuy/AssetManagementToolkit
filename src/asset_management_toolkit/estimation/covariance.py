"""Labelled sample and shrinkage covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(
    returns: pd.DataFrame,
    *,
    ddof: int = 1,
) -> pd.DataFrame:
    """Estimate the sample covariance matrix from complete periodic returns.

    The function uses one shared observation set for every asset pair. Missing
    values are rejected rather than silently producing pairwise estimates with
    different effective samples.
    """
    frame = _validated_returns(returns, ddof)
    values = np.cov(frame.to_numpy(dtype=float), rowvar=False, ddof=ddof)
    matrix = np.atleast_2d(values)
    return pd.DataFrame(
        matrix,
        index=frame.columns.copy(),
        columns=frame.columns.copy(),
        dtype=float,
    )


def constant_correlation_covariance(
    returns: pd.DataFrame,
    *,
    ddof: int = 1,
) -> pd.DataFrame:
    """Estimate covariance using one average off-diagonal correlation.

    Sample variances remain on the diagonal. Every off-diagonal covariance is
    reconstructed from the average sample correlation and the corresponding
    sample standard deviations.
    """
    sample = sample_covariance(returns, ddof=ddof)
    variances = np.diag(sample.to_numpy(dtype=float))
    if np.any(variances <= 0.0):
        raise ValueError(
            "constant-correlation covariance requires strictly positive "
            "sample variance for every asset"
        )
    if len(sample) == 1:
        return sample

    standard_deviations = np.sqrt(variances)
    sample_correlation = sample.to_numpy(dtype=float) / np.outer(
        standard_deviations,
        standard_deviations,
    )
    off_diagonal = sample_correlation[np.triu_indices(len(sample), k=1)]
    average_correlation = float(off_diagonal.mean())
    target = average_correlation * np.outer(
        standard_deviations,
        standard_deviations,
    )
    np.fill_diagonal(target, variances)
    return pd.DataFrame(
        target,
        index=sample.index.copy(),
        columns=sample.columns.copy(),
        dtype=float,
    )


def shrink_covariance(
    returns: pd.DataFrame,
    *,
    target: str = "constant_correlation",
    intensity: float = 0.5,
    ddof: int = 1,
) -> pd.DataFrame:
    """Blend sample covariance with a constant-correlation target.

    ``intensity=0`` returns the sample estimate and ``intensity=1`` returns the
    target. The intensity is caller-supplied; this function does not estimate
    an optimal Ledoit–Wolf shrinkage coefficient.
    """
    if target != "constant_correlation":
        raise ValueError("target must be 'constant_correlation'")
    shrinkage = _unit_interval(intensity, "intensity")
    sample = sample_covariance(returns, ddof=ddof)
    prior = constant_correlation_covariance(returns, ddof=ddof)
    result = (1.0 - shrinkage) * sample + shrinkage * prior
    result.index = sample.index.copy()
    result.columns = sample.columns.copy()
    return result


def ewma_covariance(
    returns: pd.DataFrame,
    *,
    decay: float = 0.94,
    demean: bool = True,
    annualization_factor: float | None = None,
) -> pd.DataFrame:
    """Estimate a normalized exponentially weighted population covariance.

    The latest observation receives the greatest weight. No Bessel or
    effective-sample correction is applied, so ``decay=1`` matches population
    covariance with ``ddof=0``.
    """
    frame = _validated_returns(returns, ddof=0)
    if len(frame) < 2:
        raise ValueError("EWMA covariance requires at least two observations")
    decay_value = _positive_real(decay, "decay")
    if decay_value > 1.0:
        raise ValueError("decay must be at most one")
    if not isinstance(demean, bool):
        raise TypeError("demean must be a bool")

    powers = np.arange(len(frame) - 1, -1, -1, dtype=float)
    weights = np.power(decay_value, powers)
    weights /= weights.sum()
    values = frame.to_numpy(dtype=float)
    center = np.average(values, axis=0, weights=weights) if demean else 0.0
    centered = values - center
    covariance = (centered * weights[:, None]).T @ centered

    if annualization_factor is not None:
        covariance *= _positive_real(
            annualization_factor,
            "annualization_factor",
        )
    return pd.DataFrame(
        covariance,
        index=frame.columns.copy(),
        columns=frame.columns.copy(),
        dtype=float,
    )


def _validated_returns(returns: pd.DataFrame, ddof: int) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain observations and assets")
    if not returns.columns.is_unique:
        raise ValueError("returns columns must be unique")
    non_numeric = [
        str(column)
        for column in returns
        if not pd.api.types.is_numeric_dtype(returns[column])
    ]
    if non_numeric:
        raise TypeError("returns columns must be numeric")
    frame = returns.astype(float).copy(deep=True)
    if frame.isna().any().any():
        raise ValueError("returns must not contain missing values")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("returns must contain only finite values")
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise TypeError("ddof must be an integer")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if len(frame) <= ddof:
        raise ValueError("returns must contain more observations than ddof")
    return frame


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return numeric


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric
