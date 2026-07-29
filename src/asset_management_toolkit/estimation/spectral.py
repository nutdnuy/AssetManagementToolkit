"""Spectral covariance denoising with explicit rank assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.estimation.covariance import sample_covariance
from asset_management_toolkit.estimation.dependence import (
    correlation_to_covariance,
    covariance_to_correlation,
)
from asset_management_toolkit.estimation.psd import apply_psd_policy


def spectral_denoised_covariance(
    returns: pd.DataFrame,
    *,
    signal_rank: int | None = None,
    effective_observations: float | None = None,
    ddof: int = 1,
    psd_policy: str = "clip",
) -> pd.DataFrame:
    """Replace noise correlation eigenvalues by their common mean.

    When ``signal_rank`` is omitted, rank is selected using the
    Marchenko–Pastur upper edge and the supplied effective observation count,
    or ``len(returns) - ddof`` by default.
    """
    covariance = sample_covariance(returns, ddof=ddof)
    correlation = covariance_to_correlation(covariance)
    values = correlation.to_numpy(dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    dimension = len(correlation)

    upper_edge: float | None = None
    if signal_rank is None:
        observations = (
            float(len(returns) - ddof)
            if effective_observations is None
            else _positive_real(effective_observations, "effective_observations")
        )
        if observations < dimension:
            raise ValueError(
                "Marchenko-Pastur selection requires effective observations "
                "at least equal to the number of assets"
            )
        upper_edge = (1.0 + np.sqrt(dimension / observations)) ** 2
        rank = int(np.count_nonzero(eigenvalues > upper_edge))
        selection = "marchenko_pastur"
    else:
        if isinstance(signal_rank, bool) or not isinstance(signal_rank, int):
            raise TypeError("signal_rank must be an integer or None")
        if not 0 <= signal_rank <= dimension:
            raise ValueError("signal_rank must be between zero and the asset count")
        rank = signal_rank
        selection = "explicit"

    denoised = eigenvalues.copy()
    if rank < dimension:
        denoised[rank:] = float(eigenvalues[rank:].mean())
    rebuilt = (eigenvectors * denoised) @ eigenvectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0
    repaired = apply_psd_policy(
        pd.DataFrame(
            rebuilt,
            index=correlation.index,
            columns=correlation.columns,
        ),
        policy=psd_policy,
    )
    scale = np.sqrt(np.diag(repaired))
    normalized = repaired.to_numpy(dtype=float) / np.outer(scale, scale)
    np.fill_diagonal(normalized, 1.0)
    clean_correlation = pd.DataFrame(
        normalized,
        index=correlation.index,
        columns=correlation.columns,
    )
    volatilities = pd.Series(
        np.sqrt(np.diag(covariance)),
        index=covariance.index,
    )
    result = correlation_to_covariance(clean_correlation, volatilities)
    result.attrs.update(
        {
            "signal_rank": rank,
            "selection": selection,
            "marchenko_pastur_upper_edge": upper_edge,
        }
    )
    return result


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return numeric
