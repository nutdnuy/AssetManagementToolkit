from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.estimation import (
    covariance_to_correlation,
    sample_covariance,
    spectral_denoised_covariance,
)


def _returns() -> pd.DataFrame:
    generator = np.random.default_rng(7)
    common = generator.normal(size=100)
    return pd.DataFrame(
        {
            "a": common + generator.normal(scale=0.2, size=100),
            "b": 0.8 * common + generator.normal(scale=0.3, size=100),
            "c": generator.normal(scale=0.7, size=100),
        }
    )


def test_explicit_rank_zero_produces_diagonal_covariance() -> None:
    returns = _returns()
    sample = sample_covariance(returns)

    result = spectral_denoised_covariance(returns, signal_rank=0)

    np.testing.assert_allclose(np.diag(result), np.diag(sample))
    np.testing.assert_allclose(
        covariance_to_correlation(result),
        np.eye(3),
        atol=1e-12,
    )
    assert result.attrs["signal_rank"] == 0
    assert result.attrs["selection"] == "explicit"


def test_marchenko_pastur_selection_is_psd_and_preserves_variance() -> None:
    returns = _returns()

    result = spectral_denoised_covariance(returns)

    np.testing.assert_allclose(
        np.diag(result),
        np.diag(sample_covariance(returns)),
    )
    assert np.linalg.eigvalsh(result).min() >= -1e-12
    assert result.attrs["selection"] == "marchenko_pastur"
    assert result.attrs["marchenko_pastur_upper_edge"] is not None


def test_spectral_denoising_does_not_mutate_returns() -> None:
    returns = _returns()
    original = returns.copy(deep=True)

    spectral_denoised_covariance(returns, signal_rank=1)

    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("rank", [-1, 4])
def test_spectral_denoising_rejects_invalid_rank(rank: int) -> None:
    with pytest.raises(ValueError, match="signal_rank"):
        spectral_denoised_covariance(_returns(), signal_rank=rank)
