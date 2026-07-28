from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.estimation import (
    constant_correlation_covariance,
    sample_covariance,
    shrink_covariance,
)


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "equity": [0.03, -0.02, 0.04, 0.01, -0.01],
            "bonds": [0.01, 0.00, -0.01, 0.02, 0.01],
            "real_assets": [0.02, -0.01, 0.01, 0.03, -0.02],
        },
        index=pd.date_range("2024-01-31", periods=5, freq="ME"),
    )


def test_sample_covariance_matches_pandas_and_preserves_labels() -> None:
    returns = _returns()

    result = sample_covariance(returns)

    pd.testing.assert_frame_equal(result, returns.cov())
    assert result.index.equals(returns.columns)
    assert result.columns.equals(returns.columns)


def test_constant_correlation_covariance_preserves_sample_variances() -> None:
    returns = _returns()
    sample = returns.cov()
    average_correlation = returns.corr().to_numpy()[np.triu_indices(3, k=1)].mean()

    result = constant_correlation_covariance(returns)

    np.testing.assert_allclose(np.diag(result), np.diag(sample))
    standard_deviations = np.sqrt(np.diag(sample))
    expected = average_correlation * np.outer(
        standard_deviations,
        standard_deviations,
    )
    np.fill_diagonal(expected, np.diag(sample))
    np.testing.assert_allclose(result, expected)


def test_shrink_covariance_has_exact_endpoints_and_convex_middle() -> None:
    returns = _returns()
    sample = sample_covariance(returns)
    target = constant_correlation_covariance(returns)

    pd.testing.assert_frame_equal(
        shrink_covariance(returns, intensity=0.0),
        sample,
    )
    pd.testing.assert_frame_equal(
        shrink_covariance(returns, intensity=1.0),
        target,
    )
    np.testing.assert_allclose(
        shrink_covariance(returns, intensity=0.25),
        0.75 * sample + 0.25 * target,
    )


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        (pd.DataFrame({"a": [0.01, np.nan], "b": [0.0, 0.01]}), "missing"),
        (pd.DataFrame({"a": [0.01], "b": [0.0]}), "observations"),
        (pd.DataFrame({"a": [0.01, 0.01], "b": [0.0, 0.01]}), "variance"),
    ],
)
def test_constant_correlation_rejects_unsafe_inputs(
    returns: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        constant_correlation_covariance(returns)


def test_covariance_estimators_do_not_mutate_input() -> None:
    returns = _returns()
    original = returns.copy(deep=True)

    shrink_covariance(returns, intensity=0.4)

    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("intensity", [-0.01, 1.01, np.nan])
def test_shrink_covariance_rejects_invalid_intensity(intensity: float) -> None:
    with pytest.raises(ValueError, match="intensity"):
        shrink_covariance(_returns(), intensity=intensity)
