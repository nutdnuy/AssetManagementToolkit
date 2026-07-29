from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.estimation import (
    correlation_to_covariance,
    covariance_to_correlation,
    factor_model_covariance,
)


def _covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.012], [0.012, 0.09]],
        index=["equity", "bonds"],
        columns=["equity", "bonds"],
    )


def test_covariance_and_correlation_round_trip_preserves_labels() -> None:
    covariance = _covariance()
    volatilities = pd.Series(
        np.sqrt(np.diag(covariance)),
        index=covariance.index,
    )

    correlation = covariance_to_correlation(covariance)
    reconstructed = correlation_to_covariance(correlation, volatilities)

    expected_correlation = pd.DataFrame(
        [[1.0, 0.2], [0.2, 1.0]],
        index=covariance.index,
        columns=covariance.columns,
    )
    pd.testing.assert_frame_equal(correlation, expected_correlation)
    pd.testing.assert_frame_equal(reconstructed, covariance)


def test_factor_model_covariance_matches_matrix_definition() -> None:
    loadings = pd.DataFrame(
        [[1.0, 0.2], [0.4, 1.1], [-0.2, 0.6]],
        index=["equity", "credit", "duration"],
        columns=["growth", "rates"],
    )
    factor_covariance = pd.DataFrame(
        [[0.0225, -0.003], [-0.003, 0.01]],
        index=loadings.columns,
        columns=loadings.columns,
    )
    specific = pd.Series(
        [0.10, 0.08, 0.05],
        index=loadings.index,
    )

    result = factor_model_covariance(loadings, factor_covariance, specific)

    expected = (
        loadings.to_numpy() @ factor_covariance.to_numpy() @ loadings.to_numpy().T
        + np.diag(np.square(specific))
    )
    np.testing.assert_allclose(result, expected)
    assert result.index.equals(loadings.index)
    assert result.columns.equals(loadings.index)


def test_dependence_functions_do_not_mutate_inputs() -> None:
    covariance = _covariance()
    original = covariance.copy(deep=True)

    covariance_to_correlation(covariance)

    pd.testing.assert_frame_equal(covariance, original)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (
            pd.DataFrame(
                [[0.04, 0.02], [0.01, 0.09]],
                index=["a", "b"],
                columns=["a", "b"],
            ),
            "symmetric",
        ),
        (
            pd.DataFrame(
                [[1.0, 2.0], [2.0, 1.0]],
                index=["a", "b"],
                columns=["a", "b"],
            ),
            "positive semidefinite",
        ),
        (
            pd.DataFrame(
                [[0.0, 0.0], [0.0, 0.09]],
                index=["a", "b"],
                columns=["a", "b"],
            ),
            "strictly positive",
        ),
    ],
)
def test_covariance_to_correlation_rejects_invalid_matrices(
    covariance: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        covariance_to_correlation(covariance)


def test_correlation_to_covariance_rejects_misaligned_volatilities() -> None:
    correlation = covariance_to_correlation(_covariance())
    volatilities = pd.Series([0.2, 0.3], index=["bonds", "equity"])

    with pytest.raises(ValueError, match="labels"):
        correlation_to_covariance(correlation, volatilities)


def test_factor_model_covariance_rejects_misaligned_factor_labels() -> None:
    loadings = pd.DataFrame(
        [[1.0], [0.5]],
        index=["a", "b"],
        columns=["factor"],
    )
    factor_covariance = pd.DataFrame(
        [[0.04]],
        index=["wrong"],
        columns=["wrong"],
    )
    specific = pd.Series([0.1, 0.1], index=loadings.index)

    with pytest.raises(ValueError, match="factor_loadings columns"):
        factor_model_covariance(loadings, factor_covariance, specific)
