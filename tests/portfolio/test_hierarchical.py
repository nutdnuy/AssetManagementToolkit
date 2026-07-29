from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import squareform

from asset_management_toolkit.portfolio import (
    condensed_correlation_distance,
    herc_weights,
    hrp_weights,
)


def _covariance() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["a", "b"],
        columns=["a", "b"],
    )


def test_condensed_correlation_distance_matches_squareform_order() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.5, -0.5], [0.5, 1.0, 0.0], [-0.5, 0.0, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    result = condensed_correlation_distance(correlation)
    expected_square = np.sqrt((1.0 - correlation) / 2.0)
    np.fill_diagonal(expected_square.values, 0.0)

    np.testing.assert_allclose(result, squareform(expected_square))


def test_two_asset_hrp_and_herc_weights_match_cluster_risk_rules() -> None:
    hrp = hrp_weights(_covariance())
    herc = herc_weights(_covariance())

    pd.testing.assert_series_equal(
        hrp,
        pd.Series([9.0 / 13.0, 4.0 / 13.0], index=["a", "b"], name="weight"),
    )
    pd.testing.assert_series_equal(
        herc,
        pd.Series([0.6, 0.4], index=["a", "b"], name="weight"),
    )


@pytest.mark.parametrize("allocator", [hrp_weights, herc_weights])
def test_hierarchical_allocators_are_fully_invested_and_labelled(allocator) -> None:
    covariance = pd.DataFrame(
        [
            [0.04, 0.006, 0.004],
            [0.006, 0.09, 0.003],
            [0.004, 0.003, 0.16],
        ],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    result = allocator(covariance, linkage_method="average")

    assert result.index.equals(covariance.index)
    assert result.sum() == pytest.approx(1.0)
    assert (result > 0.0).all()


@pytest.mark.parametrize("allocator", [hrp_weights, herc_weights])
def test_hierarchical_allocators_support_one_asset(allocator) -> None:
    covariance = pd.DataFrame([[0.04]], index=["a"], columns=["a"])

    expected = pd.Series([1.0], index=["a"], name="weight")
    pd.testing.assert_series_equal(allocator(covariance), expected)


def test_hierarchical_allocator_rejects_unsupported_linkage() -> None:
    with pytest.raises(ValueError, match="linkage_method"):
        hrp_weights(_covariance(), linkage_method="ward")
