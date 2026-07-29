from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.portfolio import (
    capitalization_weights,
    capped_equal_weights,
    equal_weights,
    inverse_volatility_weights,
)


def test_equal_weights_preserves_asset_labels() -> None:
    result = equal_weights(["equity", "bonds", "real_assets"])

    expected = pd.Series(
        [1.0 / 3.0] * 3,
        index=["equity", "bonds", "real_assets"],
        name="weight",
    )
    pd.testing.assert_series_equal(result, expected)


def test_capitalization_weights_normalizes_nonnegative_values() -> None:
    capitalizations = pd.Series(
        [60.0, 30.0, 10.0],
        index=["large", "medium", "small"],
    )

    result = capitalization_weights(capitalizations)

    expected = pd.Series(
        [0.6, 0.3, 0.1],
        index=capitalizations.index,
        name="weight",
    )
    pd.testing.assert_series_equal(result, expected)


def test_capped_equal_weights_applies_screen_and_cap() -> None:
    capitalizations = pd.Series(
        [90.0, 9.0, 1.0],
        index=["mega", "mid", "micro"],
    )

    result = capped_equal_weights(
        capitalizations,
        minimum_capitalization_weight=0.05,
        maximum_multiple_of_cap_weight=3.0,
    )

    expected = pd.Series(
        [0.73, 0.27, 0.0],
        index=capitalizations.index,
        name="weight",
    )
    pd.testing.assert_series_equal(result, expected)
    assert result.sum() == pytest.approx(1.0)


def test_capped_equal_weights_rejects_infeasible_caps() -> None:
    capitalizations = pd.Series([90.0, 10.0], index=["large", "small"])

    with pytest.raises(ValueError, match="infeasible"):
        capped_equal_weights(
            capitalizations,
            maximum_multiple_of_cap_weight=0.5,
        )


@pytest.mark.parametrize(
    "capitalizations",
    [
        pd.Series([], dtype=float),
        pd.Series([1.0, -1.0], index=["a", "b"]),
        pd.Series([0.0, 0.0], index=["a", "b"]),
    ],
)
def test_capitalization_weights_rejects_invalid_values(
    capitalizations: pd.Series,
) -> None:
    with pytest.raises(ValueError, match="market_capitalizations"):
        capitalization_weights(capitalizations)


def test_weighting_functions_reject_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        equal_weights(["a", "a"])


def test_inverse_volatility_weights_preserves_covariance_labels() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["equity", "bonds"],
        columns=["equity", "bonds"],
    )

    result = inverse_volatility_weights(covariance)

    expected = pd.Series(
        [0.6, 0.4],
        index=covariance.index,
        name="weight",
    )
    pd.testing.assert_series_equal(result, expected)


def test_inverse_volatility_weights_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        inverse_volatility_weights(np.diag([0.04, 0.0]))


def test_inverse_volatility_weights_rejects_misaligned_labels() -> None:
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09]),
        index=["a", "b"],
        columns=["b", "a"],
    )

    with pytest.raises(ValueError, match="labels"):
        inverse_volatility_weights(covariance)
