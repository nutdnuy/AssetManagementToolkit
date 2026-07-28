from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.market_regime_classification import (
    regime_episodes,
    regime_return_stats,
    regime_transition_matrix,
)


def _regimes() -> pd.Series:
    return pd.Series(
        ["Bull", "Bull", "Bear", "Bear", "Bull"],
        index=pd.date_range("2024-01-31", periods=5, freq="ME"),
        name="market_regime",
    )


def test_regime_episodes_preserves_contiguous_runs() -> None:
    regimes = _regimes()

    result = regime_episodes(regimes)

    assert result["regime"].tolist() == ["Bull", "Bear", "Bull"]
    assert result["n_observations"].tolist() == [2, 2, 1]
    assert result.loc[0, "start"] == regimes.index[0]
    assert result.loc[1, "end"] == regimes.index[3]


def test_regime_transition_matrix_returns_counts_and_probabilities() -> None:
    regimes = _regimes()

    counts = regime_transition_matrix(regimes, normalize=False)
    probabilities = regime_transition_matrix(regimes)

    expected_counts = pd.DataFrame(
        [[1, 1], [1, 1]],
        index=pd.Index(["Bull", "Bear"], name="from_regime"),
        columns=pd.Index(["Bull", "Bear"], name="to_regime"),
    )
    pd.testing.assert_frame_equal(counts, expected_counts)
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(2),
    )


def test_regime_return_stats_is_labelled_and_uses_asset_level_samples() -> None:
    regimes = _regimes()
    returns = pd.DataFrame(
        {
            "Equity": [0.03, 0.02, -0.04, -0.02, 0.01],
            "Bond": [0.01, np.nan, 0.02, 0.01, 0.00],
        },
        index=regimes.index,
    )

    result = regime_return_stats(returns, regimes, periods_per_year=12)

    assert result.index.names == ["regime", "asset"]
    assert result.loc[("Bull", "Equity"), "n_observations"] == 3
    assert result.loc[("Bull", "Bond"), "n_observations"] == 2
    assert {
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
    }.issubset(result.columns)


def test_regime_functions_reject_missing_or_unsorted_labels() -> None:
    regimes = _regimes()
    regimes.iloc[1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        regime_episodes(regimes)

    unsorted = _regimes().sort_index(ascending=False)
    with pytest.raises(ValueError, match="sorted"):
        regime_transition_matrix(unsorted)
