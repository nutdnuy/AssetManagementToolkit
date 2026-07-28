import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit import risk_return_stats


def test_summary_contains_return_and_risk_metrics() -> None:
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, -0.02, 0.03, 0.01],
            "asset_b": [0.00, 0.01, -0.01, 0.02],
        }
    )

    result = risk_return_stats(returns, periods_per_year=12)

    assert list(result.index) == ["asset_a", "asset_b"]
    assert result.index.name == "asset"
    assert result.loc["asset_a", "n_observations"] == 4
    assert {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "semivariance",
        "max_drawdown",
        "historical_var",
        "historical_cvar",
    }.issubset(result.columns)
    assert result.loc["asset_a", "semivariance"] == pytest.approx(
        result.loc["asset_a", "downside_deviation"] ** 2
    )
    assert "beta" not in result.columns


def test_summary_adds_benchmark_metrics_when_requested() -> None:
    returns = pd.Series([0.02, -0.01, 0.03, 0.00], name="portfolio")
    benchmark = pd.Series([0.01, -0.01, 0.02, 0.00], name="benchmark")

    result = risk_return_stats(
        returns,
        benchmark=benchmark,
        periods_per_year=12,
    )

    assert {
        "active_return",
        "tracking_error",
        "information_ratio",
        "beta",
        "alpha",
    }.issubset(result.columns)


def test_summary_drops_missing_values_per_asset() -> None:
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, np.nan, 0.02],
            "asset_b": [0.00, 0.01, 0.02],
        }
    )

    result = risk_return_stats(returns, periods_per_year=12)

    assert result.loc["asset_a", "n_observations"] == 2
    assert result.loc["asset_b", "n_observations"] == 3


def test_summary_rejects_invalid_simple_returns() -> None:
    with pytest.raises(ValueError, match="below -1.0"):
        risk_return_stats(pd.Series([0.01, -1.01]))


def test_summary_rejects_non_numeric_returns() -> None:
    with pytest.raises(TypeError, match="numeric"):
        risk_return_stats(pd.Series(["good", "bad"]))
