import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.analytics.risk import (
    alpha,
    annualized_volatility,
    beta,
    downside_deviation,
    historical_cvar,
    historical_var,
    information_ratio,
    max_drawdown,
    semivariance,
    sharpe_ratio,
    tracking_error,
)


def test_max_drawdown_uses_initial_wealth_peak() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    assert max_drawdown(returns) == pytest.approx(-0.20)


def test_historical_var_and_cvar_are_positive_loss_magnitudes() -> None:
    returns = pd.Series([-0.10, -0.05, 0.00, 0.05, 0.10])

    assert historical_var(returns, level=0.20) == pytest.approx(0.06)
    assert historical_cvar(returns, level=0.20) == pytest.approx(0.10)


def test_volatility_uses_sample_standard_deviation() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])

    expected = returns.std(ddof=1) * np.sqrt(12)
    assert annualized_volatility(returns, periods_per_year=12) == pytest.approx(
        expected
    )


def test_semivariance_uses_all_observations_and_annualizes() -> None:
    returns = pd.Series([-0.10, 0.05, -0.02, 0.04])

    expected_periodic = (0.10**2 + 0.02**2) / 4
    result = semivariance(returns, periods_per_year=12)

    assert result == pytest.approx(expected_periodic * 12)
    assert downside_deviation(returns, periods_per_year=12) ** 2 == pytest.approx(
        result
    )


def test_semivariance_converts_annual_hurdle_and_drops_missing_values() -> None:
    annual_hurdle = 0.12
    periodic_hurdle = (1.0 + annual_hurdle) ** (1.0 / 12) - 1.0
    returns = pd.DataFrame(
        {
            "asset_a": [0.00, np.nan, 0.02],
            "asset_b": [0.01, 0.01, 0.01],
        }
    )

    result = semivariance(
        returns,
        minimum_acceptable_return=annual_hurdle,
        periods_per_year=12,
    )

    expected_a = (periodic_hurdle**2 + 0.0) / 2 * 12
    expected_b = max(periodic_hurdle - 0.01, 0.0) ** 2 * 12
    assert result["asset_a"] == pytest.approx(expected_a)
    assert result["asset_b"] == pytest.approx(expected_b)


def test_zero_volatility_produces_nan_sharpe() -> None:
    result = sharpe_ratio(
        pd.Series([0.01, 0.01, 0.01]),
        periods_per_year=12,
    )

    assert np.isnan(result)


def test_identical_benchmark_statistics() -> None:
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.00])
    portfolio = benchmark.copy()

    assert beta(portfolio, benchmark) == pytest.approx(1.0)
    assert tracking_error(portfolio, benchmark, 12) == pytest.approx(0.0)
    assert np.isnan(information_ratio(portfolio, benchmark, 12))
    assert alpha(portfolio, benchmark, periods_per_year=12) == pytest.approx(0.0)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1])
def test_var_rejects_invalid_level(level: float) -> None:
    with pytest.raises(ValueError):
        historical_var(pd.Series([0.01, -0.01]), level)
