import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    simulate_variance_gamma_prices,
    simulate_variance_gamma_returns,
)


def test_variance_gamma_returns_are_reproducible_and_labelled() -> None:
    first = simulate_variance_gamma_returns(
        n_years=0.5,
        n_scenarios=3,
        periods_per_year=12,
        seed=42,
    )
    second = simulate_variance_gamma_returns(
        n_years=0.5,
        n_scenarios=3,
        periods_per_year=12,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (6, 3)
    assert list(first.columns) == [
        "scenario_0000",
        "scenario_0001",
        "scenario_0002",
    ]
    assert (first > -1.0).all().all()


def test_variance_gamma_log_return_moments_match_parameterization() -> None:
    theta = -0.20
    volatility = 0.30
    variance_rate = 0.40
    mean_log_return = 0.06
    returns = simulate_variance_gamma_returns(
        n_years=1,
        n_scenarios=150_000,
        mean_log_return=mean_log_return,
        theta=theta,
        volatility=volatility,
        variance_rate=variance_rate,
        periods_per_year=1,
        seed=7,
    )
    log_returns = np.log1p(returns.iloc[0].to_numpy())

    expected_variance = volatility**2 + theta**2 * variance_rate
    assert np.mean(log_returns) == pytest.approx(mean_log_return, abs=0.004)
    assert np.var(log_returns) == pytest.approx(expected_variance, abs=0.004)


def test_variance_gamma_prices_match_simulated_returns() -> None:
    returns = simulate_variance_gamma_returns(
        n_years=1,
        n_scenarios=4,
        periods_per_year=4,
        seed=9,
    )
    prices = simulate_variance_gamma_prices(
        n_years=1,
        n_scenarios=4,
        periods_per_year=4,
        initial_price=250.0,
        seed=9,
    )

    assert prices.shape == (5, 4)
    assert np.allclose(prices.iloc[0], 250.0)
    reconstructed = prices.pct_change().iloc[1:]
    reconstructed.index = returns.index
    pd.testing.assert_frame_equal(reconstructed, returns)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"variance_rate": 0.0}, ValueError),
        ({"variance_rate": -0.1}, ValueError),
        ({"volatility": -0.1}, ValueError),
        ({"theta": np.inf}, ValueError),
        ({"seed": True}, TypeError),
    ],
)
def test_variance_gamma_rejects_invalid_parameters(
    kwargs: dict,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        simulate_variance_gamma_returns(**kwargs)
