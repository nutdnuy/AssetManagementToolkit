import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    simulate_gbm_prices,
    simulate_gbm_returns,
)


def test_simulate_gbm_returns_is_reproducible_and_labelled() -> None:
    first = simulate_gbm_returns(
        n_years=0.5,
        n_scenarios=3,
        periods_per_year=12,
        seed=42,
    )
    second = simulate_gbm_returns(
        n_years=0.5,
        n_scenarios=3,
        periods_per_year=12,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (6, 3)
    assert first.index.name == "step"
    assert list(first.columns) == [
        "scenario_0000",
        "scenario_0001",
        "scenario_0002",
    ]


def test_zero_volatility_matches_the_exact_deterministic_path() -> None:
    returns = simulate_gbm_returns(
        n_years=1,
        n_scenarios=2,
        expected_return=0.12,
        volatility=0.0,
        periods_per_year=12,
        seed=1,
    )

    expected_periodic = np.expm1(0.12 / 12)
    assert np.allclose(returns.to_numpy(), expected_periodic)
    assert np.allclose((1.0 + returns).prod().to_numpy(), np.exp(0.12))


def test_prices_include_initial_value_and_match_simulated_returns() -> None:
    returns = simulate_gbm_returns(
        n_years=1,
        n_scenarios=4,
        periods_per_year=4,
        seed=7,
    )
    prices = simulate_gbm_prices(
        n_years=1,
        n_scenarios=4,
        periods_per_year=4,
        initial_price=250.0,
        seed=7,
    )

    assert prices.shape == (5, 4)
    assert np.allclose(prices.iloc[0].to_numpy(), 250.0)
    assert (prices > 0.0).all().all()
    reconstructed = prices.pct_change().iloc[1:]
    reconstructed.index = returns.index
    pd.testing.assert_frame_equal(reconstructed, returns)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"n_years": 0}, ValueError),
        ({"n_years": 0.1, "periods_per_year": 12}, ValueError),
        ({"n_scenarios": True}, TypeError),
        ({"n_scenarios": 0}, ValueError),
        ({"volatility": -0.1}, ValueError),
        ({"seed": -1}, ValueError),
    ],
)
def test_gbm_rejects_invalid_parameters(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        simulate_gbm_returns(**kwargs)


def test_prices_reject_non_positive_initial_price() -> None:
    with pytest.raises(ValueError, match="initial_price"):
        simulate_gbm_prices(initial_price=0.0)


def test_gbm_rejects_parameters_that_overflow() -> None:
    with pytest.raises(OverflowError, match="non-finite returns"):
        simulate_gbm_returns(
            n_years=1,
            n_scenarios=1,
            expected_return=1e308,
            volatility=0.0,
            periods_per_year=1,
            seed=1,
        )
