import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    simulate_gbm_returns,
    simulate_merton_jump_prices,
    simulate_merton_jump_returns,
)


def test_merton_returns_are_reproducible_and_labelled() -> None:
    first = simulate_merton_jump_returns(
        n_years=0.5,
        n_scenarios=3,
        jump_intensity=2.0,
        periods_per_year=12,
        seed=42,
    )
    second = simulate_merton_jump_returns(
        n_years=0.5,
        n_scenarios=3,
        jump_intensity=2.0,
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
    assert (first > -1.0).all().all()


def test_zero_jump_intensity_matches_gbm_exactly() -> None:
    jump_diffusion = simulate_merton_jump_returns(
        n_years=1,
        n_scenarios=20,
        expected_return=0.08,
        volatility=0.17,
        jump_intensity=0.0,
        jump_mean=-0.50,
        jump_volatility=0.80,
        periods_per_year=12,
        seed=7,
    )
    gbm = simulate_gbm_returns(
        n_years=1,
        n_scenarios=20,
        expected_return=0.08,
        volatility=0.17,
        periods_per_year=12,
        seed=7,
    )

    pd.testing.assert_frame_equal(jump_diffusion, gbm, check_exact=True)


def test_one_year_log_return_moments_match_compound_poisson_theory() -> None:
    expected_return = 0.06
    volatility = 0.12
    jump_intensity = 1.5
    jump_mean = -0.08
    jump_volatility = 0.18
    returns = simulate_merton_jump_returns(
        n_years=1,
        n_scenarios=350_000,
        expected_return=expected_return,
        volatility=volatility,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        periods_per_year=1,
        seed=19,
    )
    log_returns = np.log1p(returns.iloc[0].to_numpy())

    compensator = np.expm1(jump_mean + 0.5 * jump_volatility**2)
    theoretical_mean = (
        expected_return
        - 0.5 * volatility**2
        - jump_intensity * compensator
        + jump_intensity * jump_mean
    )
    theoretical_variance = volatility**2 + jump_intensity * (
        jump_volatility**2 + jump_mean**2
    )

    assert np.mean(log_returns) == pytest.approx(theoretical_mean, abs=0.002)
    assert np.var(log_returns) == pytest.approx(theoretical_variance, abs=0.002)


def test_jump_compensation_preserves_expected_price_growth() -> None:
    expected_return = 0.05
    returns = simulate_merton_jump_returns(
        n_years=1,
        n_scenarios=350_000,
        expected_return=expected_return,
        volatility=0.10,
        jump_intensity=2.0,
        jump_mean=-0.12,
        jump_volatility=0.20,
        periods_per_year=1,
        seed=22,
    )

    assert returns.iloc[0].mean() == pytest.approx(
        np.expm1(expected_return),
        abs=0.002,
    )


def test_merton_prices_match_simulated_returns() -> None:
    returns = simulate_merton_jump_returns(
        n_years=1,
        n_scenarios=5,
        jump_intensity=3.0,
        periods_per_year=4,
        seed=9,
    )
    prices = simulate_merton_jump_prices(
        n_years=1,
        n_scenarios=5,
        jump_intensity=3.0,
        periods_per_year=4,
        initial_price=250.0,
        seed=9,
    )

    assert prices.shape == (5, 5)
    assert np.allclose(prices.iloc[0], 250.0)
    assert (prices > 0.0).all().all()
    reconstructed = prices.pct_change().iloc[1:]
    reconstructed.index = returns.index
    pd.testing.assert_frame_equal(reconstructed, returns)


def test_deterministic_jump_size_is_supported() -> None:
    returns = simulate_merton_jump_returns(
        n_years=1,
        n_scenarios=100,
        volatility=0.0,
        jump_intensity=2.0,
        jump_mean=-0.10,
        jump_volatility=0.0,
        periods_per_year=12,
        seed=3,
    )

    assert np.isfinite(returns.to_numpy()).all()
    assert (returns > -1.0).all().all()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"jump_intensity": -0.1}, ValueError),
        ({"jump_intensity": np.nan}, ValueError),
        ({"jump_mean": np.inf}, ValueError),
        ({"jump_volatility": -0.1}, ValueError),
        ({"jump_volatility": np.nan}, ValueError),
        ({"seed": True}, TypeError),
    ],
)
def test_merton_rejects_invalid_parameters(
    kwargs: dict,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        simulate_merton_jump_returns(**kwargs)


def test_merton_rejects_jump_parameters_that_overflow() -> None:
    with pytest.raises(OverflowError, match="jump parameters"):
        simulate_merton_jump_returns(
            jump_mean=1e308,
            jump_volatility=1.0,
        )


def test_merton_prices_reject_non_positive_initial_price() -> None:
    with pytest.raises(ValueError, match="initial_price"):
        simulate_merton_jump_prices(initial_price=0.0)
