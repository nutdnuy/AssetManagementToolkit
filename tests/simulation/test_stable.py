import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    simulate_stable_prices,
    simulate_stable_returns,
    simulate_symmetric_stable_prices,
    simulate_symmetric_stable_returns,
)
from asset_management_toolkit.simulation.stable import (
    _stable_increment_location,
    _stable_s0_shocks,
)


def test_stable_returns_are_reproducible_and_labelled() -> None:
    first = simulate_stable_returns(
        n_years=0.5,
        n_scenarios=3,
        alpha=1.7,
        beta=-0.25,
        scale=0.03,
        periods_per_year=12,
        seed=42,
    )
    second = simulate_stable_returns(
        n_years=0.5,
        n_scenarios=3,
        alpha=1.7,
        beta=-0.25,
        scale=0.03,
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


def test_symmetric_wrappers_match_general_beta_zero_api() -> None:
    general = simulate_stable_returns(
        n_years=0.5,
        n_scenarios=8,
        alpha=1.6,
        beta=0.0,
        scale=0.02,
        periods_per_year=12,
        seed=27,
    )
    symmetric = simulate_symmetric_stable_returns(
        n_years=0.5,
        n_scenarios=8,
        alpha=1.6,
        scale=0.02,
        periods_per_year=12,
        seed=27,
    )
    general_prices = simulate_stable_prices(
        n_years=0.5,
        n_scenarios=8,
        alpha=1.6,
        beta=0.0,
        scale=0.02,
        periods_per_year=12,
        seed=27,
    )
    symmetric_prices = simulate_symmetric_stable_prices(
        n_years=0.5,
        n_scenarios=8,
        alpha=1.6,
        scale=0.02,
        periods_per_year=12,
        seed=27,
    )

    pd.testing.assert_frame_equal(general, symmetric, check_exact=True)
    pd.testing.assert_frame_equal(
        general_prices,
        symmetric_prices,
        check_exact=True,
    )


def test_alpha_two_has_the_gaussian_limit_moments() -> None:
    scale = 0.20
    location = 0.03
    returns = simulate_symmetric_stable_returns(
        n_years=1,
        n_scenarios=150_000,
        alpha=2.0,
        scale=scale,
        location=location,
        periods_per_year=1,
        seed=7,
    )
    log_returns = np.log1p(returns.iloc[0].to_numpy())

    assert np.mean(log_returns) == pytest.approx(location, abs=0.003)
    assert np.var(log_returns) == pytest.approx(2.0 * scale**2, abs=0.003)


def test_stable_prices_match_simulated_returns() -> None:
    returns = simulate_stable_returns(
        n_years=1,
        n_scenarios=4,
        alpha=1.8,
        beta=-0.4,
        scale=0.03,
        periods_per_year=4,
        seed=9,
    )
    prices = simulate_stable_prices(
        n_years=1,
        n_scenarios=4,
        alpha=1.8,
        beta=-0.4,
        scale=0.03,
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
    ("alpha", "beta"),
    [(1.0, -0.5), (1.4, -0.5), (1.7, 0.4)],
)
def test_stable_s0_shocks_match_theoretical_characteristic_function(
    alpha: float,
    beta: float,
) -> None:
    shocks = _stable_s0_shocks(
        alpha=alpha,
        beta=beta,
        size=(1, 300_000),
        random=np.random.default_rng(18),
    ).ravel()

    for frequency in (0.2, 0.5):
        empirical = np.mean(np.exp(1j * frequency * shocks))
        absolute_frequency = abs(frequency)
        if alpha == 1.0:
            phi = -(2.0 / np.pi) * np.log(absolute_frequency)
        else:
            phi = -np.tan(0.5 * np.pi * alpha) * (
                absolute_frequency ** (1.0 - alpha) - 1.0
            )
        theoretical = np.exp(
            -(absolute_frequency**alpha) * (1.0 - 1j * beta * np.sign(frequency) * phi)
        )

        assert empirical.real == pytest.approx(theoretical.real, abs=0.004)
        assert empirical.imag == pytest.approx(theoretical.imag, abs=0.004)


def test_negative_beta_emphasizes_left_tail() -> None:
    random_seed = 123
    negative = _stable_s0_shocks(
        alpha=1.7,
        beta=-0.7,
        size=(1, 250_000),
        random=np.random.default_rng(random_seed),
    )
    positive = _stable_s0_shocks(
        alpha=1.7,
        beta=0.7,
        size=(1, 250_000),
        random=np.random.default_rng(random_seed),
    )

    assert np.quantile(negative, 0.01) < np.quantile(positive, 0.01)
    assert np.quantile(negative, 0.99) < np.quantile(positive, 0.99)


def test_s0_increment_location_preserves_annual_levy_scaling() -> None:
    alpha = 1.7
    beta = -0.4
    scale = 0.08
    location = 0.05
    dt = 1.0 / 12.0

    step_location = _stable_increment_location(
        alpha=alpha,
        beta=beta,
        scale=scale,
        location=location,
        dt=dt,
    )
    step_scale = scale * dt ** (1.0 / alpha)
    annual_s1_location = location - beta * scale * np.tan(0.5 * np.pi * alpha)
    step_s1_location = step_location - beta * step_scale * np.tan(0.5 * np.pi * alpha)

    assert step_s1_location == pytest.approx(annual_s1_location * dt)


def test_alpha_one_s0_increment_location_preserves_annual_levy_scaling() -> None:
    beta = -0.4
    scale = 0.08
    location = 0.05
    dt = 1.0 / 12.0

    step_location = _stable_increment_location(
        alpha=1.0,
        beta=beta,
        scale=scale,
        location=location,
        dt=dt,
    )
    step_scale = scale * dt
    annual_s1_location = location - (2.0 / np.pi) * beta * scale * np.log(scale)
    step_s1_location = step_location - (
        (2.0 / np.pi) * beta * step_scale * np.log(step_scale)
    )

    assert step_s1_location == pytest.approx(annual_s1_location * dt)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"alpha": 0.0}, ValueError),
        ({"alpha": 2.1}, ValueError),
        ({"beta": -1.1}, ValueError),
        ({"beta": 1.1}, ValueError),
        ({"beta": np.nan}, ValueError),
        ({"scale": 0.0}, ValueError),
        ({"scale": -0.1}, ValueError),
        ({"location": np.nan}, ValueError),
        ({"seed": 1.5}, TypeError),
    ],
)
def test_stable_rejects_invalid_parameters(
    kwargs: dict,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        simulate_stable_returns(**kwargs)
