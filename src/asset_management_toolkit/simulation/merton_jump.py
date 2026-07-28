"""Merton jump-diffusion return and price scenario generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from asset_management_toolkit.simulation._paths import prices_from_returns
from asset_management_toolkit.simulation._validation import (
    scenario_labels,
    simulation_steps,
    validate_non_negative_real,
    validate_positive_integer,
    validate_real,
    validate_seed,
)
from asset_management_toolkit.simulation.gbm import simulate_gbm_returns


def simulate_merton_jump_returns(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    expected_return: float = 0.07,
    volatility: float = 0.15,
    jump_intensity: float = 1.0,
    jump_mean: float = -0.10,
    jump_volatility: float = 0.20,
    periods_per_year: int = 12,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate simple returns from exact Merton jump-diffusion increments.

    ``expected_return`` is the annual instantaneous price drift. Jumps arrive
    through a Poisson process with annual intensity ``jump_intensity``.
    Individual log jump sizes are normally distributed with mean
    ``jump_mean`` and standard deviation ``jump_volatility``.

    The continuous drift subtracts ``jump_intensity * kappa``, where
    ``kappa = E[exp(Y) - 1]``. This compensation keeps the expected price
    growth governed by ``expected_return`` rather than adding jump growth on
    top. The function is a statistical scenario generator, not an option
    pricer or a risk-neutral calibration routine.
    """
    steps = simulation_steps(n_years, periods_per_year)
    scenarios = validate_positive_integer(n_scenarios, "n_scenarios")
    mu = validate_real(expected_return, "expected_return")
    sigma = validate_non_negative_real(volatility, "volatility")
    intensity = validate_non_negative_real(jump_intensity, "jump_intensity")
    mean_log_jump = validate_real(jump_mean, "jump_mean")
    log_jump_sigma = validate_non_negative_real(
        jump_volatility,
        "jump_volatility",
    )
    random_seed = validate_seed(seed)

    with np.errstate(over="ignore", invalid="ignore"):
        jump_compensator = np.expm1(mean_log_jump + 0.5 * np.square(log_jump_sigma))
    if not np.isfinite(jump_compensator):
        raise OverflowError("jump parameters exceeded floating-point limits")

    if intensity == 0.0:
        return simulate_gbm_returns(
            n_years=n_years,
            n_scenarios=scenarios,
            expected_return=mu,
            volatility=sigma,
            periods_per_year=periods_per_year,
            seed=random_seed,
        )

    dt = 1.0 / periods_per_year
    random = np.random.default_rng(random_seed)
    diffusion_shocks = random.standard_normal((steps, scenarios))
    try:
        jump_counts = random.poisson(intensity * dt, size=(steps, scenarios))
    except ValueError as error:
        raise OverflowError("jump intensity exceeded numerical limits") from error

    if log_jump_sigma == 0.0:
        compound_jumps = jump_counts * mean_log_jump
    else:
        jump_shocks = random.standard_normal((steps, scenarios))
        compound_jumps = (
            jump_counts * mean_log_jump
            + np.sqrt(jump_counts) * log_jump_sigma * jump_shocks
        )

    with np.errstate(over="ignore", invalid="ignore"):
        log_returns = (
            (mu - 0.5 * np.square(sigma) - intensity * jump_compensator) * dt
            + sigma * np.sqrt(dt) * diffusion_shocks
            + compound_jumps
        )
        simple_returns = np.expm1(log_returns)
    if not np.isfinite(simple_returns).all() or np.any(simple_returns <= -1.0):
        raise OverflowError("simulation parameters exceeded floating-point limits")
    return pd.DataFrame(
        simple_returns,
        index=pd.RangeIndex(1, steps + 1, name="step"),
        columns=scenario_labels(scenarios),
        dtype=float,
    )


def simulate_merton_jump_prices(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    expected_return: float = 0.07,
    volatility: float = 0.15,
    jump_intensity: float = 1.0,
    jump_mean: float = -0.10,
    jump_volatility: float = 0.20,
    periods_per_year: int = 12,
    initial_price: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate positive Merton jump-diffusion paths including step zero."""
    returns = simulate_merton_jump_returns(
        n_years=n_years,
        n_scenarios=n_scenarios,
        expected_return=expected_return,
        volatility=volatility,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_volatility=jump_volatility,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    return prices_from_returns(returns, initial_price)
