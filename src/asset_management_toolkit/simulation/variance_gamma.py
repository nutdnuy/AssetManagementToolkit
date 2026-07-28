"""Variance Gamma return and price scenario generation."""

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
    validate_positive_real,
    validate_real,
    validate_seed,
)


def simulate_variance_gamma_returns(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    mean_log_return: float = 0.07,
    theta: float = 0.0,
    volatility: float = 0.15,
    variance_rate: float = 0.20,
    periods_per_year: int = 12,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate simple returns from a Gamma-time-changed Brownian motion.

    The Gamma clock increment has mean ``dt`` and variance
    ``variance_rate * dt``. Conditional on that clock, each log-return step is

    ``(mean_log_return - theta) * dt + theta * G + volatility * sqrt(G) * Z``.

    This parameterization keeps the unconditional mean log-return rate equal
    to ``mean_log_return`` while ``theta`` controls asymmetry.
    """
    steps = simulation_steps(n_years, periods_per_year)
    scenarios = validate_positive_integer(n_scenarios, "n_scenarios")
    mean_rate = validate_real(mean_log_return, "mean_log_return")
    clock_drift = validate_real(theta, "theta")
    sigma = validate_non_negative_real(volatility, "volatility")
    nu = validate_positive_real(variance_rate, "variance_rate")
    random_seed = validate_seed(seed)

    dt = 1.0 / periods_per_year
    random = np.random.default_rng(random_seed)
    gamma_clock = random.gamma(
        shape=dt / nu,
        scale=nu,
        size=(steps, scenarios),
    )
    shocks = random.standard_normal((steps, scenarios))
    with np.errstate(over="ignore", invalid="ignore"):
        log_returns = (
            (mean_rate - clock_drift) * dt
            + clock_drift * gamma_clock
            + sigma * np.sqrt(gamma_clock) * shocks
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


def simulate_variance_gamma_prices(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    mean_log_return: float = 0.07,
    theta: float = 0.0,
    volatility: float = 0.15,
    variance_rate: float = 0.20,
    periods_per_year: int = 12,
    initial_price: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate positive Variance Gamma price paths including step zero."""
    returns = simulate_variance_gamma_returns(
        n_years=n_years,
        n_scenarios=n_scenarios,
        mean_log_return=mean_log_return,
        theta=theta,
        volatility=volatility,
        variance_rate=variance_rate,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    return prices_from_returns(returns, initial_price)
