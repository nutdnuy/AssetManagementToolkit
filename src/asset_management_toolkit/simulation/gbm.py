"""Geometric Brownian motion scenario generation."""

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


def simulate_gbm_returns(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    expected_return: float = 0.07,
    volatility: float = 0.15,
    periods_per_year: int = 12,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate periodic simple returns from exact GBM log-return steps.

    ``expected_return`` is the annual instantaneous GBM drift ``mu`` and
    ``volatility`` is annualized ``sigma``. The output contains one scenario
    per column and one periodic simple return per row.
    """
    steps = simulation_steps(n_years, periods_per_year)
    scenarios = validate_positive_integer(n_scenarios, "n_scenarios")
    mu = validate_real(expected_return, "expected_return")
    sigma = validate_non_negative_real(volatility, "volatility")
    random_seed = validate_seed(seed)

    dt = 1.0 / periods_per_year
    random = np.random.default_rng(random_seed)
    shocks = random.standard_normal((steps, scenarios))
    with np.errstate(over="ignore", invalid="ignore"):
        log_returns = (mu - 0.5 * np.square(sigma)) * dt + sigma * np.sqrt(dt) * shocks
        simple_returns = np.expm1(log_returns)
    if not np.isfinite(simple_returns).all():
        raise OverflowError("simulation parameters produced non-finite returns")
    return pd.DataFrame(
        simple_returns,
        index=pd.RangeIndex(1, steps + 1, name="step"),
        columns=scenario_labels(scenarios),
        dtype=float,
    )


def simulate_gbm_prices(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    expected_return: float = 0.07,
    volatility: float = 0.15,
    periods_per_year: int = 12,
    initial_price: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate strictly positive GBM price paths including the initial row."""
    returns = simulate_gbm_returns(
        n_years=n_years,
        n_scenarios=n_scenarios,
        expected_return=expected_return,
        volatility=volatility,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    return prices_from_returns(returns, initial_price)
