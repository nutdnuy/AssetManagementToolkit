"""Alpha-stable return and price scenario generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from asset_management_toolkit.simulation._paths import prices_from_returns
from asset_management_toolkit.simulation._validation import (
    scenario_labels,
    simulation_steps,
    validate_positive_integer,
    validate_positive_real,
    validate_real,
    validate_seed,
)


def simulate_stable_returns(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    alpha: float = 1.7,
    beta: float = 0.0,
    scale: float = 0.10,
    location: float = 0.07,
    periods_per_year: int = 12,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate simple returns from alpha-stable log increments.

    The annual law uses Nolan's ``S0`` parameterization. ``alpha`` is the tail
    index in ``(0, 2]`` and ``beta`` is the asymmetry parameter in ``[-1, 1]``:
    negative values emphasize the left tail and positive values emphasize the
    right tail. ``beta`` is not a conventional third-moment skewness statistic.

    The increment parameters preserve the one-year ``S0`` distribution under
    Lévy-process time scaling. ``location`` is therefore a distribution
    location, not an expected return across the full supported range. Stable
    shocks are generated with the Chambers-Mallows-Stuck transformation without
    changing NumPy's global random state.
    """
    steps = simulation_steps(n_years, periods_per_year)
    scenarios = validate_positive_integer(n_scenarios, "n_scenarios")
    tail_index = validate_positive_real(alpha, "alpha")
    if tail_index > 2.0:
        raise ValueError("alpha must be less than or equal to 2")
    asymmetry = validate_real(beta, "beta")
    if not -1.0 <= asymmetry <= 1.0:
        raise ValueError("beta must be between -1 and 1")
    stable_scale = validate_positive_real(scale, "scale")
    annual_location = validate_real(location, "location")
    random_seed = validate_seed(seed)

    dt = 1.0 / periods_per_year
    step_scale = stable_scale * np.power(dt, 1.0 / tail_index)
    step_location = _stable_increment_location(
        alpha=tail_index,
        beta=asymmetry,
        scale=stable_scale,
        location=annual_location,
        dt=dt,
    )
    random = np.random.default_rng(random_seed)
    stable_shocks = _stable_s0_shocks(
        alpha=tail_index,
        beta=asymmetry,
        size=(steps, scenarios),
        random=random,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        log_returns = step_location + step_scale * stable_shocks
        simple_returns = np.expm1(log_returns)
    if not np.isfinite(simple_returns).all() or np.any(simple_returns <= -1.0):
        raise OverflowError("simulation parameters exceeded floating-point limits")
    return pd.DataFrame(
        simple_returns,
        index=pd.RangeIndex(1, steps + 1, name="step"),
        columns=scenario_labels(scenarios),
        dtype=float,
    )


def simulate_stable_prices(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    alpha: float = 1.7,
    beta: float = 0.0,
    scale: float = 0.10,
    location: float = 0.07,
    periods_per_year: int = 12,
    initial_price: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate positive alpha-stable price paths including step zero."""
    returns = simulate_stable_returns(
        n_years=n_years,
        n_scenarios=n_scenarios,
        alpha=alpha,
        beta=beta,
        scale=scale,
        location=location,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    return prices_from_returns(returns, initial_price)


def simulate_symmetric_stable_returns(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    alpha: float = 1.7,
    scale: float = 0.10,
    location: float = 0.07,
    periods_per_year: int = 12,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate symmetric alpha-stable returns; compatibility wrapper."""
    return simulate_stable_returns(
        n_years=n_years,
        n_scenarios=n_scenarios,
        alpha=alpha,
        beta=0.0,
        scale=scale,
        location=location,
        periods_per_year=periods_per_year,
        seed=seed,
    )


def simulate_symmetric_stable_prices(
    n_years: float = 10.0,
    n_scenarios: int = 1_000,
    alpha: float = 1.7,
    scale: float = 0.10,
    location: float = 0.07,
    periods_per_year: int = 12,
    initial_price: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate symmetric alpha-stable prices; compatibility wrapper."""
    return simulate_stable_prices(
        n_years=n_years,
        n_scenarios=n_scenarios,
        alpha=alpha,
        beta=0.0,
        scale=scale,
        location=location,
        periods_per_year=periods_per_year,
        initial_price=initial_price,
        seed=seed,
    )


def _stable_increment_location(
    *,
    alpha: float,
    beta: float,
    scale: float,
    location: float,
    dt: float,
) -> float:
    """Return the S0 location that preserves annual Lévy-process parameters."""
    if beta == 0.0:
        return location * dt
    if alpha == 1.0:
        return location * dt + (2.0 / np.pi) * beta * scale * dt * np.log(dt)
    return location * dt + beta * scale * np.tan(0.5 * np.pi * alpha) * (
        np.power(dt, 1.0 / alpha) - dt
    )


def _stable_s0_shocks(
    *,
    alpha: float,
    beta: float,
    size: tuple[int, int],
    random: np.random.Generator,
) -> np.ndarray:
    """Draw standardized stable shocks in Nolan's S0 parameterization."""
    lower = np.nextafter(-0.5 * np.pi, 0.0)
    upper = np.nextafter(0.5 * np.pi, 0.0)
    angles = random.uniform(lower, upper, size=size)
    if alpha == 1.0:
        if beta == 0.0:
            return np.tan(angles)
        exponentials = _unit_exponentials(size=size, random=random)
        half_pi = 0.5 * np.pi
        return (2.0 / np.pi) * (
            (half_pi + beta * angles) * np.tan(angles)
            - beta
            * np.log(
                half_pi * exponentials * np.cos(angles) / (half_pi + beta * angles)
            )
        )

    exponentials = _unit_exponentials(size=size, random=random)
    if beta == 0.0:
        numerator = np.sin(alpha * angles)
        first_scale = np.power(np.cos(angles), -1.0 / alpha)
        second_scale = np.power(
            np.cos((1.0 - alpha) * angles) / exponentials,
            (1.0 - alpha) / alpha,
        )
        return numerator * first_scale * second_scale

    tangent = beta * np.tan(0.5 * np.pi * alpha)
    angle_shift = np.arctan(tangent) / alpha
    normalizer = np.power(1.0 + tangent**2, 0.5 / alpha)
    first_term = np.sin(alpha * (angles + angle_shift))
    first_term /= np.power(np.cos(angles), 1.0 / alpha)
    second_term = np.power(
        np.cos(angles - alpha * (angles + angle_shift)) / exponentials,
        (1.0 - alpha) / alpha,
    )
    s1_shocks = normalizer * first_term * second_term
    return s1_shocks - tangent


def _unit_exponentials(
    *,
    size: tuple[int, int],
    random: np.random.Generator,
) -> np.ndarray:
    return np.maximum(
        random.exponential(scale=1.0, size=size),
        np.finfo(float).tiny,
    )
