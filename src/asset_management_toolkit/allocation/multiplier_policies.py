"""Internal multiplier policies for CPPI-family strategies."""

from __future__ import annotations

import numpy as np


def volatility_controlled_multiplier(
    prior_returns: np.ndarray,
    *,
    base_multiplier: float,
    target_volatility: float,
    periods_per_year: int,
    minimum_multiplier: float,
    maximum_multiplier: float,
    minimum_history: int,
    volatility_exponent: float,
) -> float:
    """Scale a base multiplier inversely with lagged realized volatility."""
    if prior_returns.size < minimum_history:
        return float(np.clip(base_multiplier, minimum_multiplier, maximum_multiplier))

    realized_volatility = float(
        np.std(prior_returns, ddof=1) * np.sqrt(periods_per_year)
    )
    if realized_volatility <= np.finfo(float).eps:
        return maximum_multiplier
    raw_multiplier = (
        base_multiplier
        * (target_volatility / realized_volatility) ** volatility_exponent
    )
    return float(np.clip(raw_multiplier, minimum_multiplier, maximum_multiplier))


def growth_optimal_multiplier_from_moments(
    expected_risky_return: np.ndarray,
    expected_reserve_return: np.ndarray,
    risky_volatility: np.ndarray,
    reserve_volatility: np.ndarray,
    correlation: np.ndarray,
) -> np.ndarray:
    """Return Mantilla-García growth-optimal multipliers from annual moments."""
    covariance = correlation * risky_volatility * reserve_volatility
    relative_variance = risky_volatility**2 + reserve_volatility**2 - 2.0 * covariance
    if np.any(relative_variance <= np.finfo(float).eps):
        raise ValueError(
            "risky and reserve relative variance must be greater than zero"
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        risky_growth = expected_risky_return - 0.5 * risky_volatility**2
        reserve_growth = expected_reserve_return - 0.5 * reserve_volatility**2
        relative_excess_growth = 0.5 * relative_variance
        multiplier = (
            risky_growth - reserve_growth + relative_excess_growth
        ) / relative_variance
    if not np.isfinite(multiplier).all():
        raise FloatingPointError(
            "growth-optimal multiplier produced a non-finite value"
        )
    return multiplier
