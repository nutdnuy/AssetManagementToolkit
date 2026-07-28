"""Calibration contracts for supported univariate simulation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from asset_management_toolkit.analytics._validation import validate_periods_per_year
from asset_management_toolkit.simulation._observations import (
    log_return_observations,
)
from asset_management_toolkit.simulation._validation import (
    validate_positive_integer,
    validate_positive_real,
)


@dataclass
class CalibrationResult:
    """Auditable output from one model calibration."""

    model: str
    parameters: dict[str, float]
    method: str
    success: bool
    objective: float
    n_observations: int
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    message: str = ""

    def to_series(self) -> pd.Series:
        """Return one flat review row including prefixed parameters."""
        values: dict[str, object] = {
            "model": self.model,
            "method": self.method,
            "success": self.success,
            "objective": self.objective,
            "n_observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "message": self.message,
        }
        values.update(
            {f"parameter_{name}": value for name, value in self.parameters.items()}
        )
        return pd.Series(values, name=self.model)


def calibrate_gbm(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> CalibrationResult:
    """Fit GBM drift and volatility by Gaussian log-return maximum likelihood."""
    validate_periods_per_year(periods_per_year)
    log_returns = log_return_observations(returns, min_observations=3)
    dt = 1.0 / periods_per_year
    periodic_variance = float(np.var(log_returns, ddof=0))
    if periodic_variance <= np.finfo(float).eps:
        raise ValueError("GBM calibration requires non-constant log returns")
    volatility = float(np.sqrt(periodic_variance / dt))
    expected_return = float(np.mean(log_returns) / dt + 0.5 * volatility**2)
    log_likelihood = float(
        np.sum(
            stats.norm.logpdf(
                log_returns,
                loc=(expected_return - 0.5 * volatility**2) * dt,
                scale=volatility * np.sqrt(dt),
            )
        )
    )
    aic, bic = _information_criteria(
        log_likelihood=log_likelihood,
        n_parameters=2,
        n_observations=log_returns.size,
    )
    return CalibrationResult(
        model="gbm",
        parameters={
            "expected_return": expected_return,
            "volatility": volatility,
        },
        method="maximum_likelihood",
        success=True,
        objective=-log_likelihood,
        n_observations=int(log_returns.size),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        message="closed-form Gaussian log-return maximum likelihood",
    )


def calibrate_merton_jump(
    returns: pd.Series,
    periods_per_year: int = 252,
    *,
    max_jump_intensity: float = 20.0,
    max_iterations: int = 500,
) -> CalibrationResult:
    """Fit Merton jump diffusion by truncated Poisson-mixture likelihood.

    The optimizer uses deterministic multi-start initialization. The likelihood
    includes enough Poisson components to retain at least ``1 - 1e-12`` mass at
    each candidate intensity.
    """
    validate_periods_per_year(periods_per_year)
    intensity_upper = validate_positive_real(
        max_jump_intensity,
        "max_jump_intensity",
    )
    if intensity_upper <= 1e-6:
        raise ValueError("max_jump_intensity must be greater than 1e-6")
    iterations = validate_positive_integer(max_iterations, "max_iterations")
    log_returns = log_return_observations(returns, min_observations=30)
    dt = 1.0 / periods_per_year
    gbm = calibrate_gbm(returns, periods_per_year)
    gbm_mu = gbm.parameters["expected_return"]
    gbm_sigma = gbm.parameters["volatility"]
    periodic_scale = max(float(np.std(log_returns, ddof=0)), 1e-3)
    lower_tail_shift = min(
        float(np.quantile(log_returns, 0.05) - np.median(log_returns)),
        -1e-3,
    )
    starts = [
        np.array([gbm_mu, 0.8 * gbm_sigma, 0.5, lower_tail_shift, periodic_scale]),
        np.array(
            [
                gbm_mu,
                0.7 * gbm_sigma,
                min(2.0, intensity_upper),
                -periodic_scale,
                periodic_scale,
            ]
        ),
        np.array(
            [
                gbm_mu,
                0.9 * gbm_sigma,
                min(5.0, intensity_upper),
                -2.0 * periodic_scale,
                2.0 * periodic_scale,
            ]
        ),
    ]
    lower = np.array([-5.0, 1e-6, 1e-6, -5.0, 1e-6])
    upper = np.array([5.0, 5.0, intensity_upper, 5.0, 5.0])
    starts = [np.clip(start, lower, upper) for start in starts]

    fitted = [
        optimize.minimize(
            _negative_merton_log_likelihood,
            start,
            args=(log_returns, dt),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": iterations},
        )
        for start in starts
    ]
    finite_results = [result for result in fitted if np.isfinite(result.fun)]
    if not finite_results:
        raise RuntimeError("Merton calibration failed to find a finite solution")
    best = min(finite_results, key=lambda result: float(result.fun))
    expected_return, volatility, intensity, jump_mean, jump_volatility = (
        float(value) for value in best.x
    )
    log_likelihood = -float(best.fun)
    aic, bic = _information_criteria(
        log_likelihood=log_likelihood,
        n_parameters=5,
        n_observations=log_returns.size,
    )
    return CalibrationResult(
        model="merton_jump",
        parameters={
            "expected_return": expected_return,
            "volatility": volatility,
            "jump_intensity": intensity,
            "jump_mean": jump_mean,
            "jump_volatility": jump_volatility,
        },
        method="truncated_poisson_mixture_maximum_likelihood",
        success=bool(best.success),
        objective=float(best.fun),
        n_observations=int(log_returns.size),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        message=str(best.message),
    )


def calibrate_variance_gamma(
    returns: pd.Series,
    periods_per_year: int = 252,
    *,
    max_variance_rate: float = 10.0,
    max_iterations: int = 2_000,
) -> CalibrationResult:
    """Fit Variance Gamma parameters by variance/skew/kurtosis matching.

    This method matches the second through fourth log-return cumulants. It does
    not evaluate a full density likelihood, so AIC and BIC are intentionally
    unavailable.
    """
    validate_periods_per_year(periods_per_year)
    nu_upper = validate_positive_real(max_variance_rate, "max_variance_rate")
    if nu_upper <= 1e-6:
        raise ValueError("max_variance_rate must be greater than 1e-6")
    evaluations = validate_positive_integer(max_iterations, "max_iterations")
    log_returns = log_return_observations(returns, min_observations=20)
    dt = 1.0 / periods_per_year
    centered = log_returns - np.mean(log_returns)
    target_variance = float(np.mean(centered**2))
    if target_variance <= np.finfo(float).eps:
        raise ValueError("Variance Gamma calibration requires variable returns")
    target_third = float(np.mean(centered**3))
    target_fourth_cumulant = float(np.mean(centered**4) - 3.0 * target_variance**2)
    target_skewness = target_third / target_variance**1.5
    target_excess = target_fourth_cumulant / target_variance**2
    annual_scale = float(np.sqrt(target_variance / dt))
    theta_direction = float(np.sign(target_skewness) or 1.0)

    starts = [
        np.array([0.1 * theta_direction * annual_scale, annual_scale, 0.10]),
        np.array([0.3 * theta_direction * annual_scale, annual_scale, 0.50]),
        np.array([0.5 * theta_direction * annual_scale, annual_scale, 1.00]),
    ]
    lower = np.array([-5.0, 1e-8, 1e-6])
    upper = np.array([5.0, 5.0, nu_upper])
    starts = [np.clip(start, lower, upper) for start in starts]
    fitted = [
        optimize.least_squares(
            _variance_gamma_cumulant_residuals,
            start,
            bounds=(lower, upper),
            args=(
                dt,
                target_variance,
                target_skewness,
                target_excess,
            ),
            max_nfev=evaluations,
        )
        for start in starts
    ]
    finite_results = [result for result in fitted if np.isfinite(np.sum(result.fun**2))]
    if not finite_results:
        raise RuntimeError(
            "Variance Gamma calibration failed to find a finite solution"
        )
    best = min(finite_results, key=lambda result: float(np.sum(result.fun**2)))
    theta, volatility, variance_rate = (float(value) for value in best.x)
    return CalibrationResult(
        model="variance_gamma",
        parameters={
            "mean_log_return": float(np.mean(log_returns) / dt),
            "theta": theta,
            "volatility": volatility,
            "variance_rate": variance_rate,
        },
        method="log_return_cumulant_matching",
        success=bool(best.success),
        objective=float(np.sum(best.fun**2)),
        n_observations=int(log_returns.size),
        message=str(best.message),
    )


def _negative_merton_log_likelihood(
    parameters: np.ndarray,
    log_returns: np.ndarray,
    dt: float,
) -> float:
    expected_return, volatility, intensity, jump_mean, jump_volatility = parameters
    with np.errstate(over="ignore", invalid="ignore"):
        compensator = np.expm1(jump_mean + 0.5 * jump_volatility**2)
    if not np.isfinite(compensator):
        return float("inf")
    poisson_rate = intensity * dt
    maximum_jumps = max(
        10,
        int(stats.poisson.ppf(1.0 - 1e-12, poisson_rate)),
    )
    jump_counts = np.arange(maximum_jumps + 1, dtype=float)
    conditional_means = (
        expected_return - 0.5 * volatility**2 - intensity * compensator
    ) * dt + jump_counts * jump_mean
    conditional_variances = volatility**2 * dt + jump_counts * jump_volatility**2
    if np.any(conditional_variances <= 0.0):
        return float("inf")
    log_weights = (
        -poisson_rate
        + jump_counts * np.log(poisson_rate)
        - special.gammaln(jump_counts + 1.0)
    )
    components = log_weights[:, None] + stats.norm.logpdf(
        log_returns[None, :],
        loc=conditional_means[:, None],
        scale=np.sqrt(conditional_variances)[:, None],
    )
    log_likelihood = float(np.sum(special.logsumexp(components, axis=0)))
    return -log_likelihood if np.isfinite(log_likelihood) else float("inf")


def _variance_gamma_cumulant_residuals(
    parameters: np.ndarray,
    dt: float,
    target_variance: float,
    target_skewness: float,
    target_excess: float,
) -> np.ndarray:
    theta, volatility, variance_rate = parameters
    annual_second = volatility**2 + variance_rate * theta**2
    annual_third = (
        variance_rate * theta * (2.0 * variance_rate * theta**2 + 3.0 * volatility**2)
    )
    annual_fourth = (
        3.0
        * variance_rate
        * (
            2.0 * variance_rate**2 * theta**4
            + 4.0 * variance_rate * volatility**2 * theta**2
            + volatility**4
        )
    )
    model_variance = annual_second * dt
    model_skewness = annual_third * dt / model_variance**1.5
    model_excess = annual_fourth * dt / model_variance**2
    return np.array(
        [
            np.log(model_variance / target_variance),
            (model_skewness - target_skewness) / max(1.0, abs(target_skewness)),
            (model_excess - target_excess) / max(1.0, abs(target_excess)),
        ]
    )


def _information_criteria(
    *,
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
) -> tuple[float, float]:
    aic = 2.0 * n_parameters - 2.0 * log_likelihood
    bic = np.log(n_observations) * n_parameters - 2.0 * log_likelihood
    return float(aic), float(bic)
