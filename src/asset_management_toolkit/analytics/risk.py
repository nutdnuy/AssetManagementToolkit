"""Risk and risk-adjusted performance calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm

from asset_management_toolkit.analytics._validation import (
    MetricResult,
    ReturnInput,
    align_pair,
    annual_to_periodic_rate,
    coerce_benchmark,
    coerce_returns,
    restore_metric,
    safe_ratio,
    validate_annual_rate,
    validate_periods_per_year,
    validate_probability,
)
from asset_management_toolkit.analytics.returns import annualized_return


def annualized_volatility(
    returns: ReturnInput,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate sample volatility annualized by the square-root-of-time rule."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    result = frame.apply(
        lambda series: _sample_std(series.dropna()) * np.sqrt(periods_per_year)
    )
    return restore_metric(result.astype(float), was_series)


def semivariance(
    returns: ReturnInput,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized semivariance relative to an annual hurdle.

    Returns at or above the periodic hurdle contribute zero downside. The
    squared shortfalls are averaged over all non-missing observations and
    annualized by multiplying by ``periods_per_year``.
    """
    validate_annual_rate(minimum_acceptable_return, "minimum_acceptable_return")
    validate_periods_per_year(periods_per_year)
    hurdle = annual_to_periodic_rate(
        minimum_acceptable_return,
        periods_per_year,
    )
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        clean = series.dropna().to_numpy()
        downside = np.minimum(clean - hurdle, 0.0)
        return float(np.mean(np.square(downside)) * periods_per_year)

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def downside_deviation(
    returns: ReturnInput,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized downside deviation relative to an annual hurdle."""
    result = semivariance(
        returns,
        minimum_acceptable_return,
        periods_per_year,
    )
    if isinstance(result, pd.Series):
        return np.sqrt(result).astype(float)
    return float(np.sqrt(result))


def max_drawdown(returns: ReturnInput) -> MetricResult:
    """Calculate the most negative peak-to-trough drawdown."""
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        clean = series.dropna().to_numpy()
        wealth = np.concatenate(([1.0], np.cumprod(1.0 + clean)))
        peaks = np.maximum.accumulate(wealth)
        drawdowns = wealth / peaks - 1.0
        return float(np.min(drawdowns))

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def historical_var(
    returns: ReturnInput,
    level: float = 0.05,
) -> MetricResult:
    """Calculate historical VaR as a non-negative loss magnitude."""
    validate_probability(level, "level")
    frame, was_series = coerce_returns(returns)
    result = frame.apply(
        lambda series: max(0.0, -float(series.dropna().quantile(level)))
    )
    return restore_metric(result, was_series)


def historical_cvar(
    returns: ReturnInput,
    level: float = 0.05,
) -> MetricResult:
    """Calculate historical CVaR as the mean loss beyond the VaR quantile."""
    validate_probability(level, "level")
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        clean = series.dropna()
        threshold = clean.quantile(level)
        tail = clean[clean <= threshold]
        return max(0.0, -float(tail.mean()))

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def gaussian_var(
    returns: ReturnInput,
    level: float = 0.05,
    *,
    modified: bool = False,
) -> MetricResult:
    """Calculate parametric Gaussian VaR as a non-negative loss magnitude.

    Set ``modified=True`` to apply the Cornish-Fisher adjustment using
    population skewness and kurtosis.
    """
    validate_probability(level, "level")
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        clean = series.dropna()
        z_score = float(norm.ppf(level))
        if modified:
            population_skew, population_kurtosis = _population_shape(clean)
            if not np.isfinite(population_skew) or not np.isfinite(population_kurtosis):
                return float("nan")
            z_score = float(
                z_score
                + (z_score**2 - 1.0) * population_skew / 6.0
                + (z_score**3 - 3.0 * z_score) * (population_kurtosis - 3.0) / 24.0
                - (2.0 * z_score**3 - 5.0 * z_score) * population_skew**2 / 36.0
            )
        estimate = -(float(clean.mean()) + z_score * float(clean.std(ddof=0)))
        return max(0.0, estimate)

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def cornish_fisher_var(
    returns: ReturnInput,
    level: float = 0.05,
) -> MetricResult:
    """Calculate Cornish-Fisher modified Gaussian VaR."""
    return gaussian_var(returns, level, modified=True)


def is_normal(
    returns: ReturnInput,
    significance: float = 0.01,
):
    """Apply the Jarque-Bera test and return whether normality is not rejected.

    The test has limited power in small samples; callers should interpret the
    boolean together with sample size and distribution diagnostics.
    """
    validate_probability(significance, "significance")
    frame, was_series = coerce_returns(returns)
    result = frame.apply(
        lambda series: bool(jarque_bera(series.dropna()).pvalue > significance)
    )
    if was_series:
        return bool(result.iloc[0])
    return result.astype(bool)


def skewness(returns: ReturnInput) -> MetricResult:
    """Calculate unbiased sample skewness."""
    frame, was_series = coerce_returns(returns)
    result = frame.apply(lambda series: float(series.dropna().skew()))
    return restore_metric(result, was_series)


def excess_kurtosis(returns: ReturnInput) -> MetricResult:
    """Calculate unbiased Fisher excess kurtosis."""
    frame, was_series = coerce_returns(returns)
    result = frame.apply(lambda series: float(series.dropna().kurt()))
    return restore_metric(result, was_series)


def sharpe_ratio(
    returns: ReturnInput,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized Sharpe ratio from periodic arithmetic excess returns."""
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    risk_free_periodic = annual_to_periodic_rate(risk_free_rate, periods_per_year)
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        excess = series.dropna() - risk_free_periodic
        denominator = _sample_std(excess)
        return safe_ratio(
            float(excess.mean() * np.sqrt(periods_per_year)),
            denominator,
        )

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def sortino_ratio(
    returns: ReturnInput,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized Sortino ratio relative to an annual hurdle."""
    validate_annual_rate(minimum_acceptable_return, "minimum_acceptable_return")
    validate_periods_per_year(periods_per_year)
    hurdle = annual_to_periodic_rate(
        minimum_acceptable_return,
        periods_per_year,
    )
    frame, was_series = coerce_returns(returns)

    def calculate(series: pd.Series) -> float:
        clean = series.dropna()
        excess = clean - hurdle
        downside = np.minimum(excess.to_numpy(), 0.0)
        annual_downside = float(
            np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
        )
        annual_excess = float(excess.mean() * periods_per_year)
        return safe_ratio(annual_excess, annual_downside)

    result = frame.apply(calculate)
    return restore_metric(result, was_series)


def calmar_ratio(
    returns: ReturnInput,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized return divided by absolute maximum drawdown."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    values = {}
    for column in frame:
        annual_return = float(annualized_return(frame[column], periods_per_year))
        drawdown = float(max_drawdown(frame[column]))
        values[column] = safe_ratio(annual_return, abs(drawdown))
    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def tracking_error(
    returns: ReturnInput,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized sample volatility of active returns."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    benchmark_series = coerce_benchmark(benchmark)
    values = {}
    for column in frame:
        aligned = align_pair(frame[column], benchmark_series)
        active = aligned["portfolio"] - aligned["benchmark"]
        values[column] = _sample_std(active) * np.sqrt(periods_per_year)
    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def information_ratio(
    returns: ReturnInput,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized active return divided by tracking error."""
    validate_periods_per_year(periods_per_year)
    frame, was_series = coerce_returns(returns)
    benchmark_series = coerce_benchmark(benchmark)
    values = {}
    for column in frame:
        aligned = align_pair(frame[column], benchmark_series)
        active = aligned["portfolio"] - aligned["benchmark"]
        annual_active = float(active.mean() * periods_per_year)
        annual_tracking_error = float(_sample_std(active) * np.sqrt(periods_per_year))
        values[column] = safe_ratio(annual_active, annual_tracking_error)
    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def beta(returns: ReturnInput, benchmark: pd.Series) -> MetricResult:
    """Calculate portfolio beta relative to a pairwise-aligned benchmark."""
    frame, was_series = coerce_returns(returns)
    benchmark_series = coerce_benchmark(benchmark)
    values = {}
    for column in frame:
        aligned = align_pair(frame[column], benchmark_series)
        if len(aligned) < 2:
            values[column] = float("nan")
            continue
        benchmark_variance = float(aligned["benchmark"].var(ddof=1))
        covariance = float(aligned["portfolio"].cov(aligned["benchmark"]))
        values[column] = safe_ratio(covariance, benchmark_variance)
    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def alpha(
    returns: ReturnInput,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> MetricResult:
    """Calculate annualized arithmetic Jensen alpha."""
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    risk_free_periodic = annual_to_periodic_rate(risk_free_rate, periods_per_year)
    frame, was_series = coerce_returns(returns)
    benchmark_series = coerce_benchmark(benchmark)
    values = {}
    for column in frame:
        aligned = align_pair(frame[column], benchmark_series)
        beta_value = float(beta(aligned["portfolio"], aligned["benchmark"]))
        if not np.isfinite(beta_value):
            values[column] = float("nan")
            continue
        periodic_alpha = (
            aligned["portfolio"].mean()
            - risk_free_periodic
            - beta_value * (aligned["benchmark"].mean() - risk_free_periodic)
        )
        values[column] = float(periodic_alpha * periods_per_year)
    result = pd.Series(values, dtype=float)
    return restore_metric(result, was_series)


def _sample_std(series: pd.Series) -> float:
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1))


def _population_shape(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    centered = values - values.mean()
    sigma = values.std(ddof=0)
    if np.isclose(sigma, 0.0):
        return float("nan"), float("nan")
    population_skew = float(np.mean(centered**3) / sigma**3)
    population_kurtosis = float(np.mean(centered**4) / sigma**4)
    return population_skew, population_kurtosis
