"""Combined return and risk summary façade."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from asset_management_toolkit.analytics._validation import (
    ReturnInput,
    coerce_benchmark,
    coerce_returns,
    validate_annual_rate,
    validate_periods_per_year,
    validate_probability,
)
from asset_management_toolkit.analytics.returns import (
    active_return,
    annualized_mean_return,
    annualized_return,
    total_return,
)
from asset_management_toolkit.analytics.risk import (
    alpha,
    annualized_volatility,
    beta,
    calmar_ratio,
    cornish_fisher_var,
    downside_deviation,
    excess_kurtosis,
    gaussian_var,
    historical_cvar,
    historical_var,
    information_ratio,
    max_drawdown,
    semivariance,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tracking_error,
)


def risk_return_stats(
    returns: ReturnInput,
    *,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
    var_level: float = 0.05,
) -> pd.DataFrame:
    """Build one audit-friendly return and risk table.

    Each row represents one return column. Benchmark-relative columns are
    included only when a benchmark Series is provided.
    """
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_annual_rate(
        minimum_acceptable_return,
        "minimum_acceptable_return",
    )
    validate_periods_per_year(periods_per_year)
    validate_probability(var_level, "var_level")
    frame, _ = coerce_returns(returns)

    rows = {}
    for column in frame:
        series = frame[column]
        rows[column] = {
            "n_observations": int(series.notna().sum()),
            "total_return": total_return(series),
            "annualized_return": annualized_return(series, periods_per_year),
            "annualized_mean_return": annualized_mean_return(
                series,
                periods_per_year,
            ),
            "annualized_volatility": annualized_volatility(
                series,
                periods_per_year,
            ),
            "sharpe_ratio": sharpe_ratio(
                series,
                risk_free_rate,
                periods_per_year,
            ),
            "downside_deviation": downside_deviation(
                series,
                minimum_acceptable_return,
                periods_per_year,
            ),
            "semivariance": semivariance(
                series,
                minimum_acceptable_return,
                periods_per_year,
            ),
            "sortino_ratio": sortino_ratio(
                series,
                minimum_acceptable_return,
                periods_per_year,
            ),
            "max_drawdown": max_drawdown(series),
            "calmar_ratio": calmar_ratio(series, periods_per_year),
            "skewness": skewness(series),
            "excess_kurtosis": excess_kurtosis(series),
            "historical_var": historical_var(series, var_level),
            "historical_cvar": historical_cvar(series, var_level),
            "gaussian_var": gaussian_var(series, var_level),
            "cornish_fisher_var": cornish_fisher_var(series, var_level),
        }

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "asset"

    if benchmark is not None:
        benchmark_series = coerce_benchmark(benchmark)
        for column in frame:
            series = frame[column]
            result.loc[column, "active_return"] = active_return(
                series,
                benchmark_series,
                periods_per_year,
            )
            result.loc[column, "tracking_error"] = tracking_error(
                series,
                benchmark_series,
                periods_per_year,
            )
            result.loc[column, "information_ratio"] = information_ratio(
                series,
                benchmark_series,
                periods_per_year,
            )
            result.loc[column, "beta"] = beta(series, benchmark_series)
            result.loc[column, "alpha"] = alpha(
                series,
                benchmark_series,
                risk_free_rate,
                periods_per_year,
            )

    return result
