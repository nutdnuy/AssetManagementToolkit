"""Observed-versus-simulated return-distribution diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from asset_management_toolkit.analytics._validation import (
    ReturnInput,
    validate_periods_per_year,
    validate_probability,
)
from asset_management_toolkit.simulation._observations import (
    flatten_simple_returns,
)


def return_distribution_diagnostics(
    returns: ReturnInput,
    periods_per_year: int = 252,
    *,
    tail_probability: float = 0.05,
) -> pd.Series:
    """Summarize periodic return shape, tails, and annualized log moments."""
    validate_periods_per_year(periods_per_year)
    validate_probability(tail_probability, "tail_probability")
    observations = flatten_simple_returns(returns, min_observations=4)
    if np.any(observations <= -1.0):
        raise ValueError("distribution diagnostics require returns above -1.0")
    quantiles = np.quantile(
        observations,
        [0.01, tail_probability, 0.50, 0.95, 0.99],
    )
    tail = observations[observations <= quantiles[1]]
    log_returns = np.log1p(observations)

    return pd.Series(
        {
            "n_observations": int(observations.size),
            "periodic_mean": float(np.mean(observations)),
            "periodic_standard_deviation": float(np.std(observations, ddof=1)),
            "sample_skewness": float(stats.skew(observations, bias=False)),
            "sample_excess_kurtosis": float(
                stats.kurtosis(observations, fisher=True, bias=False)
            ),
            "minimum": float(np.min(observations)),
            "q01": float(quantiles[0]),
            "q_tail": float(quantiles[1]),
            "median": float(quantiles[2]),
            "q95": float(quantiles[3]),
            "q99": float(quantiles[4]),
            "maximum": float(np.max(observations)),
            "historical_var": float(-quantiles[1]),
            "historical_cvar": float(-np.mean(tail)),
            "annualized_mean_log_return": float(
                np.mean(log_returns) * periods_per_year
            ),
            "annualized_log_volatility": float(
                np.std(log_returns, ddof=1) * np.sqrt(periods_per_year)
            ),
        },
        name="return_distribution_diagnostics",
    )


def compare_simulation_models(
    observed_returns: pd.Series,
    simulated_returns: Mapping[str, ReturnInput],
    periods_per_year: int = 252,
    *,
    tail_probability: float = 0.05,
) -> pd.DataFrame:
    """Compare simulated periodic distributions with observed returns.

    The error score combines the two-sample Kolmogorov-Smirnov statistic with
    Wasserstein distance normalized by observed sample volatility. It is a
    descriptive ranking aid, not a formal model-selection theorem.
    """
    if not isinstance(observed_returns, pd.Series):
        raise TypeError("observed_returns must be a pandas Series")
    if not isinstance(simulated_returns, Mapping) or not simulated_returns:
        raise ValueError("simulated_returns must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in simulated_returns):
        raise TypeError("simulated_returns keys must be non-empty strings")
    if "Observed" in simulated_returns:
        raise ValueError("simulated_returns cannot use the reserved name 'Observed'")

    observed = flatten_simple_returns(observed_returns, min_observations=4)
    observed_diagnostics = return_distribution_diagnostics(
        observed_returns,
        periods_per_year,
        tail_probability=tail_probability,
    )
    observed_scale = max(
        float(observed_diagnostics["periodic_standard_deviation"]),
        np.finfo(float).eps,
    )

    rows = {
        "Observed": {
            **observed_diagnostics.to_dict(),
            "mean_error": 0.0,
            "volatility_error": 0.0,
            "q01_error": 0.0,
            "q_tail_error": 0.0,
            "ks_statistic": 0.0,
            "wasserstein_distance": 0.0,
            "distribution_error_score": 0.0,
        }
    }
    for name, returns in simulated_returns.items():
        simulated = flatten_simple_returns(returns, min_observations=4)
        diagnostics = return_distribution_diagnostics(
            returns,
            periods_per_year,
            tail_probability=tail_probability,
        )
        ks_statistic = float(stats.ks_2samp(observed, simulated).statistic)
        wasserstein = float(stats.wasserstein_distance(observed, simulated))
        rows[name] = {
            **diagnostics.to_dict(),
            "mean_error": float(
                diagnostics["periodic_mean"] - observed_diagnostics["periodic_mean"]
            ),
            "volatility_error": float(
                diagnostics["periodic_standard_deviation"]
                - observed_diagnostics["periodic_standard_deviation"]
            ),
            "q01_error": float(diagnostics["q01"] - observed_diagnostics["q01"]),
            "q_tail_error": float(
                diagnostics["q_tail"] - observed_diagnostics["q_tail"]
            ),
            "ks_statistic": ks_statistic,
            "wasserstein_distance": wasserstein,
            "distribution_error_score": float(
                ks_statistic + wasserstein / observed_scale
            ),
        }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "model"
    return result
