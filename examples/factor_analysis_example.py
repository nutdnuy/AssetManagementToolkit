"""Run labelled factor regression, rolling exposure, and attribution examples."""

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics import (
    factor_regression,
    factor_return_attribution,
    regularized_factor_regression,
    rolling_factor_regression,
    rolling_returns,
)

dates = pd.date_range("2021-01-31", periods=36, freq="ME")
sequence = np.arange(len(dates))
factors = pd.DataFrame(
    {
        "Market": 0.025 * np.sin(sequence / 2.0),
        "Value": 0.018 * np.cos(sequence / 3.0),
        "Momentum": 0.015 * np.sin(sequence / 5.0 + 0.4),
    },
    index=dates,
)
fund_returns = pd.Series(
    0.001
    + 0.85 * factors["Market"]
    + 0.30 * factors["Value"]
    - 0.15 * factors["Momentum"]
    + 0.0015 * np.cos(sequence * 1.7),
    index=dates,
    name="Synthetic Fund",
)

model = factor_regression(
    fund_returns,
    factors,
    periods_per_year=12,
)
rolling = rolling_factor_regression(
    fund_returns,
    factors,
    window=18,
    step=6,
    periods_per_year=12,
)
attribution = factor_return_attribution(
    model.betas,
    factors,
    alpha=model.alpha / 12.0,
)
trailing_12_month_return = rolling_returns(fund_returns, window=12)
regularized = regularized_factor_regression(
    fund_returns,
    factors,
    method="ridge",
    cv=4,
    periods_per_year=12,
)

print("Full-sample coefficients")
print(model.coefficients.round(4))
print("\nFit diagnostics")
print(
    pd.Series(
        {
            "r_squared": model.r_squared,
            "adjusted_r_squared": model.adjusted_r_squared,
            "residual_volatility": model.residual_volatility,
        }
    ).round(4)
)
print("\nRolling factor exposures")
print(rolling.betas.round(4))
print("\nLatest model-implied attribution")
print(attribution.tail(1).round(4))
print("\nLatest trailing 12-month return")
print(round(float(trailing_12_month_return.iloc[-1]), 4))
print("\nChronologically selected Ridge exposures")
print(regularized.coefficients.round(4))
