"""Synthetic returns-based style analysis with no external data."""

import pandas as pd

from asset_management_toolkit.analytics import (
    rolling_style_exposures,
    style_exposures,
)

dates = pd.date_range("2024-01-31", periods=12, freq="ME")
style_returns = pd.DataFrame(
    {
        "equity": [
            0.030,
            -0.020,
            0.015,
            0.040,
            -0.010,
            0.025,
            0.005,
            -0.015,
            0.035,
            0.010,
            -0.005,
            0.020,
        ],
        "bond": [
            0.005,
            0.010,
            -0.002,
            0.004,
            0.012,
            0.003,
            0.008,
            0.006,
            -0.001,
            0.007,
            0.009,
            0.002,
        ],
    },
    index=dates,
)
fund_returns = (
    0.70 * style_returns["equity"] + 0.30 * style_returns["bond"] + 0.001
).rename("fund")

full_sample = style_exposures(fund_returns, style_returns)
rolling = rolling_style_exposures(
    fund_returns,
    style_returns,
    window=6,
    step=3,
)

print(full_sample.weights)
print(full_sample.r_squared)
print(rolling.weights)
