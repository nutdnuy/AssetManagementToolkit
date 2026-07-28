"""Minimal AssetManagementToolkit risk/return example."""

import pandas as pd

from asset_management_toolkit.analytics import risk_return_stats

portfolio_returns = pd.Series(
    [0.012, -0.008, 0.015, 0.004, -0.006, 0.011],
    index=pd.date_range("2026-01-01", periods=6, freq="B"),
    name="portfolio",
)

benchmark_returns = pd.Series(
    [0.010, -0.006, 0.012, 0.003, -0.005, 0.009],
    index=portfolio_returns.index,
    name="benchmark",
)

stats = risk_return_stats(
    portfolio_returns,
    benchmark=benchmark_returns,
    risk_free_rate=0.02,
    periods_per_year=252,
)

print(stats.round(4).to_string())
