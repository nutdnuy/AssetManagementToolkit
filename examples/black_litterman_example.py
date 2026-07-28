"""Synthetic Black–Litterman example with no external data."""

import pandas as pd

from asset_management_toolkit.portfolio import (
    black_litterman_posterior,
    maximum_sharpe_ratio,
)

assets = ["Equity", "Bond", "Gold"]
market_weights = pd.Series([0.50, 0.30, 0.20], index=assets)
covariance = pd.DataFrame(
    [
        [0.0400, 0.0060, 0.0040],
        [0.0060, 0.0100, 0.0015],
        [0.0040, 0.0015, 0.0225],
    ],
    index=assets,
    columns=assets,
)
pick_matrix = pd.DataFrame(
    [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    index=["Equity outperforms Bond", "Gold absolute"],
    columns=assets,
)
views = pd.Series(
    [0.04, 0.03],
    index=pick_matrix.index,
)

posterior = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    views,
    risk_aversion=2.5,
    tau=0.05,
)
weights = maximum_sharpe_ratio(
    0.0,
    posterior.posterior_returns,
    posterior.posterior_covariance,
)

print(posterior.posterior_returns)
print(weights)
