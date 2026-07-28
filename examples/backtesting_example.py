"""Minimal weight-based backtesting example."""

import pandas as pd

from asset_management_toolkit.analytics import drawdown_episodes, risk_return_stats
from asset_management_toolkit.backtesting import run_weight_backtest
from asset_management_toolkit.portfolio import risk_contributions

asset_returns = pd.DataFrame(
    {
        "bond": [0.005, 0.002, -0.001, 0.004],
        "equity": [0.020, -0.010, 0.015, 0.005],
    },
    index=pd.date_range("2026-01-01", periods=4, freq="D"),
)
target_weights = pd.DataFrame(
    [[0.4, 0.6], [0.5, 0.5]],
    index=asset_returns.index[[0, 2]],
    columns=asset_returns.columns,
)

result = run_weight_backtest(
    asset_returns,
    target_weights,
    transaction_cost_rate=0.001,
)
statistics = risk_return_stats(result.portfolio_returns, periods_per_year=252)
episodes = drawdown_episodes(result.portfolio_returns)
risk_budget = risk_contributions(
    result.weights_at_start.iloc[-1],
    asset_returns.cov(),
)

print(statistics.round(4).to_string())
print(episodes.to_string(index=False))
print(risk_budget.round(4).to_string())
