"""Summarize and visualize a synthetic observed market-regime sequence."""

import numpy as np
import pandas as pd

from asset_management_toolkit.market_regime_classification import (
    regime_episodes,
    regime_return_stats,
    regime_transition_matrix,
)
from asset_management_toolkit.visualization import regime_overlay_figure

dates = pd.date_range("2022-01-31", periods=36, freq="ME")
regimes = pd.Series(
    ["Expansion"] * 12 + ["Contraction"] * 8 + ["Recovery"] * 6 + ["Expansion"] * 10,
    index=dates,
    name="observed_regime",
)
generator = np.random.default_rng(42)
regime_means = regimes.map(
    {"Expansion": 0.012, "Contraction": -0.018, "Recovery": 0.016}
)
returns = pd.DataFrame(
    {
        "Equity": regime_means + generator.normal(0.0, 0.025, len(dates)),
        "Bond": 0.004 + generator.normal(0.0, 0.008, len(dates)),
    },
    index=dates,
)

episodes = regime_episodes(regimes)
transitions = regime_transition_matrix(regimes)
statistics = regime_return_stats(returns, regimes, periods_per_year=12)
figure = regime_overlay_figure(returns, regimes)

print("Observed episodes")
print(episodes)
print("\nTransition probabilities")
print(transitions.round(3))
print("\nConditional return statistics")
print(statistics.round(4))
print(f"\nPlotly traces: {len(figure.data)}")
