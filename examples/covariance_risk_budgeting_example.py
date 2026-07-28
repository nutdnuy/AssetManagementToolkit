"""Run covariance, weighting, and equal-risk-contribution examples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.estimation import shrink_covariance
from asset_management_toolkit.portfolio import (
    capitalization_weights,
    equal_risk_contribution_weights,
    risk_contributions,
)

rng = np.random.default_rng(20260728)
dates = pd.date_range("2020-01-31", periods=72, freq="ME")
returns = pd.DataFrame(
    rng.multivariate_normal(
        mean=[0.006, 0.002, 0.004],
        cov=[
            [0.0025, 0.0002, 0.0005],
            [0.0002, 0.0004, 0.0001],
            [0.0005, 0.0001, 0.0012],
        ],
        size=len(dates),
    ),
    index=dates,
    columns=["equity", "bonds", "real_assets"],
)

covariance = shrink_covariance(returns, intensity=0.4)
erc_weights = equal_risk_contribution_weights(covariance)
erc_contributions = risk_contributions(erc_weights, covariance)
cap_weights = capitalization_weights(
    pd.Series(
        {"equity": 700.0, "bonds": 200.0, "real_assets": 100.0},
    )
)

print("Shrinkage covariance")
print(covariance.round(6))
print("\nEqual-risk-contribution weights")
print(erc_weights.round(4))
print("\nAchieved normalized risk contributions")
print(erc_contributions.round(4))
print("\nCapitalization weights")
print(cap_weights.round(4))
