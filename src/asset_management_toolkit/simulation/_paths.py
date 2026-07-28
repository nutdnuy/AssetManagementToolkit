"""Internal helpers for converting return scenarios to price paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.simulation._validation import validate_positive_real


def prices_from_returns(
    returns: pd.DataFrame,
    initial_price: float,
) -> pd.DataFrame:
    """Compound simple-return scenarios from an explicit step-zero price."""
    starting_price = validate_positive_real(initial_price, "initial_price")
    prices = starting_price * (1.0 + returns).cumprod()
    price_values = prices.to_numpy()
    if not np.isfinite(price_values).all() or np.any(price_values <= 0.0):
        raise OverflowError("simulation parameters produced invalid prices")
    initial = pd.DataFrame(
        [np.repeat(starting_price, prices.shape[1])],
        index=pd.Index([0], name="step"),
        columns=prices.columns,
    )
    return pd.concat([initial, prices])
