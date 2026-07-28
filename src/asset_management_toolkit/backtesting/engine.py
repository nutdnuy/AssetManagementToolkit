"""Deterministic weight-based portfolio backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import coerce_returns
from asset_management_toolkit.backtesting.result import BacktestResult


def run_weight_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    initial_nav: float = 1.0,
    transaction_cost_rate: float = 0.0,
    charge_initial_trade: bool = False,
) -> BacktestResult:
    """Backtest long-only target weights applied at the start of each period.

    Target-weight dates are rebalance dates and must be present in
    ``asset_returns``. Between rebalances, weights drift with asset returns.
    Rebalance turnover is half the absolute weight change. The initial
    allocation reports turnover of one; its cost is charged only when
    ``charge_initial_trade=True``.
    """
    returns = _validate_returns(asset_returns)
    weights = _validate_weights(target_weights, returns)
    initial_value = _positive_real(initial_nav, "initial_nav")
    cost_rate = _transaction_cost_rate(transaction_cost_rate)
    if not isinstance(charge_initial_trade, bool):
        raise TypeError("charge_initial_trade must be a boolean")

    assets = returns.columns
    current_weights = np.zeros(len(assets), dtype=float)
    current_nav = initial_value
    records = {
        "portfolio_returns": [],
        "gross_returns": [],
        "nav": [],
        "weights_before": [],
        "weights_at_start": [],
        "weights_after": [],
        "trades": [],
        "turnover": [],
        "transaction_costs": [],
    }

    for position, (date, asset_return_row) in enumerate(returns.iterrows()):
        before = current_weights.copy()
        trade = np.zeros(len(assets), dtype=float)
        turnover = 0.0
        charge_cost = True

        if date in weights.index:
            start = weights.loc[date].to_numpy(dtype=float)
            trade = start - before
            if position == 0:
                turnover = float(np.abs(trade).sum())
                charge_cost = charge_initial_trade
            else:
                turnover = float(0.5 * np.abs(trade).sum())
        else:
            start = before

        cost_fraction = cost_rate * turnover if charge_cost else 0.0
        if cost_fraction >= 1.0:
            raise ValueError("transaction costs consume all portfolio wealth")
        transaction_cost = current_nav * cost_fraction
        nav_after_cost = current_nav - transaction_cost

        asset_values = 1.0 + asset_return_row.to_numpy(dtype=float)
        gross_return = float(start @ asset_return_row.to_numpy(dtype=float))
        growth = 1.0 + gross_return
        if growth <= 0.0:
            raise ValueError(
                f"portfolio wealth is fully depleted by returns at index {date!r}"
            )
        ending_nav = nav_after_cost * growth
        net_return = ending_nav / current_nav - 1.0
        after = start * asset_values / growth

        records["portfolio_returns"].append(net_return)
        records["gross_returns"].append(gross_return)
        records["nav"].append(ending_nav)
        records["weights_before"].append(before)
        records["weights_at_start"].append(start)
        records["weights_after"].append(after)
        records["trades"].append(trade)
        records["turnover"].append(turnover)
        records["transaction_costs"].append(transaction_cost)
        current_nav = ending_nav
        current_weights = after

    index = returns.index.copy()
    return BacktestResult(
        portfolio_returns=pd.Series(
            records["portfolio_returns"],
            index=index,
            name="portfolio_return",
            dtype=float,
        ),
        gross_returns=pd.Series(
            records["gross_returns"],
            index=index,
            name="gross_return",
            dtype=float,
        ),
        nav=pd.Series(records["nav"], index=index, name="nav", dtype=float),
        weights_before=_frame(records["weights_before"], index, assets),
        weights_at_start=_frame(records["weights_at_start"], index, assets),
        weights_after=_frame(records["weights_after"], index, assets),
        trades=_frame(records["trades"], index, assets),
        turnover=pd.Series(
            records["turnover"],
            index=index,
            name="turnover",
            dtype=float,
        ),
        transaction_costs=pd.Series(
            records["transaction_costs"],
            index=index,
            name="transaction_cost",
            dtype=float,
        ),
    )


def _validate_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a pandas DataFrame")
    frame, _ = coerce_returns(asset_returns)
    if frame.isna().any().any():
        raise ValueError("asset_returns must not contain missing values")
    _validate_index(frame.index, "asset_returns")
    return frame


def _validate_weights(
    target_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(target_weights, pd.DataFrame):
        raise TypeError("target_weights must be a pandas DataFrame")
    if target_weights.empty:
        raise ValueError("target_weights must contain at least one row")
    if not target_weights.columns.is_unique:
        raise ValueError("target_weights columns must be unique")
    if set(target_weights.columns) != set(asset_returns.columns):
        raise ValueError("target_weights columns must match asset_returns columns")
    weights = target_weights.loc[:, asset_returns.columns].copy(deep=True)
    non_numeric = [
        str(column)
        for column in weights
        if not pd.api.types.is_numeric_dtype(weights[column])
    ]
    if non_numeric:
        raise TypeError("target_weights columns must be numeric")
    weights = weights.astype(float)
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("target_weights must contain only finite values")
    if (weights < -1e-12).any().any() or (weights > 1.0 + 1e-12).any().any():
        raise ValueError(
            "target_weights must be long-only weights between zero and one"
        )
    if not np.allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-10):
        raise ValueError("each target_weights row must sum to one")
    _validate_index(weights.index, "target_weights")
    if weights.index[0] != asset_returns.index[0]:
        raise ValueError(
            "the first target_weights date must equal the first asset_returns date"
        )
    if not weights.index.isin(asset_returns.index).all():
        raise ValueError("every target_weights date must exist in asset_returns")
    return weights


def _validate_index(index: pd.Index, name: str) -> None:
    if not index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted in increasing order")


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


def _transaction_cost_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError("transaction_cost_rate must be a real number")
    if not np.isfinite(value) or not 0.0 <= float(value) < 1.0:
        raise ValueError(
            "transaction_cost_rate must be finite and between zero and one"
        )
    return float(value)


def _frame(
    values: list[np.ndarray],
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=columns, dtype=float)
