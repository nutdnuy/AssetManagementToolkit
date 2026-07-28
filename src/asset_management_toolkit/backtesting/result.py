"""Structured results returned by portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    """Auditable outputs from a weight-based portfolio backtest."""

    portfolio_returns: pd.Series
    gross_returns: pd.Series
    nav: pd.Series
    weights_before: pd.DataFrame
    weights_at_start: pd.DataFrame
    weights_after: pd.DataFrame
    trades: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
