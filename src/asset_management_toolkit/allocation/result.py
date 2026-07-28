"""Result containers for CPPI-family strategies."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CPPIResult:
    """Auditable paths produced by a CPPI-family allocation strategy.

    ``wealth``, ``floor``, and ``cushion`` are end-of-period values. Allocation
    weights, multiplier, turnover, and transaction costs are decisions applied
    at the start of the corresponding period.
    """

    strategy: str
    initial_wealth: float
    wealth: pd.DataFrame
    floor: pd.DataFrame
    cushion: pd.DataFrame
    risky_weight: pd.DataFrame
    safe_weight: pd.DataFrame
    multiplier: pd.DataFrame
    turnover: pd.DataFrame
    transaction_costs: pd.DataFrame
    floor_breach: pd.DataFrame
    cash_locked: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Summarize terminal outcomes, drawdowns, costs, and protection events."""
        wealth_with_start = pd.concat(
            [
                pd.DataFrame(
                    self.initial_wealth,
                    index=[0],
                    columns=self.wealth.columns,
                ),
                self.wealth.reset_index(drop=True),
            ],
            axis=0,
            ignore_index=True,
        )
        drawdowns = wealth_with_start.div(wealth_with_start.cummax()).sub(1.0)

        summary = pd.DataFrame(index=self.wealth.columns)
        summary.index.name = "path"
        summary["terminal_wealth"] = self.wealth.iloc[-1]
        summary["terminal_floor"] = self.floor.iloc[-1]
        summary["total_return"] = summary["terminal_wealth"] / self.initial_wealth - 1.0
        summary["maximum_drawdown"] = drawdowns.min().to_numpy()
        summary["minimum_cushion"] = self.cushion.min()
        summary["average_risky_weight"] = self.risky_weight.mean()
        summary["total_turnover"] = self.turnover.sum()
        summary["total_transaction_cost"] = self.transaction_costs.sum()
        summary["floor_breach_count"] = self.floor_breach.sum().astype(int)
        summary["ever_floor_breach"] = self.floor_breach.any()
        summary["cash_locked_count"] = self.cash_locked.sum().astype(int)
        summary["ever_cash_locked"] = self.cash_locked.any()
        return summary
