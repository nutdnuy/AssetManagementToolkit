"""Result containers for time-series analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesFold:
    """One chronological train/test fold."""

    fold: int
    train: pd.Series
    test: pd.Series

    @property
    def train_start(self) -> object:
        """First training label."""
        return self.train.index[0]

    @property
    def train_end(self) -> object:
        """Last training label."""
        return self.train.index[-1]

    @property
    def test_start(self) -> object:
        """First test label."""
        return self.test.index[0]

    @property
    def test_end(self) -> object:
        """Last test label."""
        return self.test.index[-1]


@dataclass(frozen=True)
class ForecastResult:
    """Fitted values, residuals, and labelled out-of-sample forecasts."""

    model: str
    fitted_values: pd.Series
    residuals: pd.Series
    forecast: pd.Series
    parameters: dict[str, object]
    aic: Optional[float] = None
    bic: Optional[float] = None


@dataclass(frozen=True)
class DecompositionResult:
    """Observed, trend, seasonal, and residual decomposition components."""

    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    model: str
    period: int
