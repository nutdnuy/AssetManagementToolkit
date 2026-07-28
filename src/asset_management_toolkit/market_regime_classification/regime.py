"""Labelled diagnostics for an observed market-regime sequence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import (
    ReturnInput,
    coerce_returns,
    validate_periods_per_year,
)
from asset_management_toolkit.analytics.returns import annualized_return
from asset_management_toolkit.analytics.risk import (
    annualized_volatility,
    max_drawdown,
)


def regime_episodes(regimes: pd.Series) -> pd.DataFrame:
    """Return contiguous observed regime episodes with labelled boundaries."""
    labels = _validated_regimes(regimes)
    changed = labels.ne(labels.shift())
    episode_ids = changed.cumsum()
    rows = []
    for _, episode in labels.groupby(episode_ids, sort=False):
        rows.append(
            {
                "regime": episode.iloc[0],
                "start": episode.index[0],
                "end": episode.index[-1],
                "n_observations": len(episode),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["regime", "start", "end", "n_observations"],
    )


def regime_transition_matrix(
    regimes: pd.Series,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """Count or row-normalize observed one-step regime transitions."""
    labels = _validated_regimes(regimes)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a boolean")
    values = list(pd.unique(labels))
    row_order = pd.Index(values, name="from_regime")
    column_order = pd.Index(values, name="to_regime")
    transitions = pd.crosstab(
        labels.iloc[:-1].rename("from_regime").reset_index(drop=True),
        labels.iloc[1:].rename("to_regime").reset_index(drop=True),
        dropna=False,
    ).reindex(index=row_order, columns=column_order, fill_value=0)
    transitions.index.name = "from_regime"
    transitions.columns.name = "to_regime"
    if not normalize:
        return transitions.astype(int)
    totals = transitions.sum(axis=1).replace(0, np.nan)
    return transitions.div(totals, axis=0).fillna(0.0).astype(float)


def regime_return_stats(
    returns: ReturnInput,
    regimes: pd.Series,
    *,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Summarize asset returns conditional on observed regime labels."""
    validate_periods_per_year(periods_per_year)
    frame, _ = coerce_returns(returns)
    if not frame.index.is_unique:
        raise ValueError("returns index must be unique")
    labels = _validated_regimes(regimes)
    overlap = frame.index.intersection(labels.index, sort=False)
    if overlap.empty:
        raise ValueError("returns and regimes have no overlapping index")
    frame = frame.reindex(overlap)
    labels = labels.reindex(overlap)

    rows = []
    for regime in pd.unique(labels):
        mask = labels.eq(regime)
        for asset in frame:
            sample = frame.loc[mask, asset].dropna()
            if sample.empty:
                continue
            rows.append(
                {
                    "regime": regime,
                    "asset": asset,
                    "n_observations": len(sample),
                    "annualized_return": annualized_return(
                        sample,
                        periods_per_year=periods_per_year,
                    ),
                    "annualized_volatility": annualized_volatility(
                        sample,
                        periods_per_year=periods_per_year,
                    ),
                    "max_drawdown": max_drawdown(sample),
                }
            )
    if not rows:
        raise ValueError("returns and regimes have no usable observations")
    return pd.DataFrame(rows).set_index(["regime", "asset"])


def _validated_regimes(regimes: pd.Series) -> pd.Series:
    if not isinstance(regimes, pd.Series):
        raise TypeError("regimes must be a pandas Series")
    if regimes.empty:
        raise ValueError("regimes must not be empty")
    if not regimes.index.is_unique:
        raise ValueError("regimes index must be unique")
    if not regimes.index.is_monotonic_increasing:
        raise ValueError("regimes index must be sorted in increasing order")
    if regimes.isna().any():
        raise ValueError("regimes must not contain missing values")
    return regimes.copy(deep=True).rename(regimes.name or "regime")
