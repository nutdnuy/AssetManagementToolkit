"""Drawdown paths and episode-level diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import coerce_returns


def drawdown_path(
    returns: pd.Series,
    *,
    initial_wealth: float = 1.0,
) -> pd.DataFrame:
    """Return wealth, running peak, and drawdown for one return series."""
    if not isinstance(initial_wealth, (int, float, np.number)) or isinstance(
        initial_wealth, bool
    ):
        raise TypeError("initial_wealth must be a real number")
    if not np.isfinite(initial_wealth) or initial_wealth <= 0.0:
        raise ValueError("initial_wealth must be finite and greater than zero")
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    frame, _ = coerce_returns(returns)
    clean = frame.iloc[:, 0].dropna()
    wealth = float(initial_wealth) * (1.0 + clean).cumprod()
    previous_peak = wealth.cummax().clip(lower=float(initial_wealth))
    drawdown = wealth / previous_peak - 1.0
    return pd.DataFrame(
        {
            "wealth": wealth,
            "previous_peak": previous_peak,
            "drawdown": drawdown,
        }
    )


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Summarize every underwater episode in one return series.

    An episode begins with the first observation below its previous wealth
    peak and ends on the first observation that regains that peak. An
    unrecovered final episode has a missing ``recovery_date``.
    """
    path = drawdown_path(returns)
    episodes: list[dict[str, object]] = []
    active_start = None

    for date, value in path["drawdown"].items():
        underwater = float(value) < -1e-12
        if active_start is None and underwater:
            active_start = date
        elif active_start is not None and not underwater:
            episodes.append(_episode_record(path, active_start, date))
            active_start = None

    if active_start is not None:
        episodes.append(_episode_record(path, active_start, None))

    columns = [
        "start_date",
        "trough_date",
        "recovery_date",
        "drawdown",
        "decline_periods",
        "recovery_periods",
        "total_periods",
        "recovered",
    ]
    return pd.DataFrame(episodes, columns=columns)


def _episode_record(
    path: pd.DataFrame,
    start_date: object,
    recovery_date: object,
) -> dict[str, object]:
    if recovery_date is None:
        episode = path.loc[start_date:]
    else:
        episode = path.loc[start_date:recovery_date]

    trough_date = episode["drawdown"].idxmin()
    decline_periods = int(
        path.index.get_loc(trough_date) - path.index.get_loc(start_date) + 1
    )
    if recovery_date is None:
        recovery_periods = None
        total_periods = int(len(episode))
    else:
        recovery_periods = int(
            path.index.get_loc(recovery_date) - path.index.get_loc(trough_date)
        )
        total_periods = int(
            path.index.get_loc(recovery_date) - path.index.get_loc(start_date) + 1
        )

    return {
        "start_date": start_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "drawdown": float(episode.loc[trough_date, "drawdown"]),
        "decline_periods": decline_periods,
        "recovery_periods": recovery_periods,
        "total_periods": total_periods,
        "recovered": recovery_date is not None,
    }
