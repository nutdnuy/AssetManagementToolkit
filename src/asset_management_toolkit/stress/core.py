"""Probability-free historical and hypothetical portfolio stress testing."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from asset_management_toolkit.stress._validation import (
    validate_return_frame,
    validate_thresholds,
    validate_weights,
)


@dataclass(frozen=True)
class StressTestResult:
    """One-period or aggregated scenario stress-test outputs."""

    summary: pd.DataFrame
    asset_contributions: pd.DataFrame
    threshold_breaches: pd.DataFrame


@dataclass(frozen=True)
class PathStressTestResult:
    """Multi-period stress-test outputs under constant-period weights."""

    summary: pd.DataFrame
    portfolio_returns: pd.DataFrame
    threshold_breaches: pd.DataFrame


def historical_stress_scenarios(
    asset_returns: pd.DataFrame,
    windows: Mapping[str, tuple[Hashable, Hashable]],
) -> pd.DataFrame:
    """Compound labelled asset returns over inclusive historical windows.

    The output contains one probability-free scenario per requested window.
    Scenario rows are directly consumable by :func:`stress_test_portfolio`.
    """
    returns = validate_return_frame(
        asset_returns,
        "asset_returns",
        require_datetime_index=True,
    )
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("windows must be a non-empty mapping")

    rows: dict[str, pd.Series] = {}
    for label, boundaries in windows.items():
        if not isinstance(label, str) or not label.strip():
            raise TypeError("window labels must be non-empty strings")
        if not isinstance(boundaries, tuple) or len(boundaries) != 2:
            raise TypeError("each window must be a (start, end) tuple")
        try:
            start = pd.Timestamp(boundaries[0])
            end = pd.Timestamp(boundaries[1])
        except (TypeError, ValueError) as error:
            raise TypeError("window boundaries must be timestamp-like") from error
        if start > end:
            raise ValueError(f"window {label!r} must have start <= end")

        selected = returns.loc[(returns.index >= start) & (returns.index <= end)]
        if selected.empty:
            raise ValueError(f"window {label!r} contains no observations")
        rows[label] = (1.0 + selected).prod(axis=0) - 1.0

    scenarios = pd.DataFrame.from_dict(rows, orient="index")
    scenarios.index.name = "scenario"
    scenarios.columns.name = asset_returns.columns.name
    return scenarios


def stress_test_portfolio(
    weights: pd.Series,
    scenarios: pd.DataFrame,
    *,
    loss_thresholds: Optional[Mapping[str, float]] = None,
) -> StressTestResult:
    """Evaluate labelled asset-return shocks against a static portfolio.

    Portfolio loss is the signed negative of portfolio return: positive values
    denote losses and negative values denote gains. Thresholds are
    probability-free governance limits, not VaR confidence levels.
    """
    scenario_returns = validate_return_frame(scenarios, "scenarios")
    aligned_weights = validate_weights(weights, scenario_returns.columns)
    thresholds = validate_thresholds(loss_thresholds)

    contributions = scenario_returns.mul(aligned_weights, axis="columns")
    contributions.columns.name = scenario_returns.columns.name
    portfolio_returns = contributions.sum(axis=1)
    portfolio_losses = -portfolio_returns

    if thresholds:
        breaches = pd.DataFrame(
            {
                label: portfolio_losses >= threshold
                for label, threshold in thresholds.items()
            },
            index=scenario_returns.index,
        )
    else:
        breaches = pd.DataFrame(index=scenario_returns.index, dtype=bool)
    breaches.columns.name = "loss_threshold"

    summary = pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "portfolio_loss": portfolio_losses,
            "worst_asset": contributions.idxmin(axis=1),
            "worst_asset_contribution": contributions.min(axis=1),
            "breached_threshold_count": breaches.sum(axis=1).astype(int),
        },
        index=scenario_returns.index,
    )
    summary.index.name = scenario_returns.index.name or "scenario"
    contributions.index.name = summary.index.name
    breaches.index.name = summary.index.name
    return StressTestResult(summary, contributions, breaches)


def stress_test_portfolio_paths(
    weights: pd.Series,
    scenario_paths: Mapping[str, pd.DataFrame],
    *,
    loss_thresholds: Optional[Mapping[str, float]] = None,
) -> PathStressTestResult:
    """Evaluate multi-period return paths with weights reset each period.

    This contract assumes constant-period rebalancing to the supplied weights,
    no transaction costs, and no probability attached to scenario labels.
    Threshold breaches are evaluated against terminal portfolio loss.
    """
    if not isinstance(scenario_paths, Mapping) or not scenario_paths:
        raise ValueError("scenario_paths must be a non-empty mapping")
    if any(not isinstance(label, str) or not label.strip() for label in scenario_paths):
        raise TypeError("scenario path labels must be non-empty strings")
    thresholds = validate_thresholds(loss_thresholds)

    periodic_returns: dict[str, pd.Series] = {}
    summary_rows: dict[str, dict[str, object]] = {}
    expected_assets: Optional[pd.Index] = None

    for label, path in scenario_paths.items():
        clean_path = validate_return_frame(path, f"scenario_paths[{label!r}]")
        if expected_assets is None:
            expected_assets = clean_path.columns
        else:
            missing = expected_assets.difference(clean_path.columns)
            extra = clean_path.columns.difference(expected_assets)
            if not missing.empty or not extra.empty:
                raise ValueError(
                    "all scenario paths must use the same assets; "
                    f"{label!r} missing={missing.tolist()}, extra={extra.tolist()}"
                )
            clean_path = clean_path.reindex(columns=expected_assets)

        aligned_weights = validate_weights(weights, clean_path.columns)
        path_returns = clean_path.mul(aligned_weights, axis="columns").sum(axis=1)
        if (path_returns < -1.0).any():
            raise ValueError(
                f"scenario path {label!r} produces a portfolio return below -1.0"
            )
        wealth = (1.0 + path_returns).cumprod()
        running_peak = (
            pd.concat(
                [pd.Series([1.0]), wealth.reset_index(drop=True)],
                ignore_index=True,
            )
            .cummax()
            .iloc[1:]
            .to_numpy()
        )
        drawdowns = wealth.to_numpy() / running_peak - 1.0
        terminal_return = float(wealth.iloc[-1] - 1.0)

        periodic_returns[label] = path_returns.reset_index(drop=True)
        summary_rows[label] = {
            "n_periods": len(path_returns),
            "terminal_return": terminal_return,
            "portfolio_loss": -terminal_return,
            "maximum_drawdown": float(np.min(drawdowns)),
            "worst_period_return": float(path_returns.min()),
        }

    summary = pd.DataFrame.from_dict(summary_rows, orient="index")
    summary.index.name = "scenario"
    portfolio_returns = pd.DataFrame(periodic_returns)
    portfolio_returns.index.name = "period"
    portfolio_returns.columns.name = "scenario"

    if thresholds:
        losses = summary["portfolio_loss"]
        breaches = pd.DataFrame(
            {label: losses >= threshold for label, threshold in thresholds.items()},
            index=summary.index,
        )
    else:
        breaches = pd.DataFrame(index=summary.index, dtype=bool)
    breaches.columns.name = "loss_threshold"
    summary["breached_threshold_count"] = breaches.sum(axis=1).astype(int)
    return PathStressTestResult(summary, portfolio_returns, breaches)
