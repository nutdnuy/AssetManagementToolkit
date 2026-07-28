"""Empirical gap-risk diagnostics for CPPI scenario results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_management_toolkit.allocation._validation import validate_real
from asset_management_toolkit.allocation.result import CPPIResult


@dataclass(frozen=True)
class CPPIGapRiskResult:
    """Scenario-level losses and aggregate gap-risk statistics."""

    confidence_level: float
    scenario_losses: pd.DataFrame
    statistics: pd.Series


def analyze_cppi_gap_risk(
    result: CPPIResult,
    *,
    confidence_level: float = 0.95,
) -> CPPIGapRiskResult:
    """Measure empirical floor-hit probability and losses across CPPI paths.

    The function summarizes supplied scenarios; it does not assign
    probabilities to generated paths or reproduce the paper's analytic Lévy
    and Fourier-inversion formulas.
    """
    if not isinstance(result, CPPIResult):
        raise TypeError("result must be a CPPIResult")
    confidence = validate_real(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    terminal_wealth = result.wealth.iloc[-1]
    terminal_floor = result.floor.iloc[-1]
    terminal_shortfall = (terminal_floor - terminal_wealth).clip(lower=0.0)
    maximum_floor_shortfall = (result.floor - result.wealth).clip(lower=0.0).max()
    breached = result.floor_breach.any()

    first_breach_period: list[object] = []
    first_breach_shortfall: list[float] = []
    for path in result.wealth.columns:
        breach_path = result.floor_breach[path]
        if breach_path.any():
            period = breach_path.index[int(np.flatnonzero(breach_path.to_numpy())[0])]
            first_breach_period.append(period)
            first_breach_shortfall.append(
                float(result.floor.loc[period, path] - result.wealth.loc[period, path])
            )
        else:
            first_breach_period.append(None)
            first_breach_shortfall.append(0.0)

    scenario_losses = pd.DataFrame(
        {
            "floor_breached": breached.astype(bool),
            "first_breach_period": first_breach_period,
            "first_breach_shortfall": first_breach_shortfall,
            "terminal_wealth": terminal_wealth,
            "terminal_floor": terminal_floor,
            "terminal_shortfall": terminal_shortfall,
            "maximum_floor_shortfall": maximum_floor_shortfall,
        },
        index=result.wealth.columns,
    )
    scenario_losses.index.name = "path"

    breached_losses = terminal_shortfall[breached]
    conditional_loss = (
        float(breached_losses.mean()) if not breached_losses.empty else np.nan
    )
    loss_quantile = float(
        terminal_shortfall.quantile(confidence, interpolation="higher")
    )
    statistics = pd.Series(
        {
            "scenario_count": int(len(scenario_losses)),
            "floor_breach_probability": float(breached.mean()),
            "expected_terminal_shortfall": float(terminal_shortfall.mean()),
            "expected_terminal_shortfall_given_breach": conditional_loss,
            "terminal_shortfall_quantile": loss_quantile,
            "expected_maximum_floor_shortfall": float(maximum_floor_shortfall.mean()),
            "worst_terminal_shortfall": float(terminal_shortfall.max()),
        },
        name="gap_risk",
    )

    return CPPIGapRiskResult(
        confidence_level=confidence,
        scenario_losses=scenario_losses,
        statistics=statistics,
    )
