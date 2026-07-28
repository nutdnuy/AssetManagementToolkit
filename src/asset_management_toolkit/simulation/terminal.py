"""Terminal-wealth calculations for scenario return paths."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from asset_management_toolkit.analytics._validation import (
    ReturnInput,
    coerce_returns,
)
from asset_management_toolkit.simulation._validation import (
    validate_non_negative_real,
    validate_positive_real,
)

TerminalWealthResult = Union[float, pd.Series]


def terminal_wealth(
    returns: ReturnInput,
    initial_wealth: float = 1.0,
) -> TerminalWealthResult:
    """Compound each scenario path into terminal wealth."""
    starting_wealth = validate_positive_real(initial_wealth, "initial_wealth")
    frame, was_series = coerce_returns(returns, default_name="scenario")
    result = starting_wealth * (1.0 + frame).prod(axis=0, skipna=True)
    result = result.astype(float)
    result.name = "terminal_wealth"
    if was_series:
        return float(result.iloc[0])
    return result


def terminal_wealth_stats(
    returns: ReturnInput,
    initial_wealth: float = 1.0,
    *,
    floor_wealth: Optional[float] = None,
    cap_wealth: Optional[float] = None,
) -> pd.Series:
    """Summarize terminal wealth across scenarios.

    Floor and cap values are absolute wealth levels in the same unit as
    ``initial_wealth``. Conditional shortfall and surplus are reported as
    positive magnitudes. Threshold metrics are NaN when their threshold is not
    supplied.
    """
    starting_wealth = validate_positive_real(initial_wealth, "initial_wealth")
    floor = _optional_threshold(floor_wealth, "floor_wealth")
    cap = _optional_threshold(cap_wealth, "cap_wealth")
    if floor is not None and cap is not None and cap <= floor:
        raise ValueError("cap_wealth must be greater than floor_wealth")

    wealth_result = terminal_wealth(returns, starting_wealth)
    if isinstance(wealth_result, pd.Series):
        wealth = wealth_result
    else:
        wealth = pd.Series([wealth_result], index=["scenario"], dtype=float)

    summary = {
        "n_scenarios": int(len(wealth)),
        "mean": float(wealth.mean()),
        "median": float(wealth.median()),
        "standard_deviation": (
            float(wealth.std(ddof=1)) if len(wealth) > 1 else float("nan")
        ),
        "minimum": float(wealth.min()),
        "maximum": float(wealth.max()),
        "floor_wealth": float(floor) if floor is not None else float("nan"),
        "probability_below_floor": float("nan"),
        "expected_shortfall_below_floor": float("nan"),
        "cap_wealth": float(cap) if cap is not None else float("nan"),
        "probability_above_cap": float("nan"),
        "expected_surplus_above_cap": float("nan"),
    }

    if floor is not None:
        below = wealth < floor
        summary["probability_below_floor"] = float(below.mean())
        summary["expected_shortfall_below_floor"] = (
            float((floor - wealth[below]).mean()) if below.any() else 0.0
        )

    if cap is not None:
        above = wealth > cap
        summary["probability_above_cap"] = float(above.mean())
        summary["expected_surplus_above_cap"] = (
            float((wealth[above] - cap).mean()) if above.any() else 0.0
        )

    return pd.Series(summary, name="terminal_wealth_stats")


def _optional_threshold(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    return validate_non_negative_real(value, name)
