"""Build the executable CPPI-family tutorial without touching other notebooks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tutorials" / "11_cppi_family_strategies.ipynb"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Level 4C — CPPI, GOPI, and Jump Gap Risk

**Learning goals**

1. distinguish fixed-maturity, open-ended, TIPP, dynamic-multiplier CPPI,
   and Growth-Optimal Portfolio Insurance (GOPI);
2. inspect floor, cushion, allocation, turnover, and breach paths;
3. separate `m=1` and `m=3` configurations from strategy-family definitions;
4. understand a locally risky reserve asset and the growth-optimal multiplier;
5. measure empirical floor-hit probability and loss across jump scenarios.

The return path is synthetic. These scenarios are research demonstrations, not
forecasts or investment recommendations.
"""
    ),
    markdown(
        """
## 1. Setup

The example uses monthly decimal simple returns and a deterministic annual safe
rate. Each allocation decision is made before observing that month's return.
"""
    ),
    code(
        """
from pathlib import Path
import sys

import pandas as pd

project_root = Path.cwd()
if not (project_root / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

pd.options.display.float_format = "{:.4f}".format

from asset_management_toolkit.allocation import (
    analyze_cppi_gap_risk,
    growth_optimal_multiplier,
    run_dynamic_multiplier_cppi,
    run_fixed_maturity_cppi,
    run_growth_optimal_cppi,
    run_open_ended_cppi,
    run_tipp,
)
"""
    ),
    code(
        """
dates = pd.date_range("2024-01-31", periods=36, freq="ME")
risky_returns = pd.Series(
    [
        0.04, 0.03, -0.02, 0.05, 0.01, -0.04,
        0.02, 0.03, 0.01, -0.02, 0.04, 0.02,
        -0.08, -0.12, -0.18, 0.06, -0.05, 0.03,
        0.08, 0.07, 0.05, 0.03, -0.01, 0.04,
        0.02, 0.01, -0.03, 0.05, 0.04, -0.02,
        0.03, 0.02, 0.01, -0.01, 0.04, 0.02,
    ],
    index=dates,
    name="synthetic_risky_asset",
)
risky_returns.to_frame().head()
"""
    ),
    markdown(
        """
## 2. Fixed-maturity CPPI

The final observation is maturity. The 80% terminal guarantee is discounted at
the safe rate. `m=1` and `m=3` are configurations of this same strategy.
"""
    ),
    code(
        """
fixed_m1 = run_fixed_maturity_cppi(
    risky_returns,
    multiplier=1.0,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)
fixed_m3 = run_fixed_maturity_cppi(
    risky_returns,
    multiplier=3.0,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)

pd.concat(
    {
        "m=1": fixed_m1.summary().iloc[0],
        "m=3": fixed_m3.summary().iloc[0],
    },
    axis=1,
)
"""
    ),
    markdown(
        """
## 3. Open-ended CPPI

This strategy has no terminal guarantee date. Its safe-rate floor resets upward
once every 12 observations and never resets downward.
"""
    ),
    code(
        """
open_ended = run_open_ended_cppi(
    risky_returns,
    multiplier=3.0,
    floor_fraction=0.80,
    reset_every=12,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)
open_ended.summary()
"""
    ),
    markdown(
        """
## 4. TIPP

TIPP ratchets its floor every period to at least 80% of the portfolio's
high-water mark. This is more responsive than a scheduled open-ended reset.
"""
    ),
    code(
        """
tipp = run_tipp(
    risky_returns,
    multiplier=3.0,
    protection_ratio=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)
tipp.summary()
"""
    ),
    markdown(
        """
## 5. Dynamic-multiplier CPPI

The multiplier uses only lagged observations. It falls when realized
volatility rises and is clipped between 1 and 5. The default volatility
exponent is one. Cont and Tankov's alpha-stable jump-hazard scaling can be
represented with `volatility_exponent=2/alpha`.
"""
    ),
    code(
        """
dynamic = run_dynamic_multiplier_cppi(
    risky_returns,
    base_multiplier=3.0,
    target_volatility=0.15,
    lookback=6,
    minimum_history=4,
    minimum_multiplier=1.0,
    maximum_multiplier=5.0,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)

stable_tail_index = 1.5
stable_hazard_dynamic = run_dynamic_multiplier_cppi(
    risky_returns,
    base_multiplier=3.0,
    target_volatility=0.15,
    lookback=6,
    minimum_history=4,
    minimum_multiplier=1.0,
    maximum_multiplier=5.0,
    volatility_exponent=2.0 / stable_tail_index,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)

pd.concat(
    [
        risky_returns.rename("risky_return"),
        dynamic.multiplier.iloc[:, 0].rename("inverse_vol_multiplier"),
        stable_hazard_dynamic.multiplier.iloc[:, 0].rename(
            "stable_hazard_multiplier"
        ),
    ],
    axis=1,
).iloc[10:22]
"""
    ),
    markdown(
        """
## 6. Growth-Optimal Portfolio Insurance (GOPI)

Mantilla-García's multiplier accounts for both the expected growth difference
and the variance of the risky asset relative to a locally risky reserve:

```text
m* = (g_risky - g_reserve + 0.5 * relative variance)
     / relative variance
```

With a risk-free reserve it reduces to expected excess return divided by risky
variance. Here the reserve is a synthetic duration-matched bond path. The
expected-return, volatility, and correlation inputs are point-in-time research
assumptions available before allocation—not values estimated from future data.
"""
    ),
    code(
        """
reserve_returns = pd.Series(
    [
        0.006, 0.004, -0.003, 0.005, 0.002, 0.001,
        0.004, -0.002, 0.003, 0.002, 0.001, 0.004,
        0.008, 0.012, 0.018, -0.004, 0.006, 0.002,
        -0.006, -0.004, 0.001, 0.003, 0.002, 0.004,
        0.003, 0.001, 0.005, -0.002, 0.002, 0.003,
        0.001, 0.002, 0.004, 0.003, 0.002, 0.001,
    ],
    index=dates,
    name="synthetic_reserve_bond",
)

constant_gopi_multiplier = growth_optimal_multiplier(
    expected_risky_return=0.08,
    expected_reserve_return=0.03,
    risky_volatility=0.18,
    reserve_volatility=0.06,
    correlation=0.25,
)
constant_gopi_multiplier
"""
    ),
    code(
        """
expected_risky_return = pd.Series(0.08, index=dates)
expected_risky_return.iloc[12:18] = 0.11

gopi = run_growth_optimal_cppi(
    risky_returns,
    reserve_returns,
    expected_risky_return=expected_risky_return,
    expected_reserve_return=0.03,
    risky_volatility=0.18,
    reserve_volatility=0.06,
    correlation=0.25,
    floor_fraction=0.80,
    initial_wealth=100.0,
    minimum_multiplier=0.0,
    maximum_multiplier=4.0,
)

pd.concat(
    [
        gopi.floor.iloc[:, 0].rename("reserve_tracking_floor"),
        gopi.multiplier.iloc[:, 0].rename("growth_optimal_multiplier"),
        gopi.risky_weight.iloc[:, 0].rename("risky_weight"),
    ],
    axis=1,
).iloc[10:20]
"""
    ),
    markdown(
        """
## 7. Empirical jump gap-risk report

Cont and Tankov show that downward price jumps create residual floor risk even
with continuous trading. Here we use four synthetic scenarios rather than
their analytic Lévy/Fourier formulas. The scenarios are explicit stress inputs,
so the resulting frequency is an empirical scenario fraction—not a calibrated
real-world probability.
"""
    ),
    code(
        """
jump_scenarios = pd.concat(
    {
        "base": risky_returns,
        "moderate_jump": risky_returns.mask(
            risky_returns.index == risky_returns.index[14], -0.35
        ),
        "severe_jump": risky_returns.mask(
            risky_returns.index == risky_returns.index[14], -0.70
        ),
        "total_gap": risky_returns.mask(
            risky_returns.index == risky_returns.index[14], -1.00
        ),
    },
    axis=1,
)

jump_cppi = run_fixed_maturity_cppi(
    jump_scenarios,
    multiplier=3.0,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)
gap_risk = analyze_cppi_gap_risk(jump_cppi, confidence_level=0.95)
gap_risk.statistics.to_frame("value")
"""
    ),
    code(
        """
gap_risk.scenario_losses[
    [
        "floor_breached",
        "first_breach_period",
        "first_breach_shortfall",
        "terminal_shortfall",
        "maximum_floor_shortfall",
    ]
]
"""
    ),
    markdown("## 8. Compare strategy paths"),
    code(
        """
results = {
    "Fixed maturity m=3": fixed_m3,
    "Open-ended": open_ended,
    "TIPP": tipp,
    "Dynamic multiplier": dynamic,
    "Growth optimal": gopi,
}

comparison = pd.concat(
    {name: result.summary().iloc[0] for name, result in results.items()},
    axis=1,
).T
comparison[
    [
        "terminal_wealth",
        "terminal_floor",
        "maximum_drawdown",
        "average_risky_weight",
        "total_turnover",
        "floor_breach_count",
        "cash_locked_count",
    ]
]
"""
    ),
    code(
        """
wealth_paths = pd.DataFrame(
    {
        name: result.wealth.iloc[:, 0]
        for name, result in results.items()
    }
)
wealth_paths.plot(
    title="Synthetic CPPI-family wealth paths",
    xlabel="Month",
    ylabel="Wealth",
    figsize=(10, 5),
)
"""
    ),
    markdown(
        """
## 9. Gap-risk exercise

Replace one return with `-1.0`. Inspect the first floor breach, the subsequent
risky weight, and cash-lock status. A floor is an allocation objective, not a
guarantee under discrete jumps.
"""
    ),
    code(
        """
gap_returns = risky_returns.copy()
gap_returns.iloc[14] = -1.0

gap_result = run_fixed_maturity_cppi(
    gap_returns,
    multiplier=3.0,
    guarantee_fraction=0.80,
    initial_wealth=100.0,
    risk_free_rate=0.03,
    periods_per_year=12,
)

gap_review = pd.concat(
    [
        gap_result.wealth.iloc[:, 0].rename("wealth"),
        gap_result.floor.iloc[:, 0].rename("floor"),
        gap_result.risky_weight.iloc[:, 0].rename("risky_weight"),
        gap_result.floor_breach.iloc[:, 0].rename("floor_breach"),
        gap_result.cash_locked.iloc[:, 0].rename("cash_locked"),
    ],
    axis=1,
)
gap_review.iloc[12:18]
"""
    ),
    markdown(
        """
## Interpretation and limitations

- Fixed maturity, open-ended, and TIPP primarily differ through their floor
  policy; dynamic multiplier changes the exposure policy.
- GOPI additionally allows a locally risky reserve asset and derives its
  multiplier from explicit point-in-time expected-return and covariance inputs.
- `m=1` and `m=3` are configurations, not additional engine families.
- The default risky-weight cap is 100%; leverage requires explicit approval.
- Less frequent rebalancing reduces trading but increases gap exposure.
- Transaction costs, cash lock, and floor breaches must be reported.
- Classical CPPI functions use one constant annual effective safe rate; GOPI's
  floor and safe holding follow the supplied reserve-asset path.
- The dynamic policy is a documented inverse-volatility rule, not a universal
  industry standard.
- `volatility_exponent=2/alpha` is a toolkit mapping of the paper's
  alpha-stable hazard scaling; the lagged realized-volatility estimator remains
  an explicit implementation choice.
- Empirical gap-risk statistics summarize supplied scenarios. They do not
  reproduce analytic Lévy jump probabilities, Fourier loss distributions, or
  option-hedging prices.
"""
    ),
]

for index, cell in enumerate(cells):
    identity = f"{OUTPUT.name}\0{index}\0{cell.cell_type}\0{cell.source}"
    cell["id"] = hashlib.sha256(identity.encode()).hexdigest()[:8]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbformat.write(notebook, OUTPUT)
