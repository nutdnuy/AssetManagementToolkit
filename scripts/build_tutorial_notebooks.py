"""Build the executable tutorial notebooks from reviewed cell definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip())


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.strip())


def write_notebook(filename: str, cells: list[nbformat.NotebookNode]) -> None:
    for index, cell in enumerate(cells):
        identity = f"{filename}\0{index}\0{cell.cell_type}\0{cell.source}"
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
    nbformat.write(notebook, TUTORIALS / filename)


BOOTSTRAP = """
from pathlib import Path
import sys

import numpy as np
import pandas as pd

project_root = Path.cwd()
if not (project_root / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

pd.options.display.float_format = "{:.4f}".format
"""


write_notebook(
    "01_risk_and_return_foundations.ipynb",
    [
        markdown(
            """
# Level 1 — Risk and Return Foundations

**Audience:** analysts who know basic Python and pandas but are new to the
`AssetManagementToolkit` API.

**Prerequisites:** Python 3.9+, NumPy, pandas, SciPy, and this repository.

**Learning goals**

1. represent periodic simple returns correctly;
2. calculate return, volatility, drawdown, VaR, and risk-adjusted ratios;
3. create one audit-friendly `risk_return_stats` table;
4. interpret frequency and benchmark assumptions.

This notebook distills the reusable ideas from legacy labs 102–106. It uses
synthetic monthly data, so no private dataset or network connection is needed.
"""
        ),
        markdown(
            """
## 1. Setup

All examples use **monthly decimal simple returns**. A value of `0.02` means
2%, and therefore `periods_per_year=12`.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.analytics.returns import (
    annualized_return,
    total_return,
)
from asset_management_toolkit.analytics.risk import (
    annualized_volatility,
    cornish_fisher_var,
    historical_cvar,
    historical_var,
    max_drawdown,
    sharpe_ratio,
)
from asset_management_toolkit.analytics import risk_return_stats
"""
        ),
        code(
            """
dates = pd.period_range("2023-01", periods=24, freq="M").to_timestamp("M")
returns = pd.DataFrame(
    {
        "Balanced": [
            0.020, -0.010, 0.015, 0.008, -0.025, 0.030,
            0.012, 0.006, -0.008, 0.018, 0.011, 0.005,
            0.014, -0.006, 0.021, 0.009, -0.018, 0.026,
            0.010, 0.004, -0.005, 0.016, 0.008, 0.013,
        ],
        "Growth": [
            0.035, -0.028, 0.024, 0.012, -0.050, 0.048,
            0.019, 0.010, -0.017, 0.031, 0.015, 0.008,
            0.023, -0.014, 0.033, 0.016, -0.036, 0.041,
            0.017, 0.007, -0.012, 0.026, 0.013, 0.020,
        ],
    },
    index=dates,
)
returns.head()
"""
        ),
        markdown(
            """
## 2. Return and volatility

`total_return` compounds the entire observed path. `annualized_return`
geometrically scales that path to one year. Volatility uses the sample standard
deviation and the square-root-of-time rule.
"""
        ),
        code(
            """
pd.DataFrame(
    {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year=12),
        "annualized_volatility": annualized_volatility(
            returns, periods_per_year=12
        ),
        "sharpe_ratio": sharpe_ratio(
            returns, risk_free_rate=0.02, periods_per_year=12
        ),
    }
)
"""
        ),
        markdown(
            """
## 3. Drawdown and tail risk

Maximum drawdown is the worst peak-to-trough loss **within the sample path**;
it is not annualized. VaR and CVaR here are positive loss magnitudes. At
`level=0.05`, the functions examine the worst 5% tail.
"""
        ),
        code(
            """
pd.DataFrame(
    {
        "max_drawdown": max_drawdown(returns),
        "historical_var_95": historical_var(returns, level=0.05),
        "historical_cvar_95": historical_cvar(returns, level=0.05),
        "cornish_fisher_var_95": cornish_fisher_var(returns, level=0.05),
    }
)
"""
        ),
        markdown(
            """
## 4. One summary table

Use the façade when you need a consistent review table rather than calling
each metric separately. The row count makes the observation window explicit.
"""
        ),
        code(
            """
stats = risk_return_stats(
    returns,
    risk_free_rate=0.02,
    minimum_acceptable_return=0.00,
    periods_per_year=12,
    var_level=0.05,
)
stats.T
"""
        ),
        markdown(
            """
## Exercise

Change the annual risk-free rate from 2% to 4%. Which columns in the summary
should change, and which should remain identical?
"""
        ),
        code(
            """
# Try it here.
stats_higher_rf = risk_return_stats(
    returns,
    risk_free_rate=0.04,
    periods_per_year=12,
)
"""
        ),
        markdown(
            """
### Answer scaffold

Compare `stats_higher_rf` with `stats`. The Sharpe ratio should change because
it uses the risk-free rate. Return, volatility, drawdown, and standalone tail
risk measures should not.
"""
        ),
        code(
            """
changed = stats_higher_rf.ne(stats).any(axis=0)
changed[changed]
"""
        ),
        markdown(
            """
## Common pitfalls and next steps

- Match `periods_per_year` to the actual sampling frequency.
- Pass decimal returns, not prices and not percentage points.
- Do not annualize maximum drawdown.
- Tail estimates from 24 observations are illustrative, not decision-grade.

Next: continue to Level 2 to turn expected returns and a covariance matrix into
portfolio weights.
"""
        ),
    ],
)


write_notebook(
    "02_portfolio_construction.ipynb",
    [
        markdown(
            """
# Level 2 — Portfolio Construction

**Audience:** users comfortable with return statistics who want to construct
long-only portfolios.

**Prerequisites:** Level 1 or equivalent knowledge of annualized return and
volatility.

**Learning goals**

1. estimate annual expected returns and covariance from periodic data;
2. evaluate a weight vector;
3. calculate minimum-volatility, maximum-Sharpe, and GMV portfolios;
4. generate and inspect an efficient frontier.

This notebook distills the reusable ideas from legacy labs 107–111 and 118.
"""
        ),
        markdown("## 1. Setup and synthetic asset returns"),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.portfolio import (
    efficient_frontier_weights,
    global_minimum_variance,
    maximum_sharpe_ratio,
    minimum_volatility,
    portfolio_return,
    portfolio_volatility,
)
"""
        ),
        code(
            """
rng = np.random.default_rng(42)
monthly_mean = np.array([0.0040, 0.0065, 0.0080, 0.0030])
monthly_cov = np.array(
    [
        [0.00040, 0.00010, 0.00008, 0.00003],
        [0.00010, 0.00090, 0.00025, 0.00004],
        [0.00008, 0.00025, 0.00160, 0.00002],
        [0.00003, 0.00004, 0.00002, 0.00016],
    ]
)
asset_names = ["Bonds", "Quality", "Equity", "Defensive"]
sample = pd.DataFrame(
    rng.multivariate_normal(monthly_mean, monthly_cov, size=120),
    columns=asset_names,
)
expected_returns = sample.mean() * 12
covariance = sample.cov() * 12
expected_returns
"""
        ),
        markdown(
            """
## 2. Evaluate an existing allocation

Weights and expected returns must have the same order. Labeled pandas inputs
make that contract visible.
"""
        ),
        code(
            """
equal_weight = pd.Series(0.25, index=asset_names)
pd.Series(
    {
        "expected_return": portfolio_return(equal_weight, expected_returns),
        "volatility": portfolio_volatility(equal_weight, covariance),
    },
    name="equal_weight",
)
"""
        ),
        markdown("## 3. Compare standard long-only portfolios"),
        code(
            """
target = float(expected_returns.median())
weights = pd.DataFrame(
    {
        "Min vol at target": minimum_volatility(
            target, expected_returns, covariance
        ),
        "Max Sharpe": maximum_sharpe_ratio(
            0.02, expected_returns, covariance
        ),
        "GMV": global_minimum_variance(covariance),
        "Equal weight": equal_weight,
    }
)
weights
"""
        ),
        code(
            """
portfolio_table = pd.DataFrame(
    {
        name: {
            "expected_return": portfolio_return(w, expected_returns),
            "volatility": portfolio_volatility(w, covariance),
        }
        for name, w in weights.items()
    }
).T
portfolio_table["ex_ante_sharpe"] = (
    portfolio_table["expected_return"] - 0.02
) / portfolio_table["volatility"]
portfolio_table
"""
        ),
        markdown(
            """
## 4. Efficient frontier

Each row below is the minimum-volatility long-only allocation for a target
return. The index stores the target return so the result remains auditable.
"""
        ),
        code(
            """
frontier_weights = efficient_frontier_weights(
    15, expected_returns, covariance
)
frontier = pd.DataFrame(
    {
        "expected_return": [
            portfolio_return(row, expected_returns)
            for _, row in frontier_weights.iterrows()
        ],
        "volatility": [
            portfolio_volatility(row, covariance)
            for _, row in frontier_weights.iterrows()
        ],
    },
    index=frontier_weights.index,
)
frontier.head()
"""
        ),
        markdown(
            """
## Exercise

Find the minimum-volatility portfolio targeting 7% annual return. Verify that
weights sum to one and that the realized model return matches the target.
"""
        ),
        code(
            """
exercise_weights = minimum_volatility(
    0.07, expected_returns, covariance
)
# Add your verification below.
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.Series(
    {
        "weight_sum": exercise_weights.sum(),
        "minimum_weight": exercise_weights.min(),
        "model_return": portfolio_return(
            exercise_weights, expected_returns
        ),
        "model_volatility": portfolio_volatility(
            exercise_weights, covariance
        ),
    }
)
"""
        ),
        markdown(
            """
## Common pitfalls and next steps

- Expected returns and covariance must use the same annualization convention.
- These optimizers are long-only and fully invested.
- Estimated expected returns are noisy; compare Max Sharpe with GMV and equal
  weight rather than treating one optimizer as truth.
- Optimization outputs are model allocations, not investment advice.

Next: Level 3 combines analytics, benchmark-relative diagnostics, and an
in-sample/out-of-sample review.
"""
        ),
    ],
)


write_notebook(
    "03_asset_management_workflow.ipynb",
    [
        markdown(
            """
# Level 3 — Asset Management Workflow

**Audience:** analysts who need a reproducible research workflow rather than a
single metric or optimizer call.

**Prerequisites:** Levels 1 and 2.

**Learning goals**

1. separate estimation and evaluation windows;
2. compare optimized portfolios with a transparent benchmark;
3. produce benchmark-relative risk/return diagnostics;
4. document assumptions and limitations.

Legacy labs 119 and 121–129 progress into CPPI, simulation, liabilities,
interest rates, and dynamic allocation. Those modules are a planned extension;
this executable notebook intentionally uses only the toolkit's current public
API.
"""
        ),
        markdown("## 1. Setup and reproducible sample"),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.analytics import risk_return_stats
from asset_management_toolkit.portfolio import (
    global_minimum_variance,
    maximum_sharpe_ratio,
)
"""
        ),
        code(
            """
rng = np.random.default_rng(7)
asset_names = ["Income", "Balanced", "Growth"]
monthly_mean = np.array([0.0035, 0.0055, 0.0075])
monthly_cov = np.array(
    [
        [0.00020, 0.00008, 0.00006],
        [0.00008, 0.00055, 0.00022],
        [0.00006, 0.00022, 0.00120],
    ]
)
all_returns = pd.DataFrame(
    rng.multivariate_normal(monthly_mean, monthly_cov, size=180),
    columns=asset_names,
    index=pd.period_range("2011-01", periods=180, freq="M").to_timestamp("M"),
)
estimation = all_returns.iloc[:120]
evaluation = all_returns.iloc[120:]
"""
        ),
        markdown(
            """
## 2. Estimate weights without looking at the evaluation period

All annualized inputs below come only from the first 120 months.
"""
        ),
        code(
            """
expected = estimation.mean() * 12
covariance = estimation.cov() * 12

weights = pd.DataFrame(
    {
        "GMV": global_minimum_variance(covariance),
        "Max Sharpe": maximum_sharpe_ratio(
            0.02, expected, covariance
        ),
        "Equal weight": pd.Series(1 / len(asset_names), index=asset_names),
    }
)
weights
"""
        ),
        markdown(
            """
## 3. Evaluate realized monthly returns

The equal-weight portfolio is the benchmark. This is an explicit comparison
choice, not a claim that it is the investable market portfolio.
"""
        ),
        code(
            """
portfolio_returns = pd.DataFrame(
    {
        name: evaluation.mul(weight, axis=1).sum(axis=1)
        for name, weight in weights.items()
    }
)
benchmark = portfolio_returns["Equal weight"].rename("Benchmark")
portfolio_returns.head()
"""
        ),
        code(
            """
review = risk_return_stats(
    portfolio_returns[["GMV", "Max Sharpe"]],
    benchmark=benchmark,
    risk_free_rate=0.02,
    periods_per_year=12,
)
review.T
"""
        ),
        markdown(
            """
## 4. A compact decision record

Record method, observation window, benchmark, and limitations alongside the
numbers. This prevents a metric table from becoming detached from its
assumptions.
"""
        ),
        code(
            """
decision_record = {
    "estimation_window": (
        str(estimation.index.min().date()),
        str(estimation.index.max().date()),
    ),
    "evaluation_window": (
        str(evaluation.index.min().date()),
        str(evaluation.index.max().date()),
    ),
    "frequency": "monthly",
    "periods_per_year": 12,
    "benchmark": "equal-weight portfolio over the same three assets",
    "constraints": "long-only, fully invested",
    "limitation": (
        "Synthetic sample; ignores costs, turnover, taxes, and capacity."
    ),
}
decision_record
"""
        ),
        markdown(
            """
## Exercise

Re-estimate the portfolios using only the first 60 months, then evaluate on the
same final 60-month window. How sensitive are the weights and realized
statistics to the estimation sample?
"""
        ),
        code(
            """
short_estimation = all_returns.iloc[:60]
short_expected = short_estimation.mean() * 12
short_covariance = short_estimation.cov() * 12

# Build short_weights, realized portfolio returns, and a new summary here.
"""
        ),
        markdown(
            """
### Answer scaffold

Use the same sequence as Sections 2–3. Compare both weights and realized
metrics; do not judge robustness from Sharpe ratio alone.
"""
        ),
        code(
            """
short_weights = pd.DataFrame(
    {
        "GMV": global_minimum_variance(short_covariance),
        "Max Sharpe": maximum_sharpe_ratio(
            0.02, short_expected, short_covariance
        ),
    }
)
weight_change = short_weights - weights[["GMV", "Max Sharpe"]]
weight_change
"""
        ),
        markdown(
            """
## Roadmap from the legacy advanced labs

The following topics are deliberately not presented as working APIs yet:

- Level 3A: CPPI and drawdown constraints (lab 119);
- Level 3B: GBM and Monte Carlo diagnostics (labs 121–123);
- Level 3C: present value, CIR rates, duration, and bonds (labs 124–127);
- Level 3D: fixed mix, glide paths, and dynamic risk budgeting (labs 128–129).

Each topic should enter the library only after provenance review, a small
public API, deterministic tests, and its own executable tutorial.
"""
        ),
    ],
)


write_notebook(
    "04_simulation_foundation.ipynb",
    [
        markdown(
            """
# Level 3A — Simulation Foundation

**Audience:** analysts who understand periodic returns and want to build
reproducible scenario analysis.

**Prerequisites:** Level 1 and basic NumPy/pandas familiarity.

**Learning goals**

1. distinguish a stochastic scenario from a forecast;
2. simulate GBM simple-return and price paths with an explicit random seed;
3. calculate terminal wealth across scenarios;
4. measure floor breaches and cap outcomes without mixing plotting into the
   simulation API.

**Outline:** model assumptions → return paths → price paths → terminal wealth
→ sensitivity exercise.

This notebook independently implements the standard GBM equation. Legacy labs
121–123 were used only to identify the curriculum topic and expected workflow.
"""
        ),
        markdown(
            """
## 1. Setup

The example uses monthly steps. `expected_return` is the annual instantaneous
GBM drift μ, while `volatility` is annualized σ.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.simulation import (
    simulate_gbm_prices,
    simulate_gbm_returns,
    terminal_wealth,
    terminal_wealth_stats,
)
"""
        ),
        markdown(
            r"""
## 2. Simulate periodic returns

For \(\Delta t = 1/P\), each simple return is generated from the exact
lognormal step:

\[
r_t = \exp\left((\mu-\tfrac{1}{2}\sigma^2)\Delta t
      + \sigma\sqrt{\Delta t}Z_t\right)-1
\]

`seed` makes the random draw reproducible; it does not make the model more
accurate.
"""
        ),
        code(
            """
simulated_returns = simulate_gbm_returns(
    n_years=5,
    n_scenarios=2_000,
    expected_return=0.07,
    volatility=0.15,
    periods_per_year=12,
    seed=42,
)
simulated_returns.iloc[:5, :4]
"""
        ),
        code(
            """
pd.Series(
    {
        "rows": simulated_returns.shape[0],
        "scenarios": simulated_returns.shape[1],
        "minimum_period_return": simulated_returns.min().min(),
        "maximum_period_return": simulated_returns.max().max(),
    }
)
"""
        ),
        markdown(
            """
## 3. Simulate price paths

Price paths contain an explicit step-zero row. Using the same seed and
parameters makes their periodic percentage changes match the return paths.
Plotting remains notebook presentation logic rather than library behavior.
"""
        ),
        code(
            """
prices = simulate_gbm_prices(
    n_years=5,
    n_scenarios=2_000,
    expected_return=0.07,
    volatility=0.15,
    periods_per_year=12,
    initial_price=100.0,
    seed=42,
)
prices.iloc[:, :20].plot(
    legend=False,
    title="Illustrative GBM price scenarios",
    xlabel="Monthly step",
    ylabel="Price index",
    figsize=(9, 4),
)
"""
        ),
        markdown(
            """
## 4. Terminal wealth

Treat each scenario column as one possible path. Floor and cap inputs below are
absolute wealth levels in the same unit as `initial_wealth`.
"""
        ),
        code(
            """
wealth = terminal_wealth(
    simulated_returns,
    initial_wealth=100.0,
)
wealth.quantile([0.05, 0.25, 0.50, 0.75, 0.95]).rename(
    "terminal_wealth"
)
"""
        ),
        code(
            """
terminal_wealth_stats(
    simulated_returns,
    initial_wealth=100.0,
    floor_wealth=80.0,
    cap_wealth=180.0,
)
"""
        ),
        markdown(
            """
## 5. Exercise — volatility sensitivity

Hold μ, the horizon, scenario count, and seed constant. Compare annual
volatility of 10% and 25%. Which terminal statistics move most, and why?
"""
        ),
        code(
            """
low_vol_returns = simulate_gbm_returns(
    n_years=5,
    n_scenarios=2_000,
    expected_return=0.07,
    volatility=0.10,
    periods_per_year=12,
    seed=7,
)
high_vol_returns = simulate_gbm_returns(
    n_years=5,
    n_scenarios=2_000,
    expected_return=0.07,
    volatility=0.25,
    periods_per_year=12,
    seed=7,
)

# Build and compare two terminal_wealth_stats Series below.
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
sensitivity = pd.DataFrame(
    {
        "10% volatility": terminal_wealth_stats(
            low_vol_returns,
            initial_wealth=100.0,
            floor_wealth=80.0,
            cap_wealth=180.0,
        ),
        "25% volatility": terminal_wealth_stats(
            high_vol_returns,
            initial_wealth=100.0,
            floor_wealth=80.0,
            cap_wealth=180.0,
        ),
    }
)
sensitivity.loc[
    [
        "mean",
        "median",
        "standard_deviation",
        "probability_below_floor",
        "expected_shortfall_below_floor",
        "probability_above_cap",
    ]
]
"""
        ),
        markdown(
            """
## Interpretation and common pitfalls

- GBM is a model of lognormal diffusion, not a prediction of future prices.
- A fixed seed supports reproducibility; use multiple seeds for implementation
  checks, not to search for a preferred outcome.
- The model assumes constant drift and volatility and independent Gaussian
  shocks. It omits regimes, jumps, fat tails, costs, taxes, and liquidity.
- Keep annual parameters and `periods_per_year` consistent.
- Report the scenario count, horizon, seed policy, assumptions, and thresholds
  alongside terminal statistics.

Next: CPPI can consume scenario returns from this module, but it belongs in a
separate allocation layer with its own floor and rebalancing contracts.
"""
        ),
    ],
)


write_notebook(
    "05_heavy_tail_and_jump_simulation.ipynb",
    [
        markdown(
            """
# Level 3B — Heavy-Tail and Jump Simulation

**Audience:** analysts who can run GBM scenarios and want models that allow
non-Gaussian log-return shapes.

**Prerequisites:** Level 3A and basic knowledge of distributions.

**Learning goals**

1. distinguish diffusion, compound-Poisson jump, random-clock, and stable-tail
   assumptions;
2. simulate Variance Gamma return and price paths;
3. simulate Merton jump-diffusion return and price paths;
4. simulate symmetric and skewed alpha-stable return and price paths;
5. interpret sample diagnostics without assuming nonexistent population
   moments.

**Outline:** model map → Variance Gamma → Merton jumps → skewed stable →
comparison → terminal wealth → exercise.

The Variance Gamma section follows Gamma-time-changed Brownian motion described
by Madan, Carr, and Chang (1998) and the owner-provided QC article. The jump
section follows Merton (1976). The stable section follows Mandelbrot's (1963)
stable Paretian proposal and uses the Chambers-Mallows-Stuck simulation
transformation.
"""
        ),
        markdown(
            """
## 1. Setup and model map

- **GBM:** continuous Gaussian log-return diffusion.
- **Variance Gamma:** Brownian motion evaluated on a random Gamma business
  clock; supports excess kurtosis and asymmetry.
- **Merton jump diffusion:** Gaussian diffusion plus a finite number of
  normally distributed log jumps arriving through a Poisson process.
- **Alpha-stable:** stable log increments with power-law tails; `beta` controls
  left/right tail asymmetry under the documented `S0` parameterization.

All functions return periodic **simple returns** so they can feed the same
terminal-wealth tools.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.analytics.risk import excess_kurtosis, skewness
from asset_management_toolkit.simulation import (
    simulate_gbm_returns,
    simulate_merton_jump_prices,
    simulate_merton_jump_returns,
    simulate_stable_prices,
    simulate_stable_returns,
    simulate_variance_gamma_prices,
    simulate_variance_gamma_returns,
    terminal_wealth_stats,
)
"""
        ),
        markdown(
            r"""
## 2. Variance Gamma: Brownian motion on a random clock

For \(\Delta t=1/P\), draw
\(G\sim\operatorname{Gamma}(\Delta t/\nu,\nu)\). Then:

\[
\Delta \log S =
(\ell-\theta)\Delta t+\theta G+\sigma\sqrt{G}Z
\]

The clock has \(E[G]=\Delta t\) and
\(\operatorname{Var}(G)=\nu\Delta t\). Parameter `theta` controls asymmetry,
while `variance_rate` (`ν`) controls how unevenly business time passes.
"""
        ),
        code(
            """
vg_returns = simulate_variance_gamma_returns(
    n_years=3,
    n_scenarios=4_000,
    mean_log_return=0.07,
    theta=-0.10,
    volatility=0.15,
    variance_rate=0.20,
    periods_per_year=12,
    seed=42,
)
vg_returns.iloc[:5, :4]
"""
        ),
        code(
            """
vg_prices = simulate_variance_gamma_prices(
    n_years=3,
    n_scenarios=4_000,
    mean_log_return=0.07,
    theta=-0.10,
    volatility=0.15,
    variance_rate=0.20,
    periods_per_year=12,
    initial_price=100.0,
    seed=42,
)
vg_prices.iloc[:, :20].plot(
    legend=False,
    title="Illustrative Variance Gamma price scenarios",
    xlabel="Monthly step",
    ylabel="Price index",
    figsize=(9, 4),
)
"""
        ),
        markdown(
            r"""
## 3. Merton jump diffusion: explicit event arrivals

For \(\Delta t=1/P\):

\[
\Delta \log S =
\left(\mu-\frac{1}{2}\sigma^2-\lambda\kappa\right)\Delta t
+\sigma\sqrt{\Delta t}Z
+\sum_{k=1}^{N_{\Delta t}}Y_k
\]

where \(N_{\Delta t}\sim\operatorname{Poisson}(\lambda\Delta t)\),
\(Y_k\sim\mathcal{N}(m_J,s_J^2)\), and
\(\kappa=\exp(m_J+s_J^2/2)-1\). The compensation term keeps expected price
growth governed by `expected_return`.
"""
        ),
        code(
            """
merton_returns = simulate_merton_jump_returns(
    n_years=3,
    n_scenarios=4_000,
    expected_return=0.07,
    volatility=0.15,
    jump_intensity=1.5,
    jump_mean=-0.12,
    jump_volatility=0.20,
    periods_per_year=12,
    seed=42,
)
merton_prices = simulate_merton_jump_prices(
    n_years=3,
    n_scenarios=4_000,
    expected_return=0.07,
    volatility=0.15,
    jump_intensity=1.5,
    jump_mean=-0.12,
    jump_volatility=0.20,
    periods_per_year=12,
    initial_price=100.0,
    seed=42,
)
merton_prices.iloc[:, :20].plot(
    legend=False,
    title="Illustrative Merton jump-diffusion price scenarios",
    xlabel="Monthly step",
    ylabel="Price index",
    figsize=(9, 4),
)
"""
        ),
        markdown(
            r"""
## 4. Symmetric and skewed alpha-stable tails

For the symmetric case, stable increments obey the simple scaling rule:

\[
\Delta \log S =
\delta\Delta t+c\Delta t^{1/\alpha}X_{\alpha}
\]

Lower `alpha` produces heavier tails. At `alpha=2`, the stable law reaches its
Gaussian limit under the stable scale convention. For `alpha < 2`, population
variance is infinite; for `alpha <= 1`, the population mean is also undefined.

The general API uses Nolan's `S0` parameterization. `beta=0` is symmetric,
`beta<0` emphasizes the left tail, and `beta>0` emphasizes the right tail.
`beta` is a distribution parameter, not the ordinary third-moment skewness
coefficient. The simulator adjusts each step's location so the stated annual
`S0` parameters remain consistent under Lévy-process time scaling.
"""
        ),
        code(
            """
stable_returns = simulate_stable_returns(
    n_years=3,
    n_scenarios=4_000,
    alpha=1.70,
    beta=0.0,
    scale=0.04,
    location=0.07,
    periods_per_year=12,
    seed=42,
)
left_skewed_returns = simulate_stable_returns(
    n_years=3,
    n_scenarios=4_000,
    alpha=1.70,
    beta=-0.60,
    scale=0.04,
    location=0.07,
    periods_per_year=12,
    seed=42,
)
left_skewed_prices = simulate_stable_prices(
    n_years=3,
    n_scenarios=4_000,
    alpha=1.70,
    beta=-0.60,
    scale=0.04,
    location=0.07,
    periods_per_year=12,
    initial_price=100.0,
    seed=42,
)
left_skewed_prices.iloc[:, :20].plot(
    legend=False,
    title="Illustrative left-skewed alpha-stable price scenarios",
    xlabel="Monthly step",
    ylabel="Price index",
    figsize=(9, 4),
)
"""
        ),
        markdown(
            """
## 5. Compare sample distributions

These are finite-sample diagnostics. They do not turn infinite stable
population moments into finite ones.
"""
        ),
        code(
            """
gbm_returns = simulate_gbm_returns(
    n_years=3,
    n_scenarios=4_000,
    expected_return=0.07,
    volatility=0.15,
    periods_per_year=12,
    seed=42,
)

model_returns = {
    "GBM": gbm_returns.stack(),
    "Variance Gamma": vg_returns.stack(),
    "Merton jumps": merton_returns.stack(),
    "Stable beta=0.00": stable_returns.stack(),
    "Stable beta=-0.60": left_skewed_returns.stack(),
}
comparison = pd.DataFrame(
    {
        name: {
            "sample_mean": values.mean(),
            "sample_std": values.std(),
            "sample_skewness": skewness(values),
            "sample_excess_kurtosis": excess_kurtosis(values),
            "sample_min": values.min(),
            "sample_max": values.max(),
        }
        for name, values in model_returns.items()
    }
).T
comparison
"""
        ),
        markdown(
            """
## 6. Compare terminal outcomes

The same terminal-wealth function accepts every return matrix. Differences
come from model assumptions and parameters, not from a change in the wealth
calculation.
"""
        ),
        code(
            """
terminal_comparison = pd.DataFrame(
    {
        name: terminal_wealth_stats(
            returns,
            initial_wealth=100.0,
            floor_wealth=80.0,
            cap_wealth=150.0,
        )
        for name, returns in {
            "GBM": gbm_returns,
            "Variance Gamma": vg_returns,
            "Merton jumps": merton_returns,
            "Stable beta=0.00": stable_returns,
            "Stable beta=-0.60": left_skewed_returns,
        }.items()
    }
)
terminal_comparison.loc[
    [
        "mean",
        "median",
        "standard_deviation",
        "probability_below_floor",
        "expected_shortfall_below_floor",
        "probability_above_cap",
    ]
]
"""
        ),
        markdown(
            """
## 7. Exercise — tail and skew sensitivity

1. Change stable `alpha` from 1.70 to 1.50 while holding `beta=-0.60`.
2. Change stable `beta` from -0.60 to +0.60.
3. Change Variance Gamma `theta` from -0.10 to +0.10.
4. Change Merton `jump_intensity` from 1.5 to 3.0.
5. Compare sample skewness, sample excess kurtosis, and floor-breach
   probability.
"""
        ),
        code(
            """
# Build the alternative scenarios here.
stable_heavier = simulate_stable_returns(
    n_years=3,
    n_scenarios=4_000,
    alpha=1.50,
    beta=-0.60,
    scale=0.04,
    location=0.07,
    periods_per_year=12,
    seed=7,
)
stable_right_skewed = simulate_stable_returns(
    n_years=3,
    n_scenarios=4_000,
    alpha=1.70,
    beta=0.60,
    scale=0.04,
    location=0.07,
    periods_per_year=12,
    seed=7,
)
vg_positive_skew = simulate_variance_gamma_returns(
    n_years=3,
    n_scenarios=4_000,
    mean_log_return=0.07,
    theta=0.10,
    volatility=0.15,
    variance_rate=0.20,
    periods_per_year=12,
    seed=7,
)
merton_more_frequent = simulate_merton_jump_returns(
    n_years=3,
    n_scenarios=4_000,
    expected_return=0.07,
    volatility=0.15,
    jump_intensity=3.0,
    jump_mean=-0.12,
    jump_volatility=0.20,
    periods_per_year=12,
    seed=7,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
exercise_returns = {
    "Stable alpha=1.50, beta=-0.60": stable_heavier,
    "Stable alpha=1.70, beta=+0.60": stable_right_skewed,
    "VG theta=+0.10": vg_positive_skew,
    "Merton intensity=3.0": merton_more_frequent,
}
pd.DataFrame(
    {
        name: {
            "sample_skewness": skewness(values.stack()),
            "sample_excess_kurtosis": excess_kurtosis(values.stack()),
            "probability_below_80": terminal_wealth_stats(
                values,
                initial_wealth=100.0,
                floor_wealth=80.0,
            )["probability_below_floor"],
        }
        for name, values in exercise_returns.items()
    }
).T
"""
        ),
        markdown(
            """
## Interpretation, pitfalls, and extensions

- These models generate scenarios; they do not identify which model will
  forecast future returns.
- Do not compare `scale` in a stable law directly with GBM volatility.
- State the stable parameterization. This toolkit uses Nolan `S0`; another
  library's `location` can differ when it uses `S1`.
- Do not report stable sample variance as a converged population variance when
  `alpha < 2`.
- Variance Gamma here is a statistical path simulator, not a calibrated
  risk-neutral option-pricing engine.
- Merton jumps here are physical/statistical scenarios. The parameters are not
  automatically risk-neutral option-pricing inputs.
- Report model, parameters, horizon, frequency, scenario count, and seed
  policy with every result.

Possible extensions include parameter calibration, CGMY, Kou double-exponential
jumps, stochastic volatility, and correlated multivariate scenarios. Each
needs a separate parameterization and validation contract.
"""
        ),
    ],
)


write_notebook(
    "06_calibration_and_model_selection.ipynb",
    [
        markdown(
            """
# Level 3C — Calibration and Model Selection

**Audience:** analysts who can simulate return paths and want an auditable way
to estimate parameters and test models out of sample.

**Prerequisites:** Levels 3A–3B, pandas, and basic distribution statistics.

**Learning goals**

1. inspect observed return shape before selecting a model;
2. calibrate GBM, Merton jump diffusion, and Variance Gamma;
3. compare observed and simulated distributions without treating one score as
   proof;
4. run leakage-aware walk-forward validation.

**Outline:** synthetic history → diagnostics → calibration → model comparison
→ walk-forward validation → exercise.

The notebook uses synthetic monthly returns so the complete workflow remains
reproducible and requires no private data or network access.
"""
        ),
        markdown(
            """
## 1. Setup and synthetic observed history

We generate one Merton jump-diffusion path and then treat it as if it were an
observed monthly asset history. In real work, replace only this Series while
preserving its frequency and decimal simple-return units.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.simulation import (
    calibrate_gbm,
    calibrate_merton_jump,
    calibrate_variance_gamma,
    compare_simulation_models,
    return_distribution_diagnostics,
    simulate_gbm_returns,
    simulate_merton_jump_returns,
    simulate_variance_gamma_returns,
    walk_forward_validate_simulation,
)
"""
        ),
        code(
            """
observed = simulate_merton_jump_returns(
    n_years=15,
    n_scenarios=1,
    expected_return=0.07,
    volatility=0.13,
    jump_intensity=1.2,
    jump_mean=-0.10,
    jump_volatility=0.18,
    periods_per_year=12,
    seed=21,
).iloc[:, 0]
observed.index = pd.period_range(
    "2011-01",
    periods=len(observed),
    freq="M",
)
observed.name = "synthetic_asset"
observed.head()
"""
        ),
        markdown(
            """
## 2. Diagnose before fitting

Diagnostics describe the sample rather than identify its true data-generating
process. Inspect tail quantiles, skewness, kurtosis, and log-return volatility
before choosing candidate models.
"""
        ),
        code(
            """
return_distribution_diagnostics(
    observed,
    periods_per_year=12,
    tail_probability=0.05,
)
"""
        ),
        markdown(
            """
## 3. Calibrate three candidate models

- GBM uses closed-form Gaussian log-return maximum likelihood.
- Merton uses deterministic multi-start maximum likelihood over a truncated
  Poisson mixture.
- Variance Gamma matches the second through fourth log-return cumulants.

Because VG does not use a full likelihood here, its AIC/BIC fields are
intentionally empty and must not be invented or compared with likelihood-based
scores.
"""
        ),
        code(
            """
gbm_fit = calibrate_gbm(observed, periods_per_year=12)
merton_fit = calibrate_merton_jump(
    observed,
    periods_per_year=12,
    max_jump_intensity=10.0,
)
vg_fit = calibrate_variance_gamma(observed, periods_per_year=12)

calibration_review = pd.DataFrame(
    [fit.to_series() for fit in [gbm_fit, merton_fit, vg_fit]]
).set_index("model")
calibration_review
"""
        ),
        markdown(
            """
## 4. Simulate from fitted parameters

Use the same horizon, scenario count, frequency, and seed policy for every
candidate. Different seeds avoid creating artificial path-by-path alignment;
distribution comparison does not require matched random numbers.
"""
        ),
        code(
            """
fitted_scenarios = {
    "GBM": simulate_gbm_returns(
        n_years=3,
        n_scenarios=2_000,
        periods_per_year=12,
        seed=101,
        **gbm_fit.parameters,
    ),
    "Merton": simulate_merton_jump_returns(
        n_years=3,
        n_scenarios=2_000,
        periods_per_year=12,
        seed=102,
        **merton_fit.parameters,
    ),
    "Variance Gamma": simulate_variance_gamma_returns(
        n_years=3,
        n_scenarios=2_000,
        periods_per_year=12,
        seed=103,
        **vg_fit.parameters,
    ),
}
"""
        ),
        code(
            """
distribution_comparison = compare_simulation_models(
    observed,
    fitted_scenarios,
    periods_per_year=12,
    tail_probability=0.05,
)
distribution_comparison[
    [
        "periodic_mean",
        "periodic_standard_deviation",
        "sample_skewness",
        "sample_excess_kurtosis",
        "q01",
        "q_tail",
        "ks_statistic",
        "wasserstein_distance",
        "distribution_error_score",
    ]
]
"""
        ),
        markdown(
            """
## 5. Walk-forward validation

The first 84 months are calibration data and each following 12-month block is
evaluated once. With an expanding window, later folds may use earlier test
periods only after those periods have become historical. No fold sees its own
future returns.
"""
        ),
        code(
            """
gbm_walk_forward = walk_forward_validate_simulation(
    observed,
    model="gbm",
    train_size=84,
    test_size=12,
    periods_per_year=12,
    n_scenarios=1_000,
    window="expanding",
    tail_probability=0.05,
    seed=200,
)
gbm_walk_forward[
    [
        "n_train",
        "n_test",
        "mean_error",
        "volatility_error",
        "tail_exceedance_rate",
        "ks_statistic",
        "actual_terminal_return",
        "simulated_terminal_median",
    ]
]
"""
        ),
        markdown(
            """
## 6. Exercise — compare out-of-sample models

Run the same walk-forward contract with `model="variance_gamma"`. Then compare:

1. average KS statistic;
2. average Wasserstein distance;
3. tail exceedance rate against the requested 5% tail;
4. simulated median terminal return versus the realized test return.

Do not select a model from one fold or one metric alone.
"""
        ),
        code(
            """
vg_walk_forward = walk_forward_validate_simulation(
    observed,
    model="variance_gamma",
    train_size=84,
    test_size=12,
    periods_per_year=12,
    n_scenarios=1_000,
    window="expanding",
    tail_probability=0.05,
    seed=201,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.DataFrame(
    {
        "GBM": {
            "mean_ks": gbm_walk_forward["ks_statistic"].mean(),
            "mean_wasserstein": gbm_walk_forward[
                "wasserstein_distance"
            ].mean(),
            "tail_exceedance_rate": (
                gbm_walk_forward["tail_exceedance_rate"].sum()
                / gbm_walk_forward["n_test"].count()
            ),
        },
        "Variance Gamma": {
            "mean_ks": vg_walk_forward["ks_statistic"].mean(),
            "mean_wasserstein": vg_walk_forward[
                "wasserstein_distance"
            ].mean(),
            "tail_exceedance_rate": (
                vg_walk_forward["tail_exceedance_rate"].sum()
                / vg_walk_forward["n_test"].count()
            ),
        },
    }
).T
"""
        ),
        markdown(
            """
## Interpretation, pitfalls, and extensions

- Calibration estimates a compact model, not future truth.
- Merton jump parameters can be weakly identified in small samples; inspect
  optimizer status and stability across windows.
- VG cumulant matching is sensitive to sample skewness and kurtosis and does
  not produce likelihood-based AIC/BIC in this toolkit.
- A lower in-sample error does not guarantee better out-of-sample tails.
- Walk-forward folds preserve time order but do not remove regime change,
  transaction costs, liquidity constraints, or model risk.
- Stable-law calibration remains deferred because parameterization and
  heavy-tail estimation need a separate numerical contract.

Next research layers include stress tests, correlated multivariate scenarios,
dynamic allocation, and rates/bond simulation.
"""
        ),
    ],
)


write_notebook(
    "07_stress_testing_foundation.ipynb",
    [
        markdown(
            """
# Level 3D — Stress Testing Foundation

**Audience:** analysts who need transparent, probability-free portfolio stress
tests after learning simulation and model validation.

**Prerequisites:** Levels 1–3C and basic pandas indexing.

**Learning goals**

1. distinguish deterministic stress scenarios from Monte Carlo probabilities;
2. construct historical and hypothetical labelled asset shocks;
3. attribute portfolio stress returns to assets and test loss limits;
4. evaluate multi-period stress paths, terminal loss, and maximum drawdown.

All data are synthetic and no scenario is investment advice or a forecast.
"""
        ),
        markdown(
            """
## 1. Setup

Returns and shocks are decimal simple returns. Portfolio weights are labelled,
fully invested, and may include short positions. Thresholds are loss limits,
not VaR confidence levels.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.stress import (
    historical_stress_scenarios,
    stress_test_portfolio,
    stress_test_portfolio_paths,
)
"""
        ),
        markdown(
            """
## 2. Build historical window scenarios

The helper compounds each asset's periodic returns over an inclusive window.
The example dates are synthetic labels; they do not recreate a real crisis.
"""
        ),
        code(
            """
dates = pd.date_range("2025-01-31", periods=12, freq="ME")
asset_returns = pd.DataFrame(
    {
        "Global Equity": [
            0.03, -0.04, -0.12, -0.08, 0.05, 0.02,
            0.01, -0.02, 0.04, 0.03, -0.01, 0.02,
        ],
        "Government Bond": [
            0.01, 0.02, 0.03, 0.01, -0.01, 0.00,
            0.01, 0.01, -0.02, 0.00, 0.01, 0.01,
        ],
        "Credit": [
            0.01, -0.01, -0.04, -0.03, 0.02, 0.01,
            0.01, 0.00, 0.02, 0.01, 0.00, 0.01,
        ],
    },
    index=dates,
)

historical = historical_stress_scenarios(
    asset_returns,
    {
        "synthetic_selloff": (dates[1], dates[3]),
        "synthetic_recovery": (dates[4], dates[6]),
    },
)
historical
"""
        ),
        markdown(
            """
## 3. Add hypothetical shocks

Hypothetical scenarios are explicit asset-return assumptions. Combining them
with historical windows does not attach probability to either set.
"""
        ),
        code(
            """
hypothetical = pd.DataFrame(
    {
        "Global Equity": [-0.25, -0.10, 0.05],
        "Government Bond": [0.04, -0.12, -0.03],
        "Credit": [-0.08, -0.15, -0.05],
    },
    index=[
        "equity_crash",
        "rates_and_credit",
        "inflation_pressure",
    ],
)

scenarios = pd.concat([historical, hypothetical])
scenarios
"""
        ),
        markdown(
            """
## 4. Test a labelled portfolio

Asset contributions are weight × scenario return and add to portfolio return.
`portfolio_loss` is the signed negative of return, so a gain appears as a
negative loss.
"""
        ),
        code(
            """
weights = pd.Series(
    {
        "Global Equity": 0.50,
        "Government Bond": 0.30,
        "Credit": 0.20,
    },
    name="strategic_weight",
)

stress = stress_test_portfolio(
    weights,
    scenarios,
    loss_thresholds={
        "warning_8pct": 0.08,
        "capital_limit_12pct": 0.12,
    },
)
stress.summary.sort_values("portfolio_loss", ascending=False)
"""
        ),
        code(
            """
stress.asset_contributions.loc[
    stress.summary["portfolio_loss"].idxmax()
].sort_values()
"""
        ),
        code(
            """
stress.threshold_breaches
"""
        ),
        markdown(
            """
## 5. Evaluate multi-period paths

Path stress testing preserves sequencing. The current contract resets to the
supplied weights each period and excludes transaction costs. Terminal loss
limits and maximum drawdown answer different questions.
"""
        ),
        code(
            """
scenario_paths = {
    "fast_crash_then_rebound": pd.DataFrame(
        {
            "Global Equity": [-0.20, -0.12, 0.10, 0.06],
            "Government Bond": [0.02, 0.01, -0.01, 0.00],
            "Credit": [-0.06, -0.04, 0.03, 0.02],
        }
    ),
    "persistent_rates_shock": pd.DataFrame(
        {
            "Global Equity": [-0.03, -0.02, 0.00, 0.01],
            "Government Bond": [-0.05, -0.04, -0.03, 0.01],
            "Credit": [-0.04, -0.03, -0.02, 0.01],
        }
    ),
}

path_stress = stress_test_portfolio_paths(
    weights,
    scenario_paths,
    loss_thresholds={"terminal_limit_10pct": 0.10},
)
path_stress.summary
"""
        ),
        code(
            """
path_stress.portfolio_returns
"""
        ),
        markdown(
            """
## 6. Exercise — concentration sensitivity

Change the portfolio to 70% Global Equity, 20% Government Bond, and 10% Credit.
Compare the worst scenario, portfolio loss, asset contributions, and threshold
breaches. Do not assume the worst asset shock is automatically the largest
portfolio contributor.
"""
        ),
        code(
            """
concentrated_weights = pd.Series(
    {
        "Global Equity": 0.70,
        "Government Bond": 0.20,
        "Credit": 0.10,
    }
)

# Run stress_test_portfolio and compare its summary with `stress.summary`.
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
concentrated = stress_test_portfolio(
    concentrated_weights,
    scenarios,
    loss_thresholds={
        "warning_8pct": 0.08,
        "capital_limit_12pct": 0.12,
    },
)

pd.concat(
    {
        "strategic": stress.summary["portfolio_loss"],
        "concentrated": concentrated.summary["portfolio_loss"],
    },
    axis=1,
).sort_values("concentrated", ascending=False)
"""
        ),
        markdown(
            """
## Interpretation and limitations

- Stress tests answer “what if,” not “how likely.”
- Scenario labels, dates, shocks, weights, limits, and aggregation rules belong
  in the decision record.
- Historical windows depend on the chosen sample and do not bound future loss.
- Hypothetical shocks should have governance ownership and documented rationale.
- Static one-period contribution is additive; multi-period compounded
  attribution needs a separate contract.
- The path API assumes constant-period rebalancing and excludes costs,
  liquidity, taxes, market impact, and forced deleveraging.
- Correlated scenario generation remains a separate future module.

Next: correlated multivariate scenarios can generate richer paths for this
stress layer without turning deterministic stresses into probabilities.
"""
        ),
    ],
)


write_notebook(
    "08_black_litterman_portfolio.ipynb",
    [
        markdown(
            """
# Level 4 — Black–Litterman Portfolio Construction

**Audience:** analysts who understand covariance matrices and long-only
Markowitz optimization but want a disciplined way to combine market
equilibrium with explicit views.

**Prerequisites:** Levels 1–3D, labelled pandas objects, and basic matrix
notation.

**Learning goals**

1. reverse-engineer market-implied equilibrium excess returns;
2. encode absolute and relative views with a pick matrix;
3. control view uncertainty and calculate posterior returns/covariance;
4. feed the posterior into the existing long-only Markowitz optimizer;
5. record assumptions and sensitivity rather than treating views as facts.

**Outline:** market prior → views → uncertainty → posterior → allocation
comparison → sensitivity → exercise.

All inputs are synthetic. The notebook requires no credentials, network access,
or private data.
"""
        ),
        markdown(
            """
## 1. Setup and unit contract

Every return and covariance input below is annual and expressed in decimal
units. `market_weights`, covariance rows/columns, pick-matrix columns, and
posterior outputs share the same asset labels and order.

The Black–Litterman layer estimates returns and covariance. Portfolio
constraints remain the responsibility of the downstream optimizer.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.portfolio import (
    black_litterman_posterior,
    implied_equilibrium_returns,
    maximum_sharpe_ratio,
    proportional_view_uncertainty,
)
"""
        ),
        markdown(
            """
## 2. Define a synthetic market portfolio

The market weights are non-negative and sum to one. The covariance matrix is
symmetric, positive semidefinite, and uses the same annual horizon as all view
returns.
"""
        ),
        code(
            """
assets = pd.Index(
    ["Global Equity", "Government Bond", "Gold"],
    name="asset",
)
market_weights = pd.Series(
    [0.50, 0.30, 0.20],
    index=assets,
    name="market_weight",
)
covariance = pd.DataFrame(
    [
        [0.0400, 0.0060, 0.0040],
        [0.0060, 0.0100, 0.0015],
        [0.0040, 0.0015, 0.0225],
    ],
    index=assets,
    columns=assets,
)

pd.DataFrame(
    {
        "market_weight": market_weights,
        "annual_volatility": np.sqrt(np.diag(covariance)),
    }
)
"""
        ),
        markdown(
            """
## 3. Reverse-optimize equilibrium excess returns

The market-implied prior is

\\[
\\pi = \\delta\\Sigma w_{market},
\\]

where `risk_aversion` \\(\\delta\\) is positive. These are equilibrium **excess**
returns under the model contract, not historical forecasts.
"""
        ),
        code(
            """
risk_aversion = 2.5
prior_returns = implied_equilibrium_returns(
    market_weights,
    covariance,
    risk_aversion=risk_aversion,
)
prior_returns.to_frame()
"""
        ),
        markdown(
            """
## 4. Encode one relative and one absolute view

Each row of \\(P\\) defines a portfolio:

- `equity_vs_bond`: Global Equity minus Government Bond;
- `gold_absolute`: Gold by itself.

The corresponding \\(Q\\) values say that the first portfolio has a 4% expected
annual excess return and Gold has a 3% expected annual excess return.
"""
        ),
        code(
            """
view_labels = pd.Index(
    ["equity_vs_bond", "gold_absolute"],
    name="view",
)
pick_matrix = pd.DataFrame(
    [
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    index=view_labels,
    columns=assets,
)
views = pd.Series(
    [0.04, 0.03],
    index=view_labels,
    name="view_return",
)

pick_matrix.assign(view_return=views)
"""
        ),
        markdown(
            """
## 5. Inspect proportional view uncertainty

The default He–Litterman heuristic uses

\\[
\\Omega = \\operatorname{diag}\\left(
\\operatorname{diag}(\\tau P\\Sigma P^\\top)
\\right).
\\]

Only the diagonal view variances are retained. With this proportional default,
`tau` cancels from posterior returns; it still affects posterior covariance.
"""
        ),
        code(
            """
tau = 0.05
default_omega = proportional_view_uncertainty(
    covariance,
    pick_matrix,
    tau=tau,
)
default_omega
"""
        ),
        markdown(
            """
## 6. Calculate the Black–Litterman posterior

The posterior adjusts the prior only by the view innovation \\(Q-P\\pi\\), scaled
through covariance and view uncertainty. The implementation uses linear solves
rather than explicit matrix inversion.
"""
        ),
        code(
            """
posterior = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    views,
    risk_aversion=risk_aversion,
    tau=tau,
)

return_comparison = pd.concat(
    [
        posterior.prior_returns.rename("prior"),
        posterior.posterior_returns.rename("posterior"),
    ],
    axis=1,
)
return_comparison["adjustment"] = (
    return_comparison["posterior"] - return_comparison["prior"]
)
return_comparison
"""
        ),
        code(
            """
posterior.posterior_covariance
"""
        ),
        markdown(
            """
## 7. Verify an equilibrium-view invariant

If \\(Q=P\\pi\\), the views contain no innovation and posterior expected returns
must equal the prior. This is a useful implementation and data-pipeline check.
"""
        ),
        code(
            """
equilibrium_views = pick_matrix @ prior_returns
equilibrium_result = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    equilibrium_views,
    risk_aversion=risk_aversion,
    tau=tau,
)

pd.DataFrame(
    {
        "prior": equilibrium_result.prior_returns,
        "posterior": equilibrium_result.posterior_returns,
        "difference": (
            equilibrium_result.posterior_returns
            - equilibrium_result.prior_returns
        ),
    }
)
"""
        ),
        markdown(
            """
## 8. Feed the posterior into long-only Markowitz

Black–Litterman does not enforce long-only weights. Here both the prior and
posterior estimates are passed to the same fully invested, long-only
maximum-Sharpe optimizer so the comparison isolates the changed estimates.
"""
        ),
        code(
            """
risk_free_rate = 0.0
prior_allocation = maximum_sharpe_ratio(
    risk_free_rate,
    posterior.prior_returns,
    covariance,
)
posterior_allocation = maximum_sharpe_ratio(
    risk_free_rate,
    posterior.posterior_returns,
    posterior.posterior_covariance,
)

allocation_comparison = pd.concat(
    [
        market_weights.rename("market"),
        prior_allocation.rename("prior_markowitz"),
        posterior_allocation.rename("black_litterman"),
    ],
    axis=1,
)
allocation_comparison
"""
        ),
        markdown(
            """
## 9. Sensitivity to custom view uncertainty

Smaller diagonal values in \\(\\Omega\\) express tighter uncertainty and move the
posterior closer to the views. Larger values keep it closer to equilibrium.
Both matrices below use the same units as squared annual returns.
"""
        ),
        code(
            """
tight_omega = pd.DataFrame(
    np.diag([0.0002, 0.0001]),
    index=view_labels,
    columns=view_labels,
)
loose_omega = pd.DataFrame(
    np.diag([0.0200, 0.0100]),
    index=view_labels,
    columns=view_labels,
)

tight = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    views,
    risk_aversion=risk_aversion,
    tau=tau,
    view_uncertainty=tight_omega,
)
loose = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    views,
    risk_aversion=risk_aversion,
    tau=tau,
    view_uncertainty=loose_omega,
)

pd.concat(
    [
        prior_returns.rename("prior"),
        loose.posterior_returns.rename("loose_views"),
        tight.posterior_returns.rename("tight_views"),
    ],
    axis=1,
)
"""
        ),
        markdown(
            """
## 10. Exercise — reverse the relative view

Change `equity_vs_bond` from +4% to −2%, keep the Gold view unchanged, and
answer:

1. Which posterior return changes most?
2. How does the long-only allocation change?
3. Does Gold change even though its view is unchanged? Explain using covariance.
4. Which input assumptions belong in a decision record?
"""
        ),
        code(
            """
exercise_views = views.copy()
exercise_views.loc["equity_vs_bond"] = -0.02

# Calculate a new posterior and allocation here.
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
exercise_result = black_litterman_posterior(
    market_weights,
    covariance,
    pick_matrix,
    exercise_views,
    risk_aversion=risk_aversion,
    tau=tau,
)
exercise_allocation = maximum_sharpe_ratio(
    risk_free_rate,
    exercise_result.posterior_returns,
    exercise_result.posterior_covariance,
)

pd.concat(
    [
        posterior.posterior_returns.rename("original_return"),
        exercise_result.posterior_returns.rename("exercise_return"),
        posterior_allocation.rename("original_weight"),
        exercise_allocation.rename("exercise_weight"),
    ],
    axis=1,
)
"""
        ),
        markdown(
            """
## Interpretation, pitfalls, and extensions

- Keep covariance, prior, views, uncertainty, and risk-free-rate conventions in
  the same units and horizon.
- `market_weights` are model inputs, not a guarantee that the observed market
  is efficient.
- A view row defines a portfolio. Check its signs and scaling before fitting.
- View uncertainty is variance, not an intuitive percentage confidence unless
  a separately documented mapping is used.
- Smaller \\(\\Omega\\) means stronger influence; it does not make a view more
  accurate.
- Posterior covariance still inherits covariance-estimation risk.
- Optimization constraints and transaction costs are downstream decisions.
- Compare allocations across priors, uncertainty settings, and holdout periods
  before treating the result as robust.

Possible extensions include confidence-to-uncertainty mappings, alternative
priors, factor views, turnover-aware optimization, and chronological validation.
Each requires its own explicit contract and tests.
"""
        ),
    ],
)


write_notebook(
    "09_graphical_analysis.ipynb",
    [
        markdown(
            """
# Level 4B — Graphical Analysis

**Audience:** analysts who want to explore conditional relationships and
clusters across a multi-asset return universe.

**Prerequisites:** Levels 1–2, basic covariance concepts, and the optional
`graphical` or `visualization` dependencies.

**Learning goals**

1. distinguish correlation from sparse partial correlation;
2. fit a labelled dependency network with deterministic clustering and layout;
3. inspect strong positive and negative conditional links;
4. create a Plotly network figure and a separate Dash application.

**Outline:** synthetic universe → fit → clusters → edges → Plotly → Dash
factory → threshold exercise.

The notebook uses synthetic monthly returns. It does not identify causal
relationships, recommend securities, or require credentials or network access.
"""
        ),
        markdown(
            """
## 1. Setup

The production calculation and visualization layers are deliberately separate:

- `graphical_analysis` estimates covariance, precision, partial correlation,
  clusters, edges, and two-dimensional coordinates;
- `visualization` converts the result into a Plotly figure;
- `dashboard` creates an optional Dash application around that figure.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.dashboard import (
    create_graphical_analysis_dashboard,
)
from asset_management_toolkit.graphical_analysis import graphical_analysis
from asset_management_toolkit.visualization import (
    dependency_network_figure,
)
"""
        ),
        markdown(
            """
## 2. Build a synthetic multi-asset universe

Two latent factors create related assets, while Gold and Cash add weaker
connections. Independent noise prevents exact duplicates. Labels remain attached
throughout the workflow.
"""
        ),
        code(
            """
generator = np.random.default_rng(17)
n_observations = 120
growth_factor = generator.normal(0.006, 0.035, n_observations)
defensive_factor = generator.normal(0.003, 0.018, n_observations)
noise = generator.normal(0.0, 0.008, (n_observations, 8))

returns = pd.DataFrame(
    {
        "US Equity": growth_factor + noise[:, 0],
        "Developed Equity": 0.85 * growth_factor + noise[:, 1],
        "Emerging Equity": 1.10 * growth_factor + noise[:, 2],
        "Government Bond": defensive_factor + noise[:, 3],
        "Credit": 0.45 * growth_factor + 0.55 * defensive_factor + noise[:, 4],
        "Infrastructure": 0.55 * growth_factor + 0.25 * defensive_factor + noise[:, 5],
        "Gold": generator.normal(0.003, 0.025, n_observations) + noise[:, 6],
        "Cash": generator.normal(0.001, 0.002, n_observations) + noise[:, 7] * 0.10,
    },
    index=pd.date_range("2016-01-31", periods=n_observations, freq="ME"),
)
returns.describe().loc[["mean", "std"]].T
"""
        ),
        markdown(
            """
## 3. Fit the sparse dependency network

`GraphicalLassoCV` estimates a sparse precision matrix after standardization.
Off-diagonal precision terms are converted to signed partial correlations.
Affinity propagation assigns cluster labels, and metric MDS supplies stable
two-dimensional coordinates for display.

The edge threshold controls what is reported and plotted; it does not refit the
underlying covariance model.
"""
        ),
        code(
            """
network = graphical_analysis(
    returns,
    edge_threshold=0.05,
    cv=5,
    random_state=7,
)

pd.DataFrame(
    {
        "cluster": network.cluster_labels,
        "x": network.embedding["x"],
        "y": network.embedding["y"],
    }
).sort_values(["cluster", "x"])
"""
        ),
        markdown(
            """
## 4. Inspect the strongest conditional links

Partial correlation asks whether two assets remain related after accounting
for the other assets in the fitted universe. A weak edge does not imply that
their ordinary pairwise correlation is zero.
"""
        ),
        code(
            """
network.edges.head(12)
"""
        ),
        code(
            """
comparison = pd.DataFrame(
    {
        "ordinary_correlation": returns.corr().stack(),
        "partial_correlation": network.partial_correlations.stack(),
    }
)
comparison = comparison[
    comparison.index.get_level_values(0)
    < comparison.index.get_level_values(1)
]
comparison.reindex(
    comparison["partial_correlation"].abs().sort_values(ascending=False).index
).head(12)
"""
        ),
        markdown(
            """
## 5. Create the Plotly research view

Edge width represents absolute partial-correlation strength. Teal edges are
positive and rose edges are negative. Node color represents cluster; hover text
reports the cluster and number of visible links.
"""
        ),
        code(
            """
figure = dependency_network_figure(
    network,
    title="Synthetic multi-asset dependency network",
)
figure
"""
        ),
        markdown(
            """
## 6. Filter without refitting

Filtering changes only the displayed subgraph. The fitted precision matrix and
cluster assignments remain those of the complete universe.
"""
        ),
        code(
            """
selected_cluster = [int(network.cluster_labels.loc["US Equity"])]
cluster_figure = dependency_network_figure(
    network,
    clusters=selected_cluster,
    title="Cluster containing US Equity",
)
cluster_figure
"""
        ),
        markdown(
            """
## 7. Create the Dash application

The factory returns a standard Dash app with a cluster selector and responsive
Plotly graph. A notebook should not start a long-running web server, so this
cell validates the app structure only.

To run it from a Python script:

```python
app = create_graphical_analysis_dashboard(network)
app.run(debug=False)
```
"""
        ),
        code(
            """
app = create_graphical_analysis_dashboard(
    network,
    title="Synthetic Asset Dependency Network",
)
{
    "app_title": app.title,
    "layout_type": type(app.layout).__name__,
    "callback_count": len(app.callback_map),
}
"""
        ),
        markdown(
            """
## 8. Exercise — change the reporting threshold

Refit with `edge_threshold=0.15`, then compare:

1. the number of reported edges;
2. cluster labels and embedding coordinates;
3. the strongest retained positive and negative links.

Which outputs should remain unchanged, and why?
"""
        ),
        code(
            """
# Try it here.
stricter = graphical_analysis(
    returns,
    edge_threshold=0.15,
    cv=5,
    random_state=7,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.Series(
    {
        "edges_at_0.05": len(network.edges),
        "edges_at_0.15": len(stricter.edges),
        "clusters_identical": network.cluster_labels.equals(
            stricter.cluster_labels
        ),
        "embedding_max_abs_difference": (
            network.embedding - stricter.embedding
        ).abs().to_numpy().max(),
    }
)
"""
        ),
        markdown(
            """
## Interpretation, pitfalls, and extensions

- Partial correlation is conditional association, not causality.
- Results depend on the selected universe, sample window, scaling, and
  regularization selected by cross-validation.
- Cluster numbers are arbitrary labels; interpret their members, not the
  numeric value.
- MDS coordinates support visualization and can rotate or reflect without
  changing pairwise geometry.
- Do not silently drop incomplete assets. Align and document the sample before
  fitting.
- A display threshold hides small edges but does not change the fitted model.
- Validate stability across chronological windows before using the network in
  a research decision.

Possible extensions include rolling networks, sector metadata, cluster
stability diagnostics, and downstream risk-budget comparisons. Each needs a
separate timing and validation contract.
"""
        ),
    ],
)


write_notebook(
    "10_returns_based_style_analysis.ipynb",
    [
        markdown(
            """
# Level 4C — Returns-Based Style Analysis

**Audience:** analysts who want to infer how a fund's returns behave relative
to investable style indices.

**Prerequisites:** Level 1, labelled pandas return series, and basic portfolio
weight interpretation.

**Learning goals**

1. distinguish return behavior from disclosed holdings;
2. estimate long-only, fully invested style exposures;
3. interpret fitted returns, residuals, and R-squared;
4. track changing exposures with rolling windows;
5. recognize index-selection, missing-data, and stability risks.

**Outline:** synthetic styles → static exposure → diagnostics → missing data →
rolling exposure → holdout review → exercise.

The notebook uses synthetic monthly decimal simple returns. It requires no
credentials, private data, or network access.
"""
        ),
        markdown(
            """
## 1. Setup and model contract

Returns-based style analysis finds a passive style-index mix whose return
variation most closely follows the fund:

\\[
\\min_w \\operatorname{Var}(r_{fund} - R_{style}w)
\\quad\\text{subject to}\\quad
\\sum_j w_j=1,\\; 0\\leq w_j\\leq1.
\\]

The estimated weights describe **return behavior over the sample**. They are
not a reconstruction of portfolio holdings.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.analytics import (
    rolling_style_exposures,
    style_exposures,
)
"""
        ),
        markdown(
            """
## 2. Create three synthetic style indices

The styles have different volatility and co-movement. Distinct return patterns
are essential: nearly identical indices cannot identify separate exposures.
"""
        ),
        code(
            """
generator = np.random.default_rng(42)
n_months = 72
dates = pd.date_range("2020-01-31", periods=n_months, freq="ME")

market = generator.normal(0.006, 0.035, n_months)
rate_factor = generator.normal(0.002, 0.012, n_months)
style_returns = pd.DataFrame(
    {
        "Global Equity": market + generator.normal(0.0, 0.010, n_months),
        "Government Bond": (
            -0.15 * market
            + rate_factor
            + generator.normal(0.0, 0.004, n_months)
        ),
        "Cash": generator.normal(0.0015, 0.0005, n_months),
    },
    index=dates,
)

style_returns.describe().loc[["mean", "std"]].T
"""
        ),
        markdown(
            """
## 3. Create a fund with a changing style

The first 36 months are equity-heavy. The final 36 months are bond-heavy.
A small constant selection return and residual noise are added so the example
is realistic without changing the intended style definition.
"""
        ),
        code(
            """
early_weights = pd.Series(
    {"Global Equity": 0.65, "Government Bond": 0.25, "Cash": 0.10}
)
late_weights = pd.Series(
    {"Global Equity": 0.25, "Government Bond": 0.60, "Cash": 0.15}
)

true_weights = pd.DataFrame(
    np.vstack(
        [
            np.repeat(early_weights.to_numpy()[None, :], 36, axis=0),
            np.repeat(late_weights.to_numpy()[None, :], 36, axis=0),
        ]
    ),
    index=dates,
    columns=style_returns.columns,
)
selection_return = 0.001
residual_noise = generator.normal(0.0, 0.002, n_months)
fund_returns = (
    (style_returns * true_weights).sum(axis=1)
    + selection_return
    + residual_noise
).rename("Synthetic Fund")

pd.concat(
    [
        true_weights.iloc[[0, -1]].set_axis(["early", "late"]),
        pd.Series(
            {
                "early": fund_returns.iloc[:36].mean(),
                "late": fund_returns.iloc[36:].mean(),
            },
            name="mean_fund_return",
        ),
    ],
    axis=1,
)
"""
        ),
        markdown(
            """
## 4. Estimate one full-sample style

One estimate summarizes the complete period. Because the fund changed style
halfway through, these weights should be interpreted as an average behavioral
exposure rather than a stable mandate.
"""
        ),
        code(
            """
full_sample = style_exposures(fund_returns, style_returns)

pd.DataFrame(
    {
        "estimated_weight": full_sample.weights,
        "early_true_weight": early_weights,
        "late_true_weight": late_weights,
    }
)
"""
        ),
        markdown(
            """
## 5. Review fit and residuals

R-squared measures in-sample return variation explained by the style mix.
Residuals retain their mean: the optimizer minimizes tracking variance rather
than forcing the average selection return to zero.
"""
        ),
        code(
            """
pd.Series(
    {
        "observations": full_sample.n_observations,
        "r_squared": full_sample.r_squared,
        "centered_residual_sum_squares": full_sample.residual_sum_squares,
        "mean_residual": full_sample.residuals.mean(),
        "residual_volatility": full_sample.residuals.std(ddof=1),
    },
    name="fit_diagnostic",
)
"""
        ),
        code(
            """
pd.concat(
    [
        fund_returns,
        full_sample.fitted_returns,
        full_sample.residuals,
    ],
    axis=1,
).head()
"""
        ),
        markdown(
            """
## 6. Confirm that average selection return does not change style

Adding a constant to every fund return changes the residual mean but not the
tracking variance. A correct Sharpe-style objective therefore leaves the
estimated exposures unchanged.
"""
        ),
        code(
            """
shifted = style_exposures(fund_returns + 0.01, style_returns)

pd.DataFrame(
    {
        "original": full_sample.weights,
        "fund_plus_1pct_each_month": shifted.weights,
        "difference": shifted.weights - full_sample.weights,
    }
)
"""
        ),
        markdown(
            """
## 7. Handle missing observations explicitly

The API aligns inputs by index and jointly drops rows with any missing fund or
style return. It never replaces a missing return with zero. Always review the
resulting observation count.
"""
        ),
        code(
            """
styles_with_gap = style_returns.copy()
styles_with_gap.loc[dates[10], "Government Bond"] = np.nan
fund_with_shorter_history = fund_returns.iloc[2:]

complete_case = style_exposures(
    fund_with_shorter_history,
    styles_with_gap,
)
pd.Series(
    {
        "fund_rows": len(fund_with_shorter_history),
        "style_rows": len(styles_with_gap),
        "complete_rows_used": complete_case.n_observations,
    }
)
"""
        ),
        markdown(
            """
## 8. Estimate rolling exposures

A 24-month trailing window is long enough to estimate the three styles while
remaining responsive to the synthetic regime change. `step=3` reports one
estimate every three complete months.
"""
        ),
        code(
            """
rolling = rolling_style_exposures(
    fund_returns,
    style_returns,
    window=24,
    step=3,
)

rolling.weights.iloc[[0, len(rolling.weights) // 2, -1]]
"""
        ),
        code(
            """
rolling_diagnostics = pd.concat(
    [rolling.r_squared, rolling.residual_sum_squares],
    axis=1,
)
rolling_diagnostics.tail()
"""
        ),
        markdown(
            """
## 9. Compare rolling estimates with known synthetic exposures

The first reported window belongs entirely to the early regime. The last
window belongs entirely to the late regime. Intermediate windows blend both.
"""
        ),
        code(
            """
comparison = pd.DataFrame(
    {
        "first_rolling": rolling.weights.iloc[0],
        "early_true": early_weights,
        "last_rolling": rolling.weights.iloc[-1],
        "late_true": late_weights,
    }
)
comparison
"""
        ),
        markdown(
            """
## 10. Chronological holdout review

An in-sample R-squared is not evidence that exposures persist. Fit only the
first 36 months, freeze those weights, and evaluate the next 12 months after
the synthetic style change begins.
"""
        ),
        code(
            """
estimation = style_exposures(
    fund_returns.iloc[:36],
    style_returns.iloc[:36],
)
holdout_styles = style_returns.iloc[36:48]
holdout_fund = fund_returns.iloc[36:48]
holdout_fitted = holdout_styles @ estimation.weights
holdout_residual = holdout_fund - holdout_fitted

pd.Series(
    {
        "estimation_r_squared": estimation.r_squared,
        "holdout_residual_mean": holdout_residual.mean(),
        "holdout_residual_volatility": holdout_residual.std(ddof=1),
    }
)
"""
        ),
        markdown(
            """
## 11. Exercise — compare window lengths

Estimate rolling exposures with 12- and 36-month windows using `step=3`.
Compare:

1. how quickly equity exposure responds after month 36;
2. the variability of each estimated weight;
3. the median rolling R-squared.

Which window is more responsive, and which is more stable?
"""
        ),
        code(
            """
# Try it here.
short_window = rolling_style_exposures(
    fund_returns,
    style_returns,
    window=12,
    step=3,
)
long_window = rolling_style_exposures(
    fund_returns,
    style_returns,
    window=36,
    step=3,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.DataFrame(
    {
        "12_month_weight_std": short_window.weights.std(),
        "36_month_weight_std": long_window.weights.std(),
        "12_month_last_weight": short_window.weights.iloc[-1],
        "36_month_last_weight": long_window.weights.iloc[-1],
    }
).join(
    pd.DataFrame(
        {
            "12_month_median_r_squared": [short_window.r_squared.median()] * 3,
            "36_month_median_r_squared": [long_window.r_squared.median()] * 3,
        },
        index=style_returns.columns,
    )
)
"""
        ),
        markdown(
            """
## Interpretation, pitfalls, and extensions

- Estimated weights describe return behavior, not holdings.
- Index selection defines the answer. Use broad, investable, economically
  distinct style indices in consistent currencies and return conventions.
- High R-squared is in-sample fit, not proof of manager skill or persistence.
- Short windows react faster but usually produce noisier exposures.
- Long windows are smoother but can conceal real style changes.
- Do not fill missing returns with zero; review complete-case sample size.
- Similar or redundant style indices can make exposures non-identifiable.
- Fund fees, timing differences, stale pricing, derivatives, leverage, and
  nonlinear exposures can appear in residuals or distort weights.

Useful extensions include confidence intervals, turnover diagnostics,
structural-break tests, and walk-forward comparison against simple style
benchmarks. Each requires its own statistical contract.
"""
        ),
    ],
)


write_notebook(
    "13_covariance_estimation.ipynb",
    [
        markdown(
            """
# Level 13 — Covariance Estimation

**Audience:** analysts who already understand sample volatility and portfolio
variance.

**Prerequisites:** Levels 1–2, NumPy, pandas, and labelled return data.

**Learning goals**

1. estimate a labelled sample covariance matrix;
2. construct a constant-correlation target;
3. blend the two with an explicit shrinkage intensity;
4. show how covariance choice changes GMV weights.

This tutorial independently implements public covariance-estimation concepts
using synthetic monthly returns. It contains no course data or network access.
"""
        ),
        markdown(
            """
## 1. Setup and synthetic observations

All three assets use the same 72 monthly observations. The random seed makes
the example exactly reproducible.
"""
        ),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.estimation import (
    constant_correlation_covariance,
    sample_covariance,
    shrink_covariance,
)
from asset_management_toolkit.portfolio import global_minimum_variance
"""
        ),
        code(
            """
rng = np.random.default_rng(20260728)
dates = pd.date_range("2020-01-31", periods=72, freq="ME")
monthly_returns = pd.DataFrame(
    rng.multivariate_normal(
        mean=[0.006, 0.002, 0.004],
        cov=[
            [0.0025, 0.0002, 0.0005],
            [0.0002, 0.0004, 0.0001],
            [0.0005, 0.0001, 0.0012],
        ],
        size=len(dates),
    ),
    index=dates,
    columns=["Equity", "Bonds", "Real assets"],
)
monthly_returns.describe().loc[["mean", "std"]]
"""
        ),
        markdown(
            """
## 2. Sample covariance

`sample_covariance` uses one complete observation set for every pair. Its
default `ddof=1` matches the usual sample covariance convention.
"""
        ),
        code(
            """
sample = sample_covariance(monthly_returns)
sample
"""
        ),
        markdown(
            """
## 3. Constant-correlation target

The target keeps each sample variance on the diagonal. It replaces all
off-diagonal correlations with their cross-sectional average.
"""
        ),
        code(
            """
constant_correlation = constant_correlation_covariance(monthly_returns)

def covariance_to_correlation(covariance):
    volatility = np.sqrt(np.diag(covariance))
    return covariance / np.outer(volatility, volatility)

pd.DataFrame(
    covariance_to_correlation(constant_correlation),
    index=constant_correlation.index,
    columns=constant_correlation.columns,
)
"""
        ),
        markdown(
            """
## 4. Caller-controlled shrinkage

An intensity of 0.40 gives 40% weight to the structured target and 60% to the
sample estimate. It is a chosen scenario, not an estimated optimal
Ledoit–Wolf coefficient.
"""
        ),
        code(
            """
shrunk = shrink_covariance(monthly_returns, intensity=0.40)
pd.DataFrame(
    {
        "sample": sample.stack(),
        "constant_correlation": constant_correlation.stack(),
        "shrunk_40_percent": shrunk.stack(),
    }
).head(9)
"""
        ),
        markdown(
            """
## 5. Portfolio sensitivity

GMV construction can amplify covariance-estimation differences. Comparing
weights makes model risk visible rather than hiding it inside one matrix.
"""
        ),
        code(
            """
gmv_comparison = pd.concat(
    {
        "Sample": global_minimum_variance(sample),
        "Constant correlation": global_minimum_variance(constant_correlation),
        "40% shrinkage": global_minimum_variance(shrunk),
    },
    axis=1,
)
gmv_comparison
"""
        ),
        markdown(
            """
## Exercise — inspect the shrinkage path

Calculate GMV weights at intensities 0, 0.25, 0.50, 0.75, and 1. Which asset's
weight is most sensitive to the covariance assumption?
"""
        ),
        code(
            """
# Try it here.
intensities = [0.0, 0.25, 0.50, 0.75, 1.0]
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.DataFrame(
    {
        intensity: global_minimum_variance(
            shrink_covariance(monthly_returns, intensity=intensity)
        )
        for intensity in intensities
    }
).T.rename_axis("shrinkage_intensity")
"""
        ),
        markdown(
            """
## Interpretation and pitfalls

- Covariance estimates depend on the sampling window and return frequency.
- Missing values are rejected to avoid inconsistent pairwise sample sizes.
- A constant-correlation target is structured, not automatically correct.
- Fixed intensity is a sensitivity parameter, not an optimal estimate.
- Portfolio-weight stability and chronological holdout risk matter more than
  in-sample matrix fit.

Next: use an estimated covariance matrix to construct explicit risk budgets.
"""
        ),
    ],
)


write_notebook(
    "14_risk_budgeting_and_erc.ipynb",
    [
        markdown(
            """
# Level 14 — Risk Budgeting and Equal Risk Contribution

**Audience:** portfolio analysts who want to allocate portfolio risk rather
than capital alone.

**Prerequisites:** Levels 2 and 12, covariance matrices, and portfolio
volatility.

**Learning goals**

1. distinguish capital weights from normalized risk contributions;
2. construct equal-risk-contribution (ERC) weights;
3. target a non-equal risk budget;
4. create transparent equal- and capitalization-weight policies.

The examples use a synthetic covariance matrix and market capitalizations.
"""
        ),
        markdown("## 1. Setup"),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.portfolio import (
    capitalization_weights,
    capped_equal_weights,
    equal_risk_contribution_weights,
    equal_weights,
    risk_contributions,
    target_risk_contribution_weights,
)
"""
        ),
        code(
            """
assets = pd.Index(["Equity", "Bonds", "Real assets"])
covariance = pd.DataFrame(
    [
        [0.0400, 0.0030, 0.0090],
        [0.0030, 0.0100, 0.0020],
        [0.0090, 0.0020, 0.0250],
    ],
    index=assets,
    columns=assets,
)
covariance
"""
        ),
        markdown(
            """
## 2. Equal capital is not equal risk

Equal weights allocate one-third of capital to every asset. Different
volatilities and correlations mean their risk contributions need not be equal.
"""
        ),
        code(
            """
equal_capital = equal_weights(assets)
pd.DataFrame(
    {
        "capital_weight": equal_capital,
        "risk_contribution": risk_contributions(
            equal_capital,
            covariance,
        ),
    }
)
"""
        ),
        markdown(
            """
## 3. Equal-risk-contribution portfolio

ERC solves for long-only, fully-invested weights whose normalized
contributions to portfolio volatility are equal.
"""
        ),
        code(
            """
erc_weights = equal_risk_contribution_weights(covariance)
pd.DataFrame(
    {
        "capital_weight": erc_weights,
        "risk_contribution": risk_contributions(
            erc_weights,
            covariance,
        ),
    }
)
"""
        ),
        markdown(
            """
## 4. Target a deliberate risk budget

The target below assigns 50% of total ex-ante risk to Equity, 20% to Bonds,
and 30% to Real assets. Risk budgets must be strictly positive and sum to one.
"""
        ),
        code(
            """
target_budget = pd.Series(
    [0.50, 0.20, 0.30],
    index=assets,
    name="target_contribution",
)
target_weights = target_risk_contribution_weights(
    target_budget,
    covariance,
)
pd.DataFrame(
    {
        "target_risk": target_budget,
        "achieved_risk": risk_contributions(
            target_weights,
            covariance,
        ),
        "capital_weight": target_weights,
    }
)
"""
        ),
        markdown(
            """
## 5. Transparent weighting policies

Capitalization weights use caller-supplied market values. Capped equal weights
can screen very small assets and prevent an equal-weight rule from exceeding a
multiple of capitalization weight.
"""
        ),
        code(
            """
market_caps = pd.Series(
    [700.0, 200.0, 100.0],
    index=assets,
    name="market_capitalization",
)
pd.concat(
    {
        "Equal": equal_weights(assets),
        "Capitalization": capitalization_weights(market_caps),
        "Screened and capped": capped_equal_weights(
            market_caps,
            minimum_capitalization_weight=0.05,
            maximum_multiple_of_cap_weight=2.0,
        ),
    },
    axis=1,
)
"""
        ),
        markdown(
            """
## Exercise — tilt the risk budget

Change the target from `[0.50, 0.20, 0.30]` to `[0.30, 0.40, 0.30]`. Which
capital weight changes most? Verify the achieved contributions directly.
"""
        ),
        code(
            """
# Try it here.
defensive_budget = pd.Series(
    [0.30, 0.40, 0.30],
    index=assets,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
defensive_weights = target_risk_contribution_weights(
    defensive_budget,
    covariance,
)
pd.DataFrame(
    {
        "original_weight": target_weights,
        "defensive_weight": defensive_weights,
        "weight_change": defensive_weights - target_weights,
        "achieved_risk": risk_contributions(
            defensive_weights,
            covariance,
        ),
    }
)
"""
        ),
        markdown(
            """
## Interpretation and pitfalls

- Risk budgets depend entirely on the covariance estimate.
- Equal risk contribution is not equal capital weight.
- Ex-ante contributions can differ from realized contributions.
- Long-only target budgets may be infeasible or numerically fragile for some
  covariance structures.
- Static weighting policies do not define rebalance timing. Feed dated target
  weights into `run_weight_backtest` to model drift, turnover, and costs.

Next: estimate fund exposures to labelled return factors.
"""
        ),
    ],
)


write_notebook(
    "15_factor_regression.ipynb",
    [
        markdown(
            """
# Level 15 — Labelled Factor Regression

**Audience:** analysts evaluating fund or strategy return exposures.

**Prerequisites:** basic linear regression, periodic decimal returns, and
chronological train/evaluation discipline.

**Learning goals**

1. recover labelled alpha and factor betas from synthetic returns;
2. interpret fit and classical inference diagnostics;
3. inspect rolling exposure changes;
4. create additive model-implied factor attribution.

No Fama–French dataset is bundled. Users remain responsible for factor
definitions, licensing, frequency, and risk-free-rate conventions.
"""
        ),
        markdown("## 1. Setup and synthetic factors"),
        code(
            BOOTSTRAP
            + """
from asset_management_toolkit.analytics import (
    factor_regression,
    factor_return_attribution,
    rolling_factor_regression,
)
"""
        ),
        code(
            """
rng = np.random.default_rng(20260728)
dates = pd.date_range("2018-01-31", periods=96, freq="ME")
factors = pd.DataFrame(
    {
        "Market": rng.normal(0.006, 0.040, len(dates)),
        "Value": rng.normal(0.002, 0.025, len(dates)),
        "Quality": rng.normal(0.002, 0.020, len(dates)),
    },
    index=dates,
)
noise = rng.normal(0.0, 0.004, len(dates))
fund_returns = pd.Series(
    0.001
    + 1.10 * factors["Market"]
    + 0.35 * factors["Value"]
    - 0.20 * factors["Quality"]
    + noise,
    index=dates,
    name="Synthetic fund",
)
pd.concat([fund_returns, factors], axis=1).head()
"""
        ),
        markdown(
            """
## 2. Static OLS exposures

The regression models fund excess return as an intercept plus labelled factor
returns. Alpha is annualized by multiplication; betas remain unitless.
"""
        ),
        code(
            """
result = factor_regression(
    fund_returns,
    factors,
    periods_per_year=12,
)
pd.DataFrame(
    {
        "coefficient": result.coefficients,
        "standard_error": result.standard_errors,
        "t_statistic": result.t_statistics,
        "p_value": result.p_values,
    }
)
"""
        ),
        code(
            """
pd.Series(
    {
        "r_squared": result.r_squared,
        "adjusted_r_squared": result.adjusted_r_squared,
        "annualized_residual_volatility": result.residual_volatility,
        "complete_observations": result.n_observations,
    }
)
"""
        ),
        markdown(
            """
## 3. Rolling exposures and a regime change

Create a second synthetic fund whose Market and Value betas change halfway
through the sample. Trailing windows reveal the transition gradually.
"""
        ),
        code(
            """
changing_fund = pd.Series(
    np.concatenate(
        [
            (
                0.001
                + 1.30 * factors["Market"].iloc[:48]
                + 0.10 * factors["Value"].iloc[:48]
                + noise[:48]
            ).to_numpy(),
            (
                -0.0005
                + 0.55 * factors["Market"].iloc[48:]
                + 0.85 * factors["Value"].iloc[48:]
                + noise[48:]
            ).to_numpy(),
        ]
    ),
    index=dates,
    name="Changing fund",
)
rolling = rolling_factor_regression(
    changing_fund,
    factors,
    window=36,
    step=6,
    periods_per_year=12,
)
rolling.betas.tail()
"""
        ),
        markdown(
            """
## 4. Additive model-implied attribution

Attribution multiplies each periodic factor return by its fitted exposure and
adds periodic alpha. Realized fund return can still differ by the residual.
"""
        ),
        code(
            """
attribution = factor_return_attribution(
    result.betas,
    factors,
    alpha=result.alpha / 12,
)
attribution.head()
"""
        ),
        code(
            """
pd.DataFrame(
    {
        "model_fitted": attribution["total"],
        "regression_fitted": result.fitted_returns,
        "realized_fund": fund_returns,
        "residual": result.residuals,
    }
).head()
"""
        ),
        markdown(
            """
## Exercise — compare stable and changing exposures

Run the same 36-month rolling regression on the stable synthetic fund. Compare
the standard deviation of rolling betas with the changing fund.
"""
        ),
        code(
            """
# Try it here.
stable_rolling = rolling_factor_regression(
    fund_returns,
    factors,
    window=36,
    step=6,
    periods_per_year=12,
)
"""
        ),
        markdown("### Answer scaffold"),
        code(
            """
pd.DataFrame(
    {
        "stable_beta_std": stable_rolling.betas.std(),
        "changing_beta_std": rolling.betas.std(),
        "stable_last_beta": stable_rolling.betas.iloc[-1],
        "changing_last_beta": rolling.betas.iloc[-1],
    }
)
"""
        ),
        markdown(
            """
## Interpretation and pitfalls

- Factor exposure is descriptive association, not proof of causality or skill.
- Factor returns and fund returns must use consistent frequency, currency, and
  return conventions.
- Alpha depends on factor choice and the risk-free-rate convention.
- Classical standard errors here assume homoskedastic residuals; HAC inference
  is not claimed.
- Rolling windows are trailing and descriptive, not forecasts.
- Collinear factors make unconstrained OLS exposures unidentified.
- High R-squared does not establish out-of-sample persistence.

Useful extensions include chronological holdout review, robust inference, and
regularized models for wider or collinear factor sets.
"""
        ),
    ],
)
