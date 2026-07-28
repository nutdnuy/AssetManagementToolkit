"""Build the executable classical time-series forecasting tutorial."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tutorials" / "12_time_series_forecasting.ipynb"


def markdown(text: str) -> nbformat.NotebookNode:
    """Create a trimmed Markdown cell."""
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbformat.NotebookNode:
    """Create a trimmed code cell."""
    return nbformat.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Level 12 — Classical Time-Series Forecasting

**Audience:** analysts who work with monthly asset, fund, economic, or business
series and want a leakage-aware forecasting workflow.

**Prerequisites:** basic pandas, labelled indexes, and train/test evaluation.
Install the tutorial dependencies with `python -m pip install -e ".[tutorials]"`.

**Learning goals**

1. validate a regular labelled time series and create trailing features;
2. split observations chronologically without shuffling;
3. compare a seasonal-naive baseline with Holt-Winters and SARIMA;
4. calculate aligned forecast metrics;
5. use rolling-origin evaluation instead of relying on in-sample fit.

**Outline:** synthetic monthly series → trailing features → holdout split →
baseline → Holt-Winters → metrics → decomposition → SARIMA → rolling origin →
exercise.

The notebook uses deterministic synthetic levels. It requires no credentials,
network access, private data, or historical model artifacts.
"""
    ),
    markdown(
        """
## 1. Setup

The production API keeps preparation, diagnostics, forecasting, and evaluation
separate. Statsmodels is used only by the optional classical-model functions.
"""
    ),
    code(
        """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path.cwd()
if not (project_root / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

pd.options.display.float_format = "{:.4f}".format

from asset_management_toolkit.time_series import (
    chronological_train_test_split,
    decompose_time_series,
    exponential_smoothing_forecast,
    forecast_metrics,
    moving_average_features,
    sarima_forecast,
    seasonal_naive_forecast,
    walk_forward_forecast,
)
"""
    ),
    markdown(
        """
## 2. Create a synthetic monthly level series

The series combines a smooth trend, a repeating 12-month seasonal pattern, and
seeded noise. A `PeriodIndex` makes the monthly timing contract explicit.
"""
    ),
    code(
        """
generator = np.random.default_rng(17)
index = pd.period_range("2018-01", periods=84, freq="M")
trend = np.linspace(100.0, 125.0, len(index))
seasonal_pattern = np.tile(
    [0.0, 2.0, 4.0, 3.0, 1.0, -1.0, -3.0, -2.0, 0.0, 1.0, 3.0, 2.0],
    7,
)
monthly_levels = pd.Series(
    trend + seasonal_pattern + generator.normal(0.0, 0.5, len(index)),
    index=index,
    name="synthetic_level",
)

monthly_levels.to_frame().head()
"""
    ),
    code(
        """
ax = monthly_levels.plot(
    figsize=(11, 4),
    title="Synthetic monthly level series",
    color="#7c5cff",
)
ax.set_xlabel("Month")
ax.set_ylabel("Level")
plt.show()
"""
    ),
    markdown(
        """
## 3. Build trailing moving-average features

Simple and exponentially weighted moving averages use only the current and
past observations. They summarize history; they are not forecasts by
themselves.
"""
    ),
    code(
        """
features = moving_average_features(
    monthly_levels,
    simple_windows=[3, 12],
    exponential_spans=[6, 12],
)
features.tail()
"""
    ),
    code(
        """
features[["synthetic_level", "sma_12", "ewma_12"]].plot(
    figsize=(11, 4),
    title="Trailing features",
)
plt.show()
"""
    ),
    markdown(
        """
## 4. Reserve the latest year as a chronological holdout

Random splitting would allow later observations to influence an earlier model.
The final 12 months are therefore held out intact.
"""
    ),
    code(
        """
train, test = chronological_train_test_split(
    monthly_levels,
    test_size=12,
)

pd.Series(
    {
        "train_start": train.index[0],
        "train_end": train.index[-1],
        "test_start": test.index[0],
        "test_end": test.index[-1],
        "n_train": len(train),
        "n_test": len(test),
    },
    name="split",
)
"""
    ),
    markdown(
        """
## 5. Establish a seasonal-naive baseline

The seasonal-naive forecast repeats the last observed 12-month cycle. A more
complex model should be judged against this transparent benchmark.
"""
    ),
    code(
        """
baseline = seasonal_naive_forecast(
    train,
    horizon=len(test),
    seasonal_period=12,
)
baseline.head()
"""
    ),
    markdown(
        """
## 6. Fit additive Holt-Winters

The additive specification assumes the trend and seasonal amplitude combine in
level units. The training sample contains six complete seasonal cycles.
"""
    ),
    code(
        """
holt_winters = exponential_smoothing_forecast(
    train,
    horizon=len(test),
    trend="add",
    seasonal="add",
    seasonal_periods=12,
)

pd.Series(
    {
        "aic": holt_winters.aic,
        "bic": holt_winters.bic,
        "residual_mean": holt_winters.residuals.mean(),
        "residual_rmse": np.sqrt(np.mean(holt_winters.residuals**2)),
    },
    name="holt_winters_fit",
)
"""
    ),
    markdown(
        """
## 7. Compare out-of-sample forecast metrics

Errors are defined as `forecast − actual`. Negative bias therefore indicates
under-forecasting. MAPE omits zero actuals; symmetric MAPE omits rows where
both values are zero.
"""
    ),
    code(
        """
holdout_metrics = pd.DataFrame(
    {
        "seasonal_naive": forecast_metrics(test, baseline),
        "holt_winters": forecast_metrics(test, holt_winters.forecast),
    }
)
holdout_metrics
"""
    ),
    code(
        """
comparison = pd.concat(
    [
        train.rename("train"),
        test.rename("actual"),
        baseline.rename("seasonal_naive"),
        holt_winters.forecast.rename("holt_winters"),
    ],
    axis=1,
)
comparison.iloc[-30:].plot(
    figsize=(11, 4),
    title="Chronological holdout forecasts",
)
plt.show()
"""
    ),
    markdown(
        """
## 8. Decompose the training sample

Classical decomposition separates an observed series into trend, seasonal, and
residual components. It is a diagnostic description, not causal evidence.
"""
    ),
    code(
        """
decomposition = decompose_time_series(
    train,
    period=12,
    model="additive",
    extrapolate_trend=1,
)

pd.DataFrame(
    {
        "observed": decomposition.observed,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.residual,
    }
).tail()
"""
    ),
    code(
        """
decomposition_frame = pd.DataFrame(
    {
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.residual,
    }
)
decomposition_frame.plot(
    subplots=True,
    figsize=(11, 7),
    title=["Trend", "Seasonal", "Residual"],
)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        """
## 9. Fit an explicitly specified SARIMA model

This API deliberately does not run automated order search. The model contract
records the selected non-seasonal and seasonal orders so order selection can be
reviewed separately from holdout evaluation.
"""
    ),
    code(
        """
sarima = sarima_forecast(
    train,
    horizon=len(test),
    order=(1, 1, 1),
    seasonal_order=(1, 0, 0, 12),
    trend="n",
)

pd.DataFrame(
    {
        "seasonal_naive": forecast_metrics(test, baseline),
        "holt_winters": forecast_metrics(test, holt_winters.forecast),
        "sarima": forecast_metrics(test, sarima.forecast),
    }
)
"""
    ),
    markdown(
        """
## 10. Use rolling-origin evaluation

One holdout can be unusually easy or difficult. Rolling-origin evaluation
repeats the forecast using only data available before each test block.
"""
    ),
    code(
        """
def seasonal_baseline_forecaster(
    history: pd.Series,
    horizon: int,
) -> pd.Series:
    return seasonal_naive_forecast(
        history,
        horizon=horizon,
        seasonal_period=12,
    )


walk_forward = walk_forward_forecast(
    monthly_levels,
    seasonal_baseline_forecaster,
    initial_train_size=48,
    test_size=12,
    window="expanding",
)
walk_forward.head()
"""
    ),
    code(
        """
fold_metrics = pd.DataFrame(
    {
        fold: forecast_metrics(
            block["actual"],
            block["forecast"],
        )
        for fold, block in walk_forward.groupby("fold", sort=True)
    }
).T
fold_metrics
"""
    ),
    markdown(
        """
## 11. Exercise — shorten the seasonal cycle

Change `seasonal_period` from 12 to 6 in the baseline and rerun the same
rolling-origin evaluation.

1. Compare mean RMSE across folds.
2. Inspect forecast bias.
3. Explain why a six-month cycle is misspecified for the synthetic generator.
"""
    ),
    code(
        """
# Try it here.
def six_month_baseline(
    history: pd.Series,
    horizon: int,
) -> pd.Series:
    return seasonal_naive_forecast(
        history,
        horizon=horizon,
        seasonal_period=6,
    )


six_month_walk_forward = walk_forward_forecast(
    monthly_levels,
    six_month_baseline,
    initial_train_size=48,
    test_size=12,
)
"""
    ),
    markdown("### Answer scaffold"),
    code(
        """
def summarize_walk_forward(result: pd.DataFrame) -> pd.Series:
    metrics_by_fold = pd.DataFrame(
        {
            fold: forecast_metrics(
                block["actual"],
                block["forecast"],
            )
            for fold, block in result.groupby("fold", sort=True)
        }
    ).T
    return metrics_by_fold.mean()


pd.DataFrame(
    {
        "12_month_cycle": summarize_walk_forward(walk_forward),
        "6_month_cycle": summarize_walk_forward(six_month_walk_forward),
    }
)
"""
    ),
    markdown(
        """
## Interpretation, pitfalls, and extensions

- Never shuffle a forecasting holdout or tune on the final test period.
- Preserve regular, unique, increasing time labels; do not silently guess an
  irregular future frequency.
- Compare complex models with a seasonal-naive baseline.
- AIC and BIC compare likelihood-based fits on the same training sample; they
  do not replace out-of-sample metrics.
- Decomposition describes recurring structure but does not identify economic
  causes.
- MAPE needs care near zero; review MAE, RMSE, bias, and sMAPE together.
- Automated ARIMA order search and neural-network forecasting remain deferred
  until they have explicit nested-validation and reproducibility contracts.

Useful extensions include forecast intervals, structural-break diagnostics,
multiple seasonalities, and expanding-versus-rolling window comparisons.
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
