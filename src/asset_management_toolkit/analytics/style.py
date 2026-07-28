"""Returns-based style analysis with long-only, fully invested exposures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize


@dataclass(frozen=True)
class StyleAnalysisResult:
    """Estimated style exposures and in-sample fit diagnostics.

    ``residual_sum_squares`` is calculated after subtracting the residual mean,
    matching the tracking-variance objective rather than penalizing average
    selection return.
    """

    weights: pd.Series
    fitted_returns: pd.Series
    residuals: pd.Series
    residual_sum_squares: float
    r_squared: float
    n_observations: int


@dataclass(frozen=True)
class RollingStyleAnalysisResult:
    """Rolling style exposures and window-level fit diagnostics."""

    weights: pd.DataFrame
    residual_sum_squares: pd.Series
    r_squared: pd.Series
    window: int


def style_exposures(
    fund_returns: pd.Series,
    style_returns: pd.DataFrame,
) -> StyleAnalysisResult:
    """Estimate long-only, fully invested returns-based style exposures.

    The function minimizes the variance of the difference between fund returns
    and the returns of a style-index portfolio. Style weights are constrained to
    the interval ``[0, 1]`` and to sum to one. The residual mean is not forced
    to zero, so a persistent return difference does not alter the exposures.

    Inputs are interpreted as decimal simple returns. They are aligned by index,
    and rows containing a missing fund or style return are excluded jointly.
    """
    aligned = _align_style_inputs(fund_returns, style_returns)
    return _fit_aligned_style(aligned.iloc[:, 0], aligned.iloc[:, 1:])


def rolling_style_exposures(
    fund_returns: pd.Series,
    style_returns: pd.DataFrame,
    window: int,
    *,
    step: int = 1,
) -> RollingStyleAnalysisResult:
    """Estimate style exposures over trailing fixed-observation windows.

    ``window`` counts jointly complete observations after index alignment.
    Each result row is labelled with the final observation in its window.
    ``step`` controls the number of complete observations between window ends.
    """
    aligned = _align_style_inputs(fund_returns, style_returns)
    _validate_positive_integer(window, "window")
    _validate_positive_integer(step, "step")
    minimum = max(2, style_returns.shape[1])
    if window < minimum:
        raise ValueError(
            f"window must contain at least {minimum} observations for "
            f"{style_returns.shape[1]} styles"
        )
    if window > len(aligned):
        raise ValueError("window cannot exceed the number of complete observations")
    if not aligned.index.is_monotonic_increasing:
        raise ValueError("rolling inputs must have an increasing index")

    results: list[StyleAnalysisResult] = []
    labels = []
    for end in range(window, len(aligned) + 1, step):
        sample = aligned.iloc[end - window : end]
        results.append(_fit_aligned_style(sample.iloc[:, 0], sample.iloc[:, 1:]))
        labels.append(sample.index[-1])

    result_index = pd.Index(labels, name=aligned.index.name)
    weights = pd.DataFrame(
        [result.weights.to_numpy() for result in results],
        index=result_index,
        columns=style_returns.columns.copy(),
        dtype=float,
    )
    weights.columns.name = style_returns.columns.name
    residual_sum_squares = pd.Series(
        [result.residual_sum_squares for result in results],
        index=result_index,
        name="residual_sum_squares",
        dtype=float,
    )
    r_squared = pd.Series(
        [result.r_squared for result in results],
        index=result_index,
        name="r_squared",
        dtype=float,
    )
    return RollingStyleAnalysisResult(
        weights=weights,
        residual_sum_squares=residual_sum_squares,
        r_squared=r_squared,
        window=window,
    )


def _align_style_inputs(
    fund_returns: pd.Series,
    style_returns: pd.DataFrame,
) -> pd.DataFrame:
    fund = _validated_fund_returns(fund_returns)
    styles = _validated_style_returns(style_returns)
    aligned = pd.concat(
        [fund.rename("__fund_return__"), styles],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("fund_returns and style_returns have no complete overlap")

    minimum = max(2, styles.shape[1])
    if len(aligned) < minimum:
        raise ValueError(
            f"at least {minimum} complete observations are required for "
            f"{styles.shape[1]} styles"
        )
    _validate_style_identifiability(aligned.iloc[:, 1:].to_numpy())
    return aligned


def _validated_fund_returns(fund_returns: pd.Series) -> pd.Series:
    if not isinstance(fund_returns, pd.Series):
        raise TypeError("fund_returns must be a pandas Series")
    if fund_returns.empty:
        raise ValueError("fund_returns must not be empty")
    if not fund_returns.index.is_unique:
        raise ValueError("fund_returns index must be unique")
    if not pd.api.types.is_numeric_dtype(fund_returns):
        raise TypeError("fund_returns must be numeric")
    result = fund_returns.astype(float).copy()
    _validate_simple_return_values(result.to_numpy(), "fund_returns")
    if result.dropna().empty:
        raise ValueError("fund_returns must contain at least one observation")
    return result


def _validated_style_returns(style_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(style_returns, pd.DataFrame):
        raise TypeError("style_returns must be a pandas DataFrame")
    if style_returns.empty or style_returns.shape[1] == 0:
        raise ValueError("style_returns must contain at least one style and one row")
    if not style_returns.index.is_unique:
        raise ValueError("style_returns index must be unique")
    if not style_returns.columns.is_unique:
        raise ValueError("style_returns columns must be unique")
    non_numeric = [
        str(column)
        for column in style_returns
        if not pd.api.types.is_numeric_dtype(style_returns[column])
    ]
    if non_numeric:
        raise TypeError(
            "style_returns columns must be numeric: " + ", ".join(non_numeric)
        )
    result = style_returns.astype(float).copy(deep=True)
    _validate_simple_return_values(result.to_numpy(), "style_returns")
    empty_columns = [str(column) for column in result if result[column].dropna().empty]
    if empty_columns:
        raise ValueError(
            "style_returns columns contain no observations: " + ", ".join(empty_columns)
        )
    return result


def _validate_simple_return_values(values: np.ndarray, name: str) -> None:
    finite_or_missing = np.isfinite(values) | np.isnan(values)
    if not finite_or_missing.all():
        raise ValueError(f"{name} must not contain infinite values")
    if np.any(values[~np.isnan(values)] < -1.0):
        raise ValueError(f"{name} simple returns cannot be below -1.0")


def _fit_aligned_style(
    fund_returns: pd.Series,
    style_returns: pd.DataFrame,
) -> StyleAnalysisResult:
    target = fund_returns.to_numpy(dtype=float)
    design = style_returns.to_numpy(dtype=float)
    _validate_style_identifiability(design, in_window=True)
    centered_target = target - target.mean()
    centered_design = design - design.mean(axis=0)
    initial = np.repeat(1.0 / design.shape[1], design.shape[1])

    def objective(weights: np.ndarray) -> float:
        residuals = centered_target - centered_design @ weights
        return float(0.5 * residuals @ residuals)

    def gradient(weights: np.ndarray) -> np.ndarray:
        residuals = centered_target - centered_design @ weights
        return -(centered_design.T @ residuals)

    optimization = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=((0.0, 1.0),) * design.shape[1],
        constraints={
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1.0,
            "jac": lambda weights: np.ones_like(weights),
        },
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    weights = _weights_from_optimization(optimization, style_returns.columns)
    fitted_values = design @ weights.to_numpy()
    residual_values = target - fitted_values
    centered_residuals = residual_values - residual_values.mean()
    residual_sum_squares = float(centered_residuals @ centered_residuals)
    total_sum_squares = float(centered_target @ centered_target)
    r_squared = (
        float("nan")
        if np.isclose(total_sum_squares, 0.0)
        else float(1.0 - residual_sum_squares / total_sum_squares)
    )
    fitted = pd.Series(
        fitted_values,
        index=fund_returns.index.copy(),
        name="style_fitted_return",
        dtype=float,
    )
    residuals = pd.Series(
        residual_values,
        index=fund_returns.index.copy(),
        name="style_residual",
        dtype=float,
    )
    return StyleAnalysisResult(
        weights=weights,
        fitted_returns=fitted,
        residuals=residuals,
        residual_sum_squares=residual_sum_squares,
        r_squared=r_squared,
        n_observations=len(fund_returns),
    )


def _weights_from_optimization(
    result: OptimizeResult,
    labels: pd.Index,
) -> pd.Series:
    if not result.success:
        raise RuntimeError(f"style exposure optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights[np.isclose(weights, 0.0, atol=1e-12)] = 0.0
    weights[np.isclose(weights, 1.0, atol=1e-12)] = 1.0
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8):
        raise RuntimeError("style exposure optimization returned invalid weights")
    weights = weights / weights.sum()
    return pd.Series(weights, index=labels.copy(), name="style_weight", dtype=float)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _validate_style_identifiability(
    design: np.ndarray,
    *,
    in_window: bool = False,
) -> None:
    if design.shape[1] == 1:
        return
    centered = design - design.mean(axis=0)
    differences = centered[:, :-1] - centered[:, [-1]]
    if np.linalg.matrix_rank(differences) < design.shape[1] - 1:
        suffix = " in every window" if in_window else ""
        raise ValueError(f"style_returns must identify distinct exposures{suffix}")
