"""Labelled time-series factor regression and return attribution."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from asset_management_toolkit.analytics._validation import (
    annual_to_periodic_rate,
    coerce_returns,
    validate_annual_rate,
    validate_periods_per_year,
)


@dataclass(frozen=True)
class FactorRegressionResult:
    """Coefficients, inference statistics, fit diagnostics, and residual path.

    The ``alpha`` coefficient and its standard error are annualized by
    multiplication. Factor coefficients are unitless exposures. Factor returns
    are interpreted as periodic factor premia; the annual risk-free rate is
    converted to the matching periodic rate and subtracted only from the
    dependent asset return.
    """

    coefficients: pd.Series
    standard_errors: pd.Series
    t_statistics: pd.Series
    p_values: pd.Series
    fitted_returns: pd.Series
    residuals: pd.Series
    r_squared: float
    adjusted_r_squared: float
    residual_volatility: float
    n_observations: int
    degrees_of_freedom: int

    @property
    def alpha(self) -> float:
        """Return annualized regression alpha."""
        return float(self.coefficients.loc["alpha"])

    @property
    def betas(self) -> pd.Series:
        """Return factor exposures without the alpha coefficient."""
        result = self.coefficients.drop("alpha").copy()
        result.name = "factor_beta"
        return result


@dataclass(frozen=True)
class RollingFactorRegressionResult:
    """Window-level factor coefficients, inference, and fit diagnostics."""

    coefficients: pd.DataFrame
    standard_errors: pd.DataFrame
    t_statistics: pd.DataFrame
    p_values: pd.DataFrame
    r_squared: pd.Series
    adjusted_r_squared: pd.Series
    residual_volatility: pd.Series
    window: int
    step: int

    @property
    def alpha(self) -> pd.Series:
        """Return annualized rolling alpha."""
        result = self.coefficients["alpha"].copy()
        result.name = "alpha"
        return result

    @property
    def betas(self) -> pd.DataFrame:
        """Return rolling factor exposures without alpha."""
        return self.coefficients.drop(columns="alpha").copy()


@dataclass(frozen=True)
class RegularizedFactorRegressionResult:
    """Penalized factor coefficients and fit diagnostics."""

    coefficients: pd.Series
    fitted_returns: pd.Series
    residuals: pd.Series
    r_squared: float
    residual_volatility: float
    selected_regularization: float
    n_observations: int
    method: str

    @property
    def alpha(self) -> float:
        """Return annualized regression alpha."""
        return float(self.coefficients.loc["alpha"])

    @property
    def betas(self) -> pd.Series:
        """Return regularized factor exposures without alpha."""
        result = self.coefficients.drop("alpha").copy()
        result.name = "factor_beta"
        return result


@dataclass(frozen=True)
class RollingRegularizedFactorRegressionResult:
    """Trailing penalized factor coefficients and fit diagnostics."""

    coefficients: pd.DataFrame
    r_squared: pd.Series
    residual_volatility: pd.Series
    selected_regularization: pd.Series
    window: int
    step: int
    method: str

    @property
    def alpha(self) -> pd.Series:
        """Return annualized rolling alpha."""
        result = self.coefficients["alpha"].copy()
        result.name = "alpha"
        return result

    @property
    def betas(self) -> pd.DataFrame:
        """Return rolling factor exposures without alpha."""
        return self.coefficients.drop(columns="alpha").copy()


def factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> FactorRegressionResult:
    """Estimate an unconstrained labelled time-series factor model with OLS.

    The fitted model is

    ``asset_return - risk_free = alpha + factor_returns @ beta + residual``.

    Inputs are aligned by index and rows containing any missing value are
    excluded jointly. Factor-return columns must be unique and linearly
    independent after adding the intercept.
    """
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    aligned = _align_factor_inputs(asset_returns, factor_returns)
    return _fit_aligned_factor_model(
        aligned.iloc[:, 0],
        aligned.iloc[:, 1:],
        risk_free_rate,
        periods_per_year,
    )


def rolling_factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int,
    *,
    step: int = 1,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> RollingFactorRegressionResult:
    """Estimate the factor model over trailing fixed-observation windows."""
    _validate_positive_integer(window, "window")
    _validate_positive_integer(step, "step")
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    aligned = _align_factor_inputs(asset_returns, factor_returns)
    minimum = factor_returns.shape[1] + 2
    if window < minimum:
        raise ValueError(
            f"window must contain at least {minimum} observations for "
            f"{factor_returns.shape[1]} factors and an intercept"
        )
    if window > len(aligned):
        raise ValueError("window cannot exceed the number of complete observations")
    if not aligned.index.is_monotonic_increasing:
        raise ValueError("rolling inputs must have an increasing index")

    results: list[FactorRegressionResult] = []
    labels = []
    for end in range(window, len(aligned) + 1, step):
        sample = aligned.iloc[end - window : end]
        results.append(
            _fit_aligned_factor_model(
                sample.iloc[:, 0],
                sample.iloc[:, 1:],
                risk_free_rate,
                periods_per_year,
                in_window=True,
            )
        )
        labels.append(sample.index[-1])

    result_index = pd.Index(labels, name=aligned.index.name)
    coefficient_labels = results[0].coefficients.index.copy()
    return RollingFactorRegressionResult(
        coefficients=_result_frame(
            results,
            "coefficients",
            result_index,
            coefficient_labels,
        ),
        standard_errors=_result_frame(
            results,
            "standard_errors",
            result_index,
            coefficient_labels,
        ),
        t_statistics=_result_frame(
            results,
            "t_statistics",
            result_index,
            coefficient_labels,
        ),
        p_values=_result_frame(
            results,
            "p_values",
            result_index,
            coefficient_labels,
        ),
        r_squared=_result_series(results, "r_squared", result_index),
        adjusted_r_squared=_result_series(
            results,
            "adjusted_r_squared",
            result_index,
        ),
        residual_volatility=_result_series(
            results,
            "residual_volatility",
            result_index,
        ),
        window=window,
        step=step,
    )


def regularized_factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    method: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    regularization: float | None = None,
    l1_ratio: float = 0.5,
    cv: int = 5,
    regularization_grid: Iterable[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> RegularizedFactorRegressionResult:
    """Estimate a penalized factor model with chronological validation.

    Factors are standardized inside each ``TimeSeriesSplit`` fold. When
    ``regularization`` is omitted, the penalty is selected by chronological
    cross-validation using negative mean squared error. Reported coefficients
    are transformed back to the original factor units.
    """
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    selected_method = _validate_regularized_method(method)
    selected_regularization = _optional_positive_real(
        regularization,
        "regularization",
    )
    ratio = _validate_l1_ratio(l1_ratio)
    folds = _validate_cv(cv)
    grid = _regularization_grid(regularization_grid)
    aligned = _align_factor_inputs(
        asset_returns,
        factor_returns,
        require_identifiable=False,
    )
    if selected_regularization is None and len(aligned) <= folds:
        raise ValueError("complete observations must exceed cv")
    return _fit_regularized_factor_model(
        aligned.iloc[:, 0],
        aligned.iloc[:, 1:],
        method=selected_method,
        regularization=selected_regularization,
        l1_ratio=ratio,
        cv=folds,
        regularization_grid=grid,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )


def rolling_regularized_factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int,
    *,
    step: int = 1,
    method: Literal["ridge", "lasso", "elastic_net"] = "ridge",
    regularization: float | None = None,
    l1_ratio: float = 0.5,
    cv: int = 5,
    regularization_grid: Iterable[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> RollingRegularizedFactorRegressionResult:
    """Estimate penalized factor exposures over trailing observation windows."""
    _validate_positive_integer(window, "window")
    _validate_positive_integer(step, "step")
    validate_annual_rate(risk_free_rate, "risk_free_rate")
    validate_periods_per_year(periods_per_year)
    selected_method = _validate_regularized_method(method)
    selected_regularization = _optional_positive_real(
        regularization,
        "regularization",
    )
    ratio = _validate_l1_ratio(l1_ratio)
    folds = _validate_cv(cv)
    grid = _regularization_grid(regularization_grid)
    aligned = _align_factor_inputs(
        asset_returns,
        factor_returns,
        require_identifiable=False,
    )
    minimum = factor_returns.shape[1] + 2
    if window < minimum:
        raise ValueError(
            f"window must contain at least {minimum} observations for "
            f"{factor_returns.shape[1]} factors and an intercept"
        )
    if window > len(aligned):
        raise ValueError("window cannot exceed the number of complete observations")
    if selected_regularization is None and window <= folds:
        raise ValueError("window must exceed cv")
    if not aligned.index.is_monotonic_increasing:
        raise ValueError("rolling inputs must have an increasing index")

    results: list[RegularizedFactorRegressionResult] = []
    labels = []
    for end in range(window, len(aligned) + 1, step):
        sample = aligned.iloc[end - window : end]
        results.append(
            _fit_regularized_factor_model(
                sample.iloc[:, 0],
                sample.iloc[:, 1:],
                method=selected_method,
                regularization=selected_regularization,
                l1_ratio=ratio,
                cv=folds,
                regularization_grid=grid,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
        )
        labels.append(sample.index[-1])

    result_index = pd.Index(labels, name=aligned.index.name)
    coefficient_labels = results[0].coefficients.index.copy()
    return RollingRegularizedFactorRegressionResult(
        coefficients=pd.DataFrame(
            [result.coefficients.to_numpy() for result in results],
            index=result_index,
            columns=coefficient_labels,
            dtype=float,
        ),
        r_squared=_regularized_result_series(
            results,
            "r_squared",
            result_index,
        ),
        residual_volatility=_regularized_result_series(
            results,
            "residual_volatility",
            result_index,
        ),
        selected_regularization=_regularized_result_series(
            results,
            "selected_regularization",
            result_index,
        ),
        window=window,
        step=step,
        method=selected_method,
    )


def factor_return_attribution(
    exposures: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    alpha: float = 0.0,
) -> pd.DataFrame:
    """Calculate periodic return contributions from static factor exposures.

    ``alpha`` is a periodic intercept in the same units as ``factor_returns``.
    The returned ``total`` is the model-implied periodic return, not realized
    return attribution with an unexplained residual.
    """
    factors = _validated_factor_returns(factor_returns)
    weights = _validated_exposures(exposures, factors.columns)
    intercept = _finite_real(alpha, "alpha")

    contributions = factors.mul(weights, axis="columns")
    contributions.insert(0, "alpha", intercept)
    contributions["total"] = contributions.sum(
        axis=1,
        min_count=contributions.shape[1],
    )
    return contributions


def _align_factor_inputs(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    require_identifiable: bool = True,
) -> pd.DataFrame:
    asset = _validated_asset_returns(asset_returns)
    factors = _validated_factor_returns(factor_returns)
    aligned = pd.concat(
        [asset.rename("__asset_return__"), factors],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("asset_returns and factor_returns have no complete overlap")

    minimum = factors.shape[1] + 2
    if len(aligned) < minimum:
        raise ValueError(
            f"at least {minimum} complete observations are required for "
            f"{factors.shape[1]} factors and an intercept"
        )
    if require_identifiable:
        _validate_factor_identifiability(aligned.iloc[:, 1:].to_numpy())
    return aligned


def _validated_asset_returns(asset_returns: pd.Series) -> pd.Series:
    if not isinstance(asset_returns, pd.Series):
        raise TypeError("asset_returns must be a pandas Series")
    if not asset_returns.index.is_unique:
        raise ValueError("asset_returns index must be unique")
    frame, _ = coerce_returns(asset_returns, default_name="asset")
    result = frame.iloc[:, 0]
    result.name = asset_returns.name
    return result


def _validated_factor_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(factor_returns, pd.DataFrame):
        raise TypeError("factor_returns must be a pandas DataFrame")
    if not factor_returns.index.is_unique:
        raise ValueError("factor_returns index must be unique")
    frame, _ = coerce_returns(factor_returns)
    reserved = [name for name in ("alpha", "total") if name in frame.columns]
    if reserved:
        raise ValueError(
            "factor_returns columns use reserved result labels: " + ", ".join(reserved)
        )
    return frame


def _validated_exposures(
    exposures: pd.Series,
    factor_labels: pd.Index,
) -> pd.Series:
    if not isinstance(exposures, pd.Series):
        raise TypeError("exposures must be a pandas Series")
    if exposures.empty:
        raise ValueError("exposures must not be empty")
    if not exposures.index.is_unique:
        raise ValueError("exposures index must be unique")
    if not pd.api.types.is_numeric_dtype(exposures):
        raise TypeError("exposures must be numeric")
    result = exposures.astype(float).copy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("exposures must contain only finite values")
    if not result.index.equals(factor_labels):
        raise ValueError(
            "exposures labels and order must exactly match factor_returns columns"
        )
    return result


def _fit_aligned_factor_model(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float,
    periods_per_year: int,
    *,
    in_window: bool = False,
) -> FactorRegressionResult:
    periodic_risk_free = annual_to_periodic_rate(
        risk_free_rate,
        periods_per_year,
    )
    target = asset_returns.to_numpy(dtype=float) - periodic_risk_free
    factors = factor_returns.to_numpy(dtype=float)
    _validate_factor_identifiability(factors, in_window=in_window)
    design = np.column_stack([np.ones(len(factors)), factors])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    fitted_excess = design @ coefficients
    residuals = target - fitted_excess
    degrees_of_freedom = len(target) - design.shape[1]
    residual_sum_squares = float(residuals @ residuals)
    residual_variance = residual_sum_squares / degrees_of_freedom
    covariance = residual_variance * np.linalg.inv(design.T @ design)
    periodic_standard_errors = np.sqrt(
        np.clip(np.diag(covariance), a_min=0.0, a_max=None)
    )
    t_statistics = _t_statistics(coefficients, periodic_standard_errors)
    p_values = 2.0 * student_t.sf(np.abs(t_statistics), degrees_of_freedom)

    coefficient_values = coefficients.copy()
    coefficient_values[0] *= periods_per_year
    standard_error_values = periodic_standard_errors.copy()
    standard_error_values[0] *= periods_per_year
    labels = pd.Index(["alpha", *factor_returns.columns])
    coefficient_series = pd.Series(
        coefficient_values,
        index=labels,
        name="coefficient",
        dtype=float,
    )
    standard_errors = pd.Series(
        standard_error_values,
        index=labels,
        name="standard_error",
        dtype=float,
    )
    t_statistic_series = pd.Series(
        t_statistics,
        index=labels,
        name="t_statistic",
        dtype=float,
    )
    p_value_series = pd.Series(
        p_values,
        index=labels,
        name="p_value",
        dtype=float,
    )

    centered_target = target - target.mean()
    total_sum_squares = float(centered_target @ centered_target)
    r_squared = (
        float("nan")
        if np.isclose(total_sum_squares, 0.0)
        else float(1.0 - residual_sum_squares / total_sum_squares)
    )
    adjusted_r_squared = (
        float("nan")
        if not np.isfinite(r_squared)
        else float(1.0 - (1.0 - r_squared) * (len(target) - 1) / degrees_of_freedom)
    )
    fitted = pd.Series(
        fitted_excess + periodic_risk_free,
        index=asset_returns.index.copy(),
        name="factor_fitted_return",
        dtype=float,
    )
    residual_series = pd.Series(
        residuals,
        index=asset_returns.index.copy(),
        name="factor_residual",
        dtype=float,
    )
    return FactorRegressionResult(
        coefficients=coefficient_series,
        standard_errors=standard_errors,
        t_statistics=t_statistic_series,
        p_values=p_value_series,
        fitted_returns=fitted,
        residuals=residual_series,
        r_squared=r_squared,
        adjusted_r_squared=adjusted_r_squared,
        residual_volatility=float(np.sqrt(residual_variance * periods_per_year)),
        n_observations=len(target),
        degrees_of_freedom=degrees_of_freedom,
    )


def _fit_regularized_factor_model(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    method: str,
    regularization: float | None,
    l1_ratio: float,
    cv: int,
    regularization_grid: np.ndarray,
    risk_free_rate: float,
    periods_per_year: int,
) -> RegularizedFactorRegressionResult:
    (
        ElasticNet,
        GridSearchCV,
        Lasso,
        Pipeline,
        Ridge,
        StandardScaler,
        TimeSeriesSplit,
    ) = _sklearn_regularized_components()
    if method == "ridge":
        estimator = Ridge()
    elif method == "lasso":
        estimator = Lasso(max_iter=20_000)
    else:
        estimator = ElasticNet(l1_ratio=l1_ratio, max_iter=20_000)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )
    factors = factor_returns.to_numpy(dtype=float)
    periodic_risk_free = annual_to_periodic_rate(
        risk_free_rate,
        periods_per_year,
    )
    target = asset_returns.to_numpy(dtype=float) - periodic_risk_free
    if regularization is None:
        search = GridSearchCV(
            pipeline,
            {"model__alpha": regularization_grid},
            cv=TimeSeriesSplit(n_splits=cv),
            scoring="neg_mean_squared_error",
            refit=True,
        )
        search.fit(factors, target)
        fitted_pipeline = search.best_estimator_
        selected = float(search.best_params_["model__alpha"])
    else:
        pipeline.set_params(model__alpha=regularization)
        fitted_pipeline = pipeline.fit(factors, target)
        selected = regularization

    scaler = fitted_pipeline.named_steps["scale"]
    model = fitted_pipeline.named_steps["model"]
    scaled_coefficients = np.asarray(model.coef_, dtype=float)
    betas = scaled_coefficients / np.asarray(scaler.scale_, dtype=float)
    periodic_alpha = float(model.intercept_ - np.sum(np.asarray(scaler.mean_) * betas))
    fitted_excess = np.asarray(fitted_pipeline.predict(factors), dtype=float)
    residual_values = target - fitted_excess
    centered_target = target - target.mean()
    total_sum_squares = float(centered_target @ centered_target)
    residual_sum_squares = float(residual_values @ residual_values)
    r_squared = (
        float("nan")
        if np.isclose(total_sum_squares, 0.0)
        else float(1.0 - residual_sum_squares / total_sum_squares)
    )
    coefficient_labels = pd.Index(["alpha", *factor_returns.columns])
    coefficients = pd.Series(
        np.concatenate([[periodic_alpha * periods_per_year], betas]),
        index=coefficient_labels,
        name="coefficient",
        dtype=float,
    )
    fitted = pd.Series(
        fitted_excess + periodic_risk_free,
        index=asset_returns.index.copy(),
        name="factor_fitted_return",
        dtype=float,
    )
    residuals = pd.Series(
        residual_values,
        index=asset_returns.index.copy(),
        name="factor_residual",
        dtype=float,
    )
    return RegularizedFactorRegressionResult(
        coefficients=coefficients,
        fitted_returns=fitted,
        residuals=residuals,
        r_squared=r_squared,
        residual_volatility=float(residuals.std(ddof=1) * np.sqrt(periods_per_year)),
        selected_regularization=float(selected),
        n_observations=len(target),
        method=method,
    )


def _validate_factor_identifiability(
    factors: np.ndarray,
    *,
    in_window: bool = False,
) -> None:
    design = np.column_stack([np.ones(len(factors)), factors])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        suffix = " in every window" if in_window else ""
        raise ValueError(
            "factor_returns must identify linearly independent factors" + suffix
        )


def _t_statistics(
    coefficients: np.ndarray,
    standard_errors: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(coefficients)
    nonzero = standard_errors > np.finfo(float).eps
    result[nonzero] = coefficients[nonzero] / standard_errors[nonzero]
    result[~nonzero] = np.where(
        np.isclose(coefficients[~nonzero], 0.0),
        0.0,
        np.sign(coefficients[~nonzero]) * np.inf,
    )
    return result


def _result_frame(
    results: list[FactorRegressionResult],
    attribute: str,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(
        [getattr(result, attribute).to_numpy() for result in results],
        index=index,
        columns=columns,
        dtype=float,
    )


def _result_series(
    results: list[FactorRegressionResult],
    attribute: str,
    index: pd.Index,
) -> pd.Series:
    return pd.Series(
        [getattr(result, attribute) for result in results],
        index=index,
        name=attribute,
        dtype=float,
    )


def _regularized_result_series(
    results: list[RegularizedFactorRegressionResult],
    attribute: str,
    index: pd.Index,
) -> pd.Series:
    return pd.Series(
        [getattr(result, attribute) for result in results],
        index=index,
        name=attribute,
        dtype=float,
    )


def _validate_regularized_method(method: str) -> str:
    if method not in {"ridge", "lasso", "elastic_net"}:
        raise ValueError("method must be one of: ridge, lasso, elastic_net")
    return method


def _validate_l1_ratio(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError("l1_ratio must be a real number")
    if not np.isfinite(value) or not 0.0 < float(value) <= 1.0:
        raise ValueError("l1_ratio must be greater than zero and at most one")
    return float(value)


def _validate_cv(value: int) -> int:
    _validate_positive_integer(value, "cv")
    if value < 2:
        raise ValueError("cv must be at least two")
    return value


def _optional_positive_real(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_real(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _regularization_grid(values: Iterable[float] | None) -> np.ndarray:
    if values is None:
        return np.logspace(-6, 1, 30)
    grid = np.asarray(list(values), dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("regularization_grid must contain at least one value")
    if not np.isfinite(grid).all() or np.any(grid <= 0.0):
        raise ValueError("regularization_grid values must be finite and positive")
    return np.unique(grid)


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _finite_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _sklearn_regularized_components():
    try:
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise ImportError(
            "regularized factor regression requires scikit-learn; "
            "install asset-management-toolkit[factor-model]"
        ) from error
    return (
        ElasticNet,
        GridSearchCV,
        Lasso,
        Pipeline,
        Ridge,
        StandardScaler,
        TimeSeriesSplit,
    )
