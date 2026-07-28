import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.time_series import (
    decompose_time_series,
    exponential_smoothing_forecast,
    sarima_forecast,
    seasonal_naive_forecast,
)


def _seasonal_series(length: int = 36) -> pd.Series:
    trend = np.linspace(10.0, 15.0, length)
    season = np.tile([0.0, 1.0, -0.5, 0.5], length // 4 + 1)[:length]
    return pd.Series(
        trend + season,
        index=pd.period_range("2023-01", periods=length, freq="M"),
        name="level",
    )


def test_seasonal_naive_forecast_repeats_last_cycle_with_future_labels() -> None:
    series = _seasonal_series(12)
    result = seasonal_naive_forecast(series, horizon=6, seasonal_period=4)

    np.testing.assert_allclose(
        result.to_numpy(),
        np.resize(series.iloc[-4:].to_numpy(), 6),
    )
    assert result.index[0] == series.index[-1] + 1
    assert result.index[-1] == series.index[-1] + 6
    assert result.name == series.name


def test_seasonal_naive_rejects_ambiguous_datetime_frequency() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-08"]),
    )
    with pytest.raises(ValueError, match="frequency"):
        seasonal_naive_forecast(series, horizon=2)


def test_exponential_smoothing_returns_complete_labelled_result() -> None:
    pytest.importorskip("statsmodels")
    series = _seasonal_series()

    result = exponential_smoothing_forecast(
        series,
        horizon=4,
        trend="add",
        seasonal="add",
        seasonal_periods=4,
    )

    assert result.model == "exponential_smoothing"
    assert result.forecast.index.equals(
        pd.period_range(series.index[-1] + 1, periods=4, freq="M")
    )
    assert len(result.fitted_values) == len(series)
    assert result.residuals.index.equals(series.index)
    assert np.isfinite(result.forecast.to_numpy()).all()
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_sarima_forecast_is_deterministic_and_labelled() -> None:
    pytest.importorskip("statsmodels")
    series = _seasonal_series(24)

    first = sarima_forecast(
        series,
        horizon=3,
        order=(1, 1, 0),
        trend="n",
    )
    second = sarima_forecast(
        series,
        horizon=3,
        order=(1, 1, 0),
        trend="n",
    )

    pd.testing.assert_series_equal(first.forecast, second.forecast)
    assert first.parameters["order"] == (1, 1, 0)
    assert first.forecast.index[0] == series.index[-1] + 1


def test_decomposition_recovers_additive_components() -> None:
    pytest.importorskip("statsmodels")
    series = _seasonal_series()

    result = decompose_time_series(
        series,
        period=4,
        model="additive",
        extrapolate_trend=1,
    )

    assert result.model == "additive"
    assert result.period == 4
    assert result.observed.index.equals(series.index)
    assert result.trend.notna().all()
    reconstructed = result.trend + result.seasonal + result.residual
    np.testing.assert_allclose(reconstructed, result.observed, atol=1e-10)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda series: seasonal_naive_forecast(
                series,
                horizon=2,
                seasonal_period=100,
            ),
            "must not exceed",
        ),
        (
            lambda series: exponential_smoothing_forecast(
                series,
                horizon=2,
                seasonal="add",
            ),
            "seasonal_periods is required",
        ),
        (
            lambda series: exponential_smoothing_forecast(
                series,
                horizon=2,
                damped_trend=True,
            ),
            "requires a trend",
        ),
        (
            lambda series: sarima_forecast(
                series,
                horizon=2,
                order=(1, -1, 0),
            ),
            "nonnegative",
        ),
        (
            lambda series: sarima_forecast(
                series,
                horizon=2,
                seasonal_order=(1, 0, 0, 1),
            ),
            "at least two",
        ),
    ],
)
def test_forecasting_rejects_invalid_contracts(call, message: str) -> None:
    pytest.importorskip("statsmodels")
    with pytest.raises((TypeError, ValueError), match=message):
        call(_seasonal_series())
