import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.time_series import (
    forecast_metrics,
    seasonal_naive_forecast,
    walk_forward_forecast,
)


def _monthly_series(length: int = 18) -> pd.Series:
    seasonal = np.tile([10.0, 12.0, 14.0], 6)[:length]
    return pd.Series(
        seasonal,
        index=pd.period_range("2024-01", periods=length, freq="M"),
        name="demand",
    )


def test_forecast_metrics_use_forecast_minus_actual_bias() -> None:
    actual = pd.Series([1.0, 2.0, 4.0], index=list("abc"))
    predicted = pd.Series([2.0, 2.0, 2.0], index=list("abc"))

    result = forecast_metrics(actual, predicted)

    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert result["bias"] == pytest.approx(-1.0 / 3.0)
    assert result["mape"] == pytest.approx(0.5)
    assert result["smape"] == pytest.approx((2 / 3 + 0 + 2 / 3) / 3)


def test_forecast_metrics_handle_zero_denominators_without_infinity() -> None:
    index = pd.RangeIndex(2)
    result = forecast_metrics(
        pd.Series([0.0, 2.0], index=index),
        pd.Series([0.0, 1.0], index=index),
    )

    assert result["mape"] == pytest.approx(0.5)
    assert result["smape"] == pytest.approx(2.0 / 3.0)


def test_walk_forward_forecast_has_no_lookahead_and_is_labelled() -> None:
    series = _monthly_series()

    def forecaster(train: pd.Series, horizon: int) -> pd.Series:
        return seasonal_naive_forecast(train, horizon, seasonal_period=3)

    result = walk_forward_forecast(
        series,
        forecaster,
        initial_train_size=9,
        test_size=3,
    )

    assert len(result) == 9
    assert (result["error"] == 0.0).all()
    assert list(result["fold"].unique()) == [0, 1, 2]
    assert (result["train_end"] < result.index).all()


def test_walk_forward_accepts_unlabelled_forecaster_output() -> None:
    result = walk_forward_forecast(
        _monthly_series(12),
        lambda train, horizon: np.repeat(train.iloc[-1], horizon),
        initial_train_size=8,
        test_size=2,
    )

    assert len(result) == 4
    assert result["forecast"].notna().all()


def test_evaluation_rejects_misaligned_or_invalid_forecasts() -> None:
    actual = pd.Series([1.0, 2.0], index=[0, 1])
    predicted = pd.Series([1.0, 2.0], index=[1, 2])
    with pytest.raises(ValueError, match="indexes must match"):
        forecast_metrics(actual, predicted)

    with pytest.raises(ValueError, match="length"):
        walk_forward_forecast(
            _monthly_series(),
            lambda train, horizon: [1.0],
            initial_train_size=9,
            test_size=3,
        )

    with pytest.raises(ValueError, match="index must match"):
        walk_forward_forecast(
            _monthly_series(),
            lambda train, horizon: pd.Series(
                np.ones(horizon),
                index=pd.RangeIndex(horizon),
            ),
            initial_train_size=9,
            test_size=3,
        )
