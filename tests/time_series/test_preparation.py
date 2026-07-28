import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.time_series import (
    chronological_train_test_split,
    moving_average_features,
    rolling_origin_splits,
)


def _monthly_series(length: int = 12) -> pd.Series:
    return pd.Series(
        np.arange(length, dtype=float),
        index=pd.period_range("2025-01", periods=length, freq="M"),
        name="value",
    )


def test_chronological_split_preserves_labels_and_does_not_mutate_input() -> None:
    series = _monthly_series()
    original = series.copy(deep=True)

    train, test = chronological_train_test_split(series, test_size=3)

    pd.testing.assert_series_equal(train, series.iloc[:9])
    pd.testing.assert_series_equal(test, series.iloc[9:])
    pd.testing.assert_series_equal(series, original)


def test_rolling_origin_splits_support_expanding_and_rolling_windows() -> None:
    expanding = rolling_origin_splits(
        _monthly_series(),
        initial_train_size=6,
        test_size=2,
    )
    rolling = rolling_origin_splits(
        _monthly_series(),
        initial_train_size=6,
        test_size=2,
        window="rolling",
    )

    assert [len(fold.train) for fold in expanding] == [6, 8, 10]
    assert [len(fold.train) for fold in rolling] == [6, 6, 6]
    assert all(fold.train_end < fold.test_start for fold in expanding)
    assert expanding[1].test_start == _monthly_series().index[8]


def test_moving_average_features_are_trailing_and_labelled() -> None:
    result = moving_average_features(
        _monthly_series(5),
        simple_windows=[2, 3],
        exponential_spans=[2],
    )

    assert list(result.columns) == ["value", "sma_2", "sma_3", "ewma_2"]
    assert np.isnan(result.loc[result.index[0], "sma_2"])
    assert result.loc[result.index[2], "sma_2"] == pytest.approx(1.5)
    assert result.loc[result.index[2], "sma_3"] == pytest.approx(1.0)
    assert result.loc[result.index[0], "ewma_2"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda series: chronological_train_test_split(series, 12),
            "leave at least one",
        ),
        (
            lambda series: rolling_origin_splits(series, 6, 3, step=2),
            "avoid overlapping",
        ),
        (
            lambda series: rolling_origin_splits(series, 6, window="anchored"),
            "window",
        ),
        (
            lambda series: moving_average_features(series, [2, 2]),
            "duplicate",
        ),
    ],
)
def test_preparation_rejects_invalid_contracts(call, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call(_monthly_series())


def test_time_series_inputs_must_be_sorted_unique_and_complete() -> None:
    series = _monthly_series()
    with pytest.raises(ValueError, match="sorted"):
        chronological_train_test_split(series.sort_index(ascending=False), 2)

    duplicated = series.copy()
    duplicated.index = [series.index[0]] * len(series)
    with pytest.raises(ValueError, match="unique"):
        chronological_train_test_split(duplicated, 2)

    missing = series.copy()
    missing.iloc[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        chronological_train_test_split(missing, 2)
