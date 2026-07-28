"""Leakage-aware preparation helpers for univariate time series."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import pandas as pd

from asset_management_toolkit.time_series._validation import (
    coerce_time_series,
    validate_positive_integer,
    validate_windows,
)
from asset_management_toolkit.time_series.result import TimeSeriesFold


def chronological_train_test_split(
    series: pd.Series,
    test_size: int,
) -> tuple[pd.Series, pd.Series]:
    """Split the final observations into a test set without shuffling."""
    values = coerce_time_series(series, min_observations=2)
    testing = validate_positive_integer(test_size, "test_size")
    if testing >= len(values):
        raise ValueError("test_size must leave at least one training observation")
    return (
        values.iloc[:-testing].copy(deep=True),
        values.iloc[-testing:].copy(deep=True),
    )


def rolling_origin_splits(
    series: pd.Series,
    initial_train_size: int,
    test_size: int = 1,
    *,
    step: Optional[int] = None,
    window: str = "expanding",
) -> tuple[TimeSeriesFold, ...]:
    """Create non-overlapping chronological folds for honest model evaluation.

    ``window='expanding'`` retains all earlier observations. ``window='rolling'``
    keeps exactly ``initial_train_size`` observations in every training window.
    """
    values = coerce_time_series(series, min_observations=2)
    training = validate_positive_integer(initial_train_size, "initial_train_size")
    testing = validate_positive_integer(test_size, "test_size")
    stride = testing if step is None else validate_positive_integer(step, "step")
    if stride < testing:
        raise ValueError("step must be at least test_size to avoid overlapping tests")
    if window not in {"expanding", "rolling"}:
        raise ValueError("window must be 'expanding' or 'rolling'")
    if training + testing > len(values):
        raise ValueError("series does not contain one complete train/test fold")

    folds: list[TimeSeriesFold] = []
    test_start = training
    fold_number = 0
    while test_start + testing <= len(values):
        train_start = 0 if window == "expanding" else test_start - training
        folds.append(
            TimeSeriesFold(
                fold=fold_number,
                train=values.iloc[train_start:test_start].copy(deep=True),
                test=values.iloc[test_start : test_start + testing].copy(deep=True),
            )
        )
        fold_number += 1
        test_start += stride
    return tuple(folds)


def moving_average_features(
    series: pd.Series,
    simple_windows: Sequence[int],
    *,
    exponential_spans: Sequence[int] = (),
) -> pd.DataFrame:
    """Build trailing SMA and EWMA features using current and past observations."""
    values = coerce_time_series(series)
    simple = validate_windows(simple_windows, "simple_windows")
    if exponential_spans:
        exponential = validate_windows(exponential_spans, "exponential_spans")
    else:
        exponential = ()

    result = values.to_frame(name=values.name)
    for window in simple:
        result[f"sma_{window}"] = values.rolling(
            window=window,
            min_periods=window,
        ).mean()
    for span in exponential:
        result[f"ewma_{span}"] = values.ewm(
            span=span,
            adjust=False,
            min_periods=1,
        ).mean()
    return result
