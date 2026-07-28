import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.analytics.returns import (
    active_return,
    aggregate_returns,
    annualized_mean_return,
    annualized_return,
    rolling_returns,
    total_return,
)


def test_total_and_annualized_return_for_series() -> None:
    returns = pd.Series([0.10, -0.05, 0.02], name="portfolio")
    expected_total = (1.10 * 0.95 * 1.02) - 1.0

    assert total_return(returns) == pytest.approx(expected_total)
    assert annualized_return(returns, periods_per_year=3) == pytest.approx(
        expected_total
    )
    assert annualized_mean_return(returns, periods_per_year=3) == pytest.approx(
        returns.mean() * 3
    )


def test_return_functions_support_dataframe_without_mutation() -> None:
    returns = pd.DataFrame(
        {
            "asset_a": [0.01, 0.02, np.nan],
            "asset_b": [-0.01, 0.00, 0.01],
        }
    )
    original = returns.copy(deep=True)

    result = total_return(returns)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ["asset_a", "asset_b"]
    pd.testing.assert_frame_equal(returns, original)


def test_active_return_aligns_benchmark_by_index() -> None:
    returns = pd.Series([0.02, 0.03], index=[1, 2])
    benchmark = pd.Series([0.01, 0.02], index=[2, 3])

    result = active_return(returns, benchmark, periods_per_year=12)

    assert result == pytest.approx((0.03 - 0.01) * 12)


def test_aggregate_returns_compounds_each_calendar_period() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.20],
        index=pd.to_datetime(["2026-01-02", "2026-01-30", "2026-02-27"]),
        name="portfolio",
    )

    result = aggregate_returns(returns, "ME")

    assert isinstance(result, pd.Series)
    assert result.name == "portfolio"
    assert result.iloc[0] == pytest.approx(1.10 * 0.95 - 1.0)
    assert result.iloc[1] == pytest.approx(0.20)


def test_aggregate_returns_preserves_missing_empty_asset_periods() -> None:
    returns = pd.DataFrame(
        {
            "asset_a": [0.10, np.nan],
            "asset_b": [np.nan, 0.20],
        },
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
    )

    result = aggregate_returns(returns, "ME")

    assert np.isnan(result.loc[pd.Timestamp("2026-02-28"), "asset_a"])
    assert np.isnan(result.loc[pd.Timestamp("2026-01-31"), "asset_b"])


def test_aggregate_returns_requires_an_ordered_datetime_index() -> None:
    returns = pd.Series([0.01, 0.02], index=[1, 2])

    with pytest.raises(TypeError, match="DatetimeIndex"):
        aggregate_returns(returns, "ME")


def test_rolling_returns_compounds_complete_trailing_windows() -> None:
    returns = pd.Series(
        [0.10, -0.05, 0.02],
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
        name="portfolio",
    )

    result = rolling_returns(returns, window=2)

    expected = pd.Series(
        [np.nan, 1.10 * 0.95 - 1.0, 0.95 * 1.02 - 1.0],
        index=returns.index,
        name="portfolio",
    )
    pd.testing.assert_series_equal(result, expected)


def test_rolling_returns_requires_complete_ordered_windows() -> None:
    returns = pd.DataFrame(
        {
            "complete": [0.01, 0.02, 0.03],
            "missing": [0.01, np.nan, 0.03],
        },
        index=[1, 2, 3],
    )

    result = rolling_returns(returns, window=2)

    assert result["complete"].iloc[-1] == pytest.approx(1.02 * 1.03 - 1.0)
    assert result["missing"].isna().all()


@pytest.mark.parametrize("window", [0, -1])
def test_rolling_returns_rejects_invalid_window(window: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        rolling_returns(pd.Series([0.01, 0.02]), window)


@pytest.mark.parametrize("periods_per_year", [0, -1])
def test_annualized_return_rejects_invalid_frequency(
    periods_per_year: int,
) -> None:
    with pytest.raises(ValueError):
        annualized_return(pd.Series([0.01]), periods_per_year)
