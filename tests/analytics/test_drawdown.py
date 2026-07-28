import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.analytics import drawdown_episodes, drawdown_path


def test_drawdown_path_uses_initial_wealth_as_the_first_peak() -> None:
    returns = pd.Series([-0.10, 0.05], index=["a", "b"])

    result = drawdown_path(returns, initial_wealth=100.0)

    assert result.loc["a", "wealth"] == pytest.approx(90.0)
    assert result.loc["a", "previous_peak"] == pytest.approx(100.0)
    assert result.loc["a", "drawdown"] == pytest.approx(-0.10)
    assert result.loc["b", "drawdown"] == pytest.approx(-0.055)


def test_drawdown_episodes_reports_recovered_and_open_episodes() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.Series([0.10, -0.20, 0.25, -0.10], index=dates)

    result = drawdown_episodes(returns)

    assert len(result) == 2
    first = result.iloc[0]
    assert first["start_date"] == dates[1]
    assert first["trough_date"] == dates[1]
    assert first["recovery_date"] == dates[2]
    assert first["drawdown"] == pytest.approx(-0.20)
    assert first["decline_periods"] == 1
    assert first["recovery_periods"] == 1
    assert first["total_periods"] == 2
    assert bool(first["recovered"])

    second = result.iloc[1]
    assert second["start_date"] == dates[3]
    assert pd.isna(second["recovery_date"])
    assert not bool(second["recovered"])
    assert second["total_periods"] == 1


def test_drawdown_episodes_returns_a_stable_empty_schema() -> None:
    result = drawdown_episodes(pd.Series([0.01, 0.02]))

    assert result.empty
    assert list(result.columns) == [
        "start_date",
        "trough_date",
        "recovery_date",
        "drawdown",
        "decline_periods",
        "recovery_periods",
        "total_periods",
        "recovered",
    ]


@pytest.mark.parametrize("initial_wealth", [0.0, -1.0, np.inf])
def test_drawdown_path_rejects_invalid_initial_wealth(
    initial_wealth: float,
) -> None:
    with pytest.raises(ValueError):
        drawdown_path(pd.Series([0.01]), initial_wealth=initial_wealth)
