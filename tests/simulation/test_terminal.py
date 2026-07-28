import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    terminal_wealth,
    terminal_wealth_stats,
)


def test_terminal_wealth_compounds_each_scenario_without_mutation() -> None:
    returns = pd.DataFrame(
        {
            "down": [-0.10, -0.20],
            "flat": [0.00, 0.00],
            "up": [0.10, 0.20],
        }
    )
    original = returns.copy(deep=True)

    result = terminal_wealth(returns, initial_wealth=100.0)

    expected = pd.Series(
        {"down": 72.0, "flat": 100.0, "up": 132.0},
        name="terminal_wealth",
    )
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_frame_equal(returns, original)


def test_terminal_wealth_restores_scalar_for_series_input() -> None:
    result = terminal_wealth(pd.Series([0.10, -0.10]), initial_wealth=100.0)

    assert result == pytest.approx(99.0)


def test_terminal_wealth_stats_reports_floor_and_cap_diagnostics() -> None:
    returns = pd.DataFrame(
        {
            "loss": [-0.25],
            "middle": [0.10],
            "gain": [0.60],
        }
    )

    stats = terminal_wealth_stats(
        returns,
        initial_wealth=100.0,
        floor_wealth=80.0,
        cap_wealth=150.0,
    )

    assert stats["n_scenarios"] == 3
    assert stats["mean"] == pytest.approx(115.0)
    assert stats["median"] == pytest.approx(110.0)
    assert stats["minimum"] == pytest.approx(75.0)
    assert stats["maximum"] == pytest.approx(160.0)
    assert stats["probability_below_floor"] == pytest.approx(1 / 3)
    assert stats["expected_shortfall_below_floor"] == pytest.approx(5.0)
    assert stats["probability_above_cap"] == pytest.approx(1 / 3)
    assert stats["expected_surplus_above_cap"] == pytest.approx(10.0)


def test_threshold_metrics_are_nan_when_thresholds_are_absent() -> None:
    stats = terminal_wealth_stats(pd.DataFrame({"one": [0.1], "two": [0.2]}))

    assert np.isnan(stats["probability_below_floor"])
    assert np.isnan(stats["probability_above_cap"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_wealth": 0.0},
        {"floor_wealth": -1.0},
        {"cap_wealth": -1.0},
        {"floor_wealth": 1.0, "cap_wealth": 1.0},
    ],
)
def test_terminal_stats_rejects_invalid_thresholds(kwargs: dict) -> None:
    returns = pd.DataFrame({"scenario": [0.01]})
    with pytest.raises(ValueError):
        terminal_wealth_stats(returns, **kwargs)
