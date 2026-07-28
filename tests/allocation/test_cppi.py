"""Deterministic tests for CPPI-family allocation strategies."""

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.allocation import (
    run_dynamic_multiplier_cppi,
    run_fixed_maturity_cppi,
    run_open_ended_cppi,
    run_tipp,
)


def test_fixed_maturity_floor_reaches_terminal_guarantee() -> None:
    returns = pd.Series(
        [0.0, 0.0, 0.0, 0.0],
        index=pd.period_range("2026Q1", periods=4, freq="Q"),
        name="base",
    )

    result = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        initial_wealth=100.0,
        risk_free_rate=0.04,
        periods_per_year=4,
    )

    periodic_safe = 1.04 ** (1.0 / 4.0) - 1.0
    expected_initial_floor = 80.0 / (1.0 + periodic_safe) ** 4
    assert result.floor.iloc[-1, 0] == pytest.approx(80.0)
    assert result.risky_weight.iloc[0, 0] == pytest.approx(
        3.0 * (100.0 - expected_initial_floor) / 100.0
    )
    assert result.strategy == "fixed_maturity_cppi"


def test_multiplier_one_and_three_are_distinct_configurations() -> None:
    returns = pd.Series([0.0], name="path")

    unleveraged = run_fixed_maturity_cppi(
        returns,
        multiplier=1.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
    )
    classical = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert unleveraged.risky_weight.iloc[0, 0] == pytest.approx(0.2)
    assert classical.risky_weight.iloc[0, 0] == pytest.approx(0.6)


def test_open_ended_floor_grows_and_resets_on_schedule() -> None:
    returns = pd.Series([0.50, 0.50, 0.00], name="growth")

    no_reset = run_open_ended_cppi(
        returns,
        multiplier=1.0,
        floor_fraction=0.8,
        reset_every=None,
        risk_free_rate=0.0,
        periods_per_year=1,
    )
    reset = run_open_ended_cppi(
        returns,
        multiplier=1.0,
        floor_fraction=0.8,
        reset_every=2,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert (no_reset.floor["growth"] == 0.8).all()
    assert reset.floor.iloc[0, 0] == pytest.approx(0.8)
    assert reset.floor.iloc[1, 0] == pytest.approx(0.8 * reset.wealth.iloc[1, 0])
    assert reset.floor.iloc[2, 0] == pytest.approx(reset.floor.iloc[1, 0])


def test_tipp_ratchets_against_high_water_mark_and_never_falls() -> None:
    returns = pd.Series([0.50, -0.20, 0.10], name="path")

    result = run_tipp(
        returns,
        multiplier=1.0,
        protection_ratio=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert result.floor.iloc[0, 0] == pytest.approx(0.8 * result.wealth.iloc[0, 0])
    assert result.floor.iloc[1, 0] == pytest.approx(result.floor.iloc[0, 0])
    assert result.floor["path"].is_monotonic_increasing
    assert result.strategy == "tipp"


def test_dynamic_multiplier_uses_only_lagged_realized_volatility() -> None:
    returns = pd.DataFrame(
        {
            "path_a": [0.10, -0.10, 0.01, 0.50],
            "path_b": [0.10, -0.10, 0.01, -0.50],
        }
    )

    result = run_dynamic_multiplier_cppi(
        returns,
        base_multiplier=3.0,
        target_volatility=0.10,
        lookback=2,
        minimum_history=2,
        minimum_multiplier=1.0,
        maximum_multiplier=6.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    expected_period_two = 3.0 * 0.10 / np.std([0.10, -0.10], ddof=1)
    assert result.multiplier.iloc[0, 0] == pytest.approx(3.0)
    assert result.multiplier.iloc[1, 0] == pytest.approx(3.0)
    assert result.multiplier.iloc[2, 0] == pytest.approx(expected_period_two)
    pd.testing.assert_series_equal(
        result.multiplier.iloc[:3, 0],
        result.multiplier.iloc[:3, 1],
        check_names=False,
    )


def test_dynamic_multiplier_rises_when_lagged_volatility_falls() -> None:
    returns = pd.Series([0.20, -0.20, 0.01, 0.01, 0.01], name="path")

    result = run_dynamic_multiplier_cppi(
        returns,
        base_multiplier=3.0,
        target_volatility=0.10,
        lookback=2,
        minimum_history=2,
        minimum_multiplier=1.0,
        maximum_multiplier=5.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert result.multiplier.iloc[2, 0] < result.multiplier.iloc[3, 0]
    assert result.multiplier.iloc[4, 0] == pytest.approx(5.0)


def test_gap_loss_reports_floor_breach_then_cash_lock() -> None:
    returns = pd.Series([-1.0, 0.0], name="gap")

    result = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert bool(result.floor_breach.iloc[0, 0])
    assert not bool(result.cash_locked.iloc[0, 0])
    assert result.risky_weight.iloc[1, 0] == pytest.approx(0.0)
    assert bool(result.cash_locked.iloc[1, 0])


def test_risky_weight_cap_prevents_borrowing() -> None:
    returns = pd.Series([1.0, 0.0], name="path")

    capped = run_fixed_maturity_cppi(
        returns,
        multiplier=5.0,
        guarantee_fraction=0.5,
        maximum_risky_weight=1.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )
    leveraged = run_fixed_maturity_cppi(
        returns,
        multiplier=5.0,
        guarantee_fraction=0.5,
        maximum_risky_weight=2.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert capped.risky_weight.max().max() <= 1.0
    assert leveraged.risky_weight.iloc[0, 0] == pytest.approx(2.0)
    assert leveraged.safe_weight.iloc[0, 0] == pytest.approx(-1.0)


def test_rebalance_interval_allows_weights_to_drift() -> None:
    returns = pd.Series([1.0, 0.0], name="path")

    daily = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
        rebalance_every=1,
    )
    delayed = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
        rebalance_every=2,
    )

    assert daily.risky_weight.iloc[1, 0] == pytest.approx(1.0)
    assert delayed.risky_weight.iloc[1, 0] == pytest.approx(0.75)
    assert delayed.turnover.iloc[1, 0] == pytest.approx(0.0)


def test_transaction_costs_reduce_wealth_and_are_reported() -> None:
    returns = pd.Series([0.0, 0.0], name="path")

    result = run_fixed_maturity_cppi(
        returns,
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
        transaction_cost_rate=0.01,
        rebalance_every=10,
    )

    assert result.turnover.iloc[0, 0] == pytest.approx(0.6)
    assert result.transaction_costs.iloc[0, 0] == pytest.approx(0.006)
    assert result.wealth.iloc[-1, 0] == pytest.approx(0.994)
    assert result.summary().loc["path", "total_transaction_cost"] == pytest.approx(
        0.006
    )


def test_summary_preserves_path_labels_and_reports_drawdown() -> None:
    returns = pd.DataFrame(
        {
            "up": [0.10, 0.10],
            "down": [-0.10, -0.10],
        },
        index=pd.Index(["period_1", "period_2"]),
    )
    result = run_tipp(
        returns,
        multiplier=2.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    summary = result.summary()

    assert list(summary.index) == ["up", "down"]
    assert summary.loc["down", "maximum_drawdown"] < 0.0
    assert summary.loc["up", "terminal_wealth"] > 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"guarantee_fraction": 1.1}, "between 0 and 1"),
        ({"multiplier": -1.0}, "greater than or equal to zero"),
        ({"minimum_risky_weight": 0.8, "maximum_risky_weight": 0.5}, "must not"),
        ({"transaction_cost_rate": 1.0}, "less than 1"),
        ({"rebalance_every": 0}, "greater than zero"),
        ({"risk_free_rate": -1.0}, "greater than -1.0"),
    ],
)
def test_fixed_maturity_validates_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        run_fixed_maturity_cppi(pd.Series([0.0]), **kwargs)


def test_dynamic_multiplier_validates_policy_parameters() -> None:
    returns = pd.Series([0.0, 0.0])
    with pytest.raises(ValueError, match="must not exceed lookback"):
        run_dynamic_multiplier_cppi(
            returns,
            lookback=2,
            minimum_history=3,
        )
    with pytest.raises(ValueError, match="must lie between"):
        run_dynamic_multiplier_cppi(
            returns,
            base_multiplier=7.0,
            maximum_multiplier=6.0,
        )


def test_returns_require_finite_ordered_simple_paths() -> None:
    with pytest.raises(ValueError, match="monotonic increasing"):
        run_tipp(pd.Series([0.0, 0.0], index=[2, 1]))
    with pytest.raises(ValueError, match="finite"):
        run_tipp(pd.Series([0.0, np.nan]))
    with pytest.raises(ValueError, match="below -1.0"):
        run_tipp(pd.Series([-1.01]))
