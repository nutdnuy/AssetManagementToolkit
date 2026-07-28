import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.backtesting import (
    BacktestResult,
    run_weight_backtest,
)


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "asset_a": [0.10, 0.00],
            "asset_b": [0.00, 0.10],
        },
        index=pd.date_range("2026-01-01", periods=2, freq="D"),
    )


def test_buy_and_hold_weights_drift_without_lookahead() -> None:
    returns = _returns()
    weights = pd.DataFrame(
        [[0.5, 0.5]],
        index=returns.index[:1],
        columns=returns.columns,
    )

    result = run_weight_backtest(returns, weights)

    assert isinstance(result, BacktestResult)
    assert result.gross_returns.iloc[0] == pytest.approx(0.05)
    assert result.weights_after.iloc[0, 0] == pytest.approx(0.55 / 1.05)
    assert result.gross_returns.iloc[1] == pytest.approx((0.50 / 1.05) * 0.10)
    assert result.nav.iloc[-1] == pytest.approx(1.10)
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.transaction_costs.iloc[0] == pytest.approx(0.0)


def test_rebalance_cost_is_deducted_before_the_period_return() -> None:
    returns = _returns()
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=returns.index,
        columns=returns.columns,
    )

    result = run_weight_backtest(
        returns,
        weights,
        transaction_cost_rate=0.01,
    )

    expected_turnover = 0.5 * (abs(0.5 - 0.55 / 1.05) + abs(0.5 - 0.50 / 1.05))
    expected_cost = 1.05 * 0.01 * expected_turnover
    expected_nav = (1.05 - expected_cost) * 1.05
    assert result.turnover.iloc[1] == pytest.approx(expected_turnover)
    assert result.transaction_costs.iloc[1] == pytest.approx(expected_cost)
    assert result.nav.iloc[-1] == pytest.approx(expected_nav)


def test_backtest_does_not_mutate_inputs_and_preserves_labels() -> None:
    returns = _returns()
    weights = pd.DataFrame(
        [[0.4, 0.6]],
        index=returns.index[:1],
        columns=returns.columns,
    )
    original_returns = returns.copy(deep=True)
    original_weights = weights.copy(deep=True)

    result = run_weight_backtest(returns, weights)

    pd.testing.assert_frame_equal(returns, original_returns)
    pd.testing.assert_frame_equal(weights, original_weights)
    assert result.weights_at_start.columns.equals(returns.columns)
    assert result.nav.index.equals(returns.index)


def test_initial_trade_cost_is_explicitly_optional() -> None:
    returns = pd.DataFrame(
        [[0.0, 0.0]],
        index=pd.date_range("2026-01-01", periods=1),
        columns=["a", "b"],
    )
    weights = pd.DataFrame(
        [[0.5, 0.5]],
        index=returns.index,
        columns=returns.columns,
    )

    result = run_weight_backtest(
        returns,
        weights,
        transaction_cost_rate=0.01,
        charge_initial_trade=True,
    )

    assert result.transaction_costs.iloc[0] == pytest.approx(0.01)
    assert result.portfolio_returns.iloc[0] == pytest.approx(-0.01)
    assert result.nav.iloc[0] == pytest.approx(0.99)


def test_backtest_requires_first_weights_on_first_return_date() -> None:
    returns = _returns()
    weights = pd.DataFrame(
        [[0.5, 0.5]],
        index=returns.index[1:],
        columns=returns.columns,
    )

    with pytest.raises(ValueError, match="first target_weights"):
        run_weight_backtest(returns, weights)


@pytest.mark.parametrize(
    "weights",
    [
        [[0.6, 0.6]],
        [[-0.1, 1.1]],
        [[np.nan, np.nan]],
    ],
)
def test_backtest_rejects_invalid_target_weights(weights: list[list[float]]) -> None:
    returns = _returns().iloc[:1]
    targets = pd.DataFrame(
        weights,
        index=returns.index,
        columns=returns.columns,
    )

    with pytest.raises(ValueError):
        run_weight_backtest(returns, targets)
