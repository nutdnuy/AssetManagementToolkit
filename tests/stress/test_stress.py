"""Tests for probability-free portfolio stress testing."""

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.stress import (
    historical_stress_scenarios,
    stress_test_portfolio,
    stress_test_portfolio_paths,
)


def test_historical_scenarios_compound_inclusive_windows() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "equity": [0.10, -0.20, 0.05, 0.01],
            "bond": [0.01, 0.02, -0.01, 0.00],
        },
        index=dates,
    )

    result = historical_stress_scenarios(
        returns,
        {
            "selloff": (dates[0], dates[1]),
            "recovery": (dates[2], dates[3]),
        },
    )

    assert result.loc["selloff", "equity"] == pytest.approx(1.10 * 0.80 - 1.0)
    assert result.loc["selloff", "bond"] == pytest.approx(1.01 * 1.02 - 1.0)
    assert result.index.name == "scenario"


def test_historical_scenarios_validate_windows_and_returns() -> None:
    returns = pd.DataFrame(
        {"asset": [0.01, 0.02]},
        index=pd.date_range("2026-01-01", periods=2),
    )

    with pytest.raises(ValueError, match="non-empty mapping"):
        historical_stress_scenarios(returns, {})
    with pytest.raises(ValueError, match="start <= end"):
        historical_stress_scenarios(
            returns,
            {"invalid": (returns.index[1], returns.index[0])},
        )
    with pytest.raises(ValueError, match="contains no observations"):
        historical_stress_scenarios(
            returns,
            {"future": ("2027-01-01", "2027-01-02")},
        )
    with pytest.raises(TypeError, match="DatetimeIndex"):
        historical_stress_scenarios(
            returns.reset_index(drop=True),
            {"window": (0, 1)},
        )


def test_portfolio_stress_reports_contributions_losses_and_breaches() -> None:
    weights = pd.Series({"bond": 0.40, "equity": 0.60})
    scenarios = pd.DataFrame(
        {
            "equity": [-0.30, 0.10],
            "bond": [-0.05, 0.02],
        },
        index=pd.Index(["risk_off", "relief"], name="scenario"),
    )

    result = stress_test_portfolio(
        weights,
        scenarios,
        loss_thresholds={"warning": 0.10, "limit": 0.20},
    )

    assert result.summary.loc["risk_off", "portfolio_return"] == pytest.approx(-0.20)
    assert result.summary.loc["risk_off", "portfolio_loss"] == pytest.approx(0.20)
    assert result.asset_contributions.loc["risk_off", "equity"] == pytest.approx(-0.18)
    assert result.summary.loc["risk_off", "worst_asset"] == "equity"
    assert bool(result.threshold_breaches.loc["risk_off", "warning"])
    assert bool(result.threshold_breaches.loc["risk_off", "limit"])
    assert not bool(result.threshold_breaches.loc["relief", "warning"])
    assert result.summary.loc["risk_off", "breached_threshold_count"] == 2


def test_portfolio_stress_aligns_labels_and_allows_short_weights() -> None:
    weights = pd.Series({"equity": 1.20, "bond": -0.20})
    scenarios = pd.DataFrame(
        {"bond": [0.10], "equity": [-0.20]},
        index=["shock"],
    )

    result = stress_test_portfolio(weights, scenarios)

    assert result.summary.loc["shock", "portfolio_return"] == pytest.approx(-0.26)
    assert result.threshold_breaches.empty


def test_portfolio_stress_rejects_bad_labels_weights_and_thresholds() -> None:
    scenarios = pd.DataFrame({"equity": [-0.20], "bond": [0.01]})

    with pytest.raises(ValueError, match="match exactly"):
        stress_test_portfolio(pd.Series({"equity": 1.0}), scenarios)
    with pytest.raises(ValueError, match="sum to 1.0"):
        stress_test_portfolio(
            pd.Series({"equity": 0.50, "bond": 0.40}),
            scenarios,
        )
    with pytest.raises(ValueError, match="non-negative"):
        stress_test_portfolio(
            pd.Series({"equity": 0.60, "bond": 0.40}),
            scenarios,
            loss_thresholds={"invalid": -0.01},
        )
    with pytest.raises(ValueError, match="below -1.0"):
        stress_test_portfolio(
            pd.Series({"equity": 0.60, "bond": 0.40}),
            pd.DataFrame({"equity": [-1.01], "bond": [0.00]}),
        )


def test_path_stress_reports_terminal_loss_drawdown_and_worst_period() -> None:
    weights = pd.Series({"equity": 0.60, "bond": 0.40})
    paths = {
        "three_step_crisis": pd.DataFrame(
            {
                "equity": [-0.20, -0.10, 0.05],
                "bond": [0.00, -0.05, 0.01],
            }
        ),
        "short_relief": pd.DataFrame(
            {
                "bond": [0.01, 0.01],
                "equity": [0.02, 0.03],
            }
        ),
    }

    result = stress_test_portfolio_paths(
        weights,
        paths,
        loss_thresholds={"capital_limit": 0.10},
    )

    crisis_periods = np.array([-0.12, -0.08, 0.034])
    expected_terminal = np.prod(1.0 + crisis_periods) - 1.0
    expected_wealth = np.cumprod(1.0 + crisis_periods)
    expected_drawdown = np.min(
        expected_wealth / np.maximum.accumulate(np.r_[1.0, expected_wealth])[:-1] - 1.0
    )

    assert result.summary.loc["three_step_crisis", "terminal_return"] == pytest.approx(
        expected_terminal
    )
    assert result.summary.loc["three_step_crisis", "maximum_drawdown"] == pytest.approx(
        expected_drawdown
    )
    assert result.summary.loc["three_step_crisis", "worst_period_return"] == -0.12
    assert result.summary.loc["three_step_crisis", "n_periods"] == 3
    assert bool(result.threshold_breaches.loc["three_step_crisis", "capital_limit"])
    assert list(result.portfolio_returns.columns) == [
        "three_step_crisis",
        "short_relief",
    ]
    assert np.isnan(result.portfolio_returns.loc[2, "short_relief"])


def test_path_stress_rejects_inconsistent_assets_and_empty_mapping() -> None:
    weights = pd.Series({"equity": 0.60, "bond": 0.40})
    with pytest.raises(ValueError, match="non-empty mapping"):
        stress_test_portfolio_paths(weights, {})
    with pytest.raises(ValueError, match="same assets"):
        stress_test_portfolio_paths(
            weights,
            {
                "first": pd.DataFrame({"equity": [-0.10], "bond": [0.00]}),
                "second": pd.DataFrame({"equity": [-0.10], "cash": [0.00]}),
            },
        )
    with pytest.raises(ValueError, match="portfolio return below -1.0"):
        stress_test_portfolio_paths(
            pd.Series({"equity": 2.0, "bond": -1.0}),
            {
                "insolvent": pd.DataFrame(
                    {"equity": [-1.0], "bond": [1.0]},
                )
            },
        )
