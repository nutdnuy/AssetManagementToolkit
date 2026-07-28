import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.portfolio import (
    efficient_frontier,
    efficient_frontier_weights,
    global_minimum_variance,
    maximum_sharpe_ratio,
    minimum_volatility,
    portfolio_return,
    portfolio_volatility,
)


def test_portfolio_return_and_volatility() -> None:
    weights = np.array([0.6, 0.4])
    expected_returns = np.array([0.05, 0.10])
    covariance = np.diag([0.04, 0.09])

    assert portfolio_return(weights, expected_returns) == pytest.approx(0.07)
    assert portfolio_volatility(weights, covariance) == pytest.approx(
        np.sqrt(0.6**2 * 0.04 + 0.4**2 * 0.09)
    )


def test_minimum_volatility_hits_target_and_preserves_labels() -> None:
    expected_returns = pd.Series([0.05, 0.10], index=["bond", "equity"])
    covariance = pd.DataFrame(
        [[0.04, 0.00], [0.00, 0.09]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = minimum_volatility(0.075, expected_returns, covariance)

    assert list(weights.index) == ["bond", "equity"]
    assert weights.sum() == pytest.approx(1.0)
    assert portfolio_return(weights, expected_returns) == pytest.approx(0.075)
    assert (weights >= 0.0).all()


def test_global_minimum_variance_for_identity_covariance_is_equal_weight() -> None:
    covariance = pd.DataFrame(
        np.eye(3),
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    weights = global_minimum_variance(covariance)

    np.testing.assert_allclose(weights.to_numpy(), np.repeat(1 / 3, 3), atol=1e-7)


def test_maximum_sharpe_ratio_is_long_only_and_fully_invested() -> None:
    expected_returns = pd.Series([0.04, 0.08, 0.12], index=["a", "b", "c"])
    covariance = pd.DataFrame(
        np.diag([0.01, 0.04, 0.09]),
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = maximum_sharpe_ratio(0.01, expected_returns, covariance)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 1.0).all()


def test_efficient_frontier_returns_requested_number_of_points() -> None:
    expected_returns = np.array([0.05, 0.10])
    covariance = np.diag([0.04, 0.09])

    weights = efficient_frontier_weights(5, expected_returns, covariance)

    assert weights.shape == (5, 2)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-7)


def test_efficient_frontier_returns_plot_ready_statistics() -> None:
    expected_returns = pd.Series([0.05, 0.10], index=["bond", "equity"])
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09]),
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    result = efficient_frontier(3, expected_returns, covariance)

    assert list(result.columns) == ["expected_return", "volatility"]
    np.testing.assert_allclose(result["expected_return"], [0.05, 0.075, 0.10])
    assert (result["volatility"] > 0.0).all()


def test_portfolio_validation_rejects_misaligned_dimensions() -> None:
    with pytest.raises(ValueError, match="weights must contain"):
        portfolio_return([0.5, 0.5], [0.05, 0.10, 0.12])

    with pytest.raises(ValueError, match="covariance"):
        portfolio_volatility([0.5, 0.5], np.eye(3))


def test_minimum_volatility_rejects_infeasible_target() -> None:
    with pytest.raises(ValueError, match="feasible"):
        minimum_volatility(0.20, [0.05, 0.10], np.eye(2))
