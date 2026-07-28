import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.portfolio import (
    BlackLittermanResult,
    black_litterman_posterior,
    implied_equilibrium_returns,
    maximum_sharpe_ratio,
    proportional_view_uncertainty,
)


@pytest.fixture
def labelled_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    assets = pd.Index(["Equity", "Bond", "Gold"], name="asset")
    weights = pd.Series([0.50, 0.30, 0.20], index=assets, name="market_weight")
    covariance = pd.DataFrame(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0100, 0.0015],
            [0.0040, 0.0015, 0.0225],
        ],
        index=assets,
        columns=assets,
    )
    view_labels = pd.Index(["equity_vs_bond", "gold_absolute"], name="view")
    picks = pd.DataFrame(
        [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        index=view_labels,
        columns=assets,
    )
    views = pd.Series([0.04, 0.03], index=view_labels, name="view_return")
    return weights, covariance, picks, views


def test_implied_equilibrium_returns_matches_matrix_formula(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    weights, covariance, _, _ = labelled_inputs

    result = implied_equilibrium_returns(
        weights,
        covariance,
        risk_aversion=2.5,
    )
    expected = 2.5 * covariance.to_numpy() @ weights.to_numpy()

    assert result.name == "implied_equilibrium_return"
    assert result.index.equals(weights.index)
    np.testing.assert_allclose(result.to_numpy(), expected)


def test_proportional_uncertainty_keeps_only_view_variances(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    _, covariance, picks, _ = labelled_inputs

    result = proportional_view_uncertainty(covariance, picks, tau=0.05)
    full = 0.05 * picks.to_numpy() @ covariance.to_numpy() @ picks.to_numpy().T

    assert result.index.equals(picks.index)
    assert result.columns.equals(picks.index)
    np.testing.assert_allclose(result.to_numpy(), np.diag(np.diag(full)))


def test_posterior_matches_independent_precision_form(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    weights, covariance, picks, views = labelled_inputs
    omega = pd.DataFrame(
        np.diag([0.0020, 0.0010]),
        index=views.index,
        columns=views.index,
    )
    tau = 0.05
    risk_aversion = 2.5

    result = black_litterman_posterior(
        weights,
        covariance,
        picks,
        views,
        risk_aversion=risk_aversion,
        tau=tau,
        view_uncertainty=omega,
    )

    sigma = covariance.to_numpy()
    p = picks.to_numpy()
    q = views.to_numpy()
    pi = risk_aversion * sigma @ weights.to_numpy()
    posterior_mean_covariance = np.linalg.inv(
        np.linalg.inv(tau * sigma) + p.T @ np.linalg.inv(omega.to_numpy()) @ p
    )
    expected_returns = posterior_mean_covariance @ (
        np.linalg.inv(tau * sigma) @ pi + p.T @ np.linalg.inv(omega.to_numpy()) @ q
    )
    expected_covariance = sigma + posterior_mean_covariance

    assert isinstance(result, BlackLittermanResult)
    np.testing.assert_allclose(result.prior_returns.to_numpy(), pi)
    np.testing.assert_allclose(
        result.posterior_returns.to_numpy(),
        expected_returns,
    )
    np.testing.assert_allclose(
        result.posterior_covariance.to_numpy(),
        expected_covariance,
    )


def test_equilibrium_views_leave_posterior_returns_at_prior(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    weights, covariance, picks, _ = labelled_inputs
    prior = implied_equilibrium_returns(weights, covariance)
    equilibrium_views = picks @ prior

    result = black_litterman_posterior(
        weights,
        covariance,
        picks,
        equilibrium_views,
    )

    pd.testing.assert_series_equal(
        result.posterior_returns.rename("implied_equilibrium_return"),
        prior,
    )


def test_posterior_integrates_with_long_only_markowitz(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
) -> None:
    weights, covariance, picks, views = labelled_inputs
    result = black_litterman_posterior(weights, covariance, picks, views)

    allocation = maximum_sharpe_ratio(
        0.0,
        result.posterior_returns,
        result.posterior_covariance,
    )

    assert allocation.index.equals(weights.index)
    assert allocation.sum() == pytest.approx(1.0)
    assert allocation.ge(0.0).all()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda w, s, p, q: implied_equilibrium_returns(w * 2.0, s),
            "sum to one",
        ),
        (
            lambda w, s, p, q: black_litterman_posterior(
                w,
                s,
                p[["Bond", "Equity", "Gold"]],
                q,
            ),
            "columns must match",
        ),
        (
            lambda w, s, p, q: black_litterman_posterior(
                w,
                s,
                p,
                q.iloc[::-1],
            ),
            "views index must match",
        ),
        (
            lambda w, s, p, q: black_litterman_posterior(
                w,
                s,
                p,
                q,
                tau=0.0,
            ),
            "tau",
        ),
    ],
)
def test_black_litterman_rejects_invalid_contracts(
    labelled_inputs: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series],
    operation: object,
    message: str,
) -> None:
    weights, covariance, picks, views = labelled_inputs
    with pytest.raises((TypeError, ValueError), match=message):
        operation(weights, covariance, picks, views)  # type: ignore[operator]
