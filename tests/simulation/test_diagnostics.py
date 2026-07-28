import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    compare_simulation_models,
    return_distribution_diagnostics,
    simulate_gbm_returns,
)


def test_return_distribution_diagnostics_reports_shape_and_tails() -> None:
    returns = pd.Series(
        [-0.08, -0.03, -0.01, 0.00, 0.01, 0.02, 0.03, 0.06],
        name="observed",
    )

    result = return_distribution_diagnostics(
        returns,
        periods_per_year=12,
        tail_probability=0.25,
    )

    assert result["n_observations"] == 8
    assert result["periodic_mean"] == pytest.approx(returns.mean())
    assert result["q_tail"] == pytest.approx(returns.quantile(0.25))
    assert result["historical_var"] == pytest.approx(-returns.quantile(0.25))
    tail = returns[returns <= returns.quantile(0.25)]
    assert result["historical_cvar"] == pytest.approx(-tail.mean())
    assert np.isfinite(result["annualized_log_volatility"])


def test_compare_simulation_models_has_observed_baseline_and_distances() -> None:
    observed = pd.Series(
        simulate_gbm_returns(
            n_years=1,
            n_scenarios=2_000,
            expected_return=0.06,
            volatility=0.15,
            periods_per_year=12,
            seed=1,
        ).iloc[0],
        name="observed",
    )
    close_model = simulate_gbm_returns(
        n_years=1,
        n_scenarios=2_000,
        expected_return=0.06,
        volatility=0.15,
        periods_per_year=12,
        seed=2,
    ).iloc[[0]]
    high_volatility = simulate_gbm_returns(
        n_years=1,
        n_scenarios=2_000,
        expected_return=0.06,
        volatility=0.50,
        periods_per_year=12,
        seed=2,
    ).iloc[[0]]

    comparison = compare_simulation_models(
        observed,
        {
            "Close": close_model,
            "High volatility": high_volatility,
        },
        periods_per_year=12,
    )

    assert list(comparison.index) == [
        "Observed",
        "Close",
        "High volatility",
    ]
    assert comparison.loc["Observed", "distribution_error_score"] == 0.0
    assert (
        comparison.loc["Close", "distribution_error_score"]
        < comparison.loc["High volatility", "distribution_error_score"]
    )
    assert comparison.loc["High volatility", "volatility_error"] > 0.0


@pytest.mark.parametrize(
    ("returns", "error"),
    [
        ([0.01, 0.02], TypeError),
        (pd.Series([0.01, 0.02, 0.03]), ValueError),
    ],
)
def test_diagnostics_reject_invalid_inputs(
    returns: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        return_distribution_diagnostics(returns)  # type: ignore[arg-type]


def test_compare_rejects_reserved_or_missing_models() -> None:
    observed = pd.Series([0.01, -0.02, 0.03, 0.00])
    with pytest.raises(ValueError, match="non-empty mapping"):
        compare_simulation_models(observed, {})
    with pytest.raises(ValueError, match="reserved"):
        compare_simulation_models(observed, {"Observed": observed})
    with pytest.raises(ValueError, match="above -1"):
        return_distribution_diagnostics(pd.Series([0.01, -1.0, 0.02, 0.03]))
