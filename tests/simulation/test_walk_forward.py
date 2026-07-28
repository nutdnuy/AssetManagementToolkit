import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    simulate_gbm_returns,
    walk_forward_validate_simulation,
)


def _sample_returns() -> pd.Series:
    returns = simulate_gbm_returns(
        n_years=10,
        n_scenarios=1,
        expected_return=0.07,
        volatility=0.15,
        periods_per_year=12,
        seed=7,
    ).iloc[:, 0]
    returns.index = pd.period_range("2016-01", periods=len(returns), freq="M")
    return returns.rename("asset")


def test_expanding_walk_forward_is_reproducible_and_has_no_lookahead() -> None:
    returns = _sample_returns()
    first = walk_forward_validate_simulation(
        returns,
        model="gbm",
        train_size=60,
        test_size=12,
        periods_per_year=12,
        n_scenarios=300,
        seed=11,
    )
    second = walk_forward_validate_simulation(
        returns,
        model="gbm",
        train_size=60,
        test_size=12,
        periods_per_year=12,
        n_scenarios=300,
        seed=11,
    )

    pd.testing.assert_frame_equal(first, second)
    assert list(first["n_train"]) == [60, 72, 84, 96, 108]
    assert (first["n_test"] == 12).all()
    assert (first["train_end"] < first["test_start"]).all()
    assert first["tail_exceedance_rate"].between(0.0, 1.0).all()
    assert first["ks_statistic"].between(0.0, 1.0).all()
    assert first["parameter_volatility"].gt(0.0).all()


def test_rolling_walk_forward_keeps_fixed_training_window() -> None:
    result = walk_forward_validate_simulation(
        _sample_returns(),
        model="gbm",
        train_size=48,
        test_size=12,
        periods_per_year=12,
        n_scenarios=100,
        window="rolling",
        seed=5,
    )

    assert (result["n_train"] == 48).all()
    assert result.iloc[1]["train_start"] > result.iloc[0]["train_start"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "stable"}, "model must be one of"),
        ({"model": "gbm", "window": "anchored"}, "window"),
        ({"model": "gbm", "train_size": 0}, "train_size"),
        ({"model": "gbm", "test_size": 0}, "test_size"),
    ],
)
def test_walk_forward_rejects_invalid_contracts(
    kwargs: dict,
    message: str,
) -> None:
    parameters = {
        "returns": _sample_returns(),
        "model": "gbm",
        "train_size": 60,
        "test_size": 12,
        "periods_per_year": 12,
        "n_scenarios": 100,
        "seed": 1,
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        walk_forward_validate_simulation(**parameters)


def test_walk_forward_requires_a_complete_fold() -> None:
    with pytest.raises(ValueError, match="complete train/test fold"):
        walk_forward_validate_simulation(
            _sample_returns().iloc[:20],
            model="gbm",
            train_size=15,
            test_size=10,
            periods_per_year=12,
        )


def test_walk_forward_requires_ordered_unique_index() -> None:
    returns = _sample_returns()
    duplicated = returns.copy()
    duplicated.index = [returns.index[0]] * len(returns)
    with pytest.raises(ValueError, match="unique"):
        walk_forward_validate_simulation(
            duplicated,
            model="gbm",
            train_size=60,
            test_size=12,
            periods_per_year=12,
        )

    with pytest.raises(ValueError, match="sorted"):
        walk_forward_validate_simulation(
            returns.sort_index(ascending=False),
            model="gbm",
            train_size=60,
            test_size=12,
            periods_per_year=12,
        )
