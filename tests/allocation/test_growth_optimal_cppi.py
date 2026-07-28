"""Tests for growth-optimal CPPI and jump gap-risk diagnostics."""

import pandas as pd
import pytest

from asset_management_toolkit.allocation import (
    analyze_cppi_gap_risk,
    growth_optimal_multiplier,
    run_dynamic_multiplier_cppi,
    run_fixed_maturity_cppi,
    run_growth_optimal_cppi,
)


def test_growth_optimal_multiplier_reduces_to_excess_return_over_variance() -> None:
    multiplier = growth_optimal_multiplier(
        expected_risky_return=0.08,
        expected_reserve_return=0.02,
        risky_volatility=0.20,
    )

    assert multiplier == pytest.approx((0.08 - 0.02) / 0.20**2)


def test_growth_optimal_multiplier_includes_risky_reserve_covariance() -> None:
    multiplier = growth_optimal_multiplier(
        expected_risky_return=0.08,
        expected_reserve_return=0.03,
        risky_volatility=0.20,
        reserve_volatility=0.10,
        correlation=0.50,
    )

    risky_growth = 0.08 - 0.5 * 0.20**2
    reserve_growth = 0.03 - 0.5 * 0.10**2
    relative_variance = 0.20**2 + 0.10**2 - 2 * 0.50 * 0.20 * 0.10
    expected = (risky_growth - reserve_growth + 0.5 * relative_variance) / (
        relative_variance
    )
    assert multiplier == pytest.approx(expected)


def test_growth_optimal_cppi_floor_tracks_locally_risky_reserve() -> None:
    index = pd.period_range("2026Q1", periods=2, freq="Q")
    risky = pd.Series([0.0, 0.0], index=index, name="scenario")
    reserve = pd.Series([0.10, -0.05], index=index, name="bond")

    result = run_growth_optimal_cppi(
        risky,
        reserve,
        expected_risky_return=0.00,
        expected_reserve_return=0.10,
        risky_volatility=0.20,
        reserve_volatility=0.10,
        correlation=0.0,
        floor_fraction=0.8,
    )

    assert result.multiplier.eq(0.0).all().all()
    assert result.floor.iloc[0, 0] == pytest.approx(0.8 * 1.10)
    assert result.floor.iloc[1, 0] == pytest.approx(0.8 * 1.10 * 0.95)
    assert result.wealth.iloc[1, 0] == pytest.approx(1.10 * 0.95)
    assert result.strategy == "growth_optimal_cppi"


def test_growth_optimal_cppi_accepts_point_in_time_moment_paths() -> None:
    index = pd.Index(["t0", "t1"])
    risky = pd.DataFrame(
        {"path_a": [0.0, 0.0], "path_b": [0.0, 0.0]},
        index=index,
    )
    reserve = pd.Series([0.0, 0.0], index=index)
    expected_risky = pd.Series([0.06, 0.10], index=index)

    result = run_growth_optimal_cppi(
        risky,
        reserve,
        expected_risky_return=expected_risky,
        expected_reserve_return=0.02,
        risky_volatility=0.20,
        reserve_volatility=0.0,
        correlation=0.0,
        maximum_multiplier=6.0,
    )

    assert result.multiplier.iloc[0, 0] == pytest.approx(1.0)
    assert result.multiplier.iloc[1, 0] == pytest.approx(2.0)
    pd.testing.assert_series_equal(
        result.multiplier["path_a"],
        result.multiplier["path_b"],
        check_names=False,
    )


def test_growth_optimal_cppi_rejects_misaligned_reserve_and_moments() -> None:
    risky = pd.Series([0.0, 0.0], index=["t0", "t1"])
    reserve = pd.Series([0.0, 0.0], index=["t1", "t2"])

    with pytest.raises(ValueError, match="index must match"):
        run_growth_optimal_cppi(
            risky,
            reserve,
            expected_risky_return=0.08,
            expected_reserve_return=0.02,
            risky_volatility=0.20,
            reserve_volatility=0.0,
            correlation=0.0,
        )

    with pytest.raises(ValueError, match="correlation must be between"):
        run_growth_optimal_cppi(
            risky,
            pd.Series([0.0, 0.0], index=risky.index),
            expected_risky_return=0.08,
            expected_reserve_return=0.02,
            risky_volatility=0.20,
            reserve_volatility=0.0,
            correlation=1.1,
        )


def test_growth_optimal_multiplier_requires_relative_risk() -> None:
    with pytest.raises(ValueError, match="relative variance"):
        growth_optimal_multiplier(
            expected_risky_return=0.05,
            expected_reserve_return=0.03,
            risky_volatility=0.10,
            reserve_volatility=0.10,
            correlation=1.0,
        )


def test_gap_risk_report_measures_breach_probability_and_losses() -> None:
    result = run_fixed_maturity_cppi(
        pd.DataFrame(
            {
                "safe": [0.0, 0.0],
                "gap": [-1.0, 0.0],
            }
        ),
        multiplier=3.0,
        guarantee_fraction=0.8,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    gap_risk = analyze_cppi_gap_risk(result, confidence_level=0.95)

    assert gap_risk.statistics["scenario_count"] == 2
    assert gap_risk.statistics["floor_breach_probability"] == pytest.approx(0.5)
    assert gap_risk.scenario_losses.loc["safe", "terminal_shortfall"] == 0.0
    assert gap_risk.scenario_losses.loc["gap", "terminal_shortfall"] > 0.0
    assert gap_risk.scenario_losses.loc["gap", "first_breach_period"] == 0
    assert gap_risk.statistics["worst_terminal_shortfall"] == pytest.approx(
        gap_risk.scenario_losses.loc["gap", "terminal_shortfall"]
    )


def test_gap_risk_report_handles_no_breach_and_validates_confidence() -> None:
    result = run_fixed_maturity_cppi(
        pd.DataFrame({"a": [0.0], "b": [0.1]}),
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    gap_risk = analyze_cppi_gap_risk(result)

    assert gap_risk.statistics["floor_breach_probability"] == 0.0
    assert pd.isna(gap_risk.statistics["expected_terminal_shortfall_given_breach"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        analyze_cppi_gap_risk(result, confidence_level=1.0)


def test_dynamic_multiplier_supports_alpha_stable_hazard_exponent() -> None:
    returns = pd.Series([0.20, -0.20, 0.01], name="path")

    linear = run_dynamic_multiplier_cppi(
        returns,
        base_multiplier=3.0,
        target_volatility=0.10,
        lookback=2,
        minimum_history=2,
        minimum_multiplier=0.0,
        maximum_multiplier=6.0,
        volatility_exponent=1.0,
        risk_free_rate=0.0,
        periods_per_year=1,
    )
    stable_hazard = run_dynamic_multiplier_cppi(
        returns,
        base_multiplier=3.0,
        target_volatility=0.10,
        lookback=2,
        minimum_history=2,
        minimum_multiplier=0.0,
        maximum_multiplier=6.0,
        volatility_exponent=2.0 / 1.5,
        risk_free_rate=0.0,
        periods_per_year=1,
    )

    assert stable_hazard.multiplier.iloc[2, 0] < linear.multiplier.iloc[2, 0]
