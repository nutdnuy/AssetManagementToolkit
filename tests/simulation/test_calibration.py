import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.simulation import (
    CalibrationResult,
    calibrate_gbm,
    calibrate_merton_jump,
    calibrate_variance_gamma,
    simulate_gbm_returns,
    simulate_merton_jump_returns,
    simulate_variance_gamma_returns,
)


def test_calibrate_gbm_recovers_synthetic_parameters_and_information_criteria() -> None:
    returns = pd.Series(
        simulate_gbm_returns(
            n_years=1,
            n_scenarios=50_000,
            expected_return=0.08,
            volatility=0.20,
            periods_per_year=12,
            seed=1,
        ).iloc[0],
        name="synthetic",
    )

    result = calibrate_gbm(returns, periods_per_year=12)

    assert isinstance(result, CalibrationResult)
    assert result.success
    assert result.parameters["expected_return"] == pytest.approx(0.08, abs=0.01)
    assert result.parameters["volatility"] == pytest.approx(0.20, abs=0.005)
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    assert result.to_series()["parameter_volatility"] == pytest.approx(
        result.parameters["volatility"]
    )


def test_calibrate_merton_recovers_synthetic_jump_direction_and_scale() -> None:
    returns = pd.Series(
        simulate_merton_jump_returns(
            n_years=1,
            n_scenarios=5_000,
            expected_return=0.07,
            volatility=0.12,
            jump_intensity=1.5,
            jump_mean=-0.10,
            jump_volatility=0.18,
            periods_per_year=12,
            seed=2,
        ).iloc[0],
        name="synthetic",
    )

    result = calibrate_merton_jump(
        returns,
        periods_per_year=12,
        max_iterations=250,
    )

    assert result.success
    assert result.parameters["volatility"] == pytest.approx(0.12, abs=0.03)
    assert result.parameters["jump_intensity"] == pytest.approx(1.5, abs=0.8)
    assert result.parameters["jump_mean"] < 0.0
    assert result.parameters["jump_volatility"] == pytest.approx(0.18, abs=0.08)
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None


def test_calibrate_variance_gamma_matches_synthetic_cumulants() -> None:
    returns = pd.Series(
        simulate_variance_gamma_returns(
            n_years=1,
            n_scenarios=80_000,
            mean_log_return=0.05,
            theta=-0.12,
            volatility=0.15,
            variance_rate=0.25,
            periods_per_year=12,
            seed=3,
        ).iloc[0],
        name="synthetic",
    )

    result = calibrate_variance_gamma(returns, periods_per_year=12)

    assert result.success
    assert result.parameters["mean_log_return"] == pytest.approx(0.05, abs=0.01)
    assert result.parameters["theta"] == pytest.approx(-0.12, abs=0.04)
    assert result.parameters["volatility"] == pytest.approx(0.15, abs=0.03)
    assert result.parameters["variance_rate"] == pytest.approx(0.25, abs=0.10)
    assert result.log_likelihood is None
    assert result.aic is None
    assert result.bic is None
    assert result.method == "log_return_cumulant_matching"


@pytest.mark.parametrize(
    "calibrator",
    [calibrate_gbm, calibrate_merton_jump, calibrate_variance_gamma],
)
def test_calibrators_require_one_series(calibrator: object) -> None:
    frame = pd.DataFrame({"a": [0.01] * 40, "b": [0.02] * 40})
    with pytest.raises(TypeError, match="pandas Series"):
        calibrator(frame, 12)  # type: ignore[operator]


def test_calibrators_reject_total_loss_and_constant_returns() -> None:
    with pytest.raises(ValueError, match="greater than -1"):
        calibrate_gbm(pd.Series([0.01, -1.0, 0.02]), 12)
    with pytest.raises(ValueError, match="non-constant"):
        calibrate_gbm(pd.Series([0.01, 0.01, 0.01]), 12)
    with pytest.raises(ValueError, match="variable returns"):
        calibrate_variance_gamma(pd.Series([0.01] * 20), 12)
    with pytest.raises(ValueError, match="missing"):
        calibrate_gbm(pd.Series([0.01, np.nan, 0.02, 0.03]), 12)


def test_merton_calibration_validates_optimizer_controls() -> None:
    returns = pd.Series(np.linspace(-0.03, 0.03, 40))
    with pytest.raises(ValueError, match="max_jump_intensity"):
        calibrate_merton_jump(returns, 12, max_jump_intensity=0.0)
    with pytest.raises(ValueError, match="greater than 1e-6"):
        calibrate_merton_jump(returns, 12, max_jump_intensity=1e-8)
    with pytest.raises(ValueError, match="max_iterations"):
        calibrate_merton_jump(returns, 12, max_iterations=0)
