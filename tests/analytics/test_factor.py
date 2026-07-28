from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.analytics import (
    factor_regression,
    factor_return_attribution,
    regularized_factor_regression,
    rolling_factor_regression,
    rolling_regularized_factor_regression,
)


def _exact_factor_fixture() -> tuple[pd.Series, pd.DataFrame]:
    index = pd.date_range("2024-01-31", periods=24, freq="ME")
    market = np.linspace(-0.04, 0.05, len(index))
    value = np.sin(np.arange(len(index))) * 0.02
    factors = pd.DataFrame(
        {"Market": market, "Value": value},
        index=index,
    )
    design = np.column_stack([np.ones(len(index)), market, value])
    raw_noise = np.cos(np.arange(len(index)) * 0.7) * 0.001
    noise = (
        raw_noise
        - design
        @ np.linalg.lstsq(
            design,
            raw_noise,
            rcond=None,
        )[0]
    )
    asset = pd.Series(
        0.001 + 1.2 * market - 0.4 * value + noise,
        index=index,
        name="Fund",
    )
    return asset, factors


def test_factor_regression_recovers_known_coefficients_and_diagnostics() -> None:
    asset, factors = _exact_factor_fixture()

    result = factor_regression(asset, factors, periods_per_year=12)

    assert result.alpha == pytest.approx(0.012)
    assert result.betas.to_dict() == pytest.approx({"Market": 1.2, "Value": -0.4})
    assert result.coefficients.index.tolist() == ["alpha", "Market", "Value"]
    assert result.standard_errors.index.equals(result.coefficients.index)
    assert result.t_statistics.index.equals(result.coefficients.index)
    assert result.p_values.between(0.0, 1.0).all()
    assert result.n_observations == 24
    assert result.degrees_of_freedom == 21
    assert result.r_squared > 0.99
    assert result.adjusted_r_squared > 0.99
    assert result.residual_volatility > 0.0
    pd.testing.assert_series_equal(
        result.fitted_returns + result.residuals,
        asset.rename("factor_fitted_return"),
        check_names=False,
    )


def test_factor_regression_aligns_and_drops_joint_missing_rows() -> None:
    asset, factors = _exact_factor_fixture()
    asset = asset.drop(asset.index[0])
    factors.loc[factors.index[1], "Value"] = np.nan
    original_asset = asset.copy(deep=True)
    original_factors = factors.copy(deep=True)

    result = factor_regression(asset, factors, periods_per_year=12)

    assert result.n_observations == 22
    assert result.fitted_returns.index.equals(asset.index[1:])
    pd.testing.assert_series_equal(asset, original_asset)
    pd.testing.assert_frame_equal(factors, original_factors)


def test_factor_regression_converts_annual_risk_free_rate() -> None:
    asset, factors = _exact_factor_fixture()
    annual_risk_free = 0.12
    periodic_risk_free = (1.0 + annual_risk_free) ** (1.0 / 12.0) - 1.0
    shifted_asset = asset + periodic_risk_free

    result = factor_regression(
        shifted_asset,
        factors,
        risk_free_rate=annual_risk_free,
        periods_per_year=12,
    )

    assert result.alpha == pytest.approx(0.012)
    assert result.betas.to_dict() == pytest.approx({"Market": 1.2, "Value": -0.4})


def test_rolling_factor_regression_tracks_exposure_change() -> None:
    index = pd.date_range("2025-01-31", periods=16, freq="ME")
    market = np.array([-0.03, 0.01, 0.04, -0.02, 0.03, 0.00, -0.01, 0.02] * 2)
    value = np.array([0.02, -0.01, 0.01, 0.03, -0.02, 0.04, 0.00, -0.03] * 2)
    factors = pd.DataFrame(
        {"Market": market, "Value": value},
        index=index,
    )
    asset = pd.Series(
        np.concatenate(
            [
                0.001 + 1.4 * market[:8] + 0.2 * value[:8],
                -0.001 + 0.4 * market[8:] + 1.1 * value[8:],
            ]
        ),
        index=index,
    )

    result = rolling_factor_regression(
        asset,
        factors,
        window=8,
        step=8,
        periods_per_year=12,
    )

    assert result.window == 8
    assert result.step == 8
    assert result.betas.iloc[0].to_dict() == pytest.approx(
        {"Market": 1.4, "Value": 0.2}
    )
    assert result.betas.iloc[1].to_dict() == pytest.approx(
        {"Market": 0.4, "Value": 1.1}
    )
    assert result.alpha.iloc[0] == pytest.approx(0.012)
    assert result.alpha.iloc[1] == pytest.approx(-0.012)


def test_factor_return_attribution_is_additive_and_labelled() -> None:
    factors = pd.DataFrame(
        {
            "Market": [0.02, -0.01],
            "Value": [-0.01, 0.03],
        },
        index=["period_1", "period_2"],
    )
    exposures = pd.Series({"Market": 1.2, "Value": -0.4})

    result = factor_return_attribution(exposures, factors, alpha=0.001)

    expected = pd.DataFrame(
        {
            "alpha": [0.001, 0.001],
            "Market": [0.024, -0.012],
            "Value": [0.004, -0.012],
            "total": [0.029, -0.023],
        },
        index=factors.index,
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("asset", "factors", "message"),
    [
        (
            pd.Series([0.01, 0.02, 0.03, 0.04]),
            pd.DataFrame(
                {
                    "A": [0.01, 0.02, 0.03, 0.04],
                    "A_copy": [0.01, 0.02, 0.03, 0.04],
                }
            ),
            "linearly independent",
        ),
        (
            pd.Series([0.01, 0.02]),
            pd.DataFrame({"A": [0.01, 0.02]}),
            "at least 3 complete observations",
        ),
    ],
)
def test_factor_regression_rejects_unidentified_or_short_samples(
    asset: pd.Series,
    factors: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factor_regression(asset, factors)


def test_factor_attribution_rejects_misaligned_exposure_labels() -> None:
    factors = pd.DataFrame({"Market": [0.01], "Value": [0.02]})
    exposures = pd.Series({"Value": 0.5, "Market": 0.5})

    with pytest.raises(ValueError, match="labels and order"):
        factor_return_attribution(exposures, factors)


@pytest.mark.parametrize("method", ["ridge", "lasso", "elastic_net"])
def test_regularized_factor_regression_is_labelled_and_additive(
    method: str,
) -> None:
    asset, factors = _exact_factor_fixture()

    result = regularized_factor_regression(
        asset,
        factors,
        method=method,
        regularization=1e-6,
        periods_per_year=12,
    )

    assert result.method == method
    assert result.coefficients.index.tolist() == ["alpha", "Market", "Value"]
    assert result.selected_regularization == pytest.approx(1e-6)
    assert result.n_observations == 24
    assert result.r_squared > 0.99
    pd.testing.assert_series_equal(
        result.fitted_returns + result.residuals,
        asset.rename("factor_fitted_return"),
        check_names=False,
    )


def test_regularized_factor_regression_uses_requested_cv_grid() -> None:
    asset, factors = _exact_factor_fixture()

    result = regularized_factor_regression(
        asset,
        factors,
        method="ridge",
        regularization_grid=[1e-6, 0.01, 1.0],
        cv=3,
        periods_per_year=12,
    )

    assert result.selected_regularization in {1e-6, 0.01, 1.0}
    assert result.r_squared > 0.99


def test_rolling_regularized_factor_regression_tracks_windows() -> None:
    asset, factors = _exact_factor_fixture()

    result = rolling_regularized_factor_regression(
        asset,
        factors,
        window=12,
        step=6,
        method="ridge",
        regularization=1e-6,
        periods_per_year=12,
    )

    assert result.coefficients.shape == (3, 3)
    assert result.coefficients.columns.tolist() == ["alpha", "Market", "Value"]
    assert result.selected_regularization.eq(1e-6).all()
    assert result.window == 12
    assert result.step == 6


def test_regularized_factor_regression_allows_collinear_factors() -> None:
    index = pd.date_range("2023-01-31", periods=12, freq="ME")
    market = np.linspace(-0.03, 0.04, len(index))
    factors = pd.DataFrame(
        {"Market": market, "Market copy": market},
        index=index,
    )
    asset = pd.Series(0.001 + 0.8 * market, index=index)

    result = regularized_factor_regression(
        asset,
        factors,
        method="ridge",
        regularization=0.01,
        periods_per_year=12,
    )

    assert np.isfinite(result.betas).all()
    assert result.r_squared > 0.99


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "bad"}, "method"),
        ({"regularization": 0.0}, "regularization"),
        ({"l1_ratio": 0.0}, "l1_ratio"),
        ({"cv": 1}, "cv"),
        ({"regularization_grid": []}, "regularization_grid"),
    ],
)
def test_regularized_factor_regression_rejects_invalid_options(
    kwargs: dict[str, object],
    message: str,
) -> None:
    asset, factors = _exact_factor_fixture()

    with pytest.raises((TypeError, ValueError), match=message):
        regularized_factor_regression(asset, factors, **kwargs)


@pytest.mark.parametrize("reserved", ["alpha", "total"])
def test_factor_returns_reject_reserved_result_labels(reserved: str) -> None:
    factors = pd.DataFrame(
        {
            "Market": [0.01, -0.01, 0.02, 0.00],
            reserved: [0.02, 0.01, -0.01, 0.03],
        }
    )
    asset = pd.Series([0.01, 0.02, -0.01, 0.03])

    with pytest.raises(ValueError, match="reserved result labels"):
        factor_regression(asset, factors)
