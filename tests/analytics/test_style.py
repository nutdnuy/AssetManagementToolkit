import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.analytics import (
    RollingStyleAnalysisResult,
    StyleAnalysisResult,
    rolling_style_exposures,
    style_exposures,
)


@pytest.fixture
def style_data() -> tuple[pd.Series, pd.DataFrame]:
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    styles = pd.DataFrame(
        {
            "equity": [
                0.030,
                -0.020,
                0.015,
                0.040,
                -0.010,
                0.025,
                0.005,
                -0.015,
                0.035,
                0.010,
                -0.005,
                0.020,
            ],
            "bond": [
                0.005,
                0.010,
                -0.002,
                0.004,
                0.012,
                0.003,
                0.008,
                0.006,
                -0.001,
                0.007,
                0.009,
                0.002,
            ],
        },
        index=index,
    )
    fund = (0.7 * styles["equity"] + 0.3 * styles["bond"]).rename("fund")
    return fund, styles


def test_style_exposures_recovers_known_mixture(
    style_data: tuple[pd.Series, pd.DataFrame],
) -> None:
    fund, styles = style_data

    result = style_exposures(fund, styles)

    assert isinstance(result, StyleAnalysisResult)
    assert result.weights.to_dict() == pytest.approx(
        {"equity": 0.7, "bond": 0.3},
        abs=1e-8,
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.residual_sum_squares == pytest.approx(0.0, abs=1e-16)
    assert result.r_squared == pytest.approx(1.0)
    assert result.n_observations == len(fund)
    pd.testing.assert_series_equal(
        result.fitted_returns,
        fund.rename("style_fitted_return"),
        check_exact=False,
        atol=1e-10,
    )


def test_style_exposures_enforces_long_only_boundary(
    style_data: tuple[pd.Series, pd.DataFrame],
) -> None:
    _, styles = style_data
    fund = (1.2 * styles["equity"] - 0.2 * styles["bond"]).rename("fund")

    result = style_exposures(fund, styles)

    assert result.weights["equity"] == pytest.approx(1.0)
    assert result.weights["bond"] == pytest.approx(0.0)
    assert result.residual_sum_squares > 0.0


def test_style_exposures_does_not_use_mean_residual_to_infer_style(
    style_data: tuple[pd.Series, pd.DataFrame],
) -> None:
    fund, styles = style_data
    fund_with_constant_selection_return = fund + 0.01

    result = style_exposures(fund_with_constant_selection_return, styles)

    assert result.weights.to_dict() == pytest.approx(
        {"equity": 0.7, "bond": 0.3},
        abs=1e-8,
    )
    np.testing.assert_allclose(result.residuals, 0.01, atol=1e-10)
    assert result.residual_sum_squares == pytest.approx(0.0, abs=1e-16)
    assert result.r_squared == pytest.approx(1.0)


def test_style_exposures_aligns_and_drops_jointly_missing_rows_without_mutation(
    style_data: tuple[pd.Series, pd.DataFrame],
) -> None:
    fund, styles = style_data
    fund = fund.iloc[1:].copy()
    styles = styles.copy()
    styles.iloc[3, 0] = np.nan
    original_fund = fund.copy(deep=True)
    original_styles = styles.copy(deep=True)

    result = style_exposures(fund, styles)

    expected_index = fund.index.intersection(styles.dropna().index)
    assert result.n_observations == len(expected_index)
    assert result.fitted_returns.index.equals(expected_index)
    assert result.weights.to_dict() == pytest.approx(
        {"equity": 0.7, "bond": 0.3},
        abs=1e-8,
    )
    pd.testing.assert_series_equal(fund, original_fund)
    pd.testing.assert_frame_equal(styles, original_styles)


def test_rolling_style_exposures_tracks_a_regime_change() -> None:
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    styles = pd.DataFrame(
        {
            "equity": [
                0.03,
                -0.02,
                0.01,
                0.04,
                -0.01,
                0.02,
                0.01,
                -0.03,
                0.05,
                0.02,
                -0.01,
                0.03,
            ],
            "bond": [
                0.00,
                0.01,
                0.005,
                -0.002,
                0.01,
                0.003,
                0.008,
                0.004,
                0.00,
                0.006,
                0.009,
                0.002,
            ],
        },
        index=index,
    )
    fund = pd.concat(
        [
            0.8 * styles["equity"].iloc[:6] + 0.2 * styles["bond"].iloc[:6],
            0.2 * styles["equity"].iloc[6:] + 0.8 * styles["bond"].iloc[6:],
        ]
    ).rename("fund")

    result = rolling_style_exposures(fund, styles, window=6, step=6)

    assert isinstance(result, RollingStyleAnalysisResult)
    assert result.window == 6
    assert result.weights.index.equals(index[[5, 11]])
    assert result.weights.iloc[0].to_dict() == pytest.approx(
        {"equity": 0.8, "bond": 0.2},
        abs=1e-8,
    )
    assert result.weights.iloc[1].to_dict() == pytest.approx(
        {"equity": 0.2, "bond": 0.8},
        abs=1e-8,
    )
    np.testing.assert_allclose(result.r_squared, 1.0)


@pytest.mark.parametrize(
    ("fund_transform", "style_transform", "message"),
    [
        (
            lambda fund: fund.rename_axis(None).set_axis([0] * len(fund)),
            lambda styles: styles,
            "index must be unique",
        ),
        (
            lambda fund: fund,
            lambda styles: styles.assign(copy=styles["equity"]).rename(
                columns={"copy": "equity"}
            ),
            "columns must be unique",
        ),
        (
            lambda fund: fund,
            lambda styles: styles.assign(bond=styles["equity"]),
            "distinct exposures",
        ),
        (
            lambda fund: fund,
            lambda styles: styles.assign(equity=np.inf),
            "infinite",
        ),
    ],
)
def test_style_exposures_rejects_invalid_inputs(
    style_data: tuple[pd.Series, pd.DataFrame],
    fund_transform: object,
    style_transform: object,
    message: str,
) -> None:
    fund, styles = style_data
    with pytest.raises((TypeError, ValueError), match=message):
        style_exposures(
            fund_transform(fund),  # type: ignore[operator]
            style_transform(styles),  # type: ignore[operator]
        )


@pytest.mark.parametrize(
    ("window", "step", "exception", "message"),
    [
        (1, 1, ValueError, "at least"),
        (13, 1, ValueError, "cannot exceed"),
        (6, 0, ValueError, "greater than zero"),
        (6, True, TypeError, "integer"),
    ],
)
def test_rolling_style_exposures_rejects_invalid_windows(
    style_data: tuple[pd.Series, pd.DataFrame],
    window: int,
    step: int,
    exception: type[Exception],
    message: str,
) -> None:
    fund, styles = style_data
    with pytest.raises(exception, match=message):
        rolling_style_exposures(fund, styles, window=window, step=step)
