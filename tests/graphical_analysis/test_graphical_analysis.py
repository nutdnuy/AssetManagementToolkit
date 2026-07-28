import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.graphical_analysis import graphical_analysis


def _clustered_returns() -> pd.DataFrame:
    generator = np.random.default_rng(42)
    first_factor = generator.normal(0.0, 0.02, 80)
    second_factor = generator.normal(0.0, 0.018, 80)
    noise = generator.normal(0.0, 0.004, (80, 6))
    values = np.column_stack(
        [
            first_factor + noise[:, 0],
            0.9 * first_factor + noise[:, 1],
            0.8 * first_factor + noise[:, 2],
            second_factor + noise[:, 3],
            0.9 * second_factor + noise[:, 4],
            0.8 * second_factor + noise[:, 5],
        ]
    )
    return pd.DataFrame(
        values,
        index=pd.date_range("2019-01-31", periods=80, freq="ME"),
        columns=["A", "B", "C", "D", "E", "F"],
    )


def test_graphical_analysis_returns_labelled_deterministic_outputs() -> None:
    returns = _clustered_returns()

    first = graphical_analysis(returns, edge_threshold=0.05, random_state=7)
    second = graphical_analysis(returns, edge_threshold=0.05, random_state=7)

    assert first.n_observations == 80
    assert first.covariance.index.equals(returns.columns)
    assert first.precision.columns.equals(returns.columns)
    assert first.embedding.shape == (6, 2)
    assert first.cluster_labels.index.equals(returns.columns)
    np.testing.assert_allclose(
        np.diag(first.partial_correlations),
        np.ones(6),
    )
    pd.testing.assert_frame_equal(first.embedding, second.embedding)
    pd.testing.assert_series_equal(first.cluster_labels, second.cluster_labels)
    assert (first.edges["absolute_strength"] > 0.05).all()


def test_graphical_analysis_keeps_input_unchanged() -> None:
    returns = _clustered_returns()
    original = returns.copy(deep=True)

    graphical_analysis(returns)

    pd.testing.assert_frame_equal(returns, original)


def test_graphical_analysis_rejects_missing_and_constant_assets() -> None:
    missing = _clustered_returns()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        graphical_analysis(missing)

    constant = _clustered_returns()
    constant["A"] = 0.01
    with pytest.raises(ValueError, match="zero-volatility"):
        graphical_analysis(constant)


@pytest.mark.parametrize("edge_threshold", [-0.1, 1.0, np.inf])
def test_graphical_analysis_rejects_invalid_threshold(
    edge_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="edge_threshold"):
        graphical_analysis(
            _clustered_returns(),
            edge_threshold=edge_threshold,
        )
