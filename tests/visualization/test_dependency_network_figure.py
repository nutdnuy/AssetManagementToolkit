import pandas as pd
import pytest

from asset_management_toolkit.graphical_analysis import GraphicalAnalysisResult
from asset_management_toolkit.visualization import dependency_network_figure


def _result() -> GraphicalAnalysisResult:
    assets = pd.Index(["A", "B", "C"])
    return GraphicalAnalysisResult(
        covariance=pd.DataFrame(0.0, index=assets, columns=assets),
        precision=pd.DataFrame(0.0, index=assets, columns=assets),
        partial_correlations=pd.DataFrame(0.0, index=assets, columns=assets),
        cluster_labels=pd.Series([0, 0, 1], index=assets, name="cluster"),
        embedding=pd.DataFrame(
            {"x": [0.0, 1.0, 0.5], "y": [0.0, 0.0, 1.0]},
            index=assets,
        ),
        edges=pd.DataFrame(
            {
                "source": ["A", "A"],
                "target": ["B", "C"],
                "partial_correlation": [0.4, -0.2],
                "absolute_strength": [0.4, 0.2],
            }
        ),
        n_observations=60,
        edge_threshold=0.05,
    )


def test_dependency_network_figure_uses_quantseras_tokens_and_filters() -> None:
    figure = dependency_network_figure(_result(), clusters=[0])

    assert figure.layout.paper_bgcolor == "#121212"
    assert len(figure.data) == 2
    assert list(figure.data[-1].text) == ["A", "B"]


def test_dependency_network_figure_rejects_empty_filters() -> None:
    with pytest.raises(ValueError, match="no assets"):
        dependency_network_figure(_result(), clusters=[99])
