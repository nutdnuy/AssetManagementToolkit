import pandas as pd

from asset_management_toolkit.dashboard import (
    create_graphical_analysis_dashboard,
)
from asset_management_toolkit.graphical_analysis import GraphicalAnalysisResult


def test_dashboard_factory_builds_cluster_filter_and_graph() -> None:
    assets = pd.Index(["A", "B", "C"])
    result = GraphicalAnalysisResult(
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
                "source": ["A"],
                "target": ["B"],
                "partial_correlation": [0.4],
                "absolute_strength": [0.4],
            }
        ),
        n_observations=60,
        edge_threshold=0.05,
    )

    app = create_graphical_analysis_dashboard(result)

    assert app.title == "Asset Dependency Network"
    assert app.layout is not None
    assert "cluster-filter" in str(app.layout)
    assert "dependency-network" in str(app.layout)
