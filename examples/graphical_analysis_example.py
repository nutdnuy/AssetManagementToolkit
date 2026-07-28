"""Fit and visualize a synthetic sparse asset-dependency network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.dashboard import (
    create_graphical_analysis_dashboard,
)
from asset_management_toolkit.graphical_analysis import graphical_analysis
from asset_management_toolkit.visualization import dependency_network_figure


def main() -> None:
    rng = np.random.default_rng(42)
    market = rng.normal(0.0003, 0.009, size=400)
    growth = rng.normal(0.0001, 0.006, size=400)
    defensive = rng.normal(0.0001, 0.005, size=400)
    returns = pd.DataFrame(
        {
            "Growth A": market + growth + rng.normal(0, 0.004, 400),
            "Growth B": market + growth + rng.normal(0, 0.004, 400),
            "Growth C": market + 0.8 * growth + rng.normal(0, 0.005, 400),
            "Defensive A": 0.5 * market + defensive + rng.normal(0, 0.004, 400),
            "Defensive B": 0.5 * market + defensive + rng.normal(0, 0.004, 400),
            "Diversifier": -0.1 * market + rng.normal(0, 0.008, 400),
        }
    )

    result = graphical_analysis(returns, edge_threshold=0.05)
    figure = dependency_network_figure(result)
    app = create_graphical_analysis_dashboard(result)

    print(result.edges.head())
    print(f"Plotly traces: {len(figure.data)}")
    print(f"Dash title: {app.title}")
    # For local exploration, call app.run(debug=True) explicitly.


if __name__ == "__main__":
    main()
