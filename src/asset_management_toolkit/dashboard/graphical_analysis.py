"""Dash application factory for graphical-analysis results."""

from __future__ import annotations

from asset_management_toolkit.graphical_analysis import GraphicalAnalysisResult
from asset_management_toolkit.visualization import dependency_network_figure


def create_graphical_analysis_dashboard(
    result: GraphicalAnalysisResult,
    *,
    title: str = "Asset Dependency Network",
):
    """Create a QuantSeras-styled Dash app with cluster filtering."""
    Dash, Input, Output, dcc, html = _dash_components()
    clusters = sorted(map(int, result.cluster_labels.unique()))
    app = Dash(__name__, title=title)
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.P("GRAPHICAL ANALYSIS", style=_EYEBROW),
                    html.H1(title, style={"margin": "6px 0 8px"}),
                    html.P(
                        "Sparse partial correlations, clusters, and a "
                        "two-dimensional research view.",
                        style=_SUPPORTING_TEXT,
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("Visible clusters", htmlFor="cluster-filter"),
                    dcc.Dropdown(
                        id="cluster-filter",
                        options=[
                            {"label": f"Cluster {cluster}", "value": cluster}
                            for cluster in clusters
                        ],
                        value=clusters,
                        multi=True,
                        clearable=False,
                    ),
                ],
                style=_CONTROL_CARD,
            ),
            dcc.Graph(
                id="dependency-network",
                figure=dependency_network_figure(result, title=title),
                config={"displaylogo": False, "responsive": True},
                style={"minHeight": "620px"},
            ),
            html.P(
                f"{result.n_observations} observations · "
                f"{len(result.cluster_labels)} assets · "
                f"{len(result.edges)} visible edges · "
                f"threshold {result.edge_threshold:.3f}",
                style=_SUPPORTING_TEXT,
            ),
        ],
        style=_PAGE,
    )

    @app.callback(
        Output("dependency-network", "figure"),
        Input("cluster-filter", "value"),
    )
    def update_network(selected_clusters):
        return dependency_network_figure(
            result,
            clusters=selected_clusters or clusters,
            title=title,
        )

    return app


_PAGE = {
    "minHeight": "100vh",
    "background": "#121212",
    "color": "rgba(255,255,255,0.87)",
    "fontFamily": "Roboto, sans-serif",
    "padding": "32px clamp(20px, 5vw, 72px)",
}
_EYEBROW = {
    "color": "#03DAC6",
    "fontSize": "13px",
    "fontWeight": 600,
    "letterSpacing": "0.12em",
    "margin": 0,
}
_SUPPORTING_TEXT = {
    "color": "rgba(255,255,255,0.60)",
    "lineHeight": 1.6,
}
_CONTROL_CARD = {
    "background": "#232323",
    "border": "1px solid rgba(255,255,255,0.12)",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "12px",
}


def _dash_components():
    try:
        from dash import Dash, Input, Output, dcc, html
    except ImportError as error:
        raise ImportError(
            "create_graphical_analysis_dashboard requires Dash and Plotly; "
            "install asset-management-toolkit[visualization]"
        ) from error
    return Dash, Input, Output, dcc, html
