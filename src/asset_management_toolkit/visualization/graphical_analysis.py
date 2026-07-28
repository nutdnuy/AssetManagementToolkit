"""Plotly figures for graphical-analysis results."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np

from asset_management_toolkit.graphical_analysis import GraphicalAnalysisResult


def dependency_network_figure(
    result: GraphicalAnalysisResult,
    *,
    assets: Iterable[Hashable] | None = None,
    clusters: Iterable[int] | None = None,
    title: str = "Asset dependency network",
):
    """Build an interactive Plotly figure from a fitted network result."""
    go = _plotly_graph_objects()
    selected = _selected_assets(result, assets, clusters)
    positions = result.embedding.loc[selected]
    cluster_labels = result.cluster_labels.loc[selected]
    edges = result.edges[
        result.edges["source"].isin(selected) & result.edges["target"].isin(selected)
    ]

    traces = []
    for _, edge in edges.sort_values("absolute_strength").iterrows():
        source = positions.loc[edge["source"]]
        target = positions.loc[edge["target"]]
        positive = float(edge["partial_correlation"]) >= 0.0
        traces.append(
            go.Scatter(
                x=[source["x"], target["x"]],
                y=[source["y"], target["y"]],
                mode="lines",
                hoverinfo="text",
                text=[
                    f"{edge['source']} ↔ {edge['target']}: "
                    f"{edge['partial_correlation']:.3f}"
                ]
                * 2,
                line={
                    "color": "#03DAC6" if positive else "#CF6679",
                    "width": 0.8 + 5.0 * float(edge["absolute_strength"]),
                },
                showlegend=False,
            )
        )

    degree = {
        asset: int((edges["source"].eq(asset) | edges["target"].eq(asset)).sum())
        for asset in selected
    }
    traces.append(
        go.Scatter(
            x=positions["x"],
            y=positions["y"],
            mode="markers+text",
            text=list(map(str, positions.index)),
            textposition="top center",
            customdata=np.column_stack(
                [
                    cluster_labels.to_numpy(),
                    [degree[asset] for asset in positions.index],
                ]
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Cluster %{customdata[0]}"
                "<br>Visible links %{customdata[1]}<extra></extra>"
            ),
            marker={
                "size": [15 + 2 * degree[asset] for asset in positions.index],
                "color": cluster_labels,
                "colorscale": [
                    [0.0, "#BB86FC"],
                    [0.5, "#03DAC6"],
                    [1.0, "#FFD166"],
                ],
                "line": {"color": "#121212", "width": 2},
                "showscale": False,
            },
            showlegend=False,
        )
    )
    figure = go.Figure(traces)
    figure.update_layout(
        title={"text": title, "x": 0.02},
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font={"family": "Roboto, sans-serif", "color": "rgba(255,255,255,0.87)"},
        hovermode="closest",
        margin={"l": 24, "r": 24, "t": 64, "b": 24},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "Teal: positive · Rose: negative partial correlation",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.04,
                "showarrow": False,
                "font": {"color": "rgba(255,255,255,0.60)", "size": 12},
            }
        ],
    )
    return figure


def _selected_assets(
    result: GraphicalAnalysisResult,
    assets: Iterable[str] | None,
    clusters: Iterable[int] | None,
) -> list[Hashable]:
    selected = list(result.embedding.index)
    if assets is not None:
        requested = list(assets)
        unknown = [asset for asset in requested if asset not in selected]
        if unknown:
            raise ValueError(f"unknown assets: {', '.join(map(str, unknown))}")
        selected = [asset for asset in selected if asset in requested]
    if clusters is not None:
        requested_clusters = {int(cluster) for cluster in clusters}
        selected = [
            asset
            for asset in selected
            if int(result.cluster_labels.loc[asset]) in requested_clusters
        ]
    if not selected:
        raise ValueError("the selected filters contain no assets")
    return selected


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "dependency_network_figure requires Plotly; "
            "install asset-management-toolkit[visualization]"
        ) from error
    return go
