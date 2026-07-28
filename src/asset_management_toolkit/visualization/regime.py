"""Plotly visualization for observed market regimes."""

from __future__ import annotations

import pandas as pd

from asset_management_toolkit.analytics._validation import ReturnInput, coerce_returns
from asset_management_toolkit.market_regime_classification import regime_episodes


def regime_overlay_figure(
    returns: ReturnInput,
    regimes: pd.Series,
    *,
    cumulative: bool = True,
    title: str = "Returns by observed market regime",
):
    """Overlay contiguous observed regimes on an interactive return chart."""
    go = _plotly_graph_objects()
    frame, _ = coerce_returns(returns)
    if not frame.index.is_unique:
        raise ValueError("returns index must be unique")
    episodes = regime_episodes(regimes)
    overlap = frame.index.intersection(regimes.index, sort=False)
    if overlap.empty:
        raise ValueError("returns and regimes have no overlapping index")
    frame = frame.reindex(overlap)
    labels = regimes.reindex(overlap)
    if frame.isna().any().any():
        raise ValueError("visualization returns must not contain missing values")
    plotted = (1.0 + frame).cumprod() if cumulative else frame

    figure = go.Figure()
    for asset in plotted:
        figure.add_trace(
            go.Scatter(
                x=plotted.index,
                y=plotted[asset],
                mode="lines",
                name=str(asset),
                line={"width": 2},
            )
        )

    regime_order = list(pd.unique(labels))
    palette = ["#BB86FC", "#03DAC6", "#FFD166", "#CF6679", "#82B1FF"]
    colors = {
        regime: palette[position % len(palette)]
        for position, regime in enumerate(regime_order)
    }
    visible_start = overlap[0]
    visible_end = overlap[-1]
    for _, episode in episodes.iterrows():
        start = max(episode["start"], visible_start)
        end = min(episode["end"], visible_end)
        if start > end:
            continue
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=colors.get(episode["regime"], "#BB86FC"),
            opacity=0.12,
            line_width=0,
            layer="below",
        )
    for regime in regime_order:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": colors[regime], "symbol": "square"},
                name=f"Regime: {regime}",
                legendgroup=f"regime-{regime}",
            )
        )

    figure.update_layout(
        title={"text": title, "x": 0.02},
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font={"family": "Roboto, sans-serif", "color": "rgba(255,255,255,0.87)"},
        hovermode="x unified",
        margin={"l": 56, "r": 24, "t": 72, "b": 48},
        legend={"orientation": "h", "y": -0.16},
        xaxis={"gridcolor": "rgba(255,255,255,0.08)"},
        yaxis={
            "gridcolor": "rgba(255,255,255,0.08)",
            "title": "Growth of 1.00" if cumulative else "Simple return",
        },
    )
    return figure


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "regime_overlay_figure requires Plotly; "
            "install asset-management-toolkit[visualization]"
        ) from error
    return go
