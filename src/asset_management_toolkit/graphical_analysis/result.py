"""Structured results for dependency-network analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class GraphicalAnalysisResult:
    """Labelled outputs from a fitted asset dependency network."""

    covariance: pd.DataFrame
    precision: pd.DataFrame
    partial_correlations: pd.DataFrame
    cluster_labels: pd.Series
    embedding: pd.DataFrame
    edges: pd.DataFrame
    n_observations: int
    edge_threshold: float
