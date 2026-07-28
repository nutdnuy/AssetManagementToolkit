"""Dependency-network, clustering, and embedding calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asset_management_toolkit.analytics._validation import coerce_returns
from asset_management_toolkit.graphical_analysis.result import (
    GraphicalAnalysisResult,
)


def graphical_analysis(
    returns: pd.DataFrame,
    *,
    edge_threshold: float = 0.02,
    cv: int = 5,
    max_iter: int = 1_000,
    random_state: int = 0,
) -> GraphicalAnalysisResult:
    """Fit a sparse partial-correlation network to labelled asset returns.

    Returns are standardized before fitting ``GraphicalLassoCV``. Affinity
    propagation clusters the fitted covariance matrix, while metric
    multidimensional scaling creates deterministic two-dimensional node
    coordinates for visualization.
    """
    GraphicalLassoCV, AffinityPropagation, MDS = _sklearn_estimators()
    frame = _validate_graphical_returns(returns, cv)
    threshold = _validate_threshold(edge_threshold)
    iterations = _positive_integer(max_iter, "max_iter")
    seed = _non_negative_integer(random_state, "random_state")

    standardized = (frame - frame.mean()) / frame.std(ddof=1)
    model = GraphicalLassoCV(cv=cv, max_iter=iterations)
    model.fit(standardized.to_numpy())

    labels = frame.columns.copy()
    covariance = pd.DataFrame(model.covariance_, index=labels, columns=labels)
    precision = pd.DataFrame(model.precision_, index=labels, columns=labels)
    partial = _partial_correlations(precision)

    cluster_model = AffinityPropagation(
        affinity="precomputed",
        random_state=seed,
    )
    clusters = pd.Series(
        cluster_model.fit_predict(covariance.to_numpy()),
        index=labels,
        name="cluster",
        dtype=int,
    )

    embedding_model = MDS(
        n_components=2,
        random_state=seed,
        n_init=4,
        dissimilarity="euclidean",
    )
    coordinates = embedding_model.fit_transform(standardized.to_numpy().T)
    embedding = pd.DataFrame(
        coordinates,
        index=labels,
        columns=["x", "y"],
        dtype=float,
    )

    return GraphicalAnalysisResult(
        covariance=covariance,
        precision=precision,
        partial_correlations=partial,
        cluster_labels=clusters,
        embedding=embedding,
        edges=_edge_table(partial, threshold),
        n_observations=len(frame),
        edge_threshold=threshold,
    )


def _validate_graphical_returns(
    returns: pd.DataFrame,
    cv: int,
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    frame, _ = coerce_returns(returns)
    if frame.shape[1] < 3:
        raise ValueError("returns must contain at least three assets")
    if frame.isna().any().any():
        raise ValueError("returns must not contain missing values")
    if not frame.index.is_unique:
        raise ValueError("returns index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order")
    folds = _positive_integer(cv, "cv")
    if folds < 2:
        raise ValueError("cv must be at least two")
    if len(frame) < folds:
        raise ValueError("returns must contain at least cv observations")
    volatility = frame.std(ddof=1).to_numpy()
    zero_volatility = frame.columns[np.isclose(volatility, 0.0, atol=1e-15)]
    if len(zero_volatility):
        names = ", ".join(map(str, zero_volatility))
        raise ValueError(f"returns contain zero-volatility assets: {names}")
    return frame


def _partial_correlations(precision: pd.DataFrame) -> pd.DataFrame:
    diagonal = np.diag(precision.to_numpy())
    scale = np.sqrt(np.outer(diagonal, diagonal))
    values = -precision.to_numpy() / scale
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=precision.index, columns=precision.columns)


def _edge_table(
    partial_correlations: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    values = partial_correlations.to_numpy()
    rows = []
    for start, end in zip(*np.triu_indices_from(values, k=1)):
        strength = float(values[start, end])
        if abs(strength) > threshold:
            rows.append(
                {
                    "source": partial_correlations.index[start],
                    "target": partial_correlations.index[end],
                    "partial_correlation": strength,
                    "absolute_strength": abs(strength),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target",
            "partial_correlation",
            "absolute_strength",
        ],
    ).sort_values(
        "absolute_strength",
        ascending=False,
        ignore_index=True,
    )


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError("edge_threshold must be a real number")
    if not np.isfinite(value) or not 0.0 <= float(value) < 1.0:
        raise ValueError("edge_threshold must be finite and between zero and one")
    return float(value)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return int(value)


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def _sklearn_estimators():
    try:
        from sklearn.cluster import AffinityPropagation
        from sklearn.covariance import GraphicalLassoCV
        from sklearn.manifold import MDS
    except ImportError as error:
        raise ImportError(
            "graphical_analysis requires the optional graphical dependencies; "
            "install asset-management-toolkit[graphical]"
        ) from error
    return GraphicalLassoCV, AffinityPropagation, MDS
