"""Ward + silhouette hub-cluster identification on the weighted PiP matrix.

Ported from ``scripts/4_consensus_cluster_per_subject.py`` (functions
``select_optimal_k``, ``silhouette_mean``, ``critical_nodes_from_matrix``) so
the toolbox has zero dependency on scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class HubClusterResult:
    """Output of :func:`pip_hub_cluster`."""

    labels: np.ndarray  # shape (N,), 1-based cluster labels (scipy convention)
    k: int
    silhouette: float
    hub_cluster: int  # 1-based label of the "top" cluster
    hub_nodes: np.ndarray  # 0-based node indices in the top cluster


def _silhouette_mean(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient (Euclidean) without sklearn."""
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = X.shape[0]
    if n <= 1 or unique.size <= 1:
        return 0.0
    dist = squareform(pdist(X, metric="euclidean"))
    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = float(np.mean(dist[i, same])) if np.any(same) else 0.0
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b = min(b, float(np.mean(dist[i, other])))
        if not np.isfinite(b) or max(a, b) == 0:
            s[i] = 0.0
        else:
            s[i] = (b - a) / max(a, b)
    return float(np.mean(s))


def ward_silhouette_cluster(
    X: np.ndarray, *, k_min: int = 2, k_max: int = 10
) -> Tuple[int, np.ndarray, float]:
    """Ward hierarchical clustering with silhouette-based ``k`` selection.

    Returns ``(best_k, labels, best_silhouette)``. ``labels`` are 1-based,
    matching ``scipy.cluster.hierarchy.fcluster``.
    """
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return 1, np.ones(n, dtype=np.int64), 0.0
    Z = linkage(X, method="ward")
    best_k = k_min
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        score = _silhouette_mean(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_labels is None:
        return 1, np.ones(n, dtype=np.int64), 0.0
    return int(best_k), np.asarray(best_labels, dtype=np.int64), float(best_score)


def pip_hub_cluster(
    node_P: np.ndarray,
    *,
    tau_factor: float = 1.0 / 6.0,
    clip_negative: bool = True,
    k_min: int = 2,
    k_max: int = 10,
) -> HubClusterResult:
    """Identify the PiP hub cluster on the time-tilted PiP matrix.

    The procedure (matching the manuscript):

    1. Crop ``node_P`` to its longest contiguous non-NaN row block.
    2. Apply the canonical tilt (``tau = max(2, S * tau_factor)`` with
       ``tau_factor = 1/6``) plus negative clipping. Each node is then
       represented by its tilted PiP row.
    3. Cluster the node-by-step matrix with Ward hierarchical clustering and
       choose ``k`` by maximizing the mean silhouette coefficient over
       ``k_min..k_max`` (default 2..10).
    4. The **hub cluster** is the cluster with the largest mean summed
       tilted-PiP weight across its member nodes.

    Parameters
    ----------
    node_P : np.ndarray, shape (S, N)
        Converged PiP matrix (rows = attack steps, columns = nodes).
    tau_factor : float, default 1/6
        Tilt factor; ``τ = max(2, S * tau_factor)``.
    clip_negative : bool, default True
        Clip negative tilted entries to zero before clustering.
    k_min, k_max : int
        Search range for the silhouette-optimal number of clusters.

    Returns
    -------
    HubClusterResult
    """
    from netpip.ranking import _crop_longest_non_nan_block

    P = np.asarray(node_P, dtype=np.float64)
    P = _crop_longest_non_nan_block(P)
    if P.size == 0:
        raise ValueError("node_P is empty after cropping non-NaN rows.")
    P = np.where(np.isfinite(P), P, 0.0)
    if clip_negative:
        P = np.where(P < 0, 0.0, P)

    S = P.shape[0]
    tau = max(2.0, S * float(tau_factor))
    s = np.arange(1, S + 1, dtype=np.float64)
    w = np.exp(-s / tau).reshape(-1, 1)
    Pt = P * w

    # Each node is one row -> transpose to (N, S)
    X = Pt.T
    k, labels, sil = ward_silhouette_cluster(X, k_min=k_min, k_max=k_max)

    node_sums = X.sum(axis=1)
    unique_labels = np.unique(labels)
    cluster_means = np.array(
        [float(np.mean(node_sums[labels == c])) for c in unique_labels]
    )
    hub_label = int(unique_labels[int(np.argmax(cluster_means))])
    hub_nodes = np.flatnonzero(labels == hub_label).astype(np.int64)

    return HubClusterResult(
        labels=labels,
        k=k,
        silhouette=sil,
        hub_cluster=hub_label,
        hub_nodes=hub_nodes,
    )
