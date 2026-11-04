"""Group nodes by the shape of their PiP profiles to find a critical set."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

__all__ = ["silhouette_score", "critical_cluster"]


def silhouette_score(X, labels) -> float:
    """Mean silhouette (Euclidean); singletons score 0, as in scikit-learn."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    uniq = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def critical_cluster(profiles, k_range=range(2, 11)):
    """Ward clustering of node profiles; returns the cluster with the highest participation.

    profiles : (nodes x steps) weighted PiP profiles (paper: linear weights).
    The number of clusters k maximises the mean silhouette over `k_range`; the
    critical cluster is the one whose nodes have the largest mean summed profile.

    Returns dict(k, silhouette, silhouette_by_k, labels, critical_label, critical_nodes)
    with 0-based node indices.
    """
    X = np.asarray(profiles, float)
    Z = linkage(X, method="ward")
    scores = {k: silhouette_score(X, fcluster(Z, k, "maxclust")) for k in k_range if k < len(X)}
    k = max(scores, key=scores.get)
    labels = fcluster(Z, k, "maxclust")
    summed = X.sum(axis=1)
    critical = max(np.unique(labels), key=lambda c: summed[labels == c].mean())
    return dict(k=int(k), silhouette=scores[k], silhouette_by_k=scores, labels=labels,
                critical_label=int(critical), critical_nodes=np.flatnonzero(labels == critical))
