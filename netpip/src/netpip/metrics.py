"""Classical hub metrics (Degree / Betweenness / PageRank) for PiP benchmarking.

NetworkX is an optional dependency. If it is not installed, importing the
metric helpers raises a clear ``RuntimeError`` directing the user to
``pip install netpip[networkx]``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

try:  # optional dependency
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - import-time check
    nx = None  # type: ignore


def _require_networkx() -> None:
    if nx is None:  # pragma: no cover
        raise RuntimeError(
            "networkx is required for classical metric benchmarking. "
            "Install with `pip install 'netpip[networkx]'`."
        )


def _break_ties_randomly_desc(
    values: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sort descending; within ties, randomize. Returns 0-based attack order."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    order: list[int] = []
    for val in np.unique(v)[::-1]:
        idx = np.flatnonzero(v == val)
        if idx.size > 1:
            rng.shuffle(idx)
        order.extend(idx.tolist())
    if len(order) != v.size:
        raise RuntimeError("Tie-break ordering failed.")
    return np.asarray(order, dtype=np.int64)


def degree_attack_order(
    adj: np.ndarray, *, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Node attack order ranked by undirected node degree (ties broken randomly)."""
    _require_networkx()
    rng = rng if rng is not None else np.random.default_rng()
    G = nx.from_numpy_array(np.asarray(adj))
    deg = np.array([d for _, d in G.degree(weight=None)], dtype=np.float64)
    return _break_ties_randomly_desc(deg, rng)


def betweenness_attack_order(
    adj: np.ndarray,
    *,
    normalized: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Node attack order ranked by betweenness centrality (ties broken randomly)."""
    _require_networkx()
    rng = rng if rng is not None else np.random.default_rng()
    G = nx.from_numpy_array(np.asarray(adj))
    bc = np.array(
        list(nx.betweenness_centrality(G, normalized=normalized).values()),
        dtype=np.float64,
    )
    return _break_ties_randomly_desc(bc, rng)


def pagerank_attack_order(
    adj: np.ndarray,
    *,
    alpha: float = 0.85,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Node attack order ranked by PageRank centrality (ties broken randomly)."""
    _require_networkx()
    rng = rng if rng is not None else np.random.default_rng()
    G = nx.from_numpy_array(np.asarray(adj))
    pr = np.array(list(nx.pagerank(G, alpha=alpha).values()), dtype=np.float64)
    return _break_ties_randomly_desc(pr, rng)


def metric_top_n_at_percolation_point(
    adj: np.ndarray,
    metric_order: np.ndarray,
    *,
    n_iter: int = 500,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, int, float]:
    """Top-n hub set for a metric, where n equals the metric's percolation point.

    The metric's percolation point is computed as the *mean* over ``n_iter``
    random tie-break resamples of the same value vector (the protocol used in
    the manuscript MATLAB / Python benchmarking scripts). For a single fixed
    ``metric_order``, the percolation point itself is deterministic; ``n_iter``
    only matters when callers want the averaged value across resamples.

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Binary, undirected adjacency matrix.
    metric_order : np.ndarray of int, shape (N,)
        A single 0-based attack order produced by e.g. :func:`degree_attack_order`.
    n_iter : int, default 500
        Unused here (kept for API parity with the MATLAB scripts).
    seed : int, optional
        Unused here; included for API symmetry.

    Returns
    -------
    top_nodes : np.ndarray of int, shape (perc_point,)
        The first ``perc_point`` nodes of ``metric_order``.
    perc_point : int
        The 1-based percolation point under ``metric_order``.
    perc_point_float : float
        Same as ``perc_point`` (returned as float for symmetry with averaged
        per-subject MATLAB outputs).
    """
    from netpip.ranking import percolation_point

    pp = percolation_point(adj, metric_order)
    top = np.asarray(metric_order, dtype=np.int64)[:pp]
    return top, int(pp), float(pp)


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    """Jaccard similarity ``|A ∩ B| / |A ∪ B|`` for two node-index sets."""
    A = set(int(x) for x in a)
    B = set(int(x) for x in b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)
