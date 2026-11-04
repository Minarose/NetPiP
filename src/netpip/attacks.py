"""Targeted attacks: remove nodes in a fixed order and measure how the network breaks.

Used to compare PiP with classical centralities (degree, betweenness, PageRank),
as in Fig. 4 and Supplementary Fig. 3 / Table 1 of the paper.
"""
from __future__ import annotations

import numpy as np

from ._kernels import _root, njit
from .io import to_csr, validate_adjacency

__all__ = ["component_trace", "percolation_point", "attack_outcome", "attack_orders",
           "evaluate_attacks", "centrality_scores", "outcome_from_trace"]


@njit(cache=True)
def _trace(order, indptr, indices):
    n = len(order)
    parent = np.arange(n)
    sizes = np.ones(n, dtype=np.int64)
    active = np.zeros(n, dtype=np.bool_)
    frequency = np.zeros(n + 1, dtype=np.int64)
    lcc = np.zeros(n + 1, dtype=np.int64)
    slcc = np.zeros(n + 1, dtype=np.int64)
    largest = 0
    for k in range(n - 1, -1, -1):
        node = order[k]
        active[node] = True
        frequency[1] += 1
        if largest < 1:
            largest = 1
        for j in range(indptr[node], indptr[node + 1]):
            other = indices[j]
            if not active[other]:
                continue
            r = _root(parent, node)
            t = _root(parent, other)
            if r == t:
                continue
            if sizes[r] < sizes[t]:
                r, t = t, r
            frequency[sizes[r]] -= 1
            frequency[sizes[t]] -= 1
            parent[t] = r
            sizes[r] += sizes[t]
            frequency[sizes[r]] += 1
            if sizes[r] > largest:
                largest = sizes[r]
        lcc[k] = largest
        if frequency[largest] >= 2:
            slcc[k] = largest
        else:
            s = largest - 1
            while s > 0 and frequency[s] == 0:
                s -= 1
            slcc[k] = s
    return lcc, slcc


def _as_csr(graph):
    if isinstance(graph, tuple) and len(graph) == 2:
        return graph
    return to_csr(graph)


def component_trace(adjacency, order):
    """LCC and SLCC sizes after k = 0..N removals along `order` (0-based node indices).

    `adjacency` may also be the (indptr, indices) pair from :func:`netpip.io.to_csr`,
    which avoids re-converting a large matrix for every order.
    Removed nodes are absent from the surviving graph. Both arrays have length N + 1.
    """
    indptr, indices = _as_csr(adjacency)
    order = np.asarray(order, dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(indptr) - 1)):
        raise ValueError("order must be a permutation of 0..N-1")
    return _trace(order, indptr, indices)


def percolation_point(lcc, slcc):
    """Removals at the percolation point of a targeted attack, or None if none occurs.

    Paper convention for ordered attacks: the first global maximum of the SLCC over
    k = 0..N-1 removals, including the intact graph (k = 0); after k >= 1 removals a
    lone surviving node counts as an SLCC of size 1. A maximum of 1 means no split.
    """
    second = np.asarray(slcc)[:-1].copy()
    second[1:] = np.maximum(second[1:], 1)
    return int(second.argmax()) if second.max() > 1 else None


def attack_outcome(adjacency, order):
    """Summary of one ordered attack.

    peak_removals      removals at the percolation point (see :func:`percolation_point`)
    lcc_at_peak        LCC size at that point
    slcc_at_peak       SLCC size at that point
    removals_to_10pct  first number of removals leaving an LCC of at most 10% of N
    lcc_area           normalized area under the LCC curve (trapezoid, both axes / N)
    """
    lcc, slcc = component_trace(adjacency, order)
    return outcome_from_trace(lcc, slcc)


def outcome_from_trace(lcc, slcc):
    """Attack summary (see :func:`attack_outcome`) from precomputed LCC/SLCC traces."""
    lcc, slcc = np.asarray(lcc), np.asarray(slcc)
    n = len(lcc) - 1
    k = percolation_point(lcc, slcc)
    return dict(peak_removals=k,
                lcc_at_peak=None if k is None else int(lcc[k]),
                slcc_at_peak=None if k is None else int(slcc[k]),
                removals_to_10pct=int(np.flatnonzero(lcc <= 0.1 * n)[0]),
                lcc_area=float(np.trapezoid(lcc / n, dx=1 / n)) if hasattr(np, "trapezoid")
                else float(np.trapz(lcc / n, dx=1 / n)))


def attack_orders(scores, *, n_tie_orders: int = 500, seed: int = 860024, decimals: int = 12):
    """Removal orders (0-based) by descending score, with random tie-breaking.

    Scores are rounded to `decimals`. If any ties remain, `n_tie_orders` orders
    are returned, each breaking ties at random (paper: 500 orders); otherwise one.
    """
    rounded = np.round(np.asarray(scores, dtype=float), decimals)
    n = len(rounded)
    repeats = n_tie_orders if len(np.unique(rounded)) < n else 1
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(repeats):
        shuffled = rng.permutation(n)
        orders.append(shuffled[np.argsort(-rounded[shuffled], kind="stable")])
    return np.array(orders)


def evaluate_attacks(adjacency, orders):
    """Mean (and s.d. across tie orders) of each attack outcome over a set of orders."""
    csr = _as_csr(adjacency)
    trials = [attack_outcome(csr, o) for o in np.atleast_2d(orders)]
    peaks = np.array([np.nan if t["peak_removals"] is None else t["peak_removals"] for t in trials], float)
    out = dict(n_orders=len(trials),
               peak_removals=float(np.nanmean(peaks)) if np.isfinite(peaks).any() else None,
               peak_removals_sd=float(np.nanstd(peaks)) if np.isfinite(peaks).any() else None,
               lcc_area=float(np.mean([t["lcc_area"] for t in trials])),
               lcc_area_sd=float(np.std([t["lcc_area"] for t in trials])),
               removals_to_10pct=float(np.mean([t["removals_to_10pct"] for t in trials])))
    out["first_order"] = trials[0]
    return out


def centrality_scores(adjacency, *, damping: float = 0.85):
    """Degree, betweenness and PageRank of the intact graph.

    Uses bctpy (the Brain Connectivity Toolbox, as in the paper) when installed:
    degrees_und, betweenness_bin and pagerank_centrality. Otherwise falls back to
    NetworkX, which gives the same rankings up to ties and numerical precision.
    """
    A = validate_adjacency(adjacency).astype(float)
    try:
        import bct

        return {"Degree": np.asarray(bct.degrees_und(A), float),
                "Betweenness": np.asarray(bct.betweenness_bin(A), float),
                "PageRank": np.asarray(bct.pagerank_centrality(A, damping), float)}
    except ImportError:
        import networkx as nx

        G = nx.from_numpy_array(A)
        n = len(A)
        bc = nx.betweenness_centrality(G, normalized=False)
        pr = nx.pagerank(G, alpha=damping)
        return {"Degree": A.sum(1),
                "Betweenness": 2 * np.array([bc[i] for i in range(n)]),
                "PageRank": np.array([pr[i] for i in range(n)])}
