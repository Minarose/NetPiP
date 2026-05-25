"""Tilted-peak ranking and percolation-point helpers.

Ported from ``scripts/plot_pip_surfaces.py::get_tilt_peak_order_amplitude`` and
the percolation-point logic in ``scripts/avg_percolation_metrics_brain_jaccard.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _crop_longest_non_nan_block(P: np.ndarray) -> np.ndarray:
    """Keep the longest contiguous block of rows containing any non-NaN value."""
    if P.size == 0:
        return P
    has_data = ~np.all(np.isnan(P), axis=1)
    if not np.any(has_data):
        return P[:0, :]
    idx = np.flatnonzero(has_data)
    starts = [int(idx[0])]
    ends: list[int] = []
    for prev, curr in zip(idx[:-1], idx[1:]):
        if curr != prev + 1:
            ends.append(int(prev))
            starts.append(int(curr))
    ends.append(int(idx[-1]))
    lens = [e - s + 1 for s, e in zip(starts, ends)]
    best = int(np.argmax(lens))
    return P[starts[best] : ends[best] + 1, :]


def tilted_peak_rank(
    node_P: np.ndarray,
    *,
    tau_factor: float = 1.0 / 6.0,
    clip_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank nodes by the peak of their time-tilted PiP trajectory.

    The tilt re-weights row ``s`` of ``node_P`` by ``exp(-s / tau)``, where
    ``tau = max(2, S * tau_factor)``. This emphasises **early** participation
    in fragmentation (the canonical PiP convention for hub identification).

    Parameters
    ----------
    node_P : np.ndarray, shape (S, N)
        Converged PiP matrix returned by :func:`netpip.run_pip`.
    tau_factor : float, default 1/6
        Multiplier in ``tau = max(2, S * tau_factor)``. The default matches the
        manuscript figures (``τ = S/6``).
    clip_negative : bool, default True
        If True, negative entries are clipped to zero before ranking (matches
        the manuscript default).

    Returns
    -------
    order : np.ndarray of int, shape (N,)
        Node indices (0-based) in descending order of tilted peak amplitude.
        ``order[0]`` is the node ranked first (highest peak).
    peak_step : np.ndarray of int, shape (N,)
        For each node, the step at which its tilted trajectory peaks (0-based).
    peak_amp : np.ndarray of float, shape (N,)
        For each node, the value of the tilted trajectory at its peak step.
    """
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

    peak_amp = Pt.max(axis=0)
    peak_step = Pt.argmax(axis=0)
    # Sort: primary key -peak_amp (desc), secondary peak_step (asc); stable
    order = np.lexsort((peak_step, -peak_amp))
    return order.astype(np.int64), peak_step.astype(np.int64), peak_amp.astype(np.float64)


def percolation_point(adj: np.ndarray, attack_order: np.ndarray) -> int:
    """1-based removal step at which the 2nd-largest component first peaks.

    Mirrors ``run_attack_once`` in
    ``scripts/compare_pip_bct_percolation_avg_single.m`` and the engine in
    :mod:`netpip.core`: if no removal ever produces a second component larger
    than one node, returns ``N``.
    """
    A = np.asarray(adj, dtype=np.float64).copy()
    order = np.asarray(attack_order, dtype=np.int64).reshape(-1)
    n = A.shape[0]
    if order.size != n:
        raise ValueError(f"attack_order length {order.size} != n {n}")

    second = np.zeros(n, dtype=np.int64)
    for i in range(n):
        _, labels = connected_components(csr_matrix(A), directed=False, return_labels=True)
        if labels.size:
            sizes = np.bincount(labels)
            if sizes.size >= 2:
                sizes.sort()
                second[i] = int(sizes[-2])
        node = int(order[i])
        A[node, :] = 0.0
        A[:, node] = 0.0

    mx = int(second.max())
    if mx > 1:
        return int(np.argmax(second) + 1)
    return int(n)


def pip_top_n_at_percolation_point(
    adj: np.ndarray,
    node_P: np.ndarray,
    *,
    tau_factor: float = 1.0 / 6.0,
    clip_negative: bool = True,
) -> np.ndarray:
    """Convenience: PiP top-n hub set where n equals the PiP percolation point.

    Returns a 0-based array of node indices of length ``perc_point`` (sorted
    by tilted peak amplitude, highest first).
    """
    order, _, _ = tilted_peak_rank(
        node_P, tau_factor=tau_factor, clip_negative=clip_negative
    )
    pp = percolation_point(adj, order)
    return order[:pp]
