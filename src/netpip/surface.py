"""From sampled counts to PiP surfaces and node scores.

`counts[k-1]` is the number of sampled attacks whose percolation event occurred
at removal step k, and `hits[k-1, i]` the number of those attacks in which node
i had been removed by step k. The PiP surface is P[k, i] = hits / counts.
"""
from __future__ import annotations

import numpy as np

__all__ = ["observed_steps", "retained_steps", "step_weights", "pip_probability",
           "weighted_surface", "pip_scores"]


def observed_steps(counts) -> np.ndarray:
    """1-based removal steps at which at least one sampled attack percolated."""
    return np.flatnonzero(np.asarray(counts)) + 1


def retained_steps(counts, rule: str = "longest_block") -> np.ndarray:
    """1-based steps kept for analysis.

    "longest_block" (paper default): the longest run of consecutive observed steps.
    "all_observed": every observed step (used for the hospital network, where
    step 2 was never observed).
    """
    steps = observed_steps(counts)
    if steps.size == 0:
        raise ValueError("no attacks recorded")
    if rule == "all_observed":
        return steps
    if rule != "longest_block":
        raise ValueError("rule must be 'longest_block' or 'all_observed'")
    runs = np.split(steps, np.flatnonzero(np.diff(steps) > 1) + 1)
    return max(runs, key=len)


def step_weights(steps, n_nodes: int | None = None, kind: str = "linear",
                 basis: str = "retained") -> np.ndarray:
    """Decreasing step weights that emphasise early fragmentation.

    basis="retained" (paper default): j = 1..S indexes the S retained steps;
        linear (S - j + 1) / S, exponential exp(-j / max(2, S/6)).
    basis="absolute": the actual removal step k and N replace j and S
        (paper's hospital network): linear (N - k + 1) / N, exponential exp(-k / max(2, N/6)).
    kind=None returns ones (unweighted).
    """
    steps = np.asarray(steps)
    if kind is None:
        return np.ones(len(steps))
    if basis == "retained":
        j, S = np.arange(1, len(steps) + 1), len(steps)
    elif basis == "absolute":
        if n_nodes is None:
            raise ValueError("basis='absolute' needs n_nodes")
        j, S = steps.astype(float), float(n_nodes)
    else:
        raise ValueError("basis must be 'retained' or 'absolute'")
    if kind == "linear":
        return (S - j + 1) / S
    if kind == "exponential":
        return np.exp(-j / max(2.0, S / 6.0))
    raise ValueError("kind must be 'linear', 'exponential' or None")


def pip_probability(counts, hits, steps=None) -> np.ndarray:
    """P[k, i] for the given 1-based steps (rows) and all nodes (columns)."""
    counts = np.asarray(counts, dtype=float)
    hits = np.asarray(hits, dtype=float)
    if steps is None:
        steps = observed_steps(counts)
    rows = np.asarray(steps) - 1
    if (counts[rows] == 0).any():
        raise ValueError("some requested steps were never observed")
    return hits[rows] / counts[rows, None]


def weighted_surface(counts, hits, kind: str | None = "linear", rule: str = "longest_block",
                     basis: str = "retained"):
    """Return (steps, W) with W[j, i] = w_j * P[k_j, i] over the retained steps."""
    steps = retained_steps(counts, rule)
    P = pip_probability(counts, hits, steps)
    w = step_weights(steps, len(counts), kind, basis)
    return steps, P * w[:, None]


def pip_scores(counts, hits, kind: str = "linear", rule: str = "longest_block",
               basis: str = "retained") -> np.ndarray:
    """Node PiP score: mean of the weighted profile across retained steps (higher = more critical)."""
    _, W = weighted_surface(counts, hits, kind, rule, basis)
    return W.mean(axis=0)
