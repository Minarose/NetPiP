"""When has sampling run long enough? RMS change of the PiP surface between checkpoints."""
from __future__ import annotations

import numpy as np

from .surface import retained_steps, step_weights

__all__ = ["rms_convergence", "first_stable_checkpoint"]


def rms_convergence(snapshots, final=None, *, rule: str = "longest_block", basis: str = "retained",
                    kind: str = "linear"):
    """RMS change in the weighted PiP surface between successive checkpoints.

    Follows the paper: the retained steps and step weights of the final
    (largest) checkpoint are held fixed, and only steps observed at every
    checkpoint (the set S) are compared:

        RMS_t = sqrt( sum_{k in S} sum_i (W_ki(t) - W_ki(t-1))^2 / (N |S|) )

    Parameters
    ----------
    snapshots : {n_attacks: (counts, hits)} (e.g. PiPResult.snapshots).
    final : (counts, hits) defining the retained steps and weights; default = largest snapshot.

    Returns a list of dicts with attacks, previous_attacks, rms and compared_rows.
    """
    items = sorted((int(a), (np.asarray(c, float), np.asarray(h, float))) for a, (c, h) in snapshots.items())
    if len(items) < 2:
        raise ValueError("need at least two checkpoints")
    fc, _ = (np.asarray(final[0], float), None) if final is not None else items[-1][1]
    n = len(fc)
    block = retained_steps(fc, rule)
    weights = step_weights(block, n, kind, basis)
    rows = block - 1
    common = np.ones(len(block), dtype=bool)
    for _, (c, _h) in items:
        common &= c[rows] > 0
    surfaces = [(a, h[rows[common]] / c[rows[common], None] * weights[common, None]) for a, (c, h) in items]
    out = []
    for (a0, w0), (a1, w1) in zip(surfaces, surfaces[1:]):
        out.append(dict(attacks=a1, previous_attacks=a0,
                        rms=float(np.sqrt(np.mean((w1 - w0) ** 2))), compared_rows=int(common.sum())))
    return out


def first_stable_checkpoint(rms_rows, threshold: float = 0.005, start: int = 0):
    """First checkpoint (attacks >= start) whose RMS change is at or below `threshold`, or None."""
    for row in rms_rows:
        if row["attacks"] >= start and row["rms"] <= threshold:
            return row["attacks"]
    return None
