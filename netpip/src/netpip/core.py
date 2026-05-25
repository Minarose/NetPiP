"""Monte Carlo Participation-in-Percolation (PiP) engine.

This is a faithful Python port of the MATLAB convergence engine
``scripts/pip_converge_posthw5_thresh.m`` used in the accompanying manuscript.

Algorithm (per attack):

1. Draw a uniformly random permutation of the N nodes (the *attack order*).
2. Remove nodes one at a time in that order. After each removal, compute the
   sizes of all connected components in the surviving graph and record the
   size of the second-largest component.
3. The **percolation step** for this attack is the first removal index whose
   second-largest-component size equals the per-attack maximum (or ``N`` if
   no removal ever produced a second component larger than 1).
4. Mark every node that was removed by (and including) the percolation step
   as a *participant* for this attack.

After ``n_attacks`` attacks the engine returns ``node_P`` of shape
``(N, N)``, where::

    node_P[p, i] = P(node i was removed by step p  |  percolation step == p)

Convergence (optional) is monitored via the Wilson 95% confidence-interval
half-width on the binomial proportions ``part_counts / counts_per_step``;
sampling stops early when that statistic plateaus (see :mod:`netpip.convergence`).

This module deliberately does **not** call into the validation module; users
are expected to call :func:`netpip.validate_adjacency` once before
``run_pip`` so the validator is not re-run on every attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from netpip.convergence import plateau_reached, wilson_half_width


@dataclass
class PiPResult:
    """Container for the output of :func:`run_pip`.

    Attributes
    ----------
    node_P : np.ndarray, shape (N, N)
        ``node_P[p, i]`` is the conditional probability that node ``i`` was
        removed by step ``p``, given that the attack's percolation step was
        ``p``. Rows with no attacks (``counts_per_step[p] == 0``) are NaN.
    counts_per_step : np.ndarray of uint64, shape (N,)
        ``counts_per_step[p]`` is the number of attacks whose percolation step
        was ``p`` (1-indexed: index 0 corresponds to MATLAB step 1).
    part_counts : np.ndarray of uint64, shape (N, N)
        ``part_counts[p, i]`` is the raw count of attacks where the percolation
        step was ``p`` *and* node ``i`` was among the first ``p`` removed.
    n_attacks : int
        Total number of Monte Carlo attacks performed.
    attacks_hist : np.ndarray, shape (n_chunks,)
        Cumulative number of attacks after each chunk.
    hw95_hist : np.ndarray, shape (n_chunks,)
        95th-percentile Wilson-95 half-width across the ``node_P`` entries
        after each chunk; used for plateau-based early stopping.
    converged : bool
        True if early stopping triggered before ``max_attacks`` was reached.
    elapsed_seconds : float
        Wall-clock runtime of ``run_pip``.
    """

    node_P: np.ndarray
    counts_per_step: np.ndarray
    part_counts: np.ndarray
    n_attacks: int
    attacks_hist: np.ndarray
    hw95_hist: np.ndarray
    converged: bool
    elapsed_seconds: float
    meta: dict = field(default_factory=dict)


def _second_largest_component_after_each_removal(
    adj: np.ndarray, attack: np.ndarray
) -> np.ndarray:
    """For attack order, return uint32 vector of length N-1 of 2nd-largest sizes."""
    n = adj.shape[0]
    mask = np.ones(n, dtype=bool)
    out = np.zeros(n - 1, dtype=np.uint32)

    A = adj
    for step in range(n - 1):
        mask[attack[step]] = False
        sub = A[np.ix_(mask, mask)]
        if not sub.any():
            out[step] = 0
            continue
        _, labels = connected_components(csr_matrix(sub), directed=False, return_labels=True)
        sizes = np.bincount(labels)
        if sizes.size < 2:
            out[step] = 0
        else:
            sizes.sort()
            out[step] = sizes[-2]
    return out


def _percolation_step_from_second_comp(second_comp: np.ndarray, n: int) -> int:
    """First 1-based index maximizing second_comp; n if max <= 1 (MATLAB convention)."""
    smax = int(second_comp.max()) if second_comp.size else 0
    if smax > 1:
        return int(np.argmax(second_comp) + 1)
    return int(n)


def run_pip(
    adj: np.ndarray,
    *,
    max_attacks: int = 1_000_000,
    chunk_size: int = 10_000,
    seed: Optional[int] = None,
    enforce_hw95: bool = False,
    hw95_tol: float = 0.05,
    range_tol: float = 0.005,
    slope_tol: float = 1e-7,
    plateau_window: int = 5,
    require_stable: int = 3,
    progress: Optional[Callable[[dict], None]] = None,
) -> PiPResult:
    """Run Monte Carlo PiP convergence on a binary undirected graph.

    Parameters
    ----------
    adj : np.ndarray, shape (N, N)
        Binary, symmetric, zero-diagonal adjacency matrix. **Validate first**
        with :func:`netpip.validate_adjacency` to surface input problems early
        (this function assumes the contract holds and does not re-check it on
        every attack).
    max_attacks : int, default 1_000_000
        Hard cap on the number of Monte Carlo attacks.
    chunk_size : int, default 10_000
        Attacks accumulated between convergence checks. Matches the MATLAB
        ``CHUNK_SIZE`` default.
    seed : int, optional
        Seed for the random permutation generator (NumPy ``default_rng``).
    enforce_hw95 : bool, default False
        If True, plateau detection additionally requires the mean Wilson-95
        half-width over ``plateau_window`` to fall below ``hw95_tol``.
        Matches the MATLAB ``ENFORCE_HW95`` option.
    hw95_tol : float, default 0.05
        Half-width tolerance used only when ``enforce_hw95=True``.
    range_tol : float, default 0.005
        Maximum allowed range of the half-width within the plateau window.
    slope_tol : float, default 1e-7
        Maximum allowed absolute slope (in HW95 units per attack) within the
        plateau window.
    plateau_window : int, default 5
        Number of consecutive chunks used for plateau detection.
    require_stable : int, default 3
        Number of consecutive plateau-positive checks before terminating.
    progress : callable, optional
        Called after each chunk with a dict summarizing chunk statistics
        (``n_attacks``, ``hw95``, ``mean_val``, ``range_val``, ``slope_val``,
        ``is_plateau``, ``stable_count``). Use for logging / progress bars.

    Returns
    -------
    PiPResult
        Container with ``node_P`` and supporting statistics.

    Notes
    -----
    The output ``node_P`` matches the ``node_P`` saved by the MATLAB driver
    (``pip_converge_posthw5_thresh.m``) up to Monte Carlo sampling noise.
    Differences in random-stream implementation between MATLAB's Threefry RNG
    and NumPy's PCG64 mean exact bit-for-bit reproducibility across the two
    engines is not expected; cohort-level results are.
    """
    import time

    A = np.asarray(adj, dtype=np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    counts_per_step = np.zeros(n, dtype=np.uint64)
    part_counts = np.zeros((n, n), dtype=np.uint64)

    attacks_hist: list[int] = []
    hw95_hist: list[float] = []
    stable_count = 0
    n_attacks = 0
    converged = False

    t0 = time.perf_counter()
    while n_attacks < max_attacks:
        this_chunk = min(chunk_size, max_attacks - n_attacks)
        if this_chunk <= 0:
            break

        psteps = np.zeros(this_chunk, dtype=np.int64)
        hits = np.zeros((this_chunk, n), dtype=np.uint16)

        for a in range(this_chunk):
            attack = rng.permutation(n)
            second = _second_largest_component_after_each_removal(A, attack)
            p_step = _percolation_step_from_second_comp(second, n)
            psteps[a] = p_step
            hits[a, attack[:p_step]] = 1

        for a in range(this_chunk):
            p = int(psteps[a]) - 1
            counts_per_step[p] += 1
            part_counts[p, :] += hits[a, :].astype(np.uint64)

        n_attacks += this_chunk

        node_P = np.full((n, n), np.nan, dtype=np.float64)
        nonzero = counts_per_step > 0
        if np.any(nonzero):
            node_P[nonzero, :] = (
                part_counts[nonzero, :].astype(np.float64)
                / counts_per_step[nonzero, None].astype(np.float64)
            )

        hw_mat = wilson_half_width(part_counts, counts_per_step)
        finite = hw_mat[np.isfinite(hw_mat)]
        hw95 = float(np.percentile(finite, 95)) if finite.size else np.nan

        attacks_hist.append(n_attacks)
        hw95_hist.append(hw95)

        is_plateau, slope_val, range_val, mean_val = plateau_reached(
            np.asarray(attacks_hist, dtype=np.float64),
            np.asarray(hw95_hist, dtype=np.float64),
            hw95_tol=hw95_tol,
            range_tol=range_tol,
            slope_tol=slope_tol,
            window=plateau_window,
            enforce_hw=enforce_hw95,
        )
        stable_count = stable_count + 1 if is_plateau else 0

        if progress is not None:
            progress(
                {
                    "n_attacks": n_attacks,
                    "hw95": hw95,
                    "mean_val": mean_val,
                    "range_val": range_val,
                    "slope_val": slope_val,
                    "is_plateau": is_plateau,
                    "stable_count": stable_count,
                }
            )

        if stable_count >= require_stable:
            converged = True
            break

    elapsed = time.perf_counter() - t0

    node_P = np.full((n, n), np.nan, dtype=np.float64)
    nonzero = counts_per_step > 0
    if np.any(nonzero):
        node_P[nonzero, :] = (
            part_counts[nonzero, :].astype(np.float64)
            / counts_per_step[nonzero, None].astype(np.float64)
        )

    return PiPResult(
        node_P=node_P,
        counts_per_step=counts_per_step,
        part_counts=part_counts,
        n_attacks=n_attacks,
        attacks_hist=np.asarray(attacks_hist, dtype=np.int64),
        hw95_hist=np.asarray(hw95_hist, dtype=np.float64),
        converged=converged,
        elapsed_seconds=elapsed,
        meta={
            "max_attacks": max_attacks,
            "chunk_size": chunk_size,
            "seed": seed,
            "enforce_hw95": enforce_hw95,
            "hw95_tol": hw95_tol,
            "range_tol": range_tol,
            "slope_tol": slope_tol,
            "plateau_window": plateau_window,
            "require_stable": require_stable,
        },
    )
