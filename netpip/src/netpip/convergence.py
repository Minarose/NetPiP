"""Wilson-95% half-width estimator + plateau-based early-stopping criterion.

Ported from the MATLAB helpers ``wilson_hw_matrix`` and ``check_hw_plateau``
in ``scripts/pip_converge_posthw5_thresh.m``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_Z = 1.96  # 95% normal quantile


def wilson_half_width(
    part_counts: np.ndarray, counts_per_step: np.ndarray
) -> np.ndarray:
    """Per-cell Wilson 95% confidence-interval half-width on binomial proportions.

    Parameters
    ----------
    part_counts : np.ndarray of integer dtype, shape (n_depths, n_nodes)
        Successes (number of attacks at that depth where node ``i`` was a
        participant).
    counts_per_step : np.ndarray of integer dtype, shape (n_depths,)
        Trials (number of attacks whose percolation step equals that depth).

    Returns
    -------
    np.ndarray, shape (n_depths, n_nodes)
        Wilson 95% half-widths; rows where ``counts_per_step == 0`` are NaN.
    """
    pc = np.asarray(part_counts, dtype=np.float64)
    cps = np.asarray(counts_per_step, dtype=np.float64).reshape(-1)
    n_depths, n_nodes = pc.shape
    if cps.shape[0] != n_depths:
        raise ValueError(
            f"counts_per_step length {cps.shape[0]} != part_counts rows {n_depths}"
        )

    out = np.full((n_depths, n_nodes), np.nan, dtype=np.float64)
    nonzero = cps > 0
    if not np.any(nonzero):
        return out

    n = cps[nonzero, None]  # (k, 1)
    k = pc[nonzero, :]
    p = np.where(n > 0, k / n, 0.0)
    p = np.where(np.isnan(p), 0.0, p)

    z2 = _Z**2
    num = _Z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n**2))
    den = 1.0 + z2 / n
    out[nonzero, :] = num / den
    return out


def plateau_reached(
    attacks_hist: np.ndarray,
    hw95_hist: np.ndarray,
    *,
    hw95_tol: float = 0.05,
    range_tol: float = 0.005,
    slope_tol: float = 1e-7,
    window: int = 5,
    enforce_hw: bool = False,
) -> Tuple[bool, float, float, float]:
    """Has the Wilson-95 half-width plateaued over the last ``window`` chunks?

    Parameters
    ----------
    attacks_hist : np.ndarray, shape (n_chunks,)
        Cumulative attack count at each chunk boundary.
    hw95_hist : np.ndarray, shape (n_chunks,)
        95th percentile of the per-cell Wilson half-widths after each chunk.
    hw95_tol : float, default 0.05
        Mean-of-window half-width tolerance (used only if ``enforce_hw=True``).
    range_tol : float, default 0.005
        Maximum allowed (max - min) of half-widths within the window.
    slope_tol : float, default 1e-7
        Maximum allowed absolute slope of half-widths vs attacks within the
        window (least-squares fit).
    window : int, default 5
        Number of trailing chunks to consider.
    enforce_hw : bool, default False
        If True, require ``mean(window) < hw95_tol`` in addition to the
        range and slope conditions.

    Returns
    -------
    is_plateau : bool
    slope_val : float
    range_val : float
    mean_val : float
    """
    hw = np.asarray(hw95_hist, dtype=np.float64).reshape(-1)
    atk = np.asarray(attacks_hist, dtype=np.float64).reshape(-1)

    if hw.size < window:
        return False, np.nan, np.nan, np.nan

    h = hw[-window:]
    a = atk[-window:]

    if not np.all(np.isfinite(h)):
        return False, np.nan, np.nan, np.nan

    mean_val = float(np.mean(h))
    range_val = float(np.max(h) - np.min(h))
    slope = np.polyfit(a, h, 1)
    slope_val = float(slope[0])

    if enforce_hw:
        is_plateau = (
            (mean_val < hw95_tol)
            and (range_val < range_tol)
            and (abs(slope_val) < slope_tol)
        )
    else:
        is_plateau = (range_val < range_tol) and (abs(slope_val) < slope_tol)

    return is_plateau, slope_val, range_val, mean_val
