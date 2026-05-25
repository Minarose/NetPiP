"""Shared PiP loading + tilted-peak ranking helpers.

These are the canonical implementations used by:
  - scripts/3_plot_pip_surfaces.py        (figures)
  - scripts/4_cluster_pip_set.py          (clustering input)
  - scripts/6_export_pip_order.py         (export MATLAB-readable PiP order)
  - scripts/6_jaccard_overlap.py          (Python overlap analysis)

They live here, with a non-numeric module name, so the numbered pipeline
scripts (1_*, 2_*, ...) can still `import` from them. The Python toolbox in
netpip/ re-implements the same algorithm as a stable public API.
"""
from __future__ import annotations

import numpy as np

try:
    import scipy.io
except Exception:  # pragma: no cover - runtime dependency
    scipy = None  # type: ignore[assignment]

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None  # type: ignore[assignment]


def load_pip_any(path: str, varname: str = "node_P") -> np.ndarray:
    """Load (steps x nodes) PiP matrix from a MAT file (classic or v7.3 HDF5)."""
    if scipy is not None:
        try:
            mat = scipy.io.loadmat(path)
            if varname in mat:
                return np.asarray(mat[varname], dtype=np.float64)
            keys = {k.lower(): k for k in mat.keys()}
            if varname.lower() in keys:
                return np.asarray(mat[keys[varname.lower()]], dtype=np.float64)
        except NotImplementedError:
            pass

    if h5py is None:
        raise RuntimeError("h5py is required to load v7.3 MAT files.")
    with h5py.File(path, "r") as f:
        name = varname if varname in f else next(
            (k for k in f.keys() if varname.lower() in k.lower()), None
        )
        if name is None:
            raise KeyError(f"{varname} not found in {path}")
        return np.array(f[name], dtype=np.float64).T


def crop_longest_non_nan_block(P: np.ndarray) -> np.ndarray:
    """Keep the longest contiguous block of steps with any non-NaN values."""
    if P.size == 0:
        return P

    has_data = ~np.all(np.isnan(P), axis=1)
    if not np.any(has_data):
        return P[:0, :]

    indices = np.flatnonzero(has_data)
    starts = [indices[0]]
    ends: list[int] = []
    for prev, curr in zip(indices[:-1], indices[1:]):
        if curr != prev + 1:
            ends.append(prev)
            starts.append(curr)
    ends.append(indices[-1])

    lengths = [end - start + 1 for start, end in zip(starts, ends)]
    best_idx = int(np.argmax(lengths))
    start = starts[best_idx]
    end = ends[best_idx]
    return P[start : end + 1, :]


def tilt_early(P: np.ndarray, tau: float, clip_negative: bool = True) -> np.ndarray:
    """Multiply each step s by exp(-s/tau) so early participation is amplified."""
    P = np.array(P, dtype=np.float64)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_negative:
        P[P < 0] = 0.0
    S = P.shape[0]
    s = np.arange(1, S + 1, dtype=np.float64)[:, None]
    W = np.exp(-s / float(tau))
    return P * W


def get_tilt_peak_order_amplitude(
    P_raw: np.ndarray,
    tau_factor: float = 1 / 6,
    clip_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank nodes by largest tilted peak (amplitude high, earlier tie-break)."""
    S, N = P_raw.shape
    tau = max(2.0, float(tau_factor) * S)
    Pt = tilt_early(P_raw, tau=tau, clip_negative=clip_negative)
    peak_idx = np.argmax(Pt, axis=0)
    peak_amp = Pt[peak_idx, np.arange(N)]
    order = np.lexsort((peak_idx, -peak_amp))
    return order, peak_idx, peak_amp
