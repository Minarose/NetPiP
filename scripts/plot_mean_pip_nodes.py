#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io
except Exception:  # pragma: no cover - runtime dependency
    scipy = None

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None

try:
    from nilearn import plotting
except Exception:  # pragma: no cover - runtime dependency
    plotting = None


def load_pip_any(path, varname="node_P"):
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


def crop_longest_non_nan_block(P):
    if P.size == 0:
        return P
    has_data = ~np.all(np.isnan(P), axis=1)
    if not np.any(has_data):
        return P[:0, :]
    indices = np.flatnonzero(has_data)
    starts = [indices[0]]
    ends = []
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


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot mean PiP nodes at 95th percentile.")
    parser.add_argument("--pip-mat", required=True)
    parser.add_argument(
        "--coords",
        default=str(repo_root / "data" / "MNI_66_coords.txt"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "figures" / "pip_mean_nodes"),
    )
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    if plotting is None:
        raise RuntimeError("nilearn is required for plotting.")

    P = load_pip_any(args.pip_mat, varname="node_P")
    P = crop_longest_non_nan_block(P)
    if P.size == 0:
        raise RuntimeError("No valid data after cropping.")

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    mean_nodes = np.mean(P, axis=0)
    thresh = np.percentile(mean_nodes, args.percentile)
    mask = mean_nodes >= thresh

    coords = np.loadtxt(args.coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coords must be N x 3.")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pip_mat))[0]
    np.savetxt(
        os.path.join(args.out_dir, f"{base}_mean_nodes.csv"),
        mean_nodes,
        delimiter=",",
    )

    plotting.plot_markers(
        mean_nodes[mask],
        coords[mask],
        node_size=40,
        node_cmap="Reds",
        node_vmin=0,
        node_vmax=float(mean_nodes.max()),
        display_mode="lzr",
        colorbar=True,
        title=f"{base} mean PiP nodes >= P{args.percentile:.0f}",
    ).savefig(os.path.join(args.out_dir, f"{base}_mean_nodes_p{int(args.percentile)}.png"))


if __name__ == "__main__":
    main()
