#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception:  # pragma: no cover - runtime dependency
    sio = None

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pip_any(path, varname="node_P"):
    if sio is not None:
        try:
            mat = sio.loadmat(path)
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


def plot_matrix(mat, out_path, title, cmap="gist_heat_r", dpi=300):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Attack Step")
    ax.set_ylabel("Node Index")
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot AVG PiP matrix heatmap.")
    parser.add_argument(
        "--pip-mat",
        default=str(repo_root / "data" / "PSI_broadband_MEG_mats" / "results_converge" /
                    "AVG_broadband_psi_adj_top10_ConvHW.mat"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "figures" / "pip_avg_matrix"),
    )
    args = parser.parse_args()

    P = load_pip_any(args.pip_mat, varname="node_P")
    P = crop_longest_non_nan_block(P)
    if P.size == 0:
        raise RuntimeError("No valid data after cropping.")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pip_mat))[0]
    out_path = os.path.join(args.out_dir, f"{base}_node_P_heatmap.png")
    plot_matrix(P, out_path, f"{base} node_P heatmap")


if __name__ == "__main__":
    main()
