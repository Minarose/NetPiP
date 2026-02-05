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


def load_mat_any(path, varname):
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


def plot_matrix(mat, out_path, title, cmap="magma", dpi=300):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot average PSI matrix.")
    parser.add_argument(
        "--avg-mat",
        default=str(repo_root / "data" / "PSI_broadband_MEG_mats" / "avg" / "AVG_broadband_psi_adj_top10.mat"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "figures" / "avg_psi"),
    )
    parser.add_argument("--cmap", default="magma")
    args = parser.parse_args()

    avg = load_mat_any(args.avg_mat, varname="avg_psi_adj")
    thr = load_mat_any(args.avg_mat, varname="psi_adj")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_matrix(avg, os.path.join(args.out_dir, "avg_psi_adj.png"), "Average PSI (raw)", cmap=args.cmap)
    plot_matrix(thr, os.path.join(args.out_dir, "avg_psi_adj_top10.png"), "Average PSI (top 10%)", cmap=args.cmap)


if __name__ == "__main__":
    main()
