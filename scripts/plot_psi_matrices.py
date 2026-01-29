#!/usr/bin/env python3
import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception:  # pragma: no cover - runtime dependency
    sio = None

try:
    import scipy.sparse as sparse
except Exception:  # pragma: no cover - runtime dependency
    sparse = None

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_dense(arr):
    if sparse is not None and sparse.issparse(arr):
        return arr.toarray()
    return np.asarray(arr)


def _find_key_case_insensitive(mapping, target):
    keys = {k.lower(): k for k in mapping.keys()}
    return keys.get(target.lower())


def extract_matrix(mat_dict):
    key = _find_key_case_insensitive(mat_dict, "psi_adj")
    if key is None:
        key = _find_key_case_insensitive(mat_dict, "A")
    if key is None:
        raise KeyError("Could not find psi_adj (or A) in MAT file.")
    return _to_dense(mat_dict[key])


def load_psi_adj_any(path):
    if sio is not None:
        try:
            mat = sio.loadmat(path)
            return extract_matrix(mat)
        except NotImplementedError:
            pass

    if h5py is None:
        raise RuntimeError("h5py is required to load v7.3 MAT files.")

    with h5py.File(path, "r") as f:
        key = _find_key_case_insensitive(f, "psi_adj")
        if key is None:
            key = _find_key_case_insensitive(f, "A")
        if key is None:
            raise KeyError(f"psi_adj (or A) not found in {path}")
        data = f[key]
        if hasattr(data, "shape"):
            return np.array(data).T
        raise ValueError(f"Unsupported MAT structure for {path}")


def plot_grid(mat_files, out_path, n_cols=5, cmap="magma", dpi=200):
    n_files = len(mat_files)
    n_rows = int(math.ceil(n_files / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
    )

    for idx, f in enumerate(mat_files):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        A = load_psi_adj_any(f)
        A = np.array(A, dtype=np.float64)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        im = ax.imshow(A, cmap=cmap)
        ax.set_title(os.path.basename(f), fontsize=8)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_files, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_individual(mat_files, out_dir, cmap="magma", dpi=200):
    os.makedirs(out_dir, exist_ok=True)

    for f in mat_files:
        A = load_psi_adj_any(f)
        A = np.array(A, dtype=np.float64)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(A, cmap=cmap)
        ax.set_title(os.path.basename(f), fontsize=10)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        base = os.path.splitext(os.path.basename(f))[0]
        out_path = os.path.join(out_dir, f"{base}.png")
        plt.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot original PSI adjacency matrices (grid + individual)."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--data-dir",
        default=str(repo_root / "data"),
        help="Directory containing *_psi_adj.mat files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_psi_adj.mat",
        help="Glob pattern for PSI MAT files.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "figures" / "PSI_matrices"),
        help="Output directory for plots.",
    )
    parser.add_argument("--cols", type=int, default=5, help="Grid columns.")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    parser.add_argument(
        "--no-grid", action="store_true", help="Skip grid subplot figure."
    )
    parser.add_argument(
        "--no-individual", action="store_true", help="Skip per-subject plots."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mat_files = sorted(glob.glob(os.path.join(args.data_dir, args.pattern)))
    if not mat_files:
        raise FileNotFoundError(
            f"No MAT files found in {args.data_dir} with pattern {args.pattern}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.no_grid:
        grid_path = os.path.join(args.out_dir, "psi_adj_grid.png")
        plot_grid(mat_files, grid_path, n_cols=args.cols, cmap=args.cmap, dpi=args.dpi)

    if not args.no_individual:
        indiv_dir = os.path.join(args.out_dir, "individual")
        plot_individual(mat_files, indiv_dir, cmap=args.cmap, dpi=args.dpi)


if __name__ == "__main__":
    main()
