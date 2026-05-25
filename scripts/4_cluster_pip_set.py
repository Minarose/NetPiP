#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist, squareform
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("scipy is required for clustering.") from exc

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None

try:
    from nilearn import plotting
except Exception:  # pragma: no cover - runtime dependency
    plotting = None


def load_pip_any(path, varname="node_P"):
    if scipy.io is not None:
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


def crop_longest_non_nan_block_cols(x):
    if x.size == 0:
        return x
    has_data = ~np.all(np.isnan(x), axis=0)
    if not np.any(has_data):
        return x[:, :0]
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
    return x[:, start : end + 1]


def silhouette_mean(X, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n = X.shape[0]
    if n <= 1 or unique.size <= 1:
        return 0.0

    dist = squareform(pdist(X, metric="euclidean"))
    s = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = float(np.mean(dist[i, same])) if np.any(same) else 0.0
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b = min(b, float(np.mean(dist[i, other])))
        if not np.isfinite(b) or max(a, b) == 0:
            s[i] = 0.0
        else:
            s[i] = (b - a) / max(a, b)
    return float(np.mean(s))


def select_optimal_k(X, k_min=2, k_max=10):
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return 1, np.ones(n, dtype=int)
    Z = sch.linkage(X, method="ward")
    best_k = k_min
    best_score = -np.inf
    best_labels = None
    for k in range(k_min, k_max + 1):
        labels = sch.fcluster(Z, k, criterion="maxclust")
        score = silhouette_mean(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    return best_k, best_labels


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Cluster avg PiP nodes and plot top cluster.")
    parser.add_argument("--pip-mat", required=True)
    parser.add_argument(
        "--coords",
        default=str(repo_root / "data" / "MNI_66_coords.txt"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(repo_root / "figures" / "pip_avg_cluster"),
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    if plotting is None:
        raise RuntimeError("nilearn is required for plotting.")

    P = load_pip_any(args.pip_mat, varname="node_P")
    node_by_step = P.T
    x_crop = crop_longest_non_nan_block_cols(node_by_step)
    if x_crop.size == 0:
        raise RuntimeError("No valid data after cropping.")

    n_steps = x_crop.shape[1]
    weights = np.linspace(1.0, 1.0 / n_steps, n_steps)
    x_weighted = x_crop * weights
    x_weighted = crop_longest_non_nan_block_cols(x_weighted)
    if x_weighted.size == 0:
        raise RuntimeError("No valid data after weighting/cropping.")

    X = np.nan_to_num(x_weighted, nan=0.0, posinf=0.0, neginf=0.0)
    best_k, labels = select_optimal_k(X, k_min=args.k_min, k_max=args.k_max)

    node_sums = np.sum(X, axis=1)
    clusters = np.unique(labels)
    cluster_means = np.array([np.mean(node_sums[labels == c]) for c in clusters])
    top_cluster = clusters[np.argmax(cluster_means)]
    top_mask = labels == top_cluster

    coords = np.loadtxt(args.coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coords must be N x 3.")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pip_mat))[0]
    np.savetxt(
        os.path.join(args.out_dir, f"{base}_cluster_labels.csv"),
        labels,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(args.out_dir, f"{base}_top_cluster_nodes.csv"),
        np.where(top_mask)[0],
        fmt="%d",
        delimiter=",",
    )
    print("Top cluster node indices:", ", ".join(str(i) for i in np.where(top_mask)[0]))

    plotting.plot_markers(
        np.ones(int(top_mask.sum())),
        coords[top_mask],
        node_size=40,
        node_cmap="Reds",
        node_vmin=0,
        node_vmax=1,
        display_mode="lzr",
        colorbar=False,
        title=f"{base} | top cluster (k={best_k}, n={int(top_mask.sum())})",
    ).savefig(os.path.join(args.out_dir, f"{base}_top_cluster_markers.png"))


if __name__ == "__main__":
    main()
