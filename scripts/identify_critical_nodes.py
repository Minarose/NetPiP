#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist, squareform
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("scipy is required for clustering.") from exc

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None


def load_mat_any(path, varname):
    if sio is not None:
        try:
            mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            if varname in mat:
                return np.asarray(mat[varname])
            keys = {k.lower(): k for k in mat.keys()}
            if varname.lower() in keys:
                return np.asarray(mat[keys[varname.lower()]])
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
        return np.array(f[name]).T


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
    parser = argparse.ArgumentParser(description="Identify critical nodes and map to labels.")
    parser.add_argument("--pip-mat", required=True)
    parser.add_argument("--labels-mat", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args()

    P = load_mat_any(args.pip_mat, varname="node_P")
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

    raw_labels = load_mat_any(args.labels_mat, varname="node_labels")
    if raw_labels.size == 0:
        raise RuntimeError("No node_labels found in labels MAT.")
    raw_labels = np.array(raw_labels).ravel()
    node_labels = [str(x) for x in raw_labels]

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.pip_mat))[0]
    out_csv = os.path.join(args.out_dir, f"{base}_critical_nodes_labels.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("index,label,cluster,is_critical\n")
        for i, label in enumerate(node_labels):
            f.write(f"{i},{label},{int(labels[i])},{int(top_mask[i])}\n")

    out_txt = os.path.join(args.out_dir, f"{base}_critical_nodes_list.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in np.where(top_mask)[0]:
            f.write(f"{i}\t{node_labels[i]}\n")

    print(f"Saved labeled critical nodes to {out_csv}")
    print(f"Saved critical node list to {out_txt}")
    print("Critical node indices:", ", ".join(str(i) for i in np.where(top_mask)[0]))


if __name__ == "__main__":
    main()
