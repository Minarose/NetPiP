#!/usr/bin/env python3
import argparse
import glob
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


def load_mat_any(path, varname):
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
        if np.any(same):
            a = float(np.mean(dist[i, same]))
        else:
            a = 0.0
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


def critical_nodes_from_matrix(node_by_step):
    x_crop = crop_longest_non_nan_block_cols(node_by_step)
    if x_crop.size == 0:
        return None
    n_steps = x_crop.shape[1]
    weights = np.linspace(1.0, 1.0 / n_steps, n_steps)
    x_weighted = x_crop * weights
    x_weighted = crop_longest_non_nan_block_cols(x_weighted)
    if x_weighted.size == 0:
        return None

    X = np.nan_to_num(x_weighted, nan=0.0, posinf=0.0, neginf=0.0)
    _, labels = select_optimal_k(X, k_min=2, k_max=10)
    if labels is None:
        return None

    node_sums = np.sum(X, axis=1)
    clusters = np.unique(labels)
    cluster_means = np.array([np.mean(node_sums[labels == c]) for c in clusters])
    cluster_sds = np.array([np.std(node_sums[labels == c]) for c in clusters])
    baseline_idx = int(np.argmin(cluster_means))
    baseline_sd = cluster_sds[baseline_idx]
    baseline_mean = cluster_means[baseline_idx]
    if baseline_sd == 0:
        cluster_z = cluster_means - baseline_mean
    else:
        cluster_z = (cluster_means - baseline_mean) / baseline_sd
    crit_cluster = clusters[np.argmax(cluster_z)]
    return labels == crit_cluster


def plot_consensus_markers(coords, consensus, out_path, threshold=None, title=""):
    if plotting is None:
        raise RuntimeError("nilearn is required for plotting.")
    if threshold is not None:
        mask = consensus >= threshold
        coords = coords[mask]
        consensus = consensus[mask]
    if coords.size == 0:
        return
    plotting.plot_markers(
        consensus,
        coords,
        node_size=40,
        node_cmap="Reds",
        node_vmin=0,
        node_vmax=1,
        display_mode="lzr",
        colorbar=True,
        title=title,
    ).savefig(out_path)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Consensus clustering across subjects.")
    parser.add_argument(
        "--results-dir",
        default=str(repo_root / "data" / "PSI_broadband_MEG_mats" / "results_converge"),
    )
    parser.add_argument(
        "--coords",
        default=str(repo_root / "data" / "MNI_66_coords.txt"),
    )
    parser.add_argument(
        "--out-results",
        default=str(repo_root / "results" / "consensus"),
    )
    parser.add_argument(
        "--out-figures",
        default=str(repo_root / "figures" / "consensus"),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
    )
    args = parser.parse_args()

    mat_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_ConvHW.mat")))
    if not mat_paths:
        raise FileNotFoundError(f"No *_ConvHW.mat files in {args.results_dir}")

    coords = np.loadtxt(args.coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coords file should be N x 3: {args.coords}")

    crit_list = []
    for path in mat_paths:
        P = load_mat_any(path, varname="node_P")
        node_by_step = P.T
        crit = critical_nodes_from_matrix(node_by_step)
        if crit is None:
            continue
        crit_list.append(crit.astype(int))

    if not crit_list:
        raise RuntimeError("No valid subjects found for consensus.")

    crit_nodes_group = np.vstack(crit_list)
    valid_subjects = np.sum(np.sum(crit_nodes_group, axis=1) > 0)
    if valid_subjects == 0:
        raise RuntimeError("No subjects with identified critical nodes.")

    consensus = np.sum(crit_nodes_group, axis=0) / float(valid_subjects)

    os.makedirs(args.out_results, exist_ok=True)
    os.makedirs(args.out_figures, exist_ok=True)
    np.save(os.path.join(args.out_results, "consensus_vector.npy"), consensus)
    np.savetxt(os.path.join(args.out_results, "consensus_vector.csv"), consensus, delimiter=",")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(consensus, marker="o", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("Node")
    plt.ylabel("Consensus frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_figures, "consensus_vector.png"), dpi=300)
    plt.close()

    plot_consensus_markers(
        coords,
        consensus,
        os.path.join(args.out_figures, "consensus_markers_all.png"),
        threshold=None,
        title="PiP consensus frequency",
    )
    for thr in args.thresholds:
        plot_consensus_markers(
            coords,
            consensus,
            os.path.join(args.out_figures, f"consensus_markers_gt_{thr:.2f}.png"),
            threshold=thr,
            title=f"PiP consensus (>= {thr:.2f})",
        )


if __name__ == "__main__":
    main()
