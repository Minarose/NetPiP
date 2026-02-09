#!/usr/bin/env python3
import argparse
import csv
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


def _extract_attacks_hist_from_meta(meta):
    if meta is None:
        return None
    if hasattr(meta, "attacks_hist"):
        return meta.attacks_hist
    if isinstance(meta, dict):
        return meta.get("attacks_hist")
    return None


def _extract_n_attacks_final_from_meta(meta):
    if meta is None:
        return None
    if hasattr(meta, "n_attacks_final"):
        return meta.n_attacks_final
    if isinstance(meta, dict):
        return meta.get("n_attacks_final")
    return None


def load_attacks_final(path):
    if scipy.io is not None:
        try:
            mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
            meta = mat.get("meta")
            n_attacks = _extract_n_attacks_final_from_meta(meta)
            if n_attacks is not None:
                return float(np.asarray(n_attacks).squeeze())
            attacks_hist = _extract_attacks_hist_from_meta(meta)
            if attacks_hist is not None:
                attacks = np.asarray(attacks_hist, dtype=np.float64).ravel()
                attacks = attacks[np.isfinite(attacks)]
                if attacks.size:
                    return float(attacks[-1])
        except NotImplementedError:
            pass

    if h5py is None:
        return None
    with h5py.File(path, "r") as f:
        if "meta" not in f:
            return None
        meta = f["meta"]
        if "n_attacks_final" in meta:
            data = np.array(meta["n_attacks_final"], dtype=np.float64).squeeze()
            if np.size(data):
                return float(np.ravel(data)[0])
        if "attacks_hist" in meta:
            data = np.array(meta["attacks_hist"], dtype=np.float64).T
            attacks = np.ravel(data)
            attacks = attacks[np.isfinite(attacks)]
            if attacks.size:
                return float(attacks[-1])
    return None


def compute_attack_outliers(attacks, method="iqr", iqr_k=1.5, z_thresh=3.0):
    attacks = np.asarray(attacks, dtype=np.float64)
    attacks = attacks[np.isfinite(attacks)]
    if attacks.size == 0:
        return np.array([], dtype=bool), None
    if method == "zscore":
        mean = float(np.mean(attacks))
        std = float(np.std(attacks))
        if std == 0:
            return np.zeros(attacks.shape, dtype=bool), mean
        z = (attacks - mean) / std
        return z > z_thresh, mean + z_thresh * std
    q1 = float(np.percentile(attacks, 25))
    q3 = float(np.percentile(attacks, 75))
    iqr = q3 - q1
    cutoff = q3 + iqr_k * iqr
    return attacks > cutoff, cutoff


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


def plot_consensus_red(coords, consensus, out_path, threshold, title=""):
    if plotting is None:
        raise RuntimeError("nilearn is required for plotting.")
    mask = consensus >= threshold
    coords = coords[mask]
    if coords.size == 0:
        return
    plotting.plot_markers(
        np.ones(coords.shape[0]),
        coords,
        node_size=40,
        node_cmap="Reds",
        node_vmin=0,
        node_vmax=1,
        display_mode="lzr",
        colorbar=False,
        title=title,
    ).savefig(out_path)


def plot_subject_critical_nodes(coords, crit_mask, out_path, title=""):
    if plotting is None:
        raise RuntimeError("nilearn is required for plotting.")
    crit_mask = np.asarray(crit_mask).astype(bool)
    if coords.size == 0 or not np.any(crit_mask):
        return
    plotting.plot_markers(
        np.ones(int(crit_mask.sum())),
        coords[crit_mask],
        node_size=40,
        node_cmap="Reds",
        node_vmin=0,
        node_vmax=1,
        display_mode="lzr",
        colorbar=False,
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
    parser.add_argument(
        "--consensus-red-threshold",
        type=float,
        default=None,
        help="Plot consensus nodes above this threshold in a single red color.",
    )
    parser.add_argument(
        "--out-subject-figs",
        default=None,
        help="Directory for per-subject critical-node brain plots.",
    )
    parser.add_argument(
        "--exclude-attack-outliers",
        action="store_true",
        help="Exclude subjects with unusually high #attacks to convergence.",
    )
    parser.add_argument(
        "--attack-outlier-method",
        choices=["iqr", "zscore"],
        default="iqr",
    )
    parser.add_argument(
        "--attack-outlier-k",
        type=float,
        default=1.5,
        help="IQR multiplier for attack outliers (method=iqr).",
    )
    parser.add_argument(
        "--attack-outlier-z",
        type=float,
        default=3.0,
        help="Z-score threshold for attack outliers (method=zscore).",
    )
    args = parser.parse_args()

    mat_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_ConvHW.mat")))
    if not mat_paths:
        raise FileNotFoundError(f"No *_ConvHW.mat files in {args.results_dir}")

    coords = np.loadtxt(args.coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coords file should be N x 3: {args.coords}")

    attacks_final = []
    for path in mat_paths:
        attacks_final.append(load_attacks_final(path))

    if args.exclude_attack_outliers:
        valid_attacks = [a for a in attacks_final if a is not None]
        outlier_mask, cutoff = compute_attack_outliers(
            valid_attacks,
            method=args.attack_outlier_method,
            iqr_k=args.attack_outlier_k,
            z_thresh=args.attack_outlier_z,
        )
        outlier_values = set(np.asarray(valid_attacks)[outlier_mask])
        excluded = []
        kept_paths = []
        kept_attacks = []
        for path, attacks in zip(mat_paths, attacks_final):
            if attacks is not None and attacks in outlier_values:
                excluded.append((path, attacks))
            else:
                kept_paths.append(path)
                kept_attacks.append(attacks)
        mat_paths = kept_paths
        attacks_final = kept_attacks
        print(f"Excluded {len(excluded)} subjects for high #attacks.")
        if cutoff is not None:
            print(f"Attack outlier cutoff ({args.attack_outlier_method}) = {cutoff:.2f}")

        os.makedirs(args.out_results, exist_ok=True)
        with open(os.path.join(args.out_results, "attack_outliers.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject_file", "final_attacks", "excluded"])
            for path, attacks in excluded:
                writer.writerow([os.path.basename(path), attacks, True])
            for path, attacks in zip(mat_paths, attacks_final):
                writer.writerow([os.path.basename(path), attacks, False])

    crit_list = []
    subject_records = []
    for path in mat_paths:
        P = load_mat_any(path, varname="node_P")
        node_by_step = P.T
        crit = critical_nodes_from_matrix(node_by_step)
        if crit is None:
            continue
        crit = np.asarray(crit).astype(bool)
        crit_list.append(crit.astype(int))
        subject_records.append((path, crit))

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

    if args.consensus_red_threshold is not None:
        out_path = os.path.join(
            args.out_figures,
            f"consensus_markers_red_ge_{args.consensus_red_threshold:.2f}.png",
        )
        plot_consensus_red(
            coords,
            consensus,
            out_path,
            threshold=args.consensus_red_threshold,
            title=f"PiP consensus (>= {args.consensus_red_threshold:.2f})",
        )

    if args.out_subject_figs is not None:
        os.makedirs(args.out_subject_figs, exist_ok=True)
        for path, crit in subject_records:
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.out_subject_figs, f"{base}_critical_nodes.png")
            plot_subject_critical_nodes(
                coords,
                crit,
                out_path,
                title=f"{base} | critical nodes",
            )


if __name__ == "__main__":
    main()
