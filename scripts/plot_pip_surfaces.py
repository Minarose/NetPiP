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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_pip_any(path, varname="node_P"):
    """Load (steps × nodes) from MAT (classic or v7.3/HDF5)."""
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
    """Keep the longest contiguous block of steps with any non-NaN values."""
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


def tilt_early(P, tau, clip_negative=True):
    """Multiply each step s by exp(-s/tau)."""
    P = np.array(P, dtype=np.float64)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_negative:
        P[P < 0] = 0.0
    S = P.shape[0]
    s = np.arange(1, S + 1, dtype=np.float64)[:, None]
    W = np.exp(-s / float(tau))
    return P * W


def get_tilt_peak_order_amplitude(P_raw, tau_factor=1 / 6, clip_negative=True):
    """Rank nodes by largest tilted peak (amplitude high, earlier tie-break)."""
    S, N = P_raw.shape
    tau = max(2.0, float(tau_factor) * S)
    Pt = tilt_early(P_raw, tau=tau, clip_negative=clip_negative)
    peak_idx = np.argmax(Pt, axis=0)
    peak_amp = Pt[peak_idx, np.arange(N)]
    order = np.lexsort((peak_idx, -peak_amp))
    return order, peak_idx, peak_amp


def plot_surface_3d(mat, fname_out, labels=None, clip_z01=False):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    steps, nodes = mat.shape
    X, Y = np.meshgrid(np.arange(steps), np.arange(nodes))

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, mat.T,
        cmap="gist_heat_r",
        linewidth=0, antialiased=False,
        rstride=1, cstride=1, alpha=0.9,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.35, pad=0.05)
    cbar.set_label("Participation", fontsize=20, labelpad=15)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel("Attack Step", fontsize=18, labelpad=20)
    ax.set_ylabel("Node Index", fontsize=18, labelpad=20)

    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.tick_params(axis="z", length=0, colors=(1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(0, steps - 1)
    ax.set_ylim(nodes - 1, 0)
    ax.invert_xaxis()
    ax.view_init(elev=20, azim=130)
    if clip_z01:
        ax.set_zlim(0.0, 1.0)

    if labels:
        xs = [s for (s, n, z, r1) in labels]
        ys = [n for (s, n, z, r1) in labels]
        zs = [z for (s, n, z, r1) in labels]
        ax.scatter(xs, ys, zs, s=50, c="k", depthshade=False)

        zmin, zmax = float(np.nanmin(mat)), float(np.nanmax(mat))
        dz = 0.03 * max(1e-9, (zmax - zmin))
        for (s, n, z, r1) in labels:
            ax.text(
                s, n, z + dz, str(r1),
                fontsize=14, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            )

    plt.tight_layout()
    plt.savefig(fname_out, dpi=400, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_heatmap_2d(mat, fname_out, labels=None):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap="gist_heat_r")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Participation", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Attack Step", fontsize=12)
    ax.set_ylabel("Node Index", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if labels:
        for (s, n, z, r1) in labels:
            ax.scatter(s, n, s=60, facecolors="none", edgecolors="k", linewidths=1.5)
            ax.text(
                s, n, str(r1),
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="none", alpha=0.9),
            )

    plt.tight_layout()
    plt.savefig(fname_out, dpi=400, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_heatmap_2d_on_ax(ax, mat, labels=None, vmin=None, vmax=None):
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    im = ax.imshow(
        mat.T,
        aspect="auto",
        origin="lower",
        cmap="gist_heat_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Attack Step", fontsize=12)
    ax.set_ylabel("Node Index", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    if labels:
        for (s, n, z, r1) in labels:
            ax.scatter(s, n, s=60, facecolors="none", edgecolors="k", linewidths=1.5)
            ax.text(
                s, n, str(r1),
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="none", alpha=0.9),
            )
    return im


def plot_tauS6_for_set(mat_paths_set, out_root, tag="postHW", k=66, clip_negative=True):
    """Process .mat files with τ=S/6 and save 2D + 3D plots."""
    out_dir = os.path.join(out_root, f"tauS6_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    for p in mat_paths_set:
        P_raw = load_pip_any(p)
        P_raw = crop_longest_non_nan_block(P_raw)
        if P_raw.size == 0:
            continue

        S = P_raw.shape[0]
        tau = max(2.0, (1 / 6) * S)
        display_mat = tilt_early(P_raw, tau=tau, clip_negative=clip_negative)

        order_nodes, pk_step, pk_amp = get_tilt_peak_order_amplitude(
            P_raw, tau_factor=1 / 6, clip_negative=clip_negative
        )
        top_nodes = order_nodes[:k]
        labels = [(pk_step[n], n, pk_amp[n], i + 1) for i, n in enumerate(top_nodes)]

        base_name = os.path.splitext(os.path.basename(p))[0]
        prefix = base_name.split("_")[0]

        base = os.path.join(out_dir, f"{prefix}_tauS6_{tag}")
        plot_surface_3d(display_mat, base + "_3D.png", labels=None)
        plot_heatmap_2d(display_mat, base + "_2D.png", labels=labels)


def _map_subject_files(file_paths, suffix):
    mapping = {}
    for path in file_paths:
        base = os.path.basename(path)
        if not base.endswith(suffix):
            continue
        prefix = base.split("_")[0]
        mapping[prefix] = path
    return mapping


def plot_tauS6_compare(old_dir, new_dir, out_root, tag="postHW", k=66):
    """Compare old vs new PiP matrices as side-by-side 2D heatmaps."""
    old_files = [
        os.path.join(old_dir, f) for f in os.listdir(old_dir)
        if f.endswith("participation_in_percolation.mat")
    ]
    new_files = [
        os.path.join(new_dir, f) for f in os.listdir(new_dir)
        if f.endswith("broadband_psi_adj_ConvHW.mat")
    ]
    old_map = _map_subject_files(old_files, "participation_in_percolation.mat")
    new_map = _map_subject_files(new_files, "broadband_psi_adj_ConvHW.mat")

    subjects = sorted(set(old_map.keys()) & set(new_map.keys()))
    if not subjects:
        raise FileNotFoundError("No overlapping subjects between old and new dirs.")

    out_dir = os.path.join(out_root, f"tauS6_{tag}", "old_vs_new")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "old_vs_new_all_subjects.pdf")

    with PdfPages(pdf_path) as pdf:
        for prefix in subjects:
            P_old = crop_longest_non_nan_block(
                load_pip_any(old_map[prefix], varname="node_participation_at_percolation")
            )
            P_new = crop_longest_non_nan_block(
                load_pip_any(new_map[prefix], varname="node_P")
            )
            if P_old.size == 0 or P_new.size == 0:
                continue

            tau_old = max(2.0, (1 / 6) * P_old.shape[0])
            tau_new = max(2.0, (1 / 6) * P_new.shape[0])
            disp_old = tilt_early(P_old, tau=tau_old, clip_negative=False)
            disp_new = tilt_early(P_new, tau=tau_new, clip_negative=False)

            order_old, pk_step_old, pk_amp_old = get_tilt_peak_order_amplitude(
                P_old, tau_factor=1 / 6, clip_negative=False
            )
            order_new, pk_step_new, pk_amp_new = get_tilt_peak_order_amplitude(
                P_new, tau_factor=1 / 6, clip_negative=False
            )
            labels_old = [(pk_step_old[n], n, pk_amp_old[n], i + 1) for i, n in enumerate(order_old[:k])]
            labels_new = [(pk_step_new[n], n, pk_amp_new[n], i + 1) for i, n in enumerate(order_new[:k])]

            vmin = float(np.nanmin([disp_old.min(), disp_new.min()]))
            vmax = float(np.nanmax([disp_old.max(), disp_new.max()]))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            plot_heatmap_2d_on_ax(axes[0], disp_old, labels=labels_old, vmin=vmin, vmax=vmax)
            axes[0].set_title(f"{prefix} old", fontsize=12)
            im1 = plot_heatmap_2d_on_ax(axes[1], disp_new, labels=labels_new, vmin=vmin, vmax=vmax)
            axes[1].set_title(f"{prefix} new", fontsize=12)

            cbar = fig.colorbar(im1, ax=axes, shrink=0.8, pad=0.02)
            cbar.set_label("Participation", fontsize=12, labelpad=10)
            cbar.ax.tick_params(labelsize=10)

            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{prefix}_old_vs_new_2D.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    default_pip_root = os.environ.get(
        "PIP_ROOT",
        "/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats",
    )

    parser = argparse.ArgumentParser(description="Plot PiP convergence surfaces.")
    parser.add_argument("--pip-root", default=default_pip_root)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--out-root", default=str(repo_root / "figures" / "pip_surfaces"))
    parser.add_argument("--tag", default="postHW")
    parser.add_argument("--k", type=int, default=66)
    parser.add_argument(
        "--include-prefix",
        nargs="+",
        default=None,
        help="Only plot files whose basename starts with one of these prefixes.",
    )
    parser.add_argument(
        "--no-clip-negative",
        action="store_true",
        help="Do not clip negative values to zero before plotting.",
    )
    parser.add_argument("--compare-old-dir", default=None)
    parser.add_argument("--compare-new-dir", default=None)
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(args.pip_root, "results_converge")
    mat_paths = sorted(
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith("broadband_psi_adj_ConvHW.mat")
    )
    if args.include_prefix:
        mat_paths = [
            p for p in mat_paths
            if os.path.basename(p).split("_")[0] in set(args.include_prefix)
        ]

    if not mat_paths:
        raise FileNotFoundError(f"No *_ConvHW.mat files found in {results_dir}")

    plot_tauS6_for_set(
        mat_paths,
        args.out_root,
        tag=args.tag,
        k=args.k,
        clip_negative=not args.no_clip_negative,
    )

    if args.compare_old_dir or args.compare_new_dir:
        old_dir = args.compare_old_dir or os.path.join(repo_root, "results", "old", "PiP_matrices")
        new_dir = args.compare_new_dir or os.path.join(args.pip_root, "results_converge")
        plot_tauS6_compare(old_dir, new_dir, args.out_root, tag=args.tag, k=args.k)


if __name__ == "__main__":
    main()
