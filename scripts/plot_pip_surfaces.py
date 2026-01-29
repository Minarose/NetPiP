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


def crop_trailing_nan_rows(P):
    """Drop rows after the first all-NaN row."""
    if P.size == 0:
        return P
    nan_rows = np.all(np.isnan(P), axis=1)
    if not np.any(nan_rows):
        return P
    first_nan = int(np.argmax(nan_rows))
    return P[:first_nan, :]


def tilt_early(P, tau):
    """Multiply each step s by exp(-s/tau)."""
    P = np.array(P, dtype=np.float64)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P[P < 0] = 0.0
    S = P.shape[0]
    s = np.arange(1, S + 1, dtype=np.float64)[:, None]
    W = np.exp(-s / float(tau))
    return P * W


def get_tilt_peak_order_amplitude(P_raw, tau_factor=1 / 6):
    """Rank nodes by largest tilted peak (amplitude high, earlier tie-break)."""
    S, N = P_raw.shape
    tau = max(2.0, float(tau_factor) * S)
    Pt = tilt_early(P_raw, tau=tau)
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


def plot_tauS6_for_set(mat_paths_set, out_root, tag="postHW", k=66):
    """Process .mat files with τ=S/6 and save 2D + 3D plots."""
    out_dir = os.path.join(out_root, f"tauS6_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    for p in mat_paths_set:
        P_raw = load_pip_any(p)
        P_raw = crop_trailing_nan_rows(P_raw)
        if P_raw.size == 0:
            continue

        S = P_raw.shape[0]
        tau = max(2.0, (1 / 6) * S)
        display_mat = tilt_early(P_raw, tau=tau)

        order_nodes, pk_step, pk_amp = get_tilt_peak_order_amplitude(P_raw, tau_factor=1 / 6)
        top_nodes = order_nodes[:k]
        labels = [(pk_step[n], n, pk_amp[n], i + 1) for i, n in enumerate(top_nodes)]

        base_name = os.path.splitext(os.path.basename(p))[0]
        prefix = base_name.split("_")[0]

        base = os.path.join(out_dir, f"{prefix}_tauS6_{tag}")
        plot_surface_3d(display_mat, base + "_3D.png", labels=None)
        plot_heatmap_2d(display_mat, base + "_2D.png", labels=labels)


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
    args = parser.parse_args()

    results_dir = args.results_dir or os.path.join(args.pip_root, "results_converge")
    mat_paths = sorted(
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith("broadband_psi_adj_ConvHW.mat")
    )

    if not mat_paths:
        raise FileNotFoundError(f"No *_ConvHW.mat files found in {results_dir}")

    plot_tauS6_for_set(mat_paths, args.out_root, tag=args.tag, k=args.k)


if __name__ == "__main__":
    main()
