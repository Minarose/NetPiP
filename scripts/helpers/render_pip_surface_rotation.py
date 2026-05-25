#!/usr/bin/env python3
"""Render the canonical group-average PiP surface as a rotating 360 deg GIF.

This is the "hero" animation embedded at the top of the repo README.md.
Reads ``node_P`` from the bundled AVG ConvHW MAT, crops the longest
non-NaN block, applies the same tau = S/6 tilted weighting used in the
paper figure (scripts/3_plot_pip_surfaces.py), and sweeps the 3D camera
azimuth from 0 to 360 deg, saving an animated GIF via Pillow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from helpers.pip_plot_utils import (  # noqa: E402
    crop_longest_non_nan_block,
    load_pip_any,
    tilt_early,
)

DEFAULT_PIP_MAT = (
    REPO
    / "results"
    / "pip_convergence"
    / "avg_giant75"
    / "AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat"
)
DEFAULT_OUT = REPO / "figures" / "pip_surface_rotation.gif"


def render(
    pip_mat: Path,
    out_path: Path,
    frames: int = 72,
    fps: int = 24,
    figsize: tuple[float, float] = (8.5, 6.0),
    dpi: int = 160,
    tau_factor: float = 1.0 / 6.0,
) -> None:
    P_raw = load_pip_any(str(pip_mat))
    P = crop_longest_non_nan_block(P_raw)
    if P.size == 0:
        raise RuntimeError(f"PiP matrix in {pip_mat} is empty after cropping.")
    S, N = P.shape
    tau = max(2.0, tau_factor * S)
    Z = tilt_early(P, tau=tau, clip_negative=True)

    X, Y = np.meshgrid(np.arange(S), np.arange(N), indexing="ij")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    ax.patch.set_alpha(0.0)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="gist_heat_r",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.95,
        shade=True,
    )
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Participation in Percolation", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    ax.set_xlabel("Attack step", fontsize=10, labelpad=6)
    ax.set_ylabel("Node index", fontsize=10, labelpad=6)
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.tick_params(axis="z", length=0, colors=(1, 1, 1, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.tick_params(labelsize=8)

    ax.set_position([-0.04, -0.02, 0.92, 1.06])
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    elev = 28.0

    def update(frame: int):
        azim = (360.0 * frame / frames) % 360.0
        ax.view_init(elev=elev, azim=azim)
        return (surf,)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer, dpi=dpi, savefig_kwargs={"transparent": True})
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path}  ({frames} frames, {fps} fps, {size_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pip-mat", default=str(DEFAULT_PIP_MAT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    render(
        pip_mat=Path(args.pip_mat),
        out_path=Path(args.out),
        frames=args.frames,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
