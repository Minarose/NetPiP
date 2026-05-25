#!/usr/bin/env python3
"""Supplementary Figure S1: binary group-average adjacency (giant-component rule, non-excluded)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scipy.io as sio
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scipy is required.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AVG_MAT = (
    REPO_ROOT
    / "data"
    / "PSI_broadband_MEG_mats"
    / "group_average"
    / "AVG_broadband_psi_adj_giant75_nonexcluded.mat"
)
DEFAULT_OUT_PNG = REPO_ROOT / "figures" / "FigureS1_giant75_avg_binary_adjacency.png"


def load_binary_adj(mat_path):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "psi_adj" not in mat:
        raise KeyError(f"psi_adj not found in {mat_path}")
    A = np.asarray(mat["psi_adj"], dtype=np.float64)
    A = np.nan_to_num(A, nan=0.0)
    return (A > 0.0).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Figure S1 — binary giant-thresholded group adjacency.")
    parser.add_argument(
        "--avg-mat",
        default=str(DEFAULT_AVG_MAT),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_PNG),
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    A = load_binary_adj(args.avg_mat)
    n = A.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    im = ax.imshow(A, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Node index", fontsize=11)
    ax.set_ylabel("Node index", fontsize=11)
    ax.set_title(
        "Figure S1. Group-average binary adjacency\n"
        "(giant component ≥ 75% of nodes; non-excluded subjects)",
        fontsize=11,
    )
    ax.set_xticks([0, n - 1])
    ax.set_yticks([0, n - 1])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cb.set_label("Edge (binary)", fontsize=10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
