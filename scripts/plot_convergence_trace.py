#!/usr/bin/env python3
"""
Plot convergence trace from a *_ConvHW.mat file:
  - attacks_hist vs chunk
  - hw95_hist vs chunk

Works for classic MAT and MATLAB v7.3 (HDF5) files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.io as sio
except Exception:  # pragma: no cover
    sio = None

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None


def _as_1d_float(x) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def _h5_resolve(f: h5py.File, data):  # type: ignore[name-defined]
    # Resolve MATLAB references when present.
    if isinstance(data, h5py.Reference):
        return f[data][()]
    if isinstance(data, np.ndarray) and data.dtype == h5py.ref_dtype:
        resolved = [f[ref][()] for ref in data.ravel()]
        return np.array(resolved, dtype=object)
    return data


def load_meta_fields(conv_mat: Path) -> tuple[np.ndarray, np.ndarray]:
    # Try scipy first (non-v7.3)
    if sio is not None:
        try:
            mat = sio.loadmat(str(conv_mat), squeeze_me=True, struct_as_record=False)
            meta = mat.get("meta")
            if meta is not None and hasattr(meta, "attacks_hist") and hasattr(meta, "hw95_hist"):
                return _as_1d_float(meta.attacks_hist), _as_1d_float(meta.hw95_hist)
        except NotImplementedError:
            pass

    if h5py is None:
        raise RuntimeError("Need h5py to read MATLAB v7.3 files.")

    with h5py.File(str(conv_mat), "r") as f:
        meta = f.get("meta")
        if meta is None:
            raise KeyError("`meta` not found in MAT file.")
        # Usually meta is a Group with datasets attacks_hist/hw95_hist
        if isinstance(meta, h5py.Group):
            ah = _h5_resolve(f, meta["attacks_hist"][()])
            hw = _h5_resolve(f, meta["hw95_hist"][()])
        else:
            raise TypeError(f"Unsupported meta type: {type(meta)}")

    # MATLAB stores as (1,n) or (n,1); resolve + squeeze
    return _as_1d_float(ah), _as_1d_float(hw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conv-mat", default=None, help="Path to *_ConvHW.mat")
    p.add_argument("--out-png", default=None, help="Output PNG (default: next to MAT)")
    p.add_argument("--title", default=None, help="Optional figure title")
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Generate a dummy schematic trace (no MAT file needed).",
    )
    p.add_argument("--dummy-chunks", type=int, default=16, help="Dummy: number of chunks.")
    p.add_argument("--dummy-attacks-final", type=float, default=8000.0, help="Dummy: final attacks.")
    p.add_argument("--dummy-hw95-start", type=float, default=0.20, help="Dummy: start HW95.")
    p.add_argument("--dummy-hw95-final", type=float, default=0.03, help="Dummy: final HW95.")
    p.add_argument("--dummy-max-attacks", type=float, default=1e7, help="Dummy: x-axis max attacks.")
    p.add_argument("--dummy-stop-attacks", type=float, default=2e5, help="Dummy: stop around this attack count.")
    p.add_argument("--dummy-chunk-size", type=float, default=1e4, help="Dummy: attacks per chunk.")
    p.add_argument("--dummy-win-len", type=int, default=5, help="Dummy: plateau window length (chunks).")
    p.add_argument("--dummy-need-stable", type=int, default=3, help="Dummy: stable plateau hits required.")
    p.add_argument("--dummy-hw95-thresh", type=float, default=0.05, help="Dummy: HW95 threshold (HW95_TOL).")
    args = p.parse_args()

    if args.dummy:
        # Methods schematic: HW95 vs attacks, stop early after plateau checks.
        chunk = float(args.dummy_chunk_size)
        stop_attacks = float(args.dummy_stop_attacks)
        n = int(max(3, np.ceil(stop_attacks / chunk)))
        attacks = np.arange(1, n + 1, dtype=float) * chunk
        attacks[-1] = stop_attacks  # ensure exact stop marker

        # HW95 decays smoothly then plateaus by ~stop_attacks.
        t = attacks / max(stop_attacks, 1.0)
        hw95 = args.dummy_hw95_final + (args.dummy_hw95_start - args.dummy_hw95_final) * np.exp(-4.5 * t)
        # Ensure the schematic actually crosses below the HW95 threshold by the stop chunk.
        thresh = float(args.dummy_hw95_thresh)
        target_end = 0.8 * thresh
        if hw95[-1] > target_end:
            # Apply a smooth downward offset that is 0 at start and full at end.
            delta = hw95[-1] - target_end
            hw95 = hw95 - delta * t

        out_png = Path(args.out_png) if args.out_png else Path("results/convergence_trace_dummy.png")
        title = args.title if args.title else "Convergence trace (schematic; stop after plateau)"
    else:
        if not args.conv_mat:
            raise SystemExit("Provide --conv-mat or use --dummy.")
        conv_mat = Path(args.conv_mat)
        if not conv_mat.is_file():
            raise FileNotFoundError(conv_mat)

        attacks, hw95 = load_meta_fields(conv_mat)
        if attacks.size == 0 or hw95.size == 0:
            raise RuntimeError("attacks_hist/hw95_hist empty or missing.")

        n = min(attacks.size, hw95.size)
        attacks = attacks[:n]
        hw95 = hw95[:n]
        x = np.arange(1, n + 1)

        out_png = Path(args.out_png) if args.out_png else conv_mat.with_suffix(".convergence_trace.png")
        title = args.title if args.title else f"Convergence trace: {conv_mat.name}"

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        # Single-axis schematic: HW95 vs attacks (0..max), with windows for checks + stop chunk.
        fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
        ax.plot(attacks, hw95, color="#d32f2f", lw=3.6)

        # Highlight when plateau checks become active: as soon as hw95 <= threshold.
        thresh = float(args.dummy_hw95_thresh)
        cross_idx = np.where(hw95 <= thresh)[0]
        if cross_idx.size:
            ix0 = int(cross_idx[0])
            ax.axvspan(attacks[ix0], attacks[-1], color="#bdbdbd", alpha=0.12)

        # Highlight plateau-check window (grey) and stop chunk (yellow).
        win_len = int(max(3, args.dummy_win_len))
        need_stable = int(max(1, args.dummy_need_stable))
        stop_i = len(attacks) - 1
        start_i = max(0, stop_i - win_len + 1)
        x0 = attacks[start_i]
        x1 = attacks[stop_i]
        ax.axvspan(x0, x1, color="#bdbdbd", alpha=0.22)

        # Highlight the stable streak (needStable consecutive chunks)
        stable_start_i = max(0, stop_i - need_stable + 1)
        ax.axvspan(
            attacks[stable_start_i],
            attacks[stop_i],
            color="#9e9e9e",
            alpha=0.18,
        )

        # "stop chunk" highlight: last chunk interval
        prev_x = attacks[stop_i - 1] if stop_i > 0 else 0.0
        ax.axvspan(prev_x, attacks[stop_i], color="#ffde59", alpha=0.35)

        # Dotted HW95 threshold line (HW95_TOL / plateau_start)
        ax.axhline(float(args.dummy_hw95_thresh), color="k", ls=":", lw=2.8, alpha=0.9)

        # Zoom into the region leading up to stopping (more interpretable than 0..10M)
        ax.set_xlim(0, stop_attacks * 1.15)
        ax.set_xlabel("Attacks (arbitrary units)")
        ax.set_ylabel("HW95")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # No legend for the schematic (paper-style)
        fig.tight_layout()
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}")
        return

    # Real data: dual-axis trace (attacks & HW95) vs chunk index
    fig, ax1 = plt.subplots(figsize=(10, 4.6), dpi=160)
    ax2 = ax1.twinx()

    x = np.arange(1, len(attacks) + 1)
    l1 = ax1.plot(x, attacks, color="#3a89dc", lw=2.2, label="attacks")
    l2 = ax2.plot(x, hw95, color="#d32f2f", lw=2.2, label="HW95")

    ax1.set_xlabel("Chunk index")
    ax1.set_ylabel("Attacks (meta.attacks_hist)")
    ax2.set_ylabel("HW95 (meta.hw95_hist)")
    ax1.set_title(title)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

