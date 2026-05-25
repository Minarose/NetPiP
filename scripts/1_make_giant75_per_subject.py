#!/usr/bin/env python3
"""
Per-subject giant-component thresholding (same rule as 1_make_giant75_avg.py).

For each included *_broadband_psi_adj.mat, symmetrize, zero diagonal, scan thresholds
with numpy.linspace(min, max, threshold_steps), and keep the most permissive threshold
such that the largest connected component has at least gcc_fraction * n nodes.

Writes one MAT per subject: <stem>_giant75.mat with binary psi_adj plus metadata.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required.") from exc

_REPO = Path(__file__).resolve().parent


def _load_giant_helpers():
    """Reuse helpers from 1_make_giant75_avg.py (same file dir)."""
    path = _REPO / "1_make_giant75_avg.py"
    spec = importlib.util.spec_from_file_location("giant_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def giant_threshold_one_matrix(gmod, W: np.ndarray, gcc_fraction: float, threshold_steps: int):
    W = np.asarray(W, dtype=np.float64)
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    max_val = float(np.nanmax(W))
    min_val = float(np.nanmin(W))
    if not np.isfinite(max_val) or max_val <= 0:
        raise RuntimeError("Matrix has no positive values for thresholding.")
    thresholds = np.linspace(min_val, max_val, int(threshold_steps))

    n = W.shape[0]
    gcc_target = int(round(gcc_fraction * n))
    last_good_idx = None
    for i, thr in enumerate(thresholds):
        bin_adj = W > thr
        if gmod.giant_component_size(bin_adj) >= gcc_target:
            last_good_idx = i

    if last_good_idx is None:
        raise RuntimeError("No threshold kept the giant component at the required size.")

    thr_val = float(thresholds[last_good_idx])
    psi_bin = (W > thr_val).astype(np.float64)
    psi_bin = ((psi_bin + psi_bin.T) / 2.0 > 0).astype(np.float64)
    np.fill_diagonal(psi_bin, 0.0)
    dens = gmod.density_und(psi_bin)
    return psi_bin, W, thr_val, gcc_target, dens, int(threshold_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pip-root",
        default="/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats",
    )
    parser.add_argument(
        "--outlier-csv",
        default="/hpf/projects/dkadis/ismail/NetPiP/results/pip_cluster/attack_outliers.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <pip-root>/per_subject_giant75_nonexcluded).",
    )
    parser.add_argument("--gcc-fraction", type=float, default=0.75)
    parser.add_argument("--threshold-steps", type=int, default=1000)
    parser.add_argument(
        "--all-mats",
        action="store_true",
        help="Process every *_broadband_psi_adj.mat under pip-root (ignore outlier CSV).",
    )
    args = parser.parse_args()

    gmod = _load_giant_helpers()
    read_included_subjects = gmod.read_included_subjects
    load_subject_matrix = gmod.load_subject_matrix

    pip_root = Path(args.pip_root)
    out_dir = Path(args.out_dir) if args.out_dir else pip_root / "per_subject_giant75_nonexcluded"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_mats:
        paths = sorted(pip_root.glob("*_broadband_psi_adj.mat"))
        if not paths:
            raise RuntimeError(f"No *_broadband_psi_adj.mat under {pip_root}")
    else:
        included = read_included_subjects(args.outlier_csv)
        if not included:
            raise RuntimeError(f"No included subjects in {args.outlier_csv}")
        paths = [pip_root / f for f in included]

    n_ok, n_skip = 0, 0
    for path in paths:
        if not path.is_file():
            print(f"skip missing: {path}", file=sys.stderr)
            n_skip += 1
            continue
        try:
            W = load_subject_matrix(path)
            psi_bin, W_sym, thr_val, gcc_target, dens, n_steps = giant_threshold_one_matrix(
                gmod, W, args.gcc_fraction, args.threshold_steps
            )
        except Exception as exc:
            print(f"skip {path.name}: {exc}", file=sys.stderr)
            n_skip += 1
            continue

        stem = path.stem
        if stem.endswith("_broadband_psi_adj"):
            out_stem = stem.replace("_broadband_psi_adj", "_broadband_psi_adj_giant75")
        else:
            out_stem = f"{stem}_giant75"
        out_path = out_dir / f"{out_stem}.mat"

        sio.savemat(
            str(out_path),
            {
                "psi_adj": psi_bin,
                "psi_adj_weighted": W_sym,
                "threshold_rule": f"giant_component_{args.gcc_fraction:.2f}",
                "threshold_value": thr_val,
                "threshold_steps": n_steps,
                "gcc_target": gcc_target,
                "density": dens,
                "source_file": path.name,
            },
            do_compression=True,
        )
        print(f"Saved {out_path}  thr={thr_val:.6g}  density={dens:.6g}")
        n_ok += 1

    print(f"Done. wrote={n_ok}  skipped={n_skip}  out_dir={out_dir}")


if __name__ == "__main__":
    main()
