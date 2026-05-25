#!/usr/bin/env python3
"""
Export PiP node order used for numeric labels on the τ=S/6 tilted 2D heatmap
(3_plot_pip_surfaces.plot_tauS6_for_set): same get_tilt_peak_order_amplitude ranking,
first k nodes = ranks 1..k on the figure.

Outputs MATLAB 1-based node indices (columns of node_P in the MAT file).

Row order for MATLAB comparison scripts: **left column = first node removed**,
**right column = last node removed** (hub-first / strongest tilted-peak first),
unless you pass --reverse.

Single file: writes .txt (one index per line) and a long-form .csv with ranks.

Batch: scans --results-dir for *_ConvHW.mat and writes one numeric matrix CSV
(no header) shaped n_subjects × k for readmatrix().
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "pip_convergence" / "avg_giant75"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "pip_cluster"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from helpers.pip_plot_utils import (  # noqa: E402
    crop_longest_non_nan_block,
    get_tilt_peak_order_amplitude,
    load_pip_any,
)


def matlab_order_row(
    conv_mat: str | Path, k: int, clip_negative: bool, reverse: bool
) -> tuple[np.ndarray, str]:
    """Return length-k int64 vector of MATLAB 1-based node indices; subject id string."""
    path = Path(conv_mat)
    P_raw = load_pip_any(str(path))
    P_raw = crop_longest_non_nan_block(P_raw)
    if P_raw.size == 0:
        raise RuntimeError(f"Empty PiP after crop: {path}")

    order_nodes, _, _ = get_tilt_peak_order_amplitude(
        P_raw, tau_factor=1.0 / 6.0, clip_negative=clip_negative
    )
    k = min(int(k), int(order_nodes.size))
    top = order_nodes[:k].astype(np.int64) + 1
    if reverse:
        top = top[::-1]

    stem = path.stem
    m = re.match(r"^([A-Za-z0-9]+)_broadband_psi_adj", stem)
    subject_id = m.group(1) if m else stem.split("_")[0]
    return top, subject_id


def cmd_single(args: argparse.Namespace) -> None:
    P_raw = load_pip_any(args.conv_mat)
    P_raw = crop_longest_non_nan_block(P_raw)
    if P_raw.size == 0:
        raise SystemExit("Empty PiP after crop.")
    order_nodes, pk_step, pk_amp = get_tilt_peak_order_amplitude(
        P_raw, tau_factor=1.0 / 6.0, clip_negative=args.clip_negative
    )
    k = min(int(args.k), int(order_nodes.size))
    seq = order_nodes[:k].copy()
    if args.reverse:
        seq = seq[::-1]

    txt_path = Path(args.out_txt)
    csv_path = Path(args.out_csv)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        for n in seq:
            f.write(f"{int(n) + 1}\n")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("plot_rank,node_matlab,peak_attack_step_matlab,tilted_peak_amplitude\n")
        for rank, n in enumerate(seq, start=1):
            f.write(
                f"{rank},{int(n) + 1},{int(pk_step[int(n)]) + 1},{pk_amp[int(n)]:.12g}\n"
            )

    print(f"Wrote {txt_path} ({k} lines, MATLAB node indices, first line = first removal)")
    print(f"Wrote {csv_path}")


def cmd_batch(args: argparse.Namespace) -> None:
    d = Path(args.results_dir)
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    paths = sorted(d.glob("*_ConvHW.mat"))
    if args.exclude_avg:
        paths = [p for p in paths if not p.name.startswith("AVG_")]
    if not paths:
        raise SystemExit(f"No *_ConvHW.mat in {d}")

    rows = []
    ids = []
    for p in paths:
        row, sid = matlab_order_row(p, args.k, args.clip_negative, args.reverse)
        rows.append(row)
        ids.append(sid)

    mat = np.vstack(rows)
    out = Path(args.out_matrix_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, mat, fmt="%d", delimiter=",")
    print(f"Wrote {out}  shape={mat.shape}  (rows=subjects, cols=removal order, MATLAB 1-based, no header)")

    side = out.with_suffix(".subject_ids.csv")
    with open(side, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index0", "subject_id", "conv_mat"])
        for i, (sid, p) in enumerate(zip(ids, paths)):
            w.writerow([i, sid, p.name])
    print(f"Wrote {side}  (row order for the matrix above)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--conv-mat",
        default=str(
            DEFAULT_RESULTS / "AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat"
        ),
        help="Single *_ConvHW.mat (used when --results-dir is not set).",
    )
    p.add_argument("--k", type=int, default=66, help="Width of each row (default 66).")
    p.add_argument(
        "--clip-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match 3_plot_pip_surfaces default (default: clip).",
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse order so last column = strongest tilted peak (weakest-first attack).",
    )
    p.add_argument(
        "--results-dir",
        default=None,
        help="If set, batch all *_ConvHW.mat here into one n×k numeric CSV (no header).",
    )
    p.add_argument(
        "--exclude-avg",
        action="store_true",
        help="In batch mode, skip AVG_*.mat files.",
    )
    p.add_argument(
        "--out-matrix-csv",
        default=str(
            DEFAULT_OUT_DIR / "per_subject_pip2d_order_matlab_noheader.csv"
        ),
        help="Batch output: readmatrix-friendly numeric CSV.",
    )
    p.add_argument(
        "--out-txt",
        default=str(
            DEFAULT_OUT_DIR / "AVG_giant75_PiP_2d_label_order_matlab_nodes.txt"
        ),
    )
    p.add_argument(
        "--out-csv",
        default=str(
            DEFAULT_OUT_DIR / "AVG_giant75_PiP_2d_label_order_matlab.csv"
        ),
    )
    args = p.parse_args()

    if args.results_dir:
        cmd_batch(args)
    else:
        cmd_single(args)


if __name__ == "__main__":
    main()
