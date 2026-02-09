#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("scipy is required to load/save MAT files.") from exc


def threshold_proportional(W, p):
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0,1].")
    W = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(W, 0.0)
    iu = np.triu_indices_from(W, k=1)
    w = W[iu]
    order = np.argsort(np.abs(w))[::-1]
    n_keep = max(1, int(round(p * w.size)))
    keep = np.zeros_like(w, dtype=bool)
    keep[order[:n_keep]] = True
    w_thr = np.zeros_like(w)
    w_thr[keep] = w[keep]
    W_thr = np.zeros_like(W)
    W_thr[iu] = w_thr
    W_thr = W_thr + W_thr.T
    return W_thr


def load_subject_matrix(path):
    mat = sio.loadmat(path)
    if "psi_adj" in mat:
        return np.asarray(mat["psi_adj"], dtype=np.float64)
    if "A" in mat:
        return np.asarray(mat["A"], dtype=np.float64)
    raise KeyError(f"psi_adj or A not found in {path}")


def read_included_subjects(csv_path):
    included = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("excluded", "").strip().lower() == "true":
                continue
            subj = row.get("subject_file", "").strip()
            if subj:
                if subj.endswith("_broadband_psi_adj_ConvHW.mat"):
                    subj = subj.replace("_broadband_psi_adj_ConvHW.mat", "_broadband_psi_adj.mat")
                else:
                    subj = subj.replace("_ConvHW.mat", "_broadband_psi_adj.mat")
                included.append(subj)
    return included


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Average PSI matrices for non-excluded subjects and apply proportional threshold."
    )
    parser.add_argument(
        "--pip-root",
        default=str(repo_root / "data" / "PSI_broadband_MEG_mats"),
    )
    parser.add_argument(
        "--outlier-csv",
        default=str(repo_root / "results" / "consensus_5pct" / "attack_outliers.csv"),
    )
    parser.add_argument("--threshold-prop", type=float, default=0.05)
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output .mat file for the averaged and thresholded PSI.",
    )
    args = parser.parse_args()

    pip_root = Path(args.pip_root)
    included_files = read_included_subjects(args.outlier_csv)
    if not included_files:
        raise RuntimeError(f"No included subjects found in {args.outlier_csv}")

    sumA = None
    used = []
    for fname in included_files:
        path = pip_root / fname
        if not path.exists():
            continue
        A = load_subject_matrix(path)
        A = np.asarray(A, dtype=np.float64)
        if sumA is None:
            sumA = np.zeros_like(A)
        sumA += A
        used.append(fname)

    if sumA is None or not used:
        raise RuntimeError("No valid matrices loaded for averaging.")

    avg_psi_adj = sumA / float(len(used))
    np.fill_diagonal(avg_psi_adj, 0.0)
    psi_adj = threshold_proportional(avg_psi_adj, args.threshold_prop)

    out_dir = pip_root / "avg"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.out_file) if args.out_file else out_dir / "AVG_broadband_psi_adj_top05_nonexcluded.mat"

    sio.savemat(
        out_file,
        {
            "psi_adj": psi_adj,
            "avg_psi_adj": avg_psi_adj,
            "threshold_prop": float(args.threshold_prop),
            "file_names": np.array(used, dtype=object),
        },
        do_compression=True,
    )
    print(f"Saved thresholded average PSI to {out_file}")
    print(f"Included {len(used)} subjects.")


if __name__ == "__main__":
    main()
