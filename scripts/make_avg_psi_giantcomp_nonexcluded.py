#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("scipy is required to load/save MAT files.") from exc


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


def load_subject_matrix(path):
    mat = sio.loadmat(path)
    if "psi_adj" in mat:
        return np.asarray(mat["psi_adj"], dtype=np.float64)
    if "A" in mat:
        return np.asarray(mat["A"], dtype=np.float64)
    raise KeyError(f"psi_adj or A not found in {path}")


def giant_component_size(bin_adj):
    graph = csr_matrix(bin_adj)
    _, labels = connected_components(graph, directed=False, return_labels=True)
    counts = np.bincount(labels)
    return int(counts.max()) if counts.size else 0


def density_und(bin_adj):
    n = bin_adj.shape[0]
    if n <= 1:
        return 0.0
    upper = np.triu(bin_adj, 1)
    edges = int(np.count_nonzero(upper))
    return edges / (n * (n - 1) / 2.0)


def main():
    parser = argparse.ArgumentParser(
        description="Average PSI matrices (non-excluded) and threshold by giant-component rule."
    )
    parser.add_argument(
        "--pip-root",
        default="/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats",
    )
    parser.add_argument(
        "--outlier-csv",
        default="/hpf/projects/dkadis/ismail/NetPiP/results/consensus_5pct/attack_outliers.csv",
    )
    parser.add_argument("--gcc-fraction", type=float, default=0.75)
    parser.add_argument("--threshold-steps", type=int, default=1000)
    parser.add_argument("--out-file", required=True)
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
        if sumA is None:
            sumA = np.zeros_like(A)
        sumA += A
        used.append(fname)

    if sumA is None or not used:
        raise RuntimeError("No valid matrices loaded for averaging.")

    avg_psi_adj = sumA / float(len(used))
    np.fill_diagonal(avg_psi_adj, 0.0)

    max_val = float(np.nanmax(avg_psi_adj))
    min_val = float(np.nanmin(avg_psi_adj))
    if not np.isfinite(max_val) or max_val <= 0:
        raise RuntimeError("Average PSI matrix has no positive values.")
    thresholds = np.linspace(min_val, max_val, args.threshold_steps)

    n = avg_psi_adj.shape[0]
    gcc_target = int(round(args.gcc_fraction * n))
    last_good_idx = None
    for i, thr in enumerate(thresholds):
        bin_adj = avg_psi_adj > thr
        if giant_component_size(bin_adj) >= gcc_target:
            last_good_idx = i

    if last_good_idx is None:
        raise RuntimeError("No threshold kept the giant component at the required size.")

    thr_val = float(thresholds[last_good_idx])
    adj_psi_percolating = (avg_psi_adj > thr_val).astype(np.float64)
    dens = density_und(adj_psi_percolating)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(
        out_path,
        {
            "psi_adj": adj_psi_percolating,
            "avg_psi_adj": avg_psi_adj,
            "threshold_rule": f"giant_component_{args.gcc_fraction:.2f}",
            "threshold_value": thr_val,
            "threshold_steps": int(args.threshold_steps),
            "gcc_target": gcc_target,
            "density": dens,
            "file_names": np.array(used, dtype=object),
        },
        do_compression=True,
    )
    print(f"Saved giant-component thresholded average PSI to {out_path}")
    print(f"Threshold value: {thr_val:.6g}  density={dens:.6g}  gcc_target={gcc_target}")


if __name__ == "__main__":
    main()
