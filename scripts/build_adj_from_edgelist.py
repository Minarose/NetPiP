#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("scipy is required to save MAT files.") from exc


def read_edges(path):
    edges = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row.get("u")
            v = row.get("v")
            w = row.get("w", 1.0)
            if u is None or v is None:
                continue
            try:
                w = float(w)
            except Exception:
                w = 1.0
            edges.append((u, v, w))
    return edges


def build_adjacency(edges, weighted=True):
    nodes = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float64)
    for u, v, w in edges:
        i = idx[u]
        j = idx[v]
        val = w if weighted else 1.0
        if i == j:
            continue
        if val == 0:
            continue
        A[i, j] = max(A[i, j], val)
        A[j, i] = max(A[j, i], val)
    return A, nodes


def main():
    parser = argparse.ArgumentParser(description="Build adjacency matrix from edge list CSV.")
    parser.add_argument("--edgelist", required=True)
    parser.add_argument("--out-mat", required=True)
    parser.add_argument("--weighted", action="store_true", help="Use edge weights if present.")
    args = parser.parse_args()

    edges = read_edges(args.edgelist)
    if not edges:
        raise RuntimeError(f"No edges read from {args.edgelist}")

    A, nodes = build_adjacency(edges, weighted=args.weighted)
    out_path = Path(args.out_mat)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(
        out_path,
        {
            "psi_adj": A,
            "node_labels": np.array(nodes, dtype=object),
        },
        do_compression=True,
    )
    print(f"Saved adjacency to {out_path} (n={A.shape[0]})")


if __name__ == "__main__":
    main()
