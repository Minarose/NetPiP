#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def load_labels(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        labels = [row[0].strip() for row in reader if row]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Map 0-based node indices to labels.")
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--indices-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    labels = load_labels(args.labels_csv)
    if not labels:
        raise RuntimeError(f"No labels found in {args.labels_csv}")

    indices = np.loadtxt(args.indices_csv, dtype=int, delimiter=",")
    indices = np.atleast_1d(indices)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index_0based", "index_1based", "label"])
        for idx in indices:
            if idx < 0 or idx >= len(labels):
                label = ""
            else:
                label = labels[idx]
            writer.writerow([int(idx), int(idx) + 1, label])

    print(f"Saved labeled indices to {out_path}")


if __name__ == "__main__":
    main()
