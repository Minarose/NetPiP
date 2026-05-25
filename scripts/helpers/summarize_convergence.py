#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np

try:
    import scipy.io as sio
except Exception:  # pragma: no cover - runtime dependency
    sio = None

try:
    import h5py
except Exception:  # pragma: no cover - runtime dependency
    h5py = None


def _extract_meta_value(meta, key):
    if meta is None:
        return None
    if hasattr(meta, key):
        return getattr(meta, key)
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _to_float(x):
    try:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if arr.size:
            return float(arr[0])
    except Exception:
        return None
    return None


def _to_last_float(x):
    try:
        arr = np.asarray(x, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return float(arr[-1])
    except Exception:
        return None
    return None


def _h5_resolve(f, data):
    if isinstance(data, h5py.Reference):
        return f[data][()]
    if isinstance(data, np.ndarray) and data.dtype == h5py.ref_dtype:
        resolved = [f[ref][()] for ref in data.ravel()]
        return np.array(resolved, dtype=object)
    if isinstance(data, np.ndarray) and data.dtype == object:
        resolved = [
            f[ref][()] if isinstance(ref, h5py.Reference) else ref
            for ref in data.ravel()
        ]
        return np.array(resolved, dtype=object)
    return data


def _h5_get_meta_field(f, meta, key):
    if meta is None or h5py is None:
        return None
    if isinstance(meta, h5py.Group):
        if key not in meta:
            return None
        data = meta[key][()]
        return _h5_resolve(f, data)
    if isinstance(meta, h5py.Dataset):
        if meta.dtype.names and key in meta.dtype.names:
            data = meta[key][()]
            return _h5_resolve(f, data)
    return None


def load_meta(path):
    if sio is not None:
        try:
            mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
            return mat.get("meta")
        except NotImplementedError:
            pass
    if h5py is None:
        return None
    with h5py.File(path, "r") as f:
        return f.get("meta")


def get_value_from_h5(path, key):
    if h5py is None:
        return None
    with h5py.File(path, "r") as f:
        meta = f.get("meta")
        return _h5_get_meta_field(f, meta, key)


def summarize_file(path):
    meta = load_meta(path)
    if hasattr(meta, "dtype") or isinstance(meta, dict):
        attacks_hist = _extract_meta_value(meta, "attacks_hist")
        hw95_hist = _extract_meta_value(meta, "hw95_hist")
        elapsed_sec = _extract_meta_value(meta, "elapsed_sec")
    else:
        attacks_hist = get_value_from_h5(path, "attacks_hist")
        hw95_hist = get_value_from_h5(path, "hw95_hist")
        elapsed_sec = get_value_from_h5(path, "elapsed_sec")

    final_attacks = _to_last_float(attacks_hist)
    final_hw95 = _to_last_float(hw95_hist)
    elapsed = _to_float(elapsed_sec)
    return final_attacks, final_hw95, elapsed


def main():
    parser = argparse.ArgumentParser(description="Summarize convergence metadata into CSV.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    mat_paths = sorted(glob.glob(os.path.join(args.results_dir, "*_ConvHW.mat")))
    if not mat_paths:
        raise FileNotFoundError(f"No *_ConvHW.mat files in {args.results_dir}")

    rows = []
    for path in mat_paths:
        final_attacks, final_hw95, elapsed = summarize_file(path)
        rows.append(
            {
                "file": os.path.basename(path),
                "final_attacks": final_attacks,
                "final_hw95": final_hw95,
                "elapsed_sec": elapsed,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "final_attacks", "final_hw95", "elapsed_sec"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved convergence summary to {out_path}")


if __name__ == "__main__":
    main()
