"""Loading and checking adjacency matrices.

PiP works on binary, undirected graphs without self-loops. These helpers read
common formats and either validate a matrix or convert a weighted one.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["load_adjacency", "load_edge_list", "validate_adjacency", "binarize", "to_csr"]


def validate_adjacency(adjacency, *, name: str = "adjacency") -> np.ndarray:
    """Return `adjacency` as a uint8 array after checking it is a valid PiP input.

    Requirements: square, binary (0/1), symmetric, zero diagonal. Disconnected
    graphs and isolated nodes are allowed; every node counts towards N.
    """
    A = np.asarray(adjacency)
    if hasattr(adjacency, "toarray"):  # scipy.sparse
        A = adjacency.toarray()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be a square N x N matrix, got shape {A.shape}")
    if A.shape[0] < 3:
        raise ValueError(f"{name} needs at least 3 nodes")
    if not np.isin(A, [0, 1]).all():
        raise ValueError(f"{name} must be binary (0/1). Use netpip.binarize() for weighted matrices.")
    if not np.array_equal(A, A.T):
        raise ValueError(f"{name} must be symmetric (undirected). Use binarize(..., symmetrize=True).")
    if np.diag(A).any():
        raise ValueError(f"{name} must have a zero diagonal (no self-loops)")
    return A.astype(np.uint8)


def binarize(weights, threshold: float = 0.0, *, absolute: bool = False,
             symmetrize: bool = True) -> np.ndarray:
    """Turn a weighted matrix into a PiP input: edge where weight > threshold.

    With `absolute=True` the magnitude is thresholded. With `symmetrize=True`
    an edge in either direction becomes an undirected edge. The diagonal is cleared.
    """
    W = np.asarray(weights, dtype=float)
    if absolute:
        W = np.abs(W)
    A = (W > threshold).astype(np.uint8)
    if symmetrize:
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return validate_adjacency(A)


def load_edge_list(path, n_nodes: int | None = None, *, one_based: bool = True,
                   delimiter: str | None = None) -> np.ndarray:
    """Build an adjacency matrix from a two-column edge list (header lines are skipped).

    Node labels must be integers. With `one_based=True` (default) the first node is 1.
    `n_nodes` sets N so that isolated nodes are kept; otherwise N = largest label.
    """
    path = Path(path)
    edges = []
    with path.open() as f:
        sample = f.read(2048)
        f.seek(0)
        if delimiter is None:
            delimiter = "," if "," in sample else None
        rows = csv.reader(f, delimiter=delimiter) if delimiter else (line.split() for line in f)
        for row in rows:
            if len(row) < 2:
                continue
            try:
                edges.append((int(row[0]), int(row[1])))
            except ValueError:
                continue  # header or comment
    if not edges:
        raise ValueError(f"no integer edges found in {path}")
    E = np.array(edges) - (1 if one_based else 0)
    if E.min() < 0:
        raise ValueError("negative node index; check `one_based`")
    n = int(n_nodes) if n_nodes is not None else int(E.max()) + 1
    A = np.zeros((n, n), dtype=np.uint8)
    A[E[:, 0], E[:, 1]] = 1
    A[E[:, 1], E[:, 0]] = 1
    np.fill_diagonal(A, 0)
    return validate_adjacency(A)


def load_adjacency(path, *, key: str | None = None) -> np.ndarray:
    """Read an adjacency matrix from .npy, .npz, .csv/.txt (dense matrix) or .mat.

    For .npz/.mat files with several arrays, pass `key`; otherwise the first
    square matrix is used. For edge lists use :func:`load_edge_list`.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        A = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as z:
            A = z[key] if key else next(z[k] for k in z.files if z[k].ndim == 2 and z[k].shape[0] == z[k].shape[1])
    elif suffix in (".csv", ".txt", ".tsv"):
        delimiter = "," if suffix == ".csv" else None
        A = np.loadtxt(path, delimiter=delimiter)
    elif suffix == ".mat":
        from scipy.io import loadmat

        z = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        A = z[key] if key else next(v for v in z.values()
                                    if np.ndim(v) == 2 and v.shape[0] == v.shape[1] and v.shape[0] > 2)
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    A = np.asarray(A)
    if np.isin(A, [0, 1]).all():
        return validate_adjacency(A.astype(np.uint8))
    raise ValueError(f"{path.name} is not binary; threshold it first with netpip.binarize()")


def to_csr(adjacency):
    """Compressed sparse row arrays (indptr, indices) used by the sampling kernel."""
    A = validate_adjacency(adjacency)
    indptr = np.r_[0, np.cumsum(A.sum(axis=1))].astype(np.int64)
    indices = np.nonzero(A)[1].astype(np.int64)
    return indptr, indices
