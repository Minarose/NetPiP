"""Read-only validation of user-supplied adjacency matrices.

The validator is intentionally **non-mutating**: it never thresholds, binarizes,
symmetrizes, or rewrites the matrix the user passes in. If the matrix does not
already satisfy the PiP requirements, ``validate_adjacency`` raises an
:class:`AdjacencyValidationError` with a human-readable explanation.

PiP requires the input adjacency to be:

* a 2D NumPy array of shape ``(N, N)`` with ``N >= 2``;
* finite (no NaN / inf);
* **binary** (entries are exactly 0 or 1);
* **symmetric** (undirected: ``A[i, j] == A[j, i]``);
* **zero-diagonal** (no self-loops);
* **sparse** (density strictly less than 1; not the complete graph); and
* characterized by a **giant connected component** that contains at least
  ``min_giant_fraction`` of the ``N`` nodes (default 0.5).

The non-binary / non-symmetric / non-zero-diagonal / has-NaN / too-dense /
no-giant-component conditions all raise. Use the convenience helpers in your
own preprocessing code (e.g. ``(W > 0).astype(int)``) to satisfy them before
passing the matrix to :func:`netpip.run_pip`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ArrayLike = Union[np.ndarray, "list"]


class AdjacencyValidationError(ValueError):
    """Raised when an adjacency matrix does not satisfy PiP's input contract."""


@dataclass(frozen=True)
class AdjacencyReport:
    """Descriptive statistics computed during validation (does not modify the matrix)."""

    n_nodes: int
    n_edges: int
    density: float
    giant_component_size: int
    giant_component_fraction: float
    is_binary: bool
    is_symmetric: bool
    zero_diagonal: bool

    def summary(self) -> str:
        return (
            f"AdjacencyReport(n_nodes={self.n_nodes}, n_edges={self.n_edges}, "
            f"density={self.density:.4f}, "
            f"giant_component_size={self.giant_component_size} "
            f"({100 * self.giant_component_fraction:.1f}% of nodes))"
        )


def _as_2d_array(adj: ArrayLike, *, dtype: type = np.float64) -> np.ndarray:
    """Coerce user input to a 2D NumPy array without otherwise touching the values."""
    arr = np.asarray(adj)
    if arr.ndim != 2:
        raise AdjacencyValidationError(
            f"Adjacency must be a 2D array; got {arr.ndim}D with shape {arr.shape}."
        )
    if arr.shape[0] != arr.shape[1]:
        raise AdjacencyValidationError(
            f"Adjacency must be square; got shape {arr.shape}."
        )
    if arr.shape[0] < 2:
        raise AdjacencyValidationError(
            f"Adjacency must have at least 2 nodes; got n={arr.shape[0]}."
        )
    return arr.astype(dtype, copy=False)


def validate_adjacency(
    adj: ArrayLike,
    *,
    min_giant_fraction: float = 0.5,
) -> AdjacencyReport:
    """Verify that ``adj`` satisfies PiP's input contract; return a descriptive report.

    This function performs **read-only** checks. It never modifies, thresholds,
    binarizes, or symmetrizes the input matrix. If any check fails, it raises
    :class:`AdjacencyValidationError` with a precise diagnostic.

    Parameters
    ----------
    adj : array_like, shape (N, N)
        Binary, symmetric, zero-diagonal, sparse, undirected adjacency matrix.
    min_giant_fraction : float, default 0.5
        Minimum required size of the largest connected component, as a fraction
        of ``N``. PiP is only meaningful when a single connected backbone
        dominates the graph; the giant-component rule used in the accompanying
        manuscript was 0.75 (see ``analysis/giant_component_avg_nonexcluded``).

    Returns
    -------
    AdjacencyReport
        Descriptive statistics (n_edges, density, giant-component size, etc.).

    Raises
    ------
    AdjacencyValidationError
        If ``adj`` is not binary, not symmetric, has self-loops, is fully dense,
        contains NaN/inf, or its largest connected component is smaller than
        ``min_giant_fraction * N`` nodes.
    """
    A = _as_2d_array(adj)

    if not np.all(np.isfinite(A)):
        raise AdjacencyValidationError(
            "Adjacency contains non-finite values (NaN / inf). "
            "PiP requires finite entries."
        )

    # Binary check (exactly {0, 1})
    unique = np.unique(A)
    is_binary = unique.size <= 2 and set(unique.tolist()).issubset({0.0, 1.0})
    if not is_binary:
        raise AdjacencyValidationError(
            f"Adjacency must be binary (entries in {{0, 1}}); got unique values "
            f"{unique[:10].tolist()}{'...' if unique.size > 10 else ''}. "
            "Binarize first, e.g. `(W > 0).astype(np.float64)`."
        )

    # Symmetry check (exact for binary matrices)
    if not np.array_equal(A, A.T):
        raise AdjacencyValidationError(
            "Adjacency must be symmetric (A == A.T). "
            "Symmetrize first, e.g. `((A + A.T) > 0).astype(np.float64)`."
        )

    # Zero diagonal
    diag = np.diagonal(A)
    if np.any(diag != 0):
        raise AdjacencyValidationError(
            "Adjacency must have zero diagonal (no self-loops). "
            "Zero the diagonal first, e.g. `np.fill_diagonal(A, 0)`."
        )

    n = int(A.shape[0])
    n_edges = int(np.count_nonzero(np.triu(A, 1)))
    max_edges = n * (n - 1) // 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    if density >= 1.0:
        raise AdjacencyValidationError(
            "Adjacency is the complete graph (density == 1.0). "
            "PiP requires a sparse network with a non-trivial fragmentation profile."
        )
    if n_edges == 0:
        raise AdjacencyValidationError(
            "Adjacency has no edges (density == 0.0). "
            "PiP requires at least one connected component of size >= 2."
        )

    # Giant component check
    _, labels = connected_components(csr_matrix(A), directed=False, return_labels=True)
    counts = np.bincount(labels)
    giant = int(counts.max()) if counts.size else 0
    giant_frac = giant / n
    if giant_frac < float(min_giant_fraction):
        raise AdjacencyValidationError(
            f"Largest connected component has {giant} / {n} nodes "
            f"({100 * giant_frac:.1f}% < required {100 * min_giant_fraction:.1f}%). "
            "Threshold the matrix less aggressively, or relax `min_giant_fraction`."
        )

    return AdjacencyReport(
        n_nodes=n,
        n_edges=n_edges,
        density=density,
        giant_component_size=giant,
        giant_component_fraction=giant_frac,
        is_binary=True,
        is_symmetric=True,
        zero_diagonal=True,
    )
