"""Shared fixtures for netpip tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def barbell_adj() -> np.ndarray:
    """Two K4 cliques joined by a single bridge edge: a textbook 'two-hub' graph.

    Nodes 0-3 = clique A, nodes 4-7 = clique B, edge (3, 4) is the bridge.
    Removing node 3 or node 4 fragments the graph into two K4-sized halves,
    which is what PiP should pick up.
    """
    n = 8
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            A[i, j] = A[j, i] = 1.0
    for i in range(4, 8):
        for j in range(i + 1, 8):
            A[i, j] = A[j, i] = 1.0
    A[3, 4] = A[4, 3] = 1.0
    return A


@pytest.fixture
def small_path_adj() -> np.ndarray:
    """Path graph on 6 nodes: 0-1-2-3-4-5. Bridge nodes are 2 and 3."""
    n = 6
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A
