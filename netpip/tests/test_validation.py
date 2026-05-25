from __future__ import annotations

import numpy as np
import pytest

from netpip import AdjacencyValidationError, validate_adjacency


def test_accepts_clean_binary_symmetric_matrix(barbell_adj):
    report = validate_adjacency(barbell_adj, min_giant_fraction=0.5)
    assert report.n_nodes == 8
    assert report.is_binary is True
    assert report.is_symmetric is True
    assert report.zero_diagonal is True
    assert report.giant_component_size == 8
    assert report.giant_component_fraction == 1.0


def test_rejects_non_binary():
    A = np.array([[0.0, 0.7], [0.7, 0.0]])
    with pytest.raises(AdjacencyValidationError, match="binary"):
        validate_adjacency(A)


def test_rejects_asymmetric():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(AdjacencyValidationError, match="symmetric"):
        validate_adjacency(A)


def test_rejects_self_loops():
    A = np.array([[1.0, 1.0], [1.0, 0.0]])
    with pytest.raises(AdjacencyValidationError, match="zero diagonal"):
        validate_adjacency(A)


def test_rejects_nan():
    A = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(AdjacencyValidationError, match="non-finite"):
        validate_adjacency(A)


def test_rejects_complete_graph():
    n = 4
    A = np.ones((n, n)) - np.eye(n)
    with pytest.raises(AdjacencyValidationError, match="complete graph"):
        validate_adjacency(A)


def test_rejects_small_giant_component():
    # Two disconnected dyads -> giant component = 2 / 4 = 50%, but we ask for 75%
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 1
    A[2, 3] = A[3, 2] = 1
    with pytest.raises(AdjacencyValidationError, match="Largest connected component"):
        validate_adjacency(A, min_giant_fraction=0.75)


def test_does_not_modify_input(barbell_adj):
    before = barbell_adj.copy()
    validate_adjacency(barbell_adj)
    np.testing.assert_array_equal(barbell_adj, before)


def test_rejects_wrong_shape():
    with pytest.raises(AdjacencyValidationError, match="square"):
        validate_adjacency(np.zeros((3, 4)))
    with pytest.raises(AdjacencyValidationError, match="2D"):
        validate_adjacency(np.zeros(5))
