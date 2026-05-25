from __future__ import annotations

import numpy as np

from netpip import percolation_point, pip_top_n_at_percolation_point, run_pip


def test_percolation_point_of_path_attacks_bridge(small_path_adj):
    # Path 0-1-2-3-4-5. By the MATLAB run_attack_once convention, the 2nd-component
    # size at step i is measured BEFORE removing node order[i-1]. So:
    #   step 1: full path, single component, 2nd comp = 0
    #   step 2: node 2 already removed -> {0,1} and {3,4,5} -> 2nd comp size = 2
    # The percolation step is the FIRST step at which the 2nd comp peaks, so pp = 2.
    order = np.array([2, 0, 1, 3, 4, 5], dtype=np.int64)
    pp = percolation_point(small_path_adj, order)
    assert pp == 2, f"expected percolation step 2 (MATLAB convention), got {pp}"


def test_pip_top_n_at_percolation_point_returns_subset(barbell_adj):
    res = run_pip(barbell_adj, max_attacks=500, chunk_size=100, seed=0)
    top = pip_top_n_at_percolation_point(barbell_adj, res.node_P)
    assert top.ndim == 1
    assert top.size >= 1
    assert top.size <= barbell_adj.shape[0]
    # All entries are valid node indices
    assert set(top.tolist()).issubset(set(range(barbell_adj.shape[0])))
