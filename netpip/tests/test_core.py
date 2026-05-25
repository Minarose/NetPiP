from __future__ import annotations

import numpy as np

from netpip import run_pip, tilted_peak_rank


def test_run_pip_shapes_and_finiteness(barbell_adj):
    res = run_pip(
        barbell_adj,
        max_attacks=200,
        chunk_size=50,
        seed=0,
    )
    n = barbell_adj.shape[0]
    assert res.node_P.shape == (n, n)
    assert res.counts_per_step.shape == (n,)
    assert res.part_counts.shape == (n, n)
    assert res.n_attacks == 200
    finite = res.node_P[np.isfinite(res.node_P)]
    assert finite.size > 0
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_pip_identifies_bridge_nodes_in_barbell(barbell_adj):
    """Nodes 3 and 4 are the bridge: PiP should rank them at the top."""
    res = run_pip(barbell_adj, max_attacks=1000, chunk_size=200, seed=42)
    order, _, _ = tilted_peak_rank(res.node_P)
    top2 = set(order[:2].tolist())
    assert top2 == {3, 4}, f"Expected bridge nodes {{3, 4}} at top; got {order[:4].tolist()}"


def test_engine_does_not_mutate_input(barbell_adj):
    before = barbell_adj.copy()
    run_pip(barbell_adj, max_attacks=100, chunk_size=50, seed=0)
    np.testing.assert_array_equal(barbell_adj, before)


def test_convergence_history_lengths(barbell_adj):
    res = run_pip(barbell_adj, max_attacks=300, chunk_size=100, seed=0)
    assert res.attacks_hist.shape == res.hw95_hist.shape
    assert res.attacks_hist[-1] == 300
