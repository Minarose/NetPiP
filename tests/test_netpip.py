"""Tests for the netpip toolbox. Run with:  python -m pytest"""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import netpip
from netpip import _kernels

ROOT = Path(__file__).resolve().parents[1]


def graphs():
    yield nx.path_graph(7)
    yield nx.star_graph(6)
    yield nx.barbell_graph(5, 2)
    yield nx.karate_club_graph()
    g = nx.gnm_random_graph(20, 22, seed=3)  # disconnected, with isolates
    yield g
    yield nx.watts_strogatz_graph(30, 4, 0.1, seed=1)


def adj(g):
    return nx.to_numpy_array(g, nodelist=sorted(g), dtype=np.uint8, weight=None)


def brute_sizes(A, removed):
    g = nx.from_numpy_array(A)
    g.remove_nodes_from(removed)
    sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True) + [0, 0]
    return sizes[0], sizes[1]


def brute_event(A, order):
    """Paper definition: first k in 1..N-1 maximising the SLCC (>1); otherwise N."""
    n = len(A)
    second = [brute_sizes(A, order[:k])[1] for k in range(1, n)]
    return int(np.argmax(second)) + 1 if max(second) > 1 else n


@pytest.mark.parametrize("g", list(graphs()))
def test_event_kernel_matches_brute_force(g):
    A = adj(g)
    indptr, indices = netpip.to_csr(A)
    rng = np.random.default_rng(0)
    orders = netpip.random_orders(rng, len(A), 15)
    events = _kernels.batch_event_steps(orders, indptr, indices)
    assert events.tolist() == [brute_event(A, o) for o in orders]


@pytest.mark.parametrize("g", list(graphs()))
def test_component_trace_matches_brute_force(g):
    A = adj(g)
    order = np.random.default_rng(1).permutation(len(A))
    lcc, slcc = netpip.component_trace(A, order)
    for k in range(len(A) + 1):
        assert (lcc[k], slcc[k]) == brute_sizes(A, order[:k])


def test_counts_are_consistent_and_reproducible():
    A = adj(nx.karate_club_graph())
    r1 = netpip.run_pip(A, 3000, seed=7, batch_size=250, checkpoints=[1000, 2000])
    r2 = netpip.run_pip(A, 3000, seed=7, batch_size=1000)
    assert int(r1.counts.sum()) == 3000
    n = len(A)
    np.testing.assert_array_equal(r1.hits.sum(1), r1.counts * np.arange(1, n + 1, dtype=np.uint64))
    np.testing.assert_array_equal(r1.counts, r2.counts)  # batch size does not change the result
    np.testing.assert_array_equal(r1.hits, r2.hits)
    assert sorted(r1.snapshots) == [1000, 2000]
    assert int(r1.snapshots[1000][0].sum()) == 1000


def test_counts_match_direct_simulation():
    A = adj(nx.barbell_graph(4, 1))
    res = netpip.run_pip(A, 500, seed=11, batch_size=500)
    orders = netpip.random_orders(np.random.default_rng(11), len(A), 500)
    counts = np.zeros(len(A), int)
    hits = np.zeros((len(A), len(A)), int)
    for o in orders:
        k = brute_event(A, o)
        counts[k - 1] += 1
        hits[k - 1, o[:k]] += 1
    np.testing.assert_array_equal(res.counts, counts)
    np.testing.assert_array_equal(res.hits, hits)


def test_probability_mean_is_k_over_n():
    A = adj(nx.karate_club_graph())
    res = netpip.run_pip(A, 2000, seed=2)
    steps = netpip.observed_steps(res.counts)
    P = netpip.pip_probability(res.counts, res.hits, steps)
    np.testing.assert_allclose(P.mean(1), steps / len(A))


def test_step_weights_and_retained_steps():
    counts = np.array([5, 0, 3, 4, 1, 0, 2])
    assert netpip.retained_steps(counts).tolist() == [3, 4, 5]
    assert netpip.retained_steps(counts, "all_observed").tolist() == [1, 3, 4, 5, 7]
    np.testing.assert_allclose(netpip.step_weights([3, 4, 5], kind="linear"), [1, 2 / 3, 1 / 3])
    np.testing.assert_allclose(netpip.step_weights([3, 4, 5], kind="exponential"), np.exp(-np.arange(1, 4) / 2))
    np.testing.assert_allclose(netpip.step_weights([1, 3], 75, "linear", "absolute"), [1, 73 / 75])
    np.testing.assert_allclose(netpip.step_weights([1, 3], 75, "exponential", "absolute"),
                               np.exp(-np.array([1, 3]) / 12.5))


def test_barbell_bridge_ranks_first():
    g = nx.barbell_graph(6, 1)  # two cliques joined through one bridge node
    res = netpip.run_pip(adj(g), 20000, seed=3)
    assert res.ranking()[0] == 6


def test_rms_convergence_decreases():
    A = adj(nx.karate_club_graph())
    res = netpip.run_pip(A, 40000, seed=5, checkpoints=[1000, 4000, 16000])
    rows = res.rms_convergence()
    assert [r["attacks"] for r in rows] == [4000, 16000, 40000]
    assert rows[-1]["rms"] < rows[0]["rms"]


def test_percolation_point_convention():
    # Intact graph with two components of size 3 and 2: SLCC is largest at k = 0.
    lcc = np.array([3, 2, 2, 1, 1, 0])
    slcc = np.array([2, 2, 1, 1, 0, 0])
    assert netpip.percolation_point(lcc, slcc) == 0
    assert netpip.percolation_point(np.array([3, 2, 1, 0]), np.array([0, 0, 0, 0])) is None


def test_attack_orders_ties():
    orders = netpip.attack_orders([3, 1, 3, 2], n_tie_orders=50, seed=1)
    assert orders.shape == (50, 4)
    assert set(map(tuple, orders[:, :2])) == {(0, 2), (2, 0)}
    assert netpip.attack_orders([4, 3, 2, 1]).shape == (1, 4)


def test_critical_cluster_two_groups():
    X = np.r_[np.full((3, 4), 1.0), np.zeros((7, 4))] + np.random.default_rng(0).normal(0, 0.01, (10, 4))
    out = netpip.critical_cluster(X)
    assert out["k"] == 2 and out["critical_nodes"].tolist() == [0, 1, 2]


def test_validation_errors(tmp_path):
    with pytest.raises(ValueError):
        netpip.validate_adjacency(np.array([[0, 2, 0], [2, 0, 0], [0, 0, 0]]))
    with pytest.raises(ValueError):
        netpip.validate_adjacency(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]))
    W = np.array([[0, 0.9, 0.1], [0.8, 0, 0.05], [0.1, 0.05, 0]])
    assert netpip.binarize(W, 0.5).tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    edges = tmp_path / "edges.csv"
    edges.write_text("source,target\n1,2\n2,3\n")
    assert netpip.load_edge_list(edges, n_nodes=4).sum() == 4


def test_save_and_load(tmp_path):
    A = adj(nx.karate_club_graph())
    res = netpip.run_pip(A, 1000, seed=1, checkpoints=[500])
    res.save(tmp_path / "r.npz")
    back = netpip.PiPResult.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(back.hits, res.hits)
    assert back.seed == 1 and 500 in back.snapshots


@pytest.mark.skipif(not (ROOT / "results/language/snapshots/1000000.npz").exists(),
                    reason="paper results not present")
def test_reproduces_paper_language_checkpoint():
    """Same seed and sampler as the paper: the first 1M attacks match the saved checkpoint exactly."""
    n = 66
    A = netpip.load_edge_list(ROOT / "data/networks/language/edges.csv", n_nodes=n)
    res = netpip.run_pip(A, 1_000_000, seed=2026092409)
    with np.load(ROOT / "results/language/snapshots/1000000.npz") as z:
        np.testing.assert_array_equal(res.counts, z["counts"])
        np.testing.assert_array_equal(res.hits, z["hits"])
