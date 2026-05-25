"""netpip quickstart: PiP on a small synthetic graph.

Run from the repository root after installing the package:

    pip install -e netpip[networkx]
    python netpip/examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from netpip import (
    betweenness_attack_order,
    degree_attack_order,
    jaccard,
    metric_top_n_at_percolation_point,
    pagerank_attack_order,
    pip_hub_cluster,
    pip_top_n_at_percolation_point,
    run_pip,
    tilted_peak_rank,
    validate_adjacency,
)


def make_barbell(k: int = 4) -> np.ndarray:
    """Two K_k cliques joined by one bridge edge -> obvious 'two-hub' graph."""
    n = 2 * k
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            A[i, j] = A[j, i] = 1.0
    for i in range(k, n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = 1.0
    A[k - 1, k] = A[k, k - 1] = 1.0
    return A


def main() -> None:
    adj = make_barbell(k=4)

    print("Validating input adjacency (read-only)...")
    report = validate_adjacency(adj, min_giant_fraction=0.75)
    print(" ", report.summary())

    print("\nRunning PiP Monte Carlo (small budget for the demo)...")
    res = run_pip(
        adj,
        max_attacks=2_000,
        chunk_size=500,
        seed=0,
        enforce_hw95=False,
        progress=lambda d: print(
            f"  chunk @ n_attacks={d['n_attacks']:>5} "
            f"hw95={d['hw95']:.4f} plateau={int(d['is_plateau'])} stable={d['stable_count']}"
        ),
    )
    print(
        f"  attacks={res.n_attacks}  converged={res.converged}  "
        f"elapsed={res.elapsed_seconds:.2f}s"
    )

    print("\nTilted-peak ranking (early-percolation hub-first):")
    order, peak_step, peak_amp = tilted_peak_rank(res.node_P)
    for rank, (idx, st, amp) in enumerate(zip(order, peak_step[order], peak_amp[order]), 1):
        print(f"  rank {rank:>2}  node {int(idx):>2}  peak_step={int(st):>2}  amp={float(amp):.4f}")

    print("\nPiP hub set (top-n at PiP percolation point):")
    pip_top = pip_top_n_at_percolation_point(adj, res.node_P)
    print(f"  nodes (0-based): {pip_top.tolist()}  (n = {pip_top.size})")

    print("\nPiP hub cluster (Ward + silhouette):")
    hub = pip_hub_cluster(res.node_P)
    print(f"  k = {hub.k}  silhouette = {hub.silhouette:.4f}")
    print(f"  hub cluster label = {hub.hub_cluster}  hub nodes = {hub.hub_nodes.tolist()}")

    print("\nClassical-metric benchmarks:")
    rng = np.random.default_rng(0)
    for name, order_fn in [
        ("Degree", degree_attack_order),
        ("Betweenness", betweenness_attack_order),
        ("PageRank", pagerank_attack_order),
    ]:
        m_order = order_fn(adj, rng=rng)
        m_top, m_pp, _ = metric_top_n_at_percolation_point(adj, m_order)
        j = jaccard(pip_top.tolist(), m_top.tolist())
        print(
            f"  {name:<12}  perc_point={m_pp:>2}  top={m_top.tolist()}  "
            f"Jaccard(PiP, {name}) = {j:.3f}"
        )


if __name__ == "__main__":
    main()
