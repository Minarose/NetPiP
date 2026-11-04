"""Quick start: PiP on Zachary's karate club network in a few seconds.

Run:  python examples/quickstart.py
"""
from pathlib import Path

import netpip

HERE = Path(__file__).resolve().parent

# 1. Load a binary, undirected network (edge list with 1-based node labels).
A = netpip.load_edge_list(HERE / "example_data" / "karate_edges.csv", n_nodes=34)

# 2. Sample random attacks. The paper used 1e9; 1e6 is plenty for a 34-node graph.
result = netpip.run_pip(A, n_attacks=1_000_000, seed=1, checkpoints=[10_000, 100_000])

# 3. Score and rank nodes: mean linear-weighted participation in early fragmentation.
scores = result.scores("linear")
top = result.ranking("linear")[:5]
print("Most critical nodes (1-based):", (top + 1).tolist())
print("Their PiP scores:", scores[top].round(4).tolist())

# 4. Has sampling converged? RMS change between checkpoints (paper threshold: 0.005).
for row in result.rms_convergence():
    print(f"RMS change at {row['attacks']:>9,} attacks: {row['rms']:.5f}")

# 5. Does a PiP-ordered attack split the network sooner than degree, betweenness or PageRank?
orders = {"PiP": netpip.attack_orders(scores)}
orders.update({name: netpip.attack_orders(s) for name, s in netpip.centrality_scores(A).items()})
for name, o in orders.items():
    stats = netpip.evaluate_attacks(A, o)
    print(f"{name:12s} reaches the percolation point after {stats['peak_removals']:.2f} removals")
