# Running PiP on your own network

## Input

PiP needs a **binary, undirected** network without self-loops:

| You have | Use |
|---|---|
| Binary matrix in `.npy`, `.npz`, `.csv`, `.txt` or `.mat` | `netpip.load_adjacency(path)` |
| Edge list (two integer columns, 1-based by default) | `netpip.load_edge_list(path, n_nodes=N)` (set `n_nodes` to keep isolated nodes) |
| Weighted matrix (e.g. correlations, coupling strengths) | `netpip.binarize(W, threshold)`, or `--threshold` / `--density` in `analyze_network.py` |

Disconnected networks are fine: PiP keeps every node, so all fractions are
relative to the full node count N. The paper's networks are sparse (densities
0.4–5.8%); in very dense binarised networks few nodes stand out and the
percolation point arrives late, so choose thresholds with that in mind.

## Scripts

* `quickstart.py`: karate club network in a few seconds: scores, ranking,
  convergence check and a comparison with degree, betweenness and PageRank.
* `analyze_network.py`: the full workflow for your own file, saving
  `node_scores.csv`, `rms_convergence.csv`, `attack_comparison.csv`,
  `pip_result.npz` and two figures.

```bash
python examples/quickstart.py
python examples/analyze_network.py examples/example_data/karate_adjacency.csv
python examples/analyze_network.py examples/example_data/karate_edges.csv --edge-list --n-nodes 34
python examples/analyze_network.py examples/example_data/weighted_modular_matrix.csv --threshold 0.5 --attacks 1e7
```

## Choosing the number of attacks

Start with 10^6 to 10^7 attacks, and pass checkpoints (`analyze_network.py`
does this automatically). If `rms_convergence` has not dropped below 0.005 by
the last checkpoint, sample more. Reuse a seed to reproduce a run exactly.

## Interpreting the output

* **PiP score** (linear weighting): how often a node has already been removed
  when the network first splits, weighted towards early splits. Rank nodes by it.
* **PiP surface** (exponential weighting, for display): participation by node
  and removal step. Tall early peaks mark the nodes that drive early fragmentation.
* **Attack comparison:** removals needed to reach the percolation point when
  nodes are removed in PiP order versus by degree, betweenness or PageRank.
  Lower means an earlier split. PiP is not guaranteed to win on every network;
  in the karate example, centralities split the network one removal sooner.
* **Peaks at rarely observed steps** rest on few attacks. Check
  `result.counts[k-1]` before interpreting them.

## Example data

* `karate_edges.csv`, `karate_adjacency.csv`: Zachary's karate club (34 nodes, 78 edges).
* `weighted_modular_matrix.csv`: synthetic 40-node weighted matrix with four
  modules (edges have weights 0.5–1, non-edges 0–0.45).
