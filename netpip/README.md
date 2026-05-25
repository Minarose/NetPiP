# netpip

**Participation in Percolation (PiP)** — a Python toolbox for identifying network hubs based on Monte Carlo node-removal attacks and percolation-based collapse analysis. Companion package to the manuscript *"Participation in Percolation: A Data-Driven Measure of Network Hubs in Functional Brain Networks"* (Williamson et al.).

This package gives you a clean, pip-installable Python API for the same PiP algorithm used in the paper. The full reproducible analysis (MEG data, MATLAB convergence engine, paper figures) lives in the parent repository under `analysis/` and `scripts/`.

---

## Install

From PyPI (when published):

```bash
pip install netpip
```

From this repository (recommended while in development):

```bash
git clone https://github.com/your-org/NetPiP.git
cd NetPiP
pip install -e netpip                  # core (numpy + scipy)
pip install -e "netpip[networkx]"      # adds Degree / Betweenness / PageRank benchmarks
pip install -e "netpip[all]"           # adds nilearn / matplotlib / h5py / matplotlib-venn
```

Python **3.9+** is required. The core engine depends only on **NumPy** and **SciPy**; everything else is optional.

---

## Input contract

`netpip` operates on a **pre-built binary adjacency matrix** that you supply. It validates the matrix but **never modifies it**. Your matrix must already be:

- a 2D NumPy array of shape `(N, N)` with `N >= 2`;
- finite (no NaN/inf);
- **binary** (entries are exactly 0 or 1);
- **symmetric** (`A == A.T`);
- **zero-diagonal** (no self-loops);
- **sparse** (not the complete graph); and
- characterized by a **giant connected component** of at least `min_giant_fraction` of `N` nodes (default 0.5).

If any check fails, `validate_adjacency` raises `AdjacencyValidationError` with a precise diagnostic. Use your own preprocessing code to make the matrix conform — `netpip` never thresholds, binarizes, or symmetrizes for you.

```python
from netpip import validate_adjacency

report = validate_adjacency(A, min_giant_fraction=0.75)
print(report.summary())
# AdjacencyReport(n_nodes=66, n_edges=80, density=0.0373,
#                 giant_component_size=51 (77.3% of nodes))
```

---

## Quickstart

```python
import numpy as np
from netpip import (
    validate_adjacency,
    run_pip,
    tilted_peak_rank,
    pip_top_n_at_percolation_point,
    pip_hub_cluster,
)

# Your binary, symmetric, zero-diagonal adjacency matrix (66x66 in the paper)
A = ...                                          # np.ndarray, shape (N, N)
validate_adjacency(A, min_giant_fraction=0.75)

# Monte Carlo PiP convergence
res = run_pip(
    A,
    max_attacks=1_000_000,    # hard cap
    chunk_size=10_000,        # plateau check between chunks
    seed=42,
    enforce_hw95=False,       # paper default for the giant-component leg
)

# Time-tilted ranking (τ = S/6, negative clipped)
order, peak_step, peak_amp = tilted_peak_rank(res.node_P)

# Hub set defined as PiP top-n at the PiP percolation point
hub_nodes = pip_top_n_at_percolation_point(A, res.node_P)

# Or: Ward + silhouette hub cluster on the time-tilted PiP trajectories
hub = pip_hub_cluster(res.node_P)
print(hub.hub_nodes)
```

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end demo on a small synthetic graph.

---

## Benchmarking against classical centralities

```python
import numpy as np
from netpip import (
    degree_attack_order, betweenness_attack_order, pagerank_attack_order,
    metric_top_n_at_percolation_point, jaccard, pip_top_n_at_percolation_point,
)

rng = np.random.default_rng(0)
pip_top = pip_top_n_at_percolation_point(A, res.node_P)
deg_order = degree_attack_order(A, rng=rng)
deg_top, deg_pp, _ = metric_top_n_at_percolation_point(A, deg_order)
print(f"PiP n={pip_top.size}  Degree n={deg_pp}  J = {jaccard(pip_top, deg_top):.3f}")
```

The benchmarking helpers require the optional `networkx` extra:

```bash
pip install "netpip[networkx]"
```

---

## Public API

| Module | Exported symbol | Purpose |
|---|---|---|
| `netpip.validation` | `validate_adjacency`, `AdjacencyReport`, `AdjacencyValidationError` | Read-only input checks |
| `netpip.core` | `run_pip`, `PiPResult` | Monte Carlo PiP engine |
| `netpip.convergence` | `wilson_half_width`, `plateau_reached` | Convergence diagnostics |
| `netpip.ranking` | `tilted_peak_rank`, `pip_top_n_at_percolation_point`, `percolation_point` | Hub ranking |
| `netpip.clustering` | `pip_hub_cluster`, `ward_silhouette_cluster` | Hub-cluster identification |
| `netpip.metrics` | `degree_attack_order`, `betweenness_attack_order`, `pagerank_attack_order`, `metric_top_n_at_percolation_point`, `jaccard` | Classical-centrality benchmarks |

All public objects are re-exported from `netpip` itself, so `from netpip import run_pip` works.

---

## How `node_P` is defined

The Monte Carlo engine accumulates two raw count tensors and one normalized matrix:

```
counts_per_step[p]   = # attacks whose percolation step (1-based) equals p
part_counts[p, i]    = # attacks where (perc_step == p) AND
                       (node i was among the first p nodes removed)

node_P[p, i]         = part_counts[p, i] / counts_per_step[p]
                     = P(node i was removed by step p | percolation step == p)
```

This is exactly the `node_P` saved by the MATLAB driver
`scripts/pip_converge_posthw5_thresh.m` in the parent repository, up to Monte
Carlo sampling noise (the two engines use different RNGs).

---

## Reproducing the paper

For the full reproducible analysis (group-average construction from per-subject MEG PSI matrices, giant-component thresholding rule, MATLAB convergence engine with Slurm wrappers, paper figures, BrainNet Viewer rendering), see the parent repository — in particular `analysis/giant_component_avg_nonexcluded/README.md` and `REPRODUCING.md` at the repo root.

---

## License

MIT. See [`LICENSE`](../LICENSE) at the repository root.

---

## Citation

If you use `netpip` in academic work, please cite both the package and the paper:

```bibtex
@software{netpip,
  author  = {Ismail, Minarose and Williamson, Brady J. and Kadis, Darren S.},
  title   = {netpip: Participation in Percolation for Network Hub Identification},
  year    = {2025},
  url     = {https://github.com/your-org/NetPiP},
  version = {0.1.0}
}

@article{williamson_pip,
  author  = {Williamson, Brady J. and Ismail, Minarose and Kadis, Darren S.},
  title   = {Participation in Percolation: A Data-Driven Measure of Network Hubs in Functional Brain Networks},
  year    = {202x},
  journal = {TBD}
}
```
