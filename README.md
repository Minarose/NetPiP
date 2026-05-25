# NetPiP — Participation in Percolation

> **A data-driven measure of network hubs based on Monte Carlo node-removal attacks and percolation-based collapse analysis.**

Brady J. Williamson¹, Minarose Ismail²,³, Darren S. Kadis²,³

¹ University of Cincinnati College of Medicine, Department of Radiology
² Neurosciences and Mental Health, Hospital for Sick Children, Toronto
³ Department of Physiology, University of Toronto

---

This repository hosts both the **reusable toolbox** and the **full reproducible analysis** for the manuscript *"Participation in Percolation: A Data-Driven Measure of Network Hubs in Functional Brain Networks"* (Williamson et al., 202x).

## What's in here

```
NetPiP/
├── netpip/              # Pip-installable Python toolbox
│   ├── src/netpip/      # Public API: validate_adjacency, run_pip, ...
│   ├── tests/           # Unit tests (pytest)
│   ├── examples/        # Quickstart on a synthetic graph
│   └── README.md        # Python install + API
│
├── matlab/              # MATLAB sibling toolbox (`+netpip` namespace)
│   ├── +netpip/         # netpip.validate_adjacency, netpip.run_pip, ...
│   ├── examples/        # MATLAB quickstart
│   └── README.md        # MATLAB install + API
│
├── analysis/            # Full reproducible analysis bundles
│   └── giant_component_avg_nonexcluded/
│       ├── METHODS.md   # Paper-ready methods prose
│       ├── README.md    # Step-by-step pipeline table
│       └── ...
│
├── scripts/             # Original analysis scripts (Slurm + Python + MATLAB)
├── data/                # 66-node MNI coords, AAL labels, attack-outlier table
├── results/             # Generated CSVs / figures (consensus, overlap, jaccard)
├── figures/             # Paper figures
├── REPRODUCING.md       # End-to-end reproduction guide
├── LICENSE              # MIT
└── CITATION.cff         # Citation metadata
```

There are essentially **two ways to use this repository**:

| You want to ... | Read |
|---|---|
| Apply PiP to your own binary adjacency matrices, no MEG data needed | [`netpip/README.md`](netpip/README.md) (Python) or [`matlab/README.md`](matlab/README.md) (MATLAB) |
| Reproduce the manuscript figures from the MEG data | [`REPRODUCING.md`](REPRODUCING.md) and [`analysis/giant_component_avg_nonexcluded/README.md`](analysis/giant_component_avg_nonexcluded/README.md) |

---

## Quickstart (Python)

```bash
pip install -e netpip[networkx]
python netpip/examples/quickstart.py
```

```python
import numpy as np
from netpip import validate_adjacency, run_pip, pip_top_n_at_percolation_point

A = ...                                    # your binary, symmetric, 0-diagonal adjacency
validate_adjacency(A, min_giant_fraction=0.75)

res = run_pip(A, max_attacks=1_000_000, chunk_size=10_000, seed=42)
hub_nodes = pip_top_n_at_percolation_point(A, res.node_P)
```

## Quickstart (MATLAB)

```matlab
addpath('matlab');
A = double(your_binary_symmetric_adjacency);
netpip.validate_adjacency(A, 'MinGiantFraction', 0.75);

res = netpip.run_pip(A, 'MaxAttacks', 1e6, 'ChunkSize', 1e4, 'Seed', 42);
[order, ~, ~] = netpip.tilted_peak_rank(res.node_P);
pp  = netpip.percolation_point(A, order);
hub = order(1:pp);
```

---

## What the toolbox does *not* do

By design, both the Python and MATLAB toolboxes **operate on a pre-built binary adjacency matrix** that you supply. They will validate it (binary, symmetric, zero diagonal, sparse, giant component present) and **error out** if it is malformed — but they will **never modify it**: no thresholding, no binarization, no symmetrization, no giant-component extraction. The PSI / giant-component / averaging pipeline used in the paper lives in `scripts/` and `analysis/` and is described in `REPRODUCING.md`.

This separation is intentional: PiP is a generic graph-theoretic hub measure that applies to any binary undirected network (functional brain networks, structural connectomes, social networks, infrastructure networks, ...), and the toolbox is meant to be a small dependency you can drop into any project.

---

## Citation

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
  title   = {Participation in Percolation: A Data-Driven Measure of Network Hubs
             in Functional Brain Networks},
  year    = {202x},
  journal = {TBD}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
