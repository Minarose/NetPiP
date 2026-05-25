# Reproducing the NetPiP analysis

This guide walks through end-to-end reproduction of the manuscript figures from the per-subject broadband PSI adjacency matrices. The toolbox itself ([`netpip/`](netpip/), [`matlab/`](matlab/)) is **not** required for reproduction — it is a generic re-implementation of the same algorithm — but using it gives you the cleanest Python-only path.

If you only want to apply PiP to your own data, see [`netpip/README.md`](netpip/README.md) instead.

## 0. Prerequisites

- Python **3.9+**
- MATLAB **R2020b+** (R2024b used in the paper) — only for the Slurm/HPC PiP convergence driver and the BCT-based metric benchmarks. The Python toolbox can substitute for the MATLAB engine if you don't have a MATLAB license.
- **Brain Connectivity Toolbox (BCT, 2019_03_03)** on the MATLAB path — only for the per-subject metric benchmarks in `scripts/compare_pip_bct_percolation_giant75.m`
- **BrainNet Viewer (v1.7 / build 20191031)** on the MATLAB path — only for the final cortical-surface figures in `scripts/render_brainnet_overlap_tifs.m`

A Slurm-managed HPC cluster is used in the paper for the per-subject MATLAB convergence runs; the Python toolbox can run the same algorithm interactively on a laptop for smaller graphs.

## 1. Clone and create environments

```bash
git clone https://github.com/your-org/NetPiP.git
cd NetPiP

# Reproducible Python environment for the analysis (the toolbox is also pip-installable
# on its own; this venv pins the exact stack used to generate the figures)
python3 -m venv analysis/giant_component_avg_nonexcluded/.venv
analysis/giant_component_avg_nonexcluded/.venv/bin/pip install --upgrade pip
analysis/giant_component_avg_nonexcluded/.venv/bin/pip install \
    -r analysis/giant_component_avg_nonexcluded/requirements-lock.txt
analysis/giant_component_avg_nonexcluded/.venv/bin/pip install -e netpip[all,dev]
```

Verify the toolbox installation:

```bash
analysis/giant_component_avg_nonexcluded/.venv/bin/pytest netpip/tests -q
analysis/giant_component_avg_nonexcluded/.venv/bin/python netpip/examples/quickstart.py
```

## 2. Inputs

| What | Where | How produced |
|---|---|---|
| Per-subject broadband PSI binary adjacencies | `data/PSI_broadband_MEG_mats/*_broadband_psi_adj.mat` | Upstream MEG → PSI pipeline (Kadis lab; see manuscript Methods §1). Each MAT contains `psi_adj` (66 × 66, binarized at `|PSInorm| ≥ 2`, L2 across bands). **These files are not redistributed in this repository.** |
| Inclusion table | `results/consensus_5pct/attack_outliers.csv` | Manual QC; rows with `excluded == True` (AD15, AD16) are dropped |
| 66-node AAL labels | `data/MNI_66_AAL_onelinestructure.csv` | One row per node |
| 66-node MNI coordinates | `data/MNI_66_coords.txt` | One row per node, `x y z` in mm |

## 3. Pipeline (step-by-step)

The full pipeline is summarized in [`analysis/giant_component_avg_nonexcluded/README.md`](analysis/giant_component_avg_nonexcluded/README.md). The minimum commands to reproduce the **giant-component group-average leg** are:

```bash
# (a) Per-subject giant-component (≥75%) thresholding
python scripts/make_per_subject_psi_giant75_nonexcluded.py \
    --pip-root data/PSI_broadband_MEG_mats \
    --outlier-csv results/consensus_5pct/attack_outliers.csv

# (b) Group-average giant-component thresholding
python scripts/make_avg_psi_giantcomp_nonexcluded.py \
    --pip-root data/PSI_broadband_MEG_mats \
    --outlier-csv results/consensus_5pct/attack_outliers.csv \
    --out-file data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_giant75_nonexcluded.mat

# (c) PiP convergence on the average graph
#     Either: launch the Slurm job (HPC)
sbatch scripts/slurm_pip_converge_avg_nonexcluded_giant_enforcehw05.sh

#     Or: run the Python port on a single matrix (laptop)
python - <<'PY'
import scipy.io as sio
from netpip import validate_adjacency, run_pip
m = sio.loadmat('data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_giant75_nonexcluded.mat')
A = m['psi_adj'].astype(float)
validate_adjacency(A, min_giant_fraction=0.75)
res = run_pip(A, max_attacks=1_000_000, chunk_size=10_000, seed=12345, enforce_hw95=False)
sio.savemat(
    'data/PSI_broadband_MEG_mats/results_converge_giant_avg_nonexcluded/'
    'AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat',
    {'node_P': res.node_P, 'counts_per_step': res.counts_per_step,
     'part_counts': res.part_counts, 'meta': res.meta}, do_compression=True)
PY

# (d) Per-subject PiP convergence (HPC array job)
sbatch scripts/slurm_pip_converge_giant75_per_subject.sh

# (e) Tilted-peak ranking + cluster hubs + brain figures
python scripts/plot_pip_surfaces.py                # 2D / 3D PiP surfaces
python scripts/avg_percolation_metrics_brain_jaccard.py \
    --perc-csv results/avg_metric_overlap/avg_percolation_points_matlab.csv \
    --top-n-from pip --jaccard-n-from metric

# (f) BCT-based metric benchmarks (MATLAB, requires BCT on path)
matlab -batch "addpath('scripts'); compare_pip_bct_percolation_avg_single"
matlab -batch "addpath('scripts'); compare_pip_bct_percolation_giant75"

# (g) BrainNet Viewer cortical figures (MATLAB, requires BrainNet Viewer on path)
matlab -batch "addpath('scripts'); render_brainnet_overlap_tifs"
```

## 4. Expected outputs

| Figure / table | File |
|---|---|
| Per-subject PiP-vs-metric percolation point (violin + bar) | `results/avg_metric_overlap/avg_percolation_point_bar.png` |
| PiP vs metric Jaccard overlap | `results/avg_metric_overlap/avg_jaccard_bar.png`, `jaccard_topn.txt` |
| PiP / Degree / Betweenness / PageRank top-n brain plots | `results/avg_metric_overlap/brain_top0?_*.png` |
| PiP-vs-metric Venn diagrams | `results/avg_metric_overlap/venn_pip_vs_*.png` |
| BrainNet-rendered overlap TIFs | `results/avg_metric_overlap/brainnet_overlap_pip_vs_*.tif` |
| Average-graph PiP cluster hubs (labeled) | `results/consensus_5pct/avg_giant_nonexcluded_top_cluster_nodes_labeled.csv` |
| Supp. Figure S1 (binary group adjacency) | `analysis/giant_component_avg_nonexcluded/figures/FigureS1_giant75_avg_binary_adjacency.png` |

## 5. Methods text

Paper-ready methods prose is maintained in [`analysis/giant_component_avg_nonexcluded/METHODS.md`](analysis/giant_component_avg_nonexcluded/METHODS.md) and covers: PSI matrix construction, group averaging, giant-component thresholding (per-subject and group), the PiP Monte Carlo engine, convergence, the tilted-peak ranking, Ward + silhouette hub clustering, and the BCT-based metric benchmarks.

## 6. Troubleshooting

- **`netpip` import fails after install**: confirm you installed in the right venv: `which python; which pytest`.
- **Slurm jobs can't find data**: set `PIP_ROOT` and `NETPIP_ROOT` env vars to your data root before `sbatch`.
- **BCT functions not found in MATLAB**: `export BCT_PATH=/path/to/BCT/2019_03_03_BCT` before launching MATLAB, or edit `bctPath` at the top of `scripts/compare_pip_bct_percolation_*.m`.
- **BrainNet Viewer crashes**: install v1.7 build 20191031; set `BRAINNET_PATH` env var; the surface mesh defaults to `Data/SurfTemplate/BrainMesh_ICBM152.nv`.
