# Reproducing the manuscript analysis

This guide reproduces the canonical pipeline reported in the paper:
**75 % giant-component binary PSI matrix → PiP → weighted-matrix / cluster hub identification → graph-theory benchmark (Degree, Betweenness, PageRank) → Jaccard overlap and brain renders.**

Everything else (5 %-threshold experiments, airports/yeast network experiments, exploratory variants) was removed from this repository. If you only want the toolbox API (no MEG data), see [`netpip/README.md`](netpip/README.md) / [`matlab/README.md`](matlab/README.md) instead.

## 1. Prerequisites

- **Python 3.11** (recommended; the Slurm wrappers in `scripts/slurm/` load `python/3.11.3`). Python 3.10+ should also work for the scripts; 3.9+ for the `netpip` toolbox itself.
- **MATLAB R2022b+** for the convergence engine, graph-theory benchmark, and BrainNet renders.
- **Brain Connectivity Toolbox (BCT, 2019_03_03)** on the MATLAB path — required only for the graph-theory benchmark in `scripts/5_graph_theory_*.m`. Set `BCT_PATH` env var or edit the `bctPath` line inside those scripts.
- **BrainNet Viewer (v1.7 / build 20191031)** on the MATLAB path — required only for the cortical-surface figures in `scripts/6_render_brainnet_overlap.m`. Set `BRAINNET_PATH` env var if it isn't auto-detected.
- **Python deps for the analysis scripts**: `numpy`, `scipy`, `h5py`, `matplotlib`, `pandas`, `seaborn`, `networkx`, `nilearn`, `matplotlib_venn`. The easiest way is to install the toolbox extras:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e netpip[all,dev]
  pip install nilearn matplotlib_venn seaborn pandas
  ```

Run the toolbox unit tests + quickstart to make sure your environment is sane:
```bash
pytest netpip/tests -q
python netpip/examples/quickstart.py
```

## 2. Input data

The repository ships everything you need to reproduce the paper figures *from the giant-75 matrices forward*:

| Item | Path | Notes |
|---|---|---|
| Per-subject giant-75 input | `data/PSI_broadband_MEG_mats/individual/AD??_..._giant75.mat` | **Pipeline starts here** for per-subject analyses |
| Group-average giant-75 input | `data/PSI_broadband_MEG_mats/group_average/AVG_broadband_psi_adj_giant75_nonexcluded.mat` | **Pipeline starts here** for group-average analyses |
| 66-node MNI coords | `data/MNI_66_coords.txt` | Used by `3_plot_pip_surfaces.py` + brain plots |
| AAL labels | `data/MNI_66_AAL_onelinestructure.csv` | Used by `helpers/label_cluster_nodes.py` |
| Inclusion table | `results/pip_cluster/attack_outliers.csv` | Manual QC; rows with `excluded == True` (AD15, AD16) are dropped from group analyses |

## 3. The pipeline

Each step has a numbered script. On HPC, use the matching wrapper under `scripts/slurm/`; on a workstation, run the Python/MATLAB driver directly.

### Step 1 — build the giant-75 inputs (optional rebuild from raw PSI)
The giant-75 matrices are already in `data/`, so step 1 can be skipped for reproduction. If you want to rebuild them from your own raw `*_broadband_psi_adj.mat` PSI matrices, drop them in `data/PSI_broadband_MEG_mats/` and run:

```bash
python scripts/1_make_giant75_per_subject.py \
    --indir  data/PSI_broadband_MEG_mats \
    --outdir data/PSI_broadband_MEG_mats/individual \
    --outlier-csv results/pip_cluster/attack_outliers.csv

python scripts/1_make_giant75_avg.py \
    --indir  data/PSI_broadband_MEG_mats \
    --outlier-csv results/pip_cluster/attack_outliers.csv \
    --out-mat data/PSI_broadband_MEG_mats/group_average/AVG_broadband_psi_adj_giant75_nonexcluded.mat
```

`scripts/1_threshold_giant75.m` is the MATLAB equivalent of the per-subject thresholder.

### Step 2 — run PiP convergence

```bash
# group average
sbatch scripts/slurm/2_pip_converge_avg.sh

# per-subject (one array job over the 19 non-excluded subjects)
sbatch scripts/slurm/2_pip_converge_per_subject.sh
```

Outputs land in:
- `results/pip_convergence/avg_giant75/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat`
- `results/pip_convergence/per_subject_giant75/AD??_broadband_psi_adj_giant75_ConvHW.mat`

Each `*_ConvHW.mat` contains the converged `node_P` matrix (steps × nodes) plus Wilson 95 % half-width history. These files are checked into the repo so you can skip step 2 entirely if you only want the downstream analyses.

### Step 3 — plot the weighted PiP matrix (PiP surfaces)
```bash
sbatch scripts/slurm/3_plot_pip_surfaces.sh
# or, directly:
python scripts/3_plot_pip_surfaces.py \
  --results-dir results/pip_convergence/avg_giant75 \
  --out-root    figures/pip_surfaces/avg_giant75 \
  --tag postHW_giant_avg_nonexcluded \
  --include-prefix AVG
```

### Step 4 — cluster the PiP trajectories to identify the hub set
```bash
python scripts/4_cluster_pip_set.py \
  --pip-mat results/pip_convergence/avg_giant75/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat \
  --out-dir figures/pip_cluster/avg_giant75 \
  --k-min 2 --k-max 10 \
  | tee results/pip_cluster/avg_giant75_top_cluster_nodes.txt

python scripts/helpers/label_cluster_nodes.py \
  --labels-csv  data/MNI_66_AAL_onelinestructure.csv \
  --indices-csv figures/pip_cluster/avg_giant75/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW_top_cluster_nodes.csv \
  --out-csv     results/pip_cluster/avg_giant75_top_cluster_nodes_labeled.csv
```

Per-subject consensus cluster:
```bash
python scripts/4_consensus_cluster_per_subject.py \
  --results-dir results/pip_convergence/per_subject_giant75 \
  --out-dir     figures/pip_cluster/per_subject
```

### Step 5 — graph-theory benchmark (Degree, Betweenness, PageRank)
```matlab
addpath('scripts');
run('scripts/5_graph_theory_avg.m');          % single AVG graph
run('scripts/5_graph_theory_per_subject.m');  % all 19 subjects + group stats
```
Writes `results/graph_theory_overlap/avg_percolation_points_matlab.csv` and per-subject MAT/CSV summaries. BCT must be on the path (set `BCT_PATH` env var).

### Step 6 — Jaccard overlap, Venn diagrams, BrainNet renders
```bash
# Export the AVG PiP attack order to a MATLAB-readable CSV
python scripts/6_export_pip_order.py \
  --conv-mat results/pip_convergence/avg_giant75/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat \
  --out-csv  results/pip_cluster/AVG_giant75_PiP_2d_label_order_matlab.csv \
  --out-txt  results/pip_cluster/AVG_giant75_PiP_2d_label_order_matlab_nodes.txt

# Per-subject PiP-order matrix (used by 5_graph_theory_per_subject.m)
python scripts/6_export_pip_order.py \
  --results-dir results/pip_convergence/per_subject_giant75 \
  --exclude-avg \
  --out-matrix-csv results/pip_convergence/per_subject_giant75/giant75_per_subject_pip2d_order_matlab_noheader.csv

# Python-side overlap analysis (PiP vs Degree/Betweenness/PageRank)
python scripts/6_jaccard_overlap.py \
  --perc-csv results/graph_theory_overlap/avg_percolation_points_matlab.csv \
  --top-n-from metric
```
Then in MATLAB, with BrainNet Viewer on the path:
```matlab
run('scripts/6_render_brainnet_overlap.m');
```

## 4. Expected outputs

After running steps 2 – 6 you should see:

| Output | Path |
|---|---|
| Converged PiP `node_P` (avg + 19 subjects) | `results/pip_convergence/avg_giant75/`, `results/pip_convergence/per_subject_giant75/` |
| PiP surfaces (2D / 3D weighted-matrix views) | `figures/pip_surfaces/avg_giant75/` |
| PiP cluster figures + per-node CSV | `figures/pip_cluster/avg_giant75/` |
| Average-graph PiP cluster hubs (labeled) | `results/pip_cluster/avg_giant75_top_cluster_nodes_labeled.csv` |
| Convergence summary | `results/pip_cluster/avg_giant75_convergence_summary.csv` |
| Per-subject PiP vs metric percolation-point bar/violin | `results/graph_theory_overlap/avg_percolation_point_bar.png` |
| PiP vs metric Jaccard overlap | `results/graph_theory_overlap/avg_jaccard_bar.png`, `jaccard_topn.txt` |
| Top-n brain plots (PiP / Degree / Betweenness / PageRank) | `results/graph_theory_overlap/brain_top0?_*.png` |
| PiP-vs-metric Venn diagrams | `results/graph_theory_overlap/venn_pip_vs_*.png` |
| BrainNet-rendered overlap TIFs | `results/graph_theory_overlap/brainnet_overlap_pip_vs_*.tif` |
| Supplementary Fig. S1 (binary giant-75 group adjacency) | `figures/FigureS1_giant75_avg_binary_adjacency.png` |

## 5. Troubleshooting

- **`results_converge_*` paths in old code/notebooks** — these moved to `results/pip_convergence/`. The numbered scripts and Slurm wrappers in this repo are already updated.
- **MATLAB v7.3 (HDF5) MAT files** — `scripts/3_plot_pip_surfaces.py`, `scripts/6_export_pip_order.py`, and `scripts/6_jaccard_overlap.py` use a `load_pip_any` helper that falls back to `h5py`. Make sure `h5py` is installed.
- **BrainNet renders fail** — make sure `BRAINNET_PATH` points to a folder containing `BrainNet_MapCfg.m`. The default surface is `BrainMesh_ICBM152.nv` under `BRAINNET_PATH/Data/SurfTemplate/`; override with `BRAINNET_SURFACE_NV` if needed.
- **`from helpers.pip_plot_utils import ...` fails** — the numbered scripts inject `scripts/` into `sys.path` at the top; if you copied a script outside the repo, copy `scripts/helpers/pip_plot_utils.py` along with it, or import directly from the `netpip` toolbox.
