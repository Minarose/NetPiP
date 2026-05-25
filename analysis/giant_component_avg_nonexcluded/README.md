# Giant-component (75%) group-average PiP — non-excluded subjects

This folder bundles one coherent analysis leg: **average broadband PSI across included participants, threshold so the largest connected component retains at least 75% of nodes, then participation-in-percolation (PiP) convergence on that group graph**, followed by **figures, clustering, and tabular summaries**.

Under `figures/`, most entries are **relative symlinks** into the main NetPiP tree; **`FigureS1_giant75_avg_binary_adjacency.png`** is generated and stored here. `data/`, `results/`, and `scripts/` use symlinks to canonical repo paths. **`data/per_subject_giant75_nonexcluded/`** links to **`data/PSI_broadband_MEG_mats/per_subject_giant75_nonexcluded/`** (per-subject giant-thresholded MATs from Python). Edit and run jobs from the repository root as usual; paths here are for navigation and documentation.

---

## Python environment (reproducible local plots)

HPC Slurm jobs use shared modules; for **laptop or CI** use the pinned stack in `requirements.txt`:

```bash
cd analysis/giant_component_avg_nonexcluded
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python ../../scripts/plot_figureS1_giant_binary_adjacency.py
```

The venv lives at `analysis/giant_component_avg_nonexcluded/.venv` (gitignored). From the **repository root** you can equivalently run:

```bash
analysis/giant_component_avg_nonexcluded/.venv/bin/pip install -r analysis/giant_component_avg_nonexcluded/requirements.txt
analysis/giant_component_avg_nonexcluded/.venv/bin/python scripts/plot_figureS1_giant_binary_adjacency.py
```

Use that interpreter for leg scripts that need **numpy / scipy / h5py / matplotlib / nilearn** (Figure S1 needs only **numpy, scipy, matplotlib**).

For a **bit-for-bit** replay of the tested stack, install from the committed lockfile (same venv steps as above, but use `-r requirements-lock.txt` instead of `requirements.txt`).

To refresh the lockfile after upgrading packages:

```bash
analysis/giant_component_avg_nonexcluded/.venv/bin/pip freeze > analysis/giant_component_avg_nonexcluded/requirements-lock.txt
```

---

## Pipeline (what produces what)

| Step | Script (Slurm job where applicable) | Primary inputs | Primary outputs |
|------|--------------------------------------|----------------|-----------------|
| 0 — inclusion list | (manual curation) | `data/attack_outliers.csv` (via `results/consensus_5pct/attack_outliers.csv`) | Rows with `excluded != true` and `subject_file` pointing at `*_broadband_psi_adj.mat` |
| 0b — **per-subject** giant (75%) | `make_per_subject_psi_giant75_nonexcluded.py` · `make_per_subject_psi_giant75_nonexcluded_job.sh` | Each included `*_broadband_psi_adj.mat` | **`data/per_subject_giant75_nonexcluded/*_giant75.mat`** (same rule as step 1, on **each** subject graph; symlink under this bundle’s `data/`) |
| 0c — PiP on each giant75 graph (cluster) | `slurm_pip_converge_giant75_per_subject.sh` | `per_subject_giant75_nonexcluded/*_giant75.mat` on `$PIP_ROOT` | **`$PIP_ROOT/results_converge_giant75_per_subject/*_ConvHW.mat`** (edit `#SBATCH --array=1-N` to number of `*_giant75.mat` files) |
| 1 — group average + giant threshold | `scripts/make_avg_psi_giantcomp_nonexcluded.py` · `make_avg_psi_giantcomp_nonexcluded_job.sh` | Per-subject `psi_adj` under `data/PSI_broadband_MEG_mats/`, outlier CSV | `data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_giant75_nonexcluded.mat` — fields include binary `psi_adj`, continuous `avg_psi_adj`, `threshold_value`, `gcc_target`, `density`, `file_names` |
| 2 — PiP convergence | `scripts/slurm_pip_converge_avg_nonexcluded_giant.sh` → MATLAB `pip_converge_posthw5_thresh.m` (Singularity) | AVG `.mat` above as `avg/AVG_broadband_psi_adj_giant75_nonexcluded.mat` | **`results/convergence_matrices/`** (symlink): `*_ConvHW.mat` with `node_P` (steps × nodes), `meta`, etc. Default HPC path: `$PIP_ROOT/results_converge_giant_avg_nonexcluded/` |
| 3 — PiP surfaces (τ = S/6 tilt) | `plot_pip_surfaces.py` inside `plot_avg_nonexcluded_giant_job.sh` | `*_ConvHW.mat` from step 2 | **`figures/pip_surfaces/`** → `tauS6_postHW_giant_avg_nonexcluded/AVG_tauS6_postHW_giant_avg_nonexcluded_{2D,3D}.png` |
| 4 — cluster search on PiP matrix | `plot_avg_cluster.py` in same job | `AVG_*_ConvHW.mat` | **`figures/pip_cluster/`** — e.g. `*_top_cluster_nodes.csv`, `*_cluster_labels.csv`, `*_top_cluster_markers.png` |
| 5 — anatomical labels | `label_cluster_nodes.py` in same job | AAL CSV + top-cluster CSV from step 4 | **`results/avg_giant_nonexcluded_top_cluster_nodes_labeled.csv`** |
| 6 — convergence summary table | `summarize_convergence.py` in same job | Directory of `*_ConvHW.mat` | **`results/avg_giant_nonexcluded_convergence_summary.csv`** |
| 7 — PSI matrix figures (optional / parallel track) | `plot_avg_psi_matrix_nonexcluded_5pct_job.sh` (second Python call) | AVG giant `.mat` | **`figures/psi_matrices/`** — `*_avg_psi_adj.png`, `*_avg_psi_adj_thr.png` |
| Supp. — **Figure S1** (binary group adjacency only) | `plot_figureS1_giant_binary_adjacency.py` | `data/AVG_broadband_psi_adj_giant75_nonexcluded.mat` (`psi_adj`) | **`figures/FigureS1_giant75_avg_binary_adjacency.png`** |
| PiP 2D label order (MATLAB indices) | `export_pip2d_label_order_matlab.py` | `*_ConvHW.mat` | **`results/AVG_giant75_PiP_2d_label_order_matlab_nodes.txt`** (+ `.csv`); batch: `--results-dir $PIP_ROOT/results_converge_5pct --exclude-avg --out-matrix-csv ...` → `readmatrix`-friendly `n×66` |

---

## Directory map (this bundle)

| Path here | Points to |
|-----------|-----------|
| `data/AVG_broadband_psi_adj_giant75_nonexcluded.mat` | Group graph after giant rule |
| `data/per_subject_giant75_nonexcluded/` | Per-subject binary `*_giant75.mat` (symlink → `data/PSI_broadband_MEG_mats/per_subject_giant75_nonexcluded/`) |
| `data/MNI_66_AAL_onelinestructure.csv` | Region labels for step 5 |
| `data/MNI_66_coords.txt` | MNI coordinates for `plot_avg_cluster.py` brain markers |
| `data/attack_outliers.csv` | Inclusion / exclusion metadata |
| `results/convergence_matrices/` | PiP trajectory MAT files (local: under `data/PSI_broadband_MEG_mats/results_converge_giant_avg_nonexcluded/` if mirrored) |
| `results/avg_giant_nonexcluded_*.csv` / `.txt` | Cluster + convergence summaries |
| `figures/pip_surfaces/` | τ=S/6 weighted PiP surfaces (symlink) |
| `figures/pip_cluster/` | Clustering outputs from `plot_avg_cluster.py` (symlink) |
| `figures/psi_matrices/` | Raw vs thresholded average adjacency visuals (symlink) |
| `figures/FigureS1_giant75_avg_binary_adjacency.png` | Supplementary Figure S1 — binary `psi_adj` only (`plot_figureS1_giant_binary_adjacency.py`) |
| `results/AVG_giant75_PiP_2d_label_order_matlab_nodes.txt` | MATLAB 1-based node order for τ=S/6 2D plot ranks 1…k (`export_pip2d_label_order_matlab.py`) |
| `scripts/*.py`, `scripts/*.sh`, `scripts/*.m` | Same files as `../../scripts/` |
| `requirements.txt` | Python dependencies for local plots when not using HPC modules |

---

## Running on the cluster

Slurm wrappers use `NETPIP_ROOT` / `PIP_ROOT` pointing at the SickKids HPF paths. For a local clone, set those to your checkout and data root before `sbatch`, or run the Python steps directly with `--pip-root` / `--results-dir` / `--out-root` as in the job scripts.

Canonical script locations remain `scripts/` at the **repository root**; this bundle only links them.

---

## Related but distinct analyses

- **`top05_nonexcluded`**: fixed proportional (5%) strongest edges on the *average* matrix, not the giant-component rule — different averaging job and convergence directory.
- **Per-subject** PiP: other Slurm scripts under `scripts/` use individual `SUBJECT_FILE` values, not the AVG giant graph.

See **`METHODS.md`** for prose suitable for a paper methods section.
