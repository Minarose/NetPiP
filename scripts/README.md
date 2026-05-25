# Scripts — paper-reproduction pipeline

Scripts are numbered in pipeline order. Run them top-to-bottom (or use the matching `slurm/` wrappers on an HPC cluster) to reproduce every figure and CSV used in the paper. See [`../REPRODUCING.md`](../REPRODUCING.md) for the full reproduction guide.

| # | File | Step | Output |
|---|------|------|--------|
| 1 | `1_make_giant75_per_subject.py` | Threshold per-subject PSI matrices at 75 % giant-component density | `data/PSI_broadband_MEG_mats/individual/AD??_..._giant75.mat`                   |
| 1 | `1_make_giant75_avg.py`         | Build the cohort-average giant-75 binary graph                     | `data/PSI_broadband_MEG_mats/group_average/AVG_..._giant75_nonexcluded.mat`     |
| 1 | `1_threshold_giant75.m`         | MATLAB equivalent of the per-subject thresholder                   | same as `1_make_giant75_per_subject.py`                                         |
| 2 | `2_pip_converge.m`              | PiP Monte Carlo + Wilson 95 % plateau convergence (HPC-aware)      | `results/pip_convergence/{avg_giant75,per_subject_giant75}/*_ConvHW.mat`        |
| 3 | `3_plot_pip_surfaces.py`        | 2D / 3D τ = S/6 tilted PiP surfaces (weighted-matrix view)         | `figures/pip_surfaces/avg_giant75/`                                             |
| 4 | `4_cluster_pip_set.py`          | Ward + silhouette clustering → PiP hub set (group average)         | `figures/pip_cluster/avg_giant75/`, `results/pip_cluster/avg_giant75_top_cluster_nodes*.csv` |
| 4 | `4_consensus_cluster_per_subject.py` | Per-subject consensus cluster                                  | `figures/pip_cluster/per_subject/`                                              |
| 5 | `5_graph_theory_avg.m`          | Degree / Betweenness / PageRank percolation-point benchmark (AVG) | `results/graph_theory_overlap/avg_percolation_points_matlab.csv`                |
| 5 | `5_graph_theory_per_subject.m`  | Same benchmark, per subject + group stats                          | `results/graph_theory_overlap/`                                                 |
| 6 | `6_export_pip_order.py`         | Export PiP node order to MATLAB-readable CSV(s)                    | `results/pip_cluster/AVG_giant75_PiP_2d_label_order_matlab*.csv` / per-subject matrix CSV |
| 6 | `6_jaccard_overlap.py`          | Jaccard, Venn diagrams, nilearn brain plots                        | `results/graph_theory_overlap/*.png` + `jaccard_topn.txt`                       |
| 6 | `6_render_brainnet_overlap.m`   | BrainNet Viewer cortical-surface TIFs                              | `results/graph_theory_overlap/brainnet_overlap_*.tif`                           |

## Subdirectories

- `helpers/` — reusable utilities and standalone validations:
  - `pip_plot_utils.py` — shared PiP loader + tilted-peak ranking (imported by `3_*`, `6_*`)
  - `label_cluster_nodes.py` — map cluster node indices → AAL labels
  - `summarize_convergence.py` — convergence-history summary table
  - `plot_convergence_trace.py` — single PDF of HW95 traces
  - `plot_figureS1_giant_binary_adjacency.py` — Supplementary Fig. S1
  - `validate_pip_cluster_drives_percolation.m` — sanity check that the PiP cluster nodes alone drive percolation
- `slurm/` — Slurm job wrappers using the same number prefixes as the canonical scripts (e.g. `2_pip_converge_avg.sh`, `3_plot_pip_surfaces.sh`).
