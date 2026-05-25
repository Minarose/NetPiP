#!/bin/bash
#SBATCH --job-name=avgGiantNEhw05Plots
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/avg_giant_nonexcluded_enforcehw05_plots_%j.out
#SBATCH --error=logs/avg_giant_nonexcluded_enforcehw05_plots_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

RESULTS_DIR="$NETPIP_ROOT/results/pip_convergence/avg_giant75_enforcehw05"
PIP_MAT="$RESULTS_DIR/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat"

python3 "$NETPIP_ROOT/scripts/3_plot_pip_surfaces.py" \
  --results-dir "$RESULTS_DIR" \
  --out-root "$NETPIP_ROOT/figures/pip_surfaces/avg_giant75_enforcehw05" \
  --tag postHW_giant_avg_nonexcluded_enforcehw05 \
  --include-prefix AVG

python3 "$NETPIP_ROOT/scripts/4_cluster_pip_set.py" \
  --pip-mat "$PIP_MAT" \
  --out-dir "$NETPIP_ROOT/figures/pip_cluster/avg_giant75_enforcehw05" \
  --k-min 2 --k-max 10 | tee "$NETPIP_ROOT/results/pip_cluster/avg_giant75_enforcehw05_top_cluster_nodes.txt"

python3 "$NETPIP_ROOT/scripts/helpers/label_cluster_nodes.py" \
  --labels-csv "$NETPIP_ROOT/data/MNI_66_AAL_onelinestructure.csv" \
  --indices-csv "$NETPIP_ROOT/figures/pip_cluster/avg_giant75_enforcehw05/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW_top_cluster_nodes.csv" \
  --out-csv "$NETPIP_ROOT/results/pip_cluster/avg_giant75_enforcehw05_top_cluster_nodes_labeled.csv"

python3 "$NETPIP_ROOT/scripts/helpers/summarize_convergence.py" \
  --results-dir "$RESULTS_DIR" \
  --out-csv "$NETPIP_ROOT/results/pip_cluster/avg_giant75_enforcehw05_convergence_summary.csv"
