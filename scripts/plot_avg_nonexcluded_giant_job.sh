#!/bin/bash
#SBATCH --job-name=avgGiantNEPlots
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/avg_giant_nonexcluded_plots_%j.out
#SBATCH --error=logs/avg_giant_nonexcluded_plots_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats

PIP_MAT="$PIP_ROOT/results_converge_giant_avg_nonexcluded/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW.mat"

python3 "$NETPIP_ROOT/scripts/plot_pip_surfaces.py" \
  --results-dir "$PIP_ROOT/results_converge_giant_avg_nonexcluded" \
  --out-root "$NETPIP_ROOT/figures/5percthresh_analysis/pip_surfaces_avg_giant_nonexcluded" \
  --tag postHW_giant_avg_nonexcluded \
  --include-prefix AVG

python3 "$NETPIP_ROOT/scripts/plot_avg_cluster.py" \
  --pip-mat "$PIP_MAT" \
  --out-dir "$NETPIP_ROOT/figures/5percthresh_analysis/pip_avg_cluster_giant_nonexcluded" \
  --k-min 2 --k-max 10 | tee "$NETPIP_ROOT/results/consensus_5pct/avg_giant_nonexcluded_top_cluster_nodes.txt"

python3 "$NETPIP_ROOT/scripts/label_cluster_nodes.py" \
  --labels-csv "$NETPIP_ROOT/data/MNI_66_AAL_onelinestructure.csv" \
  --indices-csv "$NETPIP_ROOT/figures/5percthresh_analysis/pip_avg_cluster_giant_nonexcluded/AVG_broadband_psi_adj_giant75_nonexcluded_ConvHW_top_cluster_nodes.csv" \
  --out-csv "$NETPIP_ROOT/results/consensus_5pct/avg_giant_nonexcluded_top_cluster_nodes_labeled.csv"

python3 "$NETPIP_ROOT/scripts/summarize_convergence.py" \
  --results-dir "$PIP_ROOT/results_converge_giant_avg_nonexcluded" \
  --out-csv "$NETPIP_ROOT/results/consensus_5pct/avg_giant_nonexcluded_convergence_summary.csv"
