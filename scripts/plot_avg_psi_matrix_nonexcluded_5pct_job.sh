#!/bin/bash
#SBATCH --job-name=avgPsi5pctNEPlot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/avg_psi_5pct_nonexcluded_%j.out
#SBATCH --error=logs/avg_psi_5pct_nonexcluded_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/plot_avg_psi_matrix.py" \
  --avg-mat "$NETPIP_ROOT/data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_top05_nonexcluded.mat" \
  --out-dir "$NETPIP_ROOT/figures/5percthresh_analysis/psi_matrices_avg_nonexcluded"
