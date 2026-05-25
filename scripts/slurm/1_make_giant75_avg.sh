#!/bin/bash
#SBATCH --job-name=avgPsiGiantNE
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/avg_psi_giant_nonexcluded_%j.out
#SBATCH --error=logs/avg_psi_giant_nonexcluded_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/1_make_giant75_avg.py" \
  --pip-root "$NETPIP_ROOT/data/PSI_broadband_MEG_mats" \
  --outlier-csv "$NETPIP_ROOT/results/pip_cluster/attack_outliers.csv" \
  --gcc-fraction 0.75 \
  --threshold-steps 1000 \
  --out-file "$NETPIP_ROOT/data/PSI_broadband_MEG_mats/group_average/AVG_broadband_psi_adj_giant75_nonexcluded.mat"
