#!/bin/bash
#SBATCH --job-name=avgPsiTop05NE
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/avg_psi_nonexcluded_%j.out
#SBATCH --error=logs/avg_psi_nonexcluded_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/make_avg_psi_threshold_nonexcluded.py" \
  --pip-root "$NETPIP_ROOT/data/PSI_broadband_MEG_mats" \
  --outlier-csv "$NETPIP_ROOT/results/consensus_5pct/attack_outliers.csv" \
  --threshold-prop 0.05 \
  --out-file "$NETPIP_ROOT/data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_top05_nonexcluded.mat"
