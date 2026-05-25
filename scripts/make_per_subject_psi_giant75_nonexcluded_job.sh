#!/bin/bash
#SBATCH --job-name=psiGiant75perSubjNE
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/per_subj_giant75_ne_%j.out
#SBATCH --error=logs/per_subj_giant75_ne_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/make_per_subject_psi_giant75_nonexcluded.py" \
  --pip-root "$NETPIP_ROOT/data/PSI_broadband_MEG_mats" \
  --outlier-csv "$NETPIP_ROOT/results/consensus_5pct/attack_outliers.csv" \
  --out-dir "$NETPIP_ROOT/data/PSI_broadband_MEG_mats/per_subject_giant75_nonexcluded" \
  --gcc-fraction 0.75 \
  --threshold-steps 1000
