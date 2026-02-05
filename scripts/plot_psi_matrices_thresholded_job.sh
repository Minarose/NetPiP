#!/bin/bash
#SBATCH --job-name=psiMat5pct
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi_mat_5pct_%j.out
#SBATCH --error=logs/psi_mat_5pct_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 - <<'PY'
import sys
print('python', sys.version)
for mod in ['numpy','scipy','h5py','matplotlib']:
    try:
        __import__(mod)
        print(mod, 'OK')
    except Exception as e:
        print(mod, 'MISSING', e)
PY

python3 "$NETPIP_ROOT/scripts/plot_psi_matrices_thresholded.py" \
  --data-dir "$NETPIP_ROOT/data/PSI_broadband_MEG_mats" \
  --out-dir "$NETPIP_ROOT/figures/5percthresh_analysis/psi_matrices"
