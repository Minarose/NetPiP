#!/bin/bash
#SBATCH --job-name=psiMatPlots
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/psi_mat_plots_%j.out
#SBATCH --error=logs/psi_mat_plots_%j.err

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

python3 "$NETPIP_ROOT/scripts/plot_psi_matrices.py" \
  --data-dir "$NETPIP_ROOT/data/PSI_broadband_MEG_mats" \
  --out-dir "$NETPIP_ROOT/figures/PSI_matrices"
