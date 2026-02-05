#!/bin/bash
#SBATCH --job-name=avgPiPMatrix
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/avg_pip_matrix_%j.out
#SBATCH --error=logs/avg_pip_matrix_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats

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

python3 "$NETPIP_ROOT/scripts/plot_avg_pip_matrix.py" \
  --pip-mat "$PIP_ROOT/results_converge/AVG_broadband_psi_adj_top10_ConvHW.mat" \
  --out-dir "$NETPIP_ROOT/figures/pip_avg_matrix"
