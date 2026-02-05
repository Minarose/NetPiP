#!/bin/bash
#SBATCH --job-name=avgPsiPlot
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/avg_psi_plot_%j.out
#SBATCH --error=logs/avg_psi_plot_%j.err

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

python3 "$NETPIP_ROOT/scripts/plot_avg_psi_matrix.py" \
  --avg-mat "$NETPIP_ROOT/data/PSI_broadband_MEG_mats/avg/AVG_broadband_psi_adj_top10.mat" \
  --out-dir "$NETPIP_ROOT/figures/avg_psi"
