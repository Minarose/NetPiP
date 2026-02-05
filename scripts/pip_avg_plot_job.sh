#!/bin/bash
#SBATCH --job-name=pipAvgPlots
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pip_avg_plots_%j.out
#SBATCH --error=logs/pip_avg_plots_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats

python3 - <<'PY'
import sys
print('python', sys.version)
for mod in ['numpy','scipy','h5py','matplotlib','nilearn']:
    try:
        __import__(mod)
        print(mod, 'OK')
    except Exception as e:
        print(mod, 'MISSING', e)
PY

python3 "$NETPIP_ROOT/scripts/plot_pip_surfaces.py" \
  --results-dir "$PIP_ROOT/results_converge" \
  --out-root "$NETPIP_ROOT/figures/pip_surfaces_avg" \
  --include-prefix "AVG"

python3 "$NETPIP_ROOT/scripts/plot_mean_pip_nodes.py" \
  --pip-mat "$PIP_ROOT/results_converge/AVG_broadband_psi_adj_top10_ConvHW.mat" \
  --coords "$NETPIP_ROOT/data/MNI_66_coords.txt" \
  --out-dir "$NETPIP_ROOT/figures/pip_mean_nodes" \
  --percentile 95
