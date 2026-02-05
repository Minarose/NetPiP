#!/bin/bash
#SBATCH --job-name=pipPyPlots5pct
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pip_py_plots_5pct_%j.out
#SBATCH --error=logs/pip_py_plots_5pct_%j.err

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

python3 "$NETPIP_ROOT/scripts/plot_pip_surfaces.py" \
  --results-dir "$PIP_ROOT/results_converge_5pct" \
  --out-root "$NETPIP_ROOT/figures/5percthresh_analysis/pip_surfaces" \
  --tag "postHW_5pct"
