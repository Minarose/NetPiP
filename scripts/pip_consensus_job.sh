#!/bin/bash
#SBATCH --job-name=pipConsensus
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pip_consensus_%j.out
#SBATCH --error=logs/pip_consensus_%j.err

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

python3 "$NETPIP_ROOT/scripts/pip_consensus_cluster.py" \
  --results-dir "$PIP_ROOT/results_converge" \
  --coords "$NETPIP_ROOT/data/MNI_66_coords.txt" \
  --out-results "$NETPIP_ROOT/results/consensus" \
  --out-figures "$NETPIP_ROOT/figures/consensus" \
  --thresholds 0.05 0.10 0.20
