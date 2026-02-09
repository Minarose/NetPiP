#!/bin/bash
#SBATCH --job-name=netPiPSurfaces
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/net_pip_surfaces_%j.out
#SBATCH --error=logs/net_pip_surfaces_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/plot_pip_surfaces.py" \
  --results-dir "$NETPIP_ROOT/results/networks_airports" \
  --out-root "$NETPIP_ROOT/figures/networks_airports" \
  --tag postHW_airports \
  --include-prefix airports \
  --only-3d \
  --max-steps 200 \
  --max-nodes 200

python3 "$NETPIP_ROOT/scripts/plot_pip_surfaces.py" \
  --results-dir "$NETPIP_ROOT/results/networks_yeast" \
  --out-root "$NETPIP_ROOT/figures/networks_yeast" \
  --tag postHW_yeast \
  --include-prefix yeastPPI \
  --only-3d \
  --max-steps 200 \
  --max-nodes 200
