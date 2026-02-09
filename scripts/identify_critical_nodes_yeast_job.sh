#!/bin/bash
#SBATCH --job-name=critNodesYeast
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/crit_nodes_yeast_%j.out
#SBATCH --error=logs/crit_nodes_yeast_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

python3 "$NETPIP_ROOT/scripts/identify_critical_nodes.py" \
  --pip-mat "$NETPIP_ROOT/results/networks_yeast/yeastPPI_LCC_ConvHW.mat" \
  --labels-mat "$NETPIP_ROOT/data/networks/yeastPPI_LCC.mat" \
  --out-dir "$NETPIP_ROOT/results/networks_yeast" \
  --k-min 2 --k-max 10

python3 "$NETPIP_ROOT/scripts/summarize_convergence.py" \
  --results-dir "$NETPIP_ROOT/results/networks_yeast" \
  --out-csv "$NETPIP_ROOT/results/networks_yeast/yeast_convergence_summary.csv"
