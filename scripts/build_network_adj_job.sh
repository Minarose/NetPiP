#!/bin/bash
#SBATCH --job-name=buildNetAdj
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/build_net_adj_%j.out
#SBATCH --error=logs/build_net_adj_%j.err

set -euo pipefail

module load python/3.11.3

export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
export DATA_ROOT=/hpf/projects/dkadis/ismail

python3 "$NETPIP_ROOT/scripts/build_adj_from_edgelist.py" \
  --edgelist "$DATA_ROOT/pip_airports_LCC_edgelist.csv" \
  --out-mat "$NETPIP_ROOT/data/networks/airports_LCC.mat" \
  --weighted

python3 "$NETPIP_ROOT/scripts/build_adj_from_edgelist.py" \
  --edgelist "$DATA_ROOT/pip_yeastPPI_LCC_edgelist.csv" \
  --out-mat "$NETPIP_ROOT/data/networks/yeastPPI_LCC.mat" \
  --weighted
