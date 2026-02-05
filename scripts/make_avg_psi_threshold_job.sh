#!/bin/bash
#SBATCH --job-name=avgPsiTop10
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/avg_psi_%j.out
#SBATCH --error=logs/avg_psi_%j.err

set -euo pipefail

module load Singularity/3.11.3_slurm

export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
SCRIPT=$NETPIP_ROOT/scripts/make_avg_psi_threshold.m

singularity exec \
  --env MLM_LICENSE_FILE=27000@imaginglab-mgt.ccm.sickkids.ca \
  --env PIP_ROOT=$PIP_ROOT \
  --bind $PIP_ROOT:$PIP_ROOT \
  --bind $NETPIP_ROOT:$NETPIP_ROOT \
  $SIF matlab -batch "run('$SCRIPT');"
