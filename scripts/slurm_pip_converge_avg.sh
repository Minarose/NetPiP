#!/bin/bash
#SBATCH --job-name=pipAvgConv
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=70:00:00
#SBATCH --output=logs/conv_avg_%j.out
#SBATCH --error=logs/conv_avg_%j.err
#SBATCH --mail-user=minarose.ismail@sickkids.ca
#SBATCH --mail-type=END,FAIL

module load Singularity/3.11.3_slurm

export SCRATCH_DIR=/scratch/$USER/${SLURM_JOB_ID}_avg
mkdir -p "$SCRATCH_DIR"
export TMPDIR=$SCRATCH_DIR
export MCR_CACHE_ROOT=$SCRATCH_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
SCRIPT=$NETPIP_ROOT/scripts/pip_converge_posthw5.m

export SUBJECT_FILE=avg/AVG_broadband_psi_adj_top10.mat

export MAX_ATTACKS=100000000
export CHUNK_SIZE=10000
export RELFROB_TOL=1e-3
export RHO_TOL=0.97
export HW95_TOL=0.05
export REQUIRE_STABLE=3
export BASE_SEED=12345

singularity exec \
  --env MLM_LICENSE_FILE=27000@imaginglab-mgt.ccm.sickkids.ca \
  --env TMPDIR=$TMPDIR \
  --env MCR_CACHE_ROOT=$MCR_CACHE_ROOT \
  --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
  --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
  --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
  --env PIP_ROOT=$PIP_ROOT \
  --env SUBJECT_FILE=$SUBJECT_FILE \
  --env MAX_ATTACKS=$MAX_ATTACKS \
  --env CHUNK_SIZE=$CHUNK_SIZE \
  --env RELFROB_TOL=$RELFROB_TOL \
  --env RHO_TOL=$RHO_TOL \
  --env HW95_TOL=$HW95_TOL \
  --env REQUIRE_STABLE=$REQUIRE_STABLE \
  --env BASE_SEED=$BASE_SEED \
  --bind $SCRATCH_DIR:$SCRATCH_DIR \
  --bind $PIP_ROOT:$PIP_ROOT \
  --bind $NETPIP_ROOT:$NETPIP_ROOT \
  $SIF matlab -batch "run('$SCRIPT');"
