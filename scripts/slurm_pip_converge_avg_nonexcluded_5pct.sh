#!/bin/bash
#SBATCH --job-name=pipConvAvg5pctNE
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=70:00:00
#SBATCH --output=logs/conv5pct_avg_nonexcluded_%j.out
#SBATCH --error=logs/conv5pct_avg_nonexcluded_%j.err
#SBATCH --mail-user=minarose.ismail@sickkids.ca
#SBATCH --mail-type=END,FAIL

module load Singularity/3.11.3_slurm

export SCRATCH_DIR=/scratch/$USER/${SLURM_JOB_ID}
mkdir -p "$SCRATCH_DIR"
export TMPDIR=$SCRATCH_DIR
export MCR_CACHE_ROOT=$SCRATCH_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
SCRIPT=$NETPIP_ROOT/scripts/pip_converge_posthw5_thresh.m

export SUBJECT_FILE=avg/AVG_broadband_psi_adj_top05_nonexcluded.mat

echo ">>> Running avg matrix: $SUBJECT_FILE"
echo ">>> CPUs: $SLURM_CPUS_PER_TASK  Scratch: $SCRATCH_DIR"

export MAX_ATTACKS=100000000
export CHUNK_SIZE=10000
export HW95_TOL=0.05
export REQUIRE_STABLE=3
export BASE_SEED=12345
export THRESH_PROP=1.0
export OUT_DIR=$PIP_ROOT/results_converge_5pct_avg_nonexcluded

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
    --env HW95_TOL=$HW95_TOL \
    --env REQUIRE_STABLE=$REQUIRE_STABLE \
    --env BASE_SEED=$BASE_SEED \
    --env THRESH_PROP=$THRESH_PROP \
    --env OUT_DIR=$OUT_DIR \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    --bind $PIP_ROOT:$PIP_ROOT \
    --bind $NETPIP_ROOT:$NETPIP_ROOT \
    $SIF matlab -batch "run('$SCRIPT');"
