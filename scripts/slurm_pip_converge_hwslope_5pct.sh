#!/bin/bash
#SBATCH --job-name=pipConv5pct
#SBATCH --array=1-21%4
#SBATCH --cpus-per-task=40
#SBATCH --mem=50G
#SBATCH --time=70:00:00
#SBATCH --output=logs/conv5pct_%A_%a.out
#SBATCH --error=logs/conv5pct_%A_%a.err
#SBATCH --mail-user=minarose.ismail@sickkids.ca
#SBATCH --mail-type=END,FAIL

module load Singularity/3.11.3_slurm

export SCRATCH_DIR=/scratch/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SCRATCH_DIR"
export TMPDIR=$SCRATCH_DIR
export MCR_CACHE_ROOT=$SCRATCH_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP
SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
SCRIPT=$NETPIP_ROOT/scripts/pip_converge_posthw5_thresh.m

mapfile -t FILES < <(ls -1 $PIP_ROOT/*_broadband_psi_adj.mat | sort)
SUBJECT_COUNT=${#FILES[@]}

if (( SLURM_ARRAY_TASK_ID > SUBJECT_COUNT )); then
    echo "Array index $SLURM_ARRAY_TASK_ID exceeds subject count $SUBJECT_COUNT"
    exit 1
fi

export SUBJECT_FILE=$(basename "${FILES[$SLURM_ARRAY_TASK_ID-1]}")

echo ">>> Running subject: $SUBJECT_FILE"
echo ">>> CPUs: $SLURM_CPUS_PER_TASK  Scratch: $SCRATCH_DIR"

export MAX_ATTACKS=100000000
export CHUNK_SIZE=10000
export HW95_TOL=0.05
export REQUIRE_STABLE=3
export BASE_SEED=12345
export THRESH_PROP=0.05
export OUT_DIR=$PIP_ROOT/results_converge_5pct

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
