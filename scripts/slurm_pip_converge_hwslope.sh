#!/bin/bash
#SBATCH --job-name=pipConvSlope
#SBATCH --array=1-21%4                     # 21 subjects, 4 in parallel
#SBATCH --cpus-per-task=40                 # 30 CPUs per subject
#SBATCH --mem=50G
#SBATCH --time=70:00:00
#SBATCH --output=logs/conv_%A_%a.out
#SBATCH --error=logs/conv_%A_%a.err
#SBATCH --mail-user=minarose.ismail@sickkids.ca
#SBATCH --mail-type=END,FAIL

module load Singularity/3.11.3_slurm

# ---------------- SCRATCH DIRECTORY ----------------
export SCRATCH_DIR=/scratch/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$SCRATCH_DIR"

export TMPDIR=$SCRATCH_DIR
export MCR_CACHE_ROOT=$SCRATCH_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ---------------- PATHS ----------------
export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
SCRIPT=pip_converge_posthw5.m

# ---------------- SUBJECT SELECTION ----------------
mapfile -t FILES < <(ls -1 $PIP_ROOT/*_broadband_psi_adj.mat | sort)
SUBJECT_COUNT=${#FILES[@]}

if (( SLURM_ARRAY_TASK_ID > SUBJECT_COUNT )); then
    echo "Array index $SLURM_ARRAY_TASK_ID exceeds subject count $SUBJECT_COUNT"
    exit 1
fi

export SUBJECT_FILE=$(basename "${FILES[$SLURM_ARRAY_TASK_ID-1]}")

echo ">>> Running subject: $SUBJECT_FILE"
echo ">>> CPUs: $SLURM_CPUS_PER_TASK  Scratch: $SCRATCH_DIR"

# ---------------- CONVERGENCE PARAMS ----------------
export MAX_ATTACKS=100000000
export CHUNK_SIZE=10000
export RELFROB_TOL=1e-3
export RHO_TOL=0.97
export HW95_TOL=0.05
export REQUIRE_STABLE=3
export BASE_SEED=12345

# ---------------- RUN MATLAB IN SINGULARITY ----------------
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
    $SIF matlab -batch "run('$SCRIPT');"
