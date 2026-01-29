#!/bin/bash
#SBATCH --job-name=pipPlots
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pip_plots_%j.out
#SBATCH --error=logs/pip_plots_%j.err
#SBATCH --account=imaginglab

module load Singularity/3.11.3_slurm

# Root folder with your PiP results
export PIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP/data/PSI_broadband_MEG_mats
export NETPIP_ROOT=/hpf/projects/dkadis/ismail/NetPiP

MATLAB_SIF=/hpf/projects/imaginglab/matlab/imaging-lab_matlab_r2024b.sif
PLOT_SCRIPT_1=$NETPIP_ROOT/scripts/barplot_convg.m
PLOT_SCRIPT_2=$NETPIP_ROOT/scripts/barnattack.m

mkdir -p logs

singularity exec \
    --env MLM_LICENSE_FILE=27000@imaginglab-mgt.ccm.sickkids.ca \
    --env PIP_ROOT=$PIP_ROOT \
    --bind $PIP_ROOT:$PIP_ROOT \
    --bind $NETPIP_ROOT:$NETPIP_ROOT \
    $MATLAB_SIF matlab -batch "run('$PLOT_SCRIPT_1'); run('$PLOT_SCRIPT_2');"
