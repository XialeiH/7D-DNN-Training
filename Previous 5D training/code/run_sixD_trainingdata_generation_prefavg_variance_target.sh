#!/bin/bash
#SBATCH --job-name=6dvar_tgt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=torch_pr_482_general
#SBATCH --output=/scratch/xh2906/logs/%x_%j.out
#SBATCH --error=/scratch/xh2906/logs/%x_%j.err

set -euo pipefail
TARGET=${1:?need target S/C/I}
STAMP=${2:?need dataset stamp}
case "$TARGET" in S|C|I) ;; *) echo "bad target $TARGET" >&2; exit 2 ;; esac
mkdir -p /scratch/xh2906/logs
cd /scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data
module purge
module load matlab/2025b
export TARGET_ONLY="$TARGET"
export DATASET_STAMP="$STAMP"
matlab -batch "cd('/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data'); sixD_trainingdata_generation_prefavg_variance;"
