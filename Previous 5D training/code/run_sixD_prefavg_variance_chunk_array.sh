#!/bin/bash
#SBATCH --job-name=6dvar_chunk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=torch_pr_482_general
#SBATCH --array=1-147%40
#SBATCH --output=/scratch/xh2906/logs/%x_%A_%a.out
#SBATCH --error=/scratch/xh2906/logs/%x_%A_%a.err

set -euo pipefail
TARGET=${1:?need target C/I/S}
STAMP=${2:?need dataset stamp}
case "$TARGET" in S|C|I) ;; *) echo "bad target $TARGET" >&2; exit 2 ;; esac
mkdir -p /scratch/xh2906/logs
cd /scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data
module purge
module load matlab/2025b
export TARGET_ONLY="$TARGET"
export DATASET_STAMP="$STAMP"
export CASE_INDEX="${SLURM_ARRAY_TASK_ID}"
matlab -batch "cd('/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data'); sixD_trainingdata_generation_prefavg_variance_chunk;"
