#!/bin/bash
#SBATCH --job-name=sixd_var_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --account=torch_pr_482_general
#SBATCH --output=/scratch/xh2906/librarySCI_logs/%x_%j.out
#SBATCH --error=/scratch/xh2906/librarySCI_logs/%x_%j.err

set -euo pipefail
STAMP=${1:?need dataset stamp}
DATA_DIR=/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/prefavg_variance_DS10_21contr_${STAMP}
OUT_DIR=${2:-$DATA_DIR/splits_randomrows_upperonly_seed42}
SCRIPT_DIR=/scratch/xh2906/NYU-Vision-2Drive-main/python-training_DNN/sixd_prefAngle
mkdir -p /scratch/xh2906/librarySCI_logs "$OUT_DIR"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
export PATH="/scratch/xh2906/conda/envs/py310/bin:$PATH"
hash -r
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE

python "$SCRIPT_DIR/build_random_row_split_variance_independent.py" \
  --out_dir "$OUT_DIR" \
  --mat_s "$DATA_DIR/S_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part_prefavg_variance_DS10_21contr_${STAMP}.mat" \
  --mat_c "$DATA_DIR/C_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part_prefavg_variance_DS10_21contr_${STAMP}.mat" \
  --mat_i "$DATA_DIR/I_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part_prefavg_variance_DS10_21contr_${STAMP}.mat" \
  --target_high_s 30 \
  --target_high_c 100 \
  --target_high_i 140
