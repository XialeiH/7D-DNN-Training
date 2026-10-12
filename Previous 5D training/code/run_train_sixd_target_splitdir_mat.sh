#!/bin/bash
#SBATCH --job-name=sixd_pref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=xh2906@nyu.edu
#SBATCH --account=torch_pr_482_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|h200
#SBATCH --output=/scratch/xh2906/librarySCI_logs/%x_%j.out
#SBATCH --error=/scratch/xh2906/librarySCI_logs/%x_%j.err

set -euo pipefail
mkdir -p /scratch/xh2906/librarySCI_logs

TARGET=${1:?need target S/C/I}
RUN_ROOT=${2:?need run root}
SPLIT_DIR=${3:?need split dir}
TRAIN_MAT=${4:?need train mat}
RUN_MODE=${5:-full}
shift 5 || true
EXTRA_TRAIN_ARGS=("$@")

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
export PATH="/scratch/xh2906/conda/envs/py310/bin:$PATH"
hash -r
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
mkdir -p /scratch/xh2906/pip-cache /scratch/xh2906/tmp
export PIP_CACHE_DIR=/scratch/xh2906/pip-cache
export TMPDIR=/scratch/xh2906/tmp

SCRIPT_DIR=/scratch/xh2906/NYU-Vision-2Drive-main/python-training_DNN/sixd_prefAngle
OUT_DIR="$RUN_ROOT/train/$TARGET"
mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
BATCH_SIZE=32768
EVAL_BATCH_SIZE=65536
if [[ "$RUN_MODE" == "smoke" ]]; then
  EXTRA_ARGS+=(--epochs 1 --stage_epochs 1 --max_stages 1 --max_train_batches 2 --max_eval_batches 2 --diag_sample_size 50000)
elif [[ "$RUN_MODE" == "bs8192_steps40" ]]; then
  BATCH_SIZE=8192
  EXTRA_ARGS+=(--epochs 5 --stage_epochs 5 --max_stages 30 --max_train_batches 40 --max_eval_batches 10 --diag_sample_size 200000)
else
  EXTRA_ARGS+=(--epochs 5 --stage_epochs 5 --max_stages 30 --max_train_batches 10 --max_eval_batches 10 --diag_sample_size 200000)
fi

python "$SCRIPT_DIR/train_sixd_single.py" \
  --target "$TARGET" \
  --train_mat "$TRAIN_MAT" \
  --split_dir "$SPLIT_DIR" \
  --out_dir "$OUT_DIR" \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --lr 5e-3 \
  --weight_decay 1e-5 \
  --lr_stage_values 5e-3,2e-3,9e-4,6e-4,2e-4 \
  --min_improve 0.05 \
  --min_best_rel_gain 0.002 \
  "${EXTRA_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}"
