#!/usr/bin/env bash
set -euo pipefail

BASE_RUN="/scratch/xh2906/librarySCI_runs/sixd_prefAngle_varaware_singlemlp_h96_final70_smooth2d_low005_stratmid_torch_20260518_044134"
SPLIT_DIR="/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/prefavg_variance_final70_smooth2d_20260514_032346/splits_randomrows_low005_upper_seed42_chunked"
TRAIN_MAT="/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/prefavg_variance_final70_smooth2d_20260514_032346/chunks_S/S_case001_Ang0.00_Contr14_full70Small.mat"
RUN_ROOT_BASE="/scratch/xh2906/librarySCI_runs"
CODE_DIR="/scratch/xh2906/NYU-Vision-2Drive-main/python-training_DNN/sixd_prefAngle"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

submit_variant() {
  local label="$1"
  local hidden_dims="$2"
  local activation_seq="$3"
  local residual_skip_every="$4"
  local job_prefix="$5"

  local run_root="${RUN_ROOT_BASE}/sixd_prefAngle_varaware_singlemlp_${label}_final70_smooth2d_low005_stratmid_torch_${TS}"
  mkdir -p "${run_root}/train" "${run_root}/logs"
  printf '%s\n' "${BASE_RUN}" > "${run_root}/base_run.txt"
  printf '%s\n' "${SPLIT_DIR}" > "${run_root}/split_dir.txt"
  cat > "${run_root}/arch_variant_config.tsv" <<CFG
base_run	${BASE_RUN}
hidden_dims	${hidden_dims}
activation_seq	${activation_seq}
residual_skip_every	${residual_skip_every}
feature_mode	cosonly5d
variance_weight_mode	soft_inv_sem
sampler_mode	target_rate_stratified
rmse_space	raw
w_log	1.0
w_rel	3.0
low_penalty_lambda	0.05
mono_pair_lambda	0.1
from_scratch	true
CFG

  local job_ids=()
  for target in S C I; do
    mkdir -p "${run_root}/train/${target}" "${run_root}/logs"
    local script="${run_root}/logs/train_${target}.sh"
    cat > "${script}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${job_prefix}_${target}
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
#SBATCH --output=${run_root}/logs/train_${target}.%j.out
#SBATCH --error=${run_root}/logs/train_${target}.%j.err

set -euo pipefail
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate py310
export PATH="/scratch/xh2906/conda/envs/py310/bin:\$PATH"
hash -r
export PYTHONNOUSERSITE=1
unset PYTHONUSERBASE
export PIP_CACHE_DIR=/scratch/xh2906/pip-cache
export TMPDIR=/scratch/xh2906/tmp
export SIXD_ARCH_VARIANT="${label}"
export SIXD_RESIDUAL_SKIP_EVERY="${residual_skip_every}"

cd "${CODE_DIR}"
python train_sixd_archvariant.py \\
  --target "${target}" \\
  --train_mat "${TRAIN_MAT}" \\
  --split_dir "${SPLIT_DIR}" \\
  --out_dir "${run_root}/train/${target}" \\
  --device cuda \\
  --feature_mode cosonly5d \\
  --hidden_dims "${hidden_dims}" \\
  --activation_seq "${activation_seq}" \\
  --seed 42 \\
  --epochs 5 \\
  --stage_epochs 5 \\
  --max_stages 30 \\
  --lr 5e-3 \\
  --weight_decay 1e-5 \\
  --batch_size 8192 \\
  --eval_batch_size 65536 \\
  --max_train_batches 40 \\
  --max_eval_batches 10 \\
  --lr_stage_values 5e-3,2e-3,9e-4,6e-4,2e-4 \\
  --min_improve 0.05 \\
  --min_best_rel_gain 0.002 \\
  --rmse_space raw \\
  --w_log 1.0 \\
  --w_rel 3.0 \\
  --low_penalty_lambda 0.05 \\
  --low_penalty_tau 3.0 \\
  --mono_pair_lambda 0.1 \\
  --variance_weight_mode soft_inv_sem \\
  --variance_min_weight 0.25 \\
  --variance_ref_quantile 0.50 \\
  --sampler_mode target_rate_stratified \\
  --target_rate_bins 0,0.5,2,5,15,inf \\
  --strata_policy equal \\
  --diag_sample_size 200000
EOF
    chmod +x "${script}"
    local jid
    jid="$(sbatch --parsable "${script}")"
    job_ids+=("${jid}")
  done

  printf '%s\t%s\t%s\n' "${label}" "${run_root}" "${job_ids[*]}" | tee -a "${RUN_ROOT_BASE}/sixd_arch_variants_${TS}_submitted.tsv"
  printf '%s\n' "${job_ids[@]}" > "${run_root}/train_job_ids.txt"
}

submit_variant "h128x4_silu_res2" "128,128,128,128" "silu,silu,silu,silu" "2" "h128sir"
submit_variant "h96_rational" "96,96,96" "rational,rational,rational" "0" "h96rat"
submit_variant "h128x4_rational_res2" "128,128,128,128" "rational,rational,rational,rational" "2" "h128rar"
