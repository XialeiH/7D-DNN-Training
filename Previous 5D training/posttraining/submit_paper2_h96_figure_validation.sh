#!/bin/bash
set -euo pipefail

RUN_ROOT=${1:?usage: submit_paper2_h96_figure_validation.sh RUN_ROOT}
PROJECT_ROOT=/scratch/xh2906/NYU-Vision-2Drive-main
CODE_DIR="$RUN_ROOT/code"
SIM_DIR="$RUN_ROOT/snn_simulations"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
PDF_DIR="$RUN_ROOT/pdf"
LOG_DIR="$RUN_ROOT/logs"
RUNTIME=/scratch/xh2906/librarySCI_runs/sixd_prefAngle_varaware_singlemlp_h96_final70_smooth2d_low005_stratmid_torch_20260518_044134/hpc_runtime_bundle_extended_corrected_h96_ikp7_l6p5_matchedlib_20260702_010052
UTILS_DIR="$CODE_DIR/current_h96_utils"
PAIR_DATA="$ARTIFACT_DIR/1.2_1.3_real_paired_data_source.mat"
mkdir -p "$SIM_DIR" "$ARTIFACT_DIR" "$PDF_DIR" "$LOG_DIR"

SIM_JOB=$(sbatch --parsable --account=torch_pr_482_general --partition=cpu_short --qos=cpu48 \
  --array=1-8%8 --cpus-per-task=1 --mem=20G --time=02:00:00 \
  --job-name=paper2_snn_h96fig --output="$LOG_DIR/snn_%A_%a.out" --error="$LOG_DIR/snn_%A_%a.err" \
  --wrap="module load matlab/2025b; angles=(0 7.5 15 22.5); task=\$SLURM_ARRAY_TASK_ID; ai=\$(( (task-1)/2 )); rep=\$(( (task-1)%2+1 )); export PAPER2_PROJECT_ROOT='$PROJECT_ROOT' SNN_OUTPUT_DIR='$SIM_DIR' SNN_ANGLE=\${angles[\$ai]} SNN_REPLICATE=\$rep SNN_DURATION_MS=10000 SNN_ANALYSIS_MS=9000; matlab -batch \"addpath('$CODE_DIR'); simulate_paper2_snn_pixels\"")

EVAL_JOB=$(sbatch --parsable --account=torch_pr_482_general --partition=cpu_short --qos=cpu48 \
  --dependency=afterok:$SIM_JOB --cpus-per-task=1 --mem=8G --time=01:00:00 \
  --job-name=h96_pair_eval --output="$LOG_DIR/evaluate_%j.out" --error="$LOG_DIR/evaluate_%j.err" \
  --wrap="module load matlab/2025b; export H96_CORRECTED_RUNTIME='$RUNTIME' H96_CURRENT_UTILS='$UTILS_DIR' SNN_OUTPUT_DIR='$SIM_DIR' PREMODEL_PAIR_DATA='$PAIR_DATA'; matlab -batch \"addpath('$CODE_DIR'); evaluate_h96_premodel_pairs\"")

PLOT_JOB=$(sbatch --parsable --account=torch_pr_482_general --partition=cpu_short --qos=cpu48 \
  --dependency=afterok:$EVAL_JOB --cpus-per-task=1 --mem=8G --time=01:00:00 \
  --job-name=h96_fig12_13 --output="$LOG_DIR/plot_%j.out" --error="$LOG_DIR/plot_%j.err" \
  --wrap="module load matlab/2025b; export PREMODEL_PAIR_DATA='$PAIR_DATA' FIGURE_PDF_DIR='$PDF_DIR' FIGURE_ARTIFACT_DIR='$ARTIFACT_DIR'; matlab -batch \"addpath('$CODE_DIR'); plot_1_2_premodel_vs_nn_cg\"")

MANIFEST="$RUN_ROOT/submission_manifest.tsv"
printf 'stage\tjob_id\tpath\n' > "$MANIFEST"
printf 'snn_simulation\t%s\t%s\n' "$SIM_JOB" "$SIM_DIR" >> "$MANIFEST"
printf 'h96_evaluation\t%s\t%s\n' "$EVAL_JOB" "$PAIR_DATA" >> "$MANIFEST"
printf 'plot\t%s\t%s\n' "$PLOT_JOB" "$PDF_DIR" >> "$MANIFEST"
printf '%s\n%s\n%s\n%s\n' "$SIM_JOB" "$EVAL_JOB" "$PLOT_JOB" "$MANIFEST"
