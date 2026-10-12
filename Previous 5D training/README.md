# Previous 5D training — manuscript h96 model

## Which model is this?

The user identified `Spectral_Analysis_Report/Figures/Draft` as the current manuscript source. `manuscript_examples/Figure_0A.m` puts `local_h96_manual_runtime` first on the MATLAB path; `posttraining/evaluate_h96_premodel_pairs.m` independently asserts the same predictor and corrected-wrapper hashes. Those exact four inference files are preserved in `runtime/`.

The model has **five inputs** `[OriCos2, Contrast, L6, L4E, L4I]`, **three 96-unit tanh hidden layers**, and a softplus output. Physical predictor arguments are `[L4E, L4I, L6, ContrastEff, OriCos2]`; the exporter rearranges them to training order. `sixd_prefAngle`, `predict_pref6D_*` and `LocalResponse_6D_*` are historical names. The training flag is `--feature_mode cosonly5d`. This is not the older 15-per-LGN `MLP5D_allOrientation...part_tanh_relu` family and not a seven-input model with TF/SF silently fixed.

Historical trained run:

```
/scratch/xh2906/librarySCI_runs/sixd_prefAngle_varaware_singlemlp_h96_final70_smooth2d_low005_stratmid_torch_20260518_044134
```

Corrected runtime on Torch:

```
<run>/hpc_runtime_bundle_extended_corrected_h96_ikp7_l6p5_matchedlib_20260702_010052
```

The corrected LocalResponse SHA is `41c7fdc837617cd66ee81b0a7f7dc2bc3af45ed8c38e67bc3e85abed8b6bb161`. Never substitute the uncorrected base run's LocalResponse. The three predictor hashes and source locations are in `../provenance/source_manifest.json`.

## Coverage and provenance limits

The exact manuscript inference source is hash-confirmed. The `code/` training/data files are preserved copies of the shared local variance-aware preferred-angle implementation. **The original May18 h96 training submission receipt, checkpoint training metadata and final70 dataset-generation patch were not available in the local run archive**, and the existing Torch SSH connection was unavailable during repository assembly. Consequently this repository does not claim bit-for-bit retraining of that checkpoint from the generic scripts' defaults.

In particular, the retained MATLAB chunk generator defaults to **147 cases (7 angles ×21 contrasts)**, whereas the manuscript split is named `final70_smooth2d`. Do not run the147-case generator and label its output as the exact final70 dataset. The full-library smoothing/reference routines are included in `library_reference/`; recover the final70 run-specific generation/configuration receipts from Torch before claiming exact dataset reproduction. Do not guess the meaning of `low005` solely from its folder name.

The later May22 architecture-variant submission is included under `historical_jobs/` as evidence of the shared recipe and the same h96 parent/data paths. It launches different architectures; **it is not the original h96 submission**. Its explicit settings (variance weighting, stratified sampling, loss coefficients) are evidence for that later job only until cross-checked with May18 metadata.

## Dataset acquisition and split code

| File | Purpose |
|---|---|
| `library_reference/Paper3b_0_FuncPrepare_fullnods_smooth2d_one.m` | Full native-grid library preparation / smoothing stage |
| `library_reference/eigenvalue_6D_library_only_70_fullnods_smooth2d.m`, `run_case.m` | Library-only reference equilibria with condition overrides |
| `code/sixD_trainingdata_generation_prefavg_variance.m` | Aggregate preferred-angle response and response-variance labels from library MATs |
| `code/sixD_trainingdata_generation_prefavg_variance_chunk.m` | Independent target/condition chunks, avoiding one huge serialized MAT write |
| `run_sixD_trainingdata_generation_prefavg_variance_target.sh` | Torch MATLAB target job |
| `run_sixD_prefavg_variance_chunk_array.sh` | Torch MATLAB array, default1–147; historical147-case dataset |
| `build_random_row_split_variance_independent.py` | Stream variance-aware MATs into train/validation/test HDF5 shards |
| `build_split_prefavg_variance_from_smat_and_chunks.py` | Alternative assembly from complete S MAT plus C/I chunks |
| `run_build_sixd_random_split_prefavg_variance.sh` | CPU Slurm split job for that historical dataset layout |

These generators read the external `Paper3PlotingData` library under the MATLAB model tree. Labels include `FR_Response`, `FR_ResponseVar`, `FR_ResponseN` plus input/condition metadata. The network uses only the five `cosonly5d` inputs, even if OriSin2 or Gamma is present in the shards.

The manuscript's already-built split is:

```
/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/prefavg_variance_final70_smooth2d_20260514_032346/splits_randomrows_low005_upper_seed42_chunked
```

For a new dataset, after staging the correct library/generator and reviewing its condition set, the retained wrapper interface is:

```bash
# Torch only; these submit the generic retained dataset workflow, NOT verified May18 reproduction.
sbatch run_sixD_prefavg_variance_chunk_array.sh S NEW_DATASET_STAMP
sbatch run_sixD_prefavg_variance_chunk_array.sh C NEW_DATASET_STAMP
sbatch run_sixD_prefavg_variance_chunk_array.sh I NEW_DATASET_STAMP
# Wait for validated chunks before split assembly. Do not run both assembly alternatives.
python build_split_prefavg_variance_from_smat_and_chunks.py --help
python build_random_row_split_variance_independent.py --help
```

Run the shell scripts from their staged historical model-tree layout after rebinding paths. The archive does not contain the underlying library MATs. Keep split metadata and normalization with checkpoints; never refit normalization during export.

## Training

`code/train_sixd_single.py` contains the single-MLP training loop, variance-aware loss and target-rate stratified sampling. `common.py` defines model/configuration, normalization and feature modes; `dataset_hdf5.py` loads shards. The `soft_inv_sem` weighting derives bounded weights from response variance/count, with separately estimated reference scales for the loss spaces. Do not transfer 7D `.5/.5 +5TF+5SF` settings into this older model by assumption.

Use the supplied GPU wrapper interface after staging all three Python modules together:

```bash
# Torch; replace placeholders and append settings verified from the desired source run.
sbatch run_train_sixd_target_splitdir_mat.sh \
  S "$NEW_RUN" "$SPLIT" "$TRAIN_MAT" bs8192_steps40 \
  --feature_mode cosonly5d --hidden_dims 96,96,96 --activation_seq tanh,tanh,tanh
```

Repeat for C and I with target-appropriate input MATs and independently verified arguments. The command above is an **architecture example**, not the recovered complete May18 command. The wrapper's generic defaults do not establish the manuscript's exact variance/sampler/loss settings. Recover `<historical run>/train/{S,C,I}` configuration and checkpoint `train_config` before an exact repeat. Preserve the recorded source version and submitted shell script with each new run.

The historical resource shape is GPU1,8CPU,64GB,24h. Use the managed Torch Python environment described in `../docs/TORCH.md`. Review the wrapper's hardcoded code/log/TMP paths; its original paths are intentionally preserved.

## Export and evaluate

For a newly verified 5D checkpoint set, the supplied exporter interface is:

```bash
python code/export_pref6d_matlab.py \
  --model_s "$RUN/train/S/best.pt" \
  --model_c "$RUN/train/C/best.pt" \
  --model_i "$RUN/train/I/best.pt" \
  --out_dir "$RUN/new_matlab_export"
```

Export alone is not the full corrected h96 runtime: confirm normalization/architecture parity and use the verified extended-domain/high-L4E wrapper and fixed-point support. Do not overwrite the frozen `runtime/` manuscript predictors with a newly trained model.

For the preserved manuscript model in MATLAB:

```matlab
runtime = fullfile(repo, 'Previous 5D training', 'runtime');
addpath(runtime, '-begin');
which LocalResponse_6D_MLP_prefAngle
which predict_pref6D_S
% Argument order is physical, not training-column order:
yS = predict_pref6D_S(20000, 15000, 20, 42, 1);
```

The runtime wrapper uses the corrected high-L4E guard, IKp7/L6{5} in the evaluation driver and existing h96 parameter conventions. Current manuscript evaluation's three hash-matched utility dependencies are supplied in `posttraining/current_h96_utils/`. The full MATLAB network tree and simulation inputs remain external.

`posttraining/submit_paper2_h96_figure_validation.sh RUN_ROOT` submits an eight-task SNN simulation array, then a dependent corrected-h96 paired evaluation, then plotting. Before using it, stage `posttraining/*.m` and `current_h96_utils/` into `RUN_ROOT/code`, bind the project/runtime paths, and create the required external MATLAB assets. The evaluator asserts exact runtime and utility hashes, produces4680 paired points per population, and saves parameter/provenance metadata. This figure validation is separate from the70-condition fixed-point HC audit.

For matched library70 validation use:

```
/scratch/xh2906/librarySCI_runs/library_only_70_fullnods_h96_ikp7_l6p5_extmatch_20260630_235041
```

The historical completed h96 HC evaluation is under the trained run's `hpc_runtime_bundle/equilibrium_70_fullnods_smooth2d_extended_corrected_h96_ikp7_l6p5_matchedlib_20260702_010052`. HC is `mean(err_ode_true_all(end,251:end))`, with finite300-iteration traces and matched library parameters. Do not substitute an old77-condition cache or an unmatched full-library equilibrium. The complete original70-case NN Slurm driver is another remote-only dependency; the current figure evaluator and fixed-point routines included here do not reconstruct that scheduler receipt.

`manuscript_examples/` contains selected Draft scripts as lineage evidence. They retain external artifact paths and are not a portable figure-data bundle. In particular Figure0A uses a1000-iteration diagnostic solve; that does not change the separate300-iteration HC audit convention.
