# Full 7D post-training on Torch

## Scientific contract

Freeze one verified S/C/I checkpoint set after all relevant training jobs end. Use the extended-corrected h96 network, high-L4E guard, IKp7, L6{5}, canonical geometric LGN mapping at all contrasts, original initialization/damping, and 300 iterations. Preserve condition, network-parameter, runtime and reference hashes in every saved result.

The current all615 evaluation contains **139 atlas conditions** with SF≤10, **97 dense conditions**, full-angle orientation at model contrast95 for three default pixels, and **88 available matched representative references** at the October5 delivery. Reinventory representatives for any new delivery; `representative192` is a historical directory name, not proof of 192 usable references.

Both TF/SF x axes and response y axes are **linear** for these deliveries. Contrast plots use the actual model-to-real contrast conversion and PCHIP interpolation. Never infer axis transforms from an inherited script name.

## Sources and stages

| Stage | Code / expected evidence |
|---|---|
| Terminal gate and freeze | `../experiments/{smoothness/posttraining,c_seed_retry}/checkpoint_gate.py`; terminal accounting, finite checkpoints, selection and lineage |
| Launcher | `launch_trial_post.py`, `launch_smooth_post.py`, or `launch_retry_post.py`; frozen `checkpoints/{S,C,I}.pt`, `checkpoint_provenance.json`, pinned code |
| Export | `template_code/export_pref6d_matlab.py`; SCI predictors plus bundle and LocalResponse |
| Mapping / parity | `check_h96_library_mapping.m`, `verify_runtime.m`; 36 mapping cases, 64 physical MATLAB/PyTorch samples including TF32, five exact baseline triggers |
| Atlas | `prepare_submit_logtfsf_atlas102.py`, `build_submit_logtfsf_atlas102.py`; cases.tsv, runtime provenance, 139 saved MATs, HC CSV/summary |
| Dense curves | `submit_dense.py`, `run_dense_task.m`, `dense_nn_driver.m`, `plot_dense.py`; cases.tsv, 97 saved maps and curves |
| Orientation / sparse TF-SF | `submit_runtime_tuning_curves_quota.py`, `submit_runtime_dense_tuning.py`, MATLAB plotting scripts |
| Representative comparisons | `representative_source`; launcher rebinds paths and uses only inventoried matched references |
| Numerical audit | `../experiments/c_seed_retry/audit_saved_retry.py` or smoothness `audit_saved_outputs.py` |
| Vector PDFs and archive | `prepare_archive.py`, `run_archive_exports.py`, `export_archive_curves*.m`, `export_comparisons.m` |

`reference_atlas` preserves an executed all615 h512 atlas driver and runtime support, without its fitted predictor weights. It documents the fixed-point calculation. It is **not** the 1024×4 model's frozen runtime. Generate predictors from the chosen checkpoints and verify parity.

## Stage a new evaluation

The historical attachment scripts create a specific run layout and record exclusive submission intents. They depend on parent manifests and live training job IDs; do not call an attachment or launcher twice. The normal layout is:

```
RUN/
  code/                         # matching training source
  train/{S,C,I}/best.pt
  submitted_jobs.json
  posttraining_7d_tf32_globalall7/
    checkpoint_gate_passed.json
    template_code/              # copied + path-rebound templates and SHA receipt
    code/
    checkpoints/{S,C,I}.pt
    atlas_139/runtime_h96iter/
    dense_tuning/
    representative192/
    tuning_curves/
    local_archive/
```

Read each chosen `attach_*.py` to stage its setup directory. For example, retry attachment expects `checkpoint_gate.py`, `learning_health.py`, `launch_retry_post.py`, `run_archive_exports.py`, `local_reference_inventory.json`, and `template_code/` as siblings; the repository organizes these for browsing, so they must be assembled into that setup layout. The inventory must be freshly generated from available reference data; a placeholder list is not valid. The attachment writes `template_source_sha.json` after rebinding paths. Smoothness and retry gates are deliberately specific to their training lineages and must not be applied unchanged to an arbitrary new scratch run.

The new training-only wrapper in `jobs/` does not supply those run-specific post-training receipts. For a new scratch run, inspect terminal checkpoints/configuration, freeze them, stage the full reference/runtime assets and parity inputs, and use the ordinary trial workflow as the starting point. Do not fabricate a `checkpoint_gate_passed.json` to bypass validation.

## Manual stage commands (already prepared run)

These commands show the actual stage interfaces. They assume the freeze, reference inventory, code/path rebinding and runtime prerequisites have been completed. They are **not** a substitute for those preparations, and submission commands create real jobs.

```bash
P="$RUN/posttraining_7d_tf32_globalall7"
CODE="$P/code"
A="$P/atlas_139"
EXTRA=/scratch/xh2906/librarySCI_runs/h96_tfsf7d_extra50_full_libequ70_20260905

python "$CODE/prepare_submit_logtfsf_atlas102.py" \
  --profile all615post --atlas-root "$A" \
  --model-s "$P/checkpoints/S.pt" --model-c "$P/checkpoints/C.pt" \
  --model-i "$P/checkpoints/I.pt" --require-transform identity \
  --require-tf-max 32 --save-operating-points --extra50-root "$EXTRA" --mode prepare

# Run verify_runtime(P) in a MATLAB CPU job with the launcher's parity_inputs.mat.
# Require runtime_verification.json to pass before submitting simulations.

python "$CODE/build_submit_logtfsf_atlas102.py" \
  --profile all615post --root "$A" --runtime "$A/runtime_h96iter" \
  --save-operating-points --extra50-root "$EXTRA" --mode submit

python "$CODE/submit_dense.py" --post-root "$P" --runtime "$A/runtime_h96iter"
```

The atlas builder also reads the historical source atlas (`SOURCE` constant: `h96_tfsf7d_h128x3_full_library_fixedpoint_atlas_102_20260726`) for base cases and driver. Sparse TF/SF submission reads `SOURCE_TUNING` (`h96_tfsf7d_capacity_h128x4_logtfsf_nodistill_combined_20260727/tfsf_tuning_angle0_contr100`). Preserve those source assets or explicitly stage/rebind verified copies. Do not replace their condition tables with guessed lists. The launcher performs necessary wrapper adaptations; invoking an unprepared raw template can retain historical plotting/path behavior.

The finalizer may schedule a successor while arrays/tuning finish. Derive child IDs from `launch.json`, `launch_complete.json`, atlas/dense/representative submission manifests, `tuning_curve_submission.json`, `archive_submission.json` and corresponding intent receipts. Missing maps while an array is active are pending work, not numerical failure.

## Required numerical and visual checks

- Confirm frozen checkpoint SHA equals the evaluated runtime provenance, model input order and normalization. MATLAB/PyTorch physical parity tolerance is `<2e-4`; include TF32 and fixed baseline triggers.
- Independently recompute **atlas HC** as `mean(err_ode_true_all(end,251:end))`. Require at least 300 finite trace columns, finite SCI maps and agreement with the per-case CSV and aggregate summary for all139 cases.
- **Representative final-state HC** is `sqrt(mean(.8*(E_library-E_NN).^2 + .2*(I_library-I_NN).^2))`; this is a different quantity. Compare exact matched reference hashes, conditions, parameters, L6 settings, iteration count and geometric mapping.
- Require all97 dense maps finite with300 iterations. Curves cover28 contrasts at two angles,21 TF samples including32,20 SF samples up to10. Verify E=.6923S+.3077C and three default orientation pixels.
- Report `final_step_max > .01` as an unsettled residual, separately from HC. Read actual TF24→32 values at both default TF/SF pixels; do not assume response decreases.
- Optional one-step library operating-target/point queries may be outside library support. They must be empty and `valid:false` when unavailable. This does not allow missing/NaN core equilibria, HC traces, or invented diagnostic values.
- Decode and visually inspect all four main vector tuning PDFs, every current representative paired PDF, and all12 final tuning PNGs. Check readable linear axes, legends, clipping and shared comparison color scales. A 150dpi PNG rendered from a verified vector PDF is acceptable with recorded provenance.
- Verify all cached extra50 reference MAT/PNG hashes afresh; reuse a prior visual review only for exact matching hashes. Keep PNG/render QA outside the main PDF-only delivery folder.

The recovered archive wrapper is a separately preserved version used after valid saved maps survived a MATLAB shutdown error. Do not weaken numerical requirements to make a failed wrapper appear successful.
