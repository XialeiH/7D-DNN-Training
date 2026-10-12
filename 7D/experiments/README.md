# Executed experiment families

All scripts here retain historical paths and source checks. Inspect them and stage a separate setup directory before creating a new experiment. Never remove receipts to make a launcher run again.

## Architecture and activation/LR

`architecture/submit_variants.py` copies verified parent training code, changes only architecture/output paths, verifies the scientific settings, and submits S/C/I separately. `submit_trials.py` and `verify_trials.py` implement the subsequent ReLU / tanh-LR trials. They depend on original parent `submitted_jobs.json`, generated target shell scripts, submission receipts, normalization and dataset verification. These are not self-contained dataset builders.

ReLU runs retain LR `.01,.005,.002,.001,.0005,.0002`; lower-LR tanh starts at `.001`. Use each recorded command/diff as authority rather than defaults in the generic trainer.

## Smoothness fine-tuning

`prepare_submit.py` calibrates and freezes the source, tests the regularizer, builds run-local wrappers, and separates preparation from submission. Read its CLI dispatch before invocation. The source is the trained 512×5 ReLU S71/C47/I66 model, not a random initialization. `code/train_smooth.py` wraps the original objective; `code/smoothness.py` implements equally balanced finite-difference curvature of `log1p(response)` over periodic physical orientation, positive contrast, TF, SF.

| Strength | S lambda | C lambda | I lambda |
|---|---:|---:|---:|
| Low | .007418810285198047 | .00856027078682517 | .010909750929757177 |
| Medium | .02967524114079219 | .03424108314730068 | .04363900371902871 |
| High | .11870096456316875 | .1369643325892027 | .17455601487611483 |

Axis scales come from the original CPU source calibration. Nominal 5/20/80% calibration shares are not exact initial GPU shares because deterministic CPU/CUDA banks differ. Compare curvature against each job's actual initial GPU bank. Selection is **fitting validation + lambda × fixed-bank R**; report fitting loss, raw RMSE, R and combined score separately. LR `2e-4 → 5e-5 → 1e-5` follows the original fitting plateau rule. Epoch-0 fallback retains trained source weights.

`posttraining/checkpoint_gate.py`, `attach_post.py` and `launch_smooth_post.py` freeze and evaluate only after all S/C/I jobs end. Missing/invalid checkpoints stop the gate. `audit_saved_outputs.py` independently checks saved maps/HC/provenance. Copy the shared post-training templates into the staged layout expected by the attachment code; the archive does not contain live receipts or completed run state.

## C-seed retry

The original 1024×4 ReLU C seed42 saturated at zero output. `c_seed_retry/learning_health.py` defines the retry criterion: at least 10 logged epochs, last three validation losses flat within `1e-8`, and last three post-initial log/relative/TF/SF gradient diagnostics all zero. Slow learning with nonzero gradients is not collapse. Infrastructure and administrative failures do not automatically count as failed seeds.

The historical authorized range was 43–47, five total attempts. `submit_C_retry.py prepare SEED` then `submit SEED` preserves exact S84/I65 checkpoint hashes, original training code/settings and resources; only C seed and output paths change. A next seed requires a documented terminal learning failure of the previous attempt. `attach_retry_post.py` installs the terminal checkpoint gate and post-training continuation.

**This search ended successfully at seed43.** It selected C epoch40, validation loss `0.17127785384818736`, raw RMSE `5.501602491779959`; training was administratively cancelled after logged epoch50. S84/I65 were unchanged. Do not execute seeds44–47 or restart the historical seed43 run. The policy and scripts are preserved as provenance, not an active retry request.

`recover_retry_archive_20261005.py` and `posttraining/recovered_archive/run_archive_exports.py` preserve the later presentation-only recovery after valid maps had been saved but MATLAB shutdown failed. Their presence is not permission to repeat the recovery; validate saved data and reconcile exclusive receipts first.
