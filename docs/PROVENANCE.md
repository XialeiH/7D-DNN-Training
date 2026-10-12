# Provenance and verification

Repository assembled from the local Neuroscience_Project source archive on2026-10-12. `provenance/source_manifest.json` records each imported file's original project-relative location, role and SHA-256. Imported source is unmodified, including historical filenames and paths. Newly written README files, `jobs/submit_training.py` and `tools/` are repository organization/validation helpers, not scripts claimed to have run in the historical experiments.

## Strongly verified identities

- The seven all615 trainer/support source hashes match the delivered1024×4 ReLU Cseed43 package's `Training records/submitted_jobs.json`.
- The all615 normalization SHA is recorded and checked by the repository verifier and training submission helper.
- The corrected5D LocalResponse and three predictors match the hardcoded hashes in the manuscript paired evaluator; the wrapper hash also matches the corrected h96 runtime record.
- The three current manuscript evaluation utilities were located by their required hashes, rather than selected by filename alone.
- `configs/*.json` arguments were extracted from executed submission records. Split/output paths alone are placeholders. The C-only and smoothness recipes retain lineage dependencies and are not generic scratch recipes.

## What has not been claimed

A source SHA proves which bytes were copied, not that a newly assembled deployment has passed MATLAB/HPC end-to-end execution. The source-integrity/static checks and CPU contract checks are recorded in `provenance/validation_report.json`. No training, dataset rebuilding, fixed-point simulations or email delivery was initiated while organizing this repository.

The original May18 5D training config/receipt and final70 data-generation patch were absent locally, and remote SSH access was unavailable. Their absence is explicit in the5D guide. The later architecture-variant recipe is not relabeled as the original h96 training command. Existing fitted manuscript inference remains precisely identified and included.

Large raw/generated data, `.pt` checkpoints, PDFs/PNGs, private delivery/email receipts, SSH/authentication material, monitor state and completed run packages are excluded. External absolute paths in historical source are dependency locations, not a deployment guarantee. This repository does not grant a new license for external model/data assets; existing source notices remain intact.
