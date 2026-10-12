# all615 data acquisition and original training sources

## Dataset definition

This workflow keeps all **615 valid native DS10 input points per block**; it does not thin them to 128, interpolate targets, or add a second 20% pool. There are 4,267 unique source units, each with 40 L6 blocks, for 170,680 native blocks. Unique eligible seven-input rows enter the dataset once.

The 17 `(TF,SF)` pairs are:

```
(2,2) (2,2.5)
(4,1) (4,2) (4,2.5) (4,3)
(10,1) (10,2) (10,2.5) (10,3)
(16,1) (16,2) (16,2.5) (16,3)
(24,2) (24,2.5) (24,3)
```

TF32 is an evaluation condition, not a supervised label in this split. Source orientations are `0:3.75:22.5`, foreground relative angles `0:3.75:90`, and model contrast values `[14,24,33,42,52,61,71,80,89,95]`. `plan.json` is the complete historical specification. The split uses seed 42 and row-wise 80/10/10 assignment after filtering; this is not a held-out-condition split.

| Target | Training rows | Validation rows | Test rows |
|---|---:|---:|---:|
| S | 36,762,205 | 4,598,608 | 4,596,151 |
| C | 48,307,951 | 6,042,483 | 6,037,552 |
| I | 46,965,488 | 5,871,988 | 5,868,732 |

Response ranges are S `[0.5,30]`, C `[0.5,100]`, I `[0.5,140]`, with positive L4I. Input normalization is copied from the original split rather than refitted: `feature_norm.json` SHA-256 `b34059b1260e5d7003ac8d642425c689b83c96e745247702b5ca2ae82bc91b0f`. Metadata includes additional fields that are not network inputs.

## Code map and execution order

| Source | Function |
|---|---|
| `code/preflight.py` | Extract original native grid in MATLAB; verify exactly 615 finite unique points |
| `code/source_simulator_original.m` | Preserved original simulator used for contract comparison |
| `code/run_ds10_all615.m` | Run each source unit over all native points; preserve existing simulator conventions |
| `code/run_source_unit.py` | Run/resume the MATLAB source unit, isolate temporary files, package results |
| `code/source_bundle.py` | Validate grid, source condition, finite rates and block consistency |
| `code/source_completion.py` | Verify all required source blocks before releasing assembly |
| `code/build_all615.py` | Stream blocks into S/C/I train/val/test HDF5 shards and write verification records |
| `code/pipeline.py` | Submit grid → pilot → bounded source waves → assembly → three GPU trainers; receipt-based controller |
| `submit_historical.py` | Original local-to-Torch staging/SSH launcher, with historical paths |
| `code/train_lowtfsf_grad01.py` | Executed training entry point; wraps the original trainer's gradient behavior |
| `code/train_sixd_single_longrun.py`, `common.py`, `dataset_hdf5.py` | Training loop, architecture/loss primitives, HDF5 loading |
| `source_trainer.py` | Earlier reference trainer retained by acquisition preflight |

Prefer **reusing the verified split** for architecture/LR comparisons. Rebuilding the dataset is expensive and unnecessary if the split and normalization hashes already match.

## Rebuild on Torch (only when deliberately needed)

1. Make a fresh run directory outside the repository, with `code/`, `logs/`, and `receipts/`. Copy this folder's `code/` and `source_trainer.py`, and make a run-local copy of `plan.json`.
2. Rebind `plan.json`'s run/output paths. Confirm `reference_run`, `reference_split`, source reuse manifest, MATLAB model tree and LGN libraries exist. Preserve the scientific fields and recorded hashes. Review hardcoded paths in `preflight.py`, `run_ds10_all615.m`, and `pipeline.py` for the account in use.
3. **Do not precreate `data_split_whitelist17` for `pipeline.py init`**: `init()` creates it itself. The normalization file included in the checkout is a reference for inspection and contract tests.
4. Activate `py310`; then, from the Torch login node, invoke the staged controller once:

```bash
python "$RUN/code/pipeline.py" init
```

This command submits real jobs, including held assembly and dependent S/C/I training. It is not a dry run. `pipeline.py advance` is the controller continuation, not a command to run repeatedly by hand.

5. Inspect `grid_validation.json`, source bundle status, `submitted_jobs.json`, and individual exclusive receipts. The pilot verifies L6 blocks 1 and 40 before full waves. Source failures are bounded to three attempts per unit; missing outputs do not become zero-valued training samples.
6. Require complete source validation, `split_manifest.json`, `dataset_validation.json` with `verified: true`, all nine HDF5 shards, and the normalization hash before standalone training.

For training on an existing split, use [the training guide](../../docs/TRAINING.md).
