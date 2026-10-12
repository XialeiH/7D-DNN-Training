# 7D DNN training and evaluation

Source archive and operating guide for the all615 h96 TF/SF surrogate experiments, with the earlier manuscript h96 5D workflow in **[Previous 5D training](Previous%205D%20training/README.md)**.

Three networks predict S, C, and I responses. The 7D input order is `[OriCos2, Contrast, L6, L4E, L4I, lgnTF, lgnSF]`; combined excitatory response is `E = 0.6923*S + 0.3077*C`. Network width/depth is an experiment setting, not the spatial h96 model size.

## Start here

1. [Torch environment, storage, and dependencies](docs/TORCH.md).
2. [Acquire and verify the all615 dataset](7D/all615/README.md).
3. [Train a new model and submit GPU jobs](docs/TRAINING.md).
4. [Export, evaluate, and plot a completed model](7D/posttraining/README.md).
5. [Architecture, smoothness, and seed experiments](7D/experiments/README.md).
6. [Previous 5D training and manuscript lineage](Previous%205D%20training/README.md).
7. [Provenance, verification, and known gaps](docs/PROVENANCE.md).

## Repository layout

| Directory | Contents | Where to run |
|---|---|---|
| `7D/all615/code` | Native DS10 simulation, source validation, HDF5 splits, original trainer and loss tests | Torch CPU/MATLAB for data; GPU for training |
| `7D/all615/plan.json` | Historical 17-pair / 615-point acquisition plan | Read before staging a new run |
| `7D/experiments` | Executed architecture/LR, smoothness, and C-seed submission/gate scripts | Torch; historical paths need review |
| `7D/posttraining` | MATLAB export, mapping/parity, 139-case atlas, dense curves, representative comparisons, archive export | Torch CPU/MATLAB |
| `configs` | Recorded training arguments, with only output/split placeholders substituted | Readable recipes |
| `jobs` | New repository entry point for submitting the recorded scratch-training recipe | Torch login node; training runs on GPU nodes |
| `Previous 5D training` | Earlier 5-input trainer, acquisition scripts, corrected manuscript predictors, evaluation and Slurm code | Torch; lightweight predictor calls also work locally in MATLAB |
| `tools` | Source-integrity/static checks | Local or Torch, no jobs submitted |
| `provenance` | SHA-256 and original project-relative path for every imported source | Audit |

## What is preserved

Imported scientific files are byte-for-byte copies; filenames such as `sixd`, `pref6D`, `logtfsf`, `atlas102`, and `representative192` are historical. Read each workflow's README for actual input dimension, transform, and case count. Do not infer these from filenames.

This is a code repository, **not a standalone copy of the laboratory's MATLAB network/data tree**. External LGN spike libraries, network parameter MAT files, training HDF5 shards, reference equilibria, and 7D `.pt` weights remain outside Git. The small 5D MATLAB predictors do include the fitted coefficients actually resolved by manuscript evaluation, so that exact inference source is preserved.

Historical launchers retain their original absolute paths and may submit many jobs immediately. They are included for reproducibility and inspection, not as a one-command installer. Use a fresh output directory and read the stage-specific instructions. Repository assembly did not start or change any cluster jobs.

## Validate the checkout

```bash
python3 tools/verify_repository.py
python3 tools/run_contract_checks.py
python3 tools/test_submission.py
```

The second command requires NumPy, PyTorch, h5py, SciPy and Matplotlib. It runs existing scientific contract tests in temporary copies, without modifying archived source or contacting Slurm. Full MATLAB/HPC evaluation requires the external assets listed in the guide.
