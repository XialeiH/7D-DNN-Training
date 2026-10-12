# Training and GPU submission

## Original all615 objective

The executed entry point is `7D/all615/code/train_lowtfsf_grad01.py`, which imports the long-run trainer beside it. Use its supplied `common.py` and `dataset_hdf5.py`; mixing similarly named modules from another experiment can change model or loss behavior.

The objective is `0.5 * variance-weighted log1p RMSE + 0.5 * variance-weighted relative RMSE + 5 * TF-unimodality + 5 * SF-unimodality`. The relative denominator is `y + 1`; raw MSE has coefficient zero. The model uses a softplus output. Existing fixed baseline rules return S=2.5, C=4.5, I=12.5 under their defined trigger conditions; the known baseline/label conflict was retained, not corrected in these trials.

The wrapper applies output-gradient factor 0.1 when **both** physical TF and SF lie in `{0.5,1}`. Forward values are unchanged. The whitelist is enforced by `whitelist_pairs.py`. TF/SF feature normalization is linear (`identity`), despite some inherited filenames saying `logtfsf`.

Original scratch settings: AdamW, AMP, weight decay `1e-5`, batch 8192, evaluation batch 65536, up to 40 train and 10 evaluation batches per logged epoch. Stages are 200 epochs, at most 40 stages (8,000 cap), with fitting-validation plateau logic. A logged epoch is not a full pass over the enormous dataset. For the 1024×4 ReLU trial, LR values are `.01,.005,.002,.001,.0005,.0002`; the exact arguments are in `configs/h1024x4_relu_original.json`.

## Run a new scratch experiment on the verified split

The new repository convenience wrapper preserves the recorded S/C/I argument lists, replacing only interpreter/script location, split and output paths. It supports the three-target original scratch recipe. It refuses C-only retry and smoothness recipes: those need their dedicated lineage/preflight gates.

On the Torch login node, with `py310` active:

```bash
export REPO=/path/to/7D-DNN-Training
export SPLIT=/path/to/verified/data_split_whitelist17
export RUN=/home/your_user/librarySCI_runs/new_unique_run_name

# Prints three full sbatch commands and scripts; creates nothing and submits nothing.
python "$REPO/jobs/submit_training.py" \
  --recipe "$REPO/configs/h1024x4_relu_original.json" \
  --split "$SPLIT" --run-root "$RUN"

# After checking the printed configuration, submit exactly once.
python "$REPO/jobs/submit_training.py" \
  --recipe "$REPO/configs/h1024x4_relu_original.json" \
  --split "$SPLIT" --run-root "$RUN" --submit
```

Use `--account` or `--python` only to select the allocated account or existing interpreter. The wrapper checks the split's verified status and normalization hash, copies the exact trainer sources into the new run, and saves generated scripts and submission receipts. It rejects an existing run directory. If submission is interrupted after an intent is written, inspect that receipt and Slurm before retrying; do not delete the directory and blindly resubmit.

This convenience wrapper submits training only. It does not attach a post-training controller automatically, because the historical controllers require additional frozen-template/reference/gate state. Follow the post-training guide after all three targets are terminal.

## What to inspect

For every target, preserve `best.pt`, configuration, normalization, train/evaluation logs, gradient diagnostics and final scheduler accounting. Independently load checkpoint metadata and finite tensors. Compare the selected metric to logged minima, accounting for the trainer's selection rule; do not substitute raw RMSE for the objective used for selection.

A valid checkpoint can survive an administrative cancellation or time limit. Report selected and last logged epochs plus terminal reason; do not claim completion of the 8,000-epoch cap. In a scratch run, selected epoch 0 is untrained. In a warm-start smoothness run, epoch 0 is the trained source fallback.

## Executed variants

- `7D/experiments/architecture/submit_variants.py`: architecture-only trials from verified parent sources/data.
- `submit_trials.py`: ReLU activation and lower-LR tanh variants, preserving the original objective.
- `7D/experiments/smoothness`: added calibrated finite-difference curvature with frozen source S71/C47/I66 and fresh optimizers.
- `7D/experiments/c_seed_retry`: C-only random-seed retry, reusing exactly evaluated S84/I65; successful seed 43 selected C40.

The historical recipes are examples of actual executed commands. They are not instructions to rerun completed experiments. See [experiment notes](../7D/experiments/README.md).
