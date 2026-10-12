# Torch execution guide

## Environment

The recorded runs use NYU Torch, account `torch_pr_482_general`, MATLAB `2025b`, and the existing Python 3.10 conda environment. Authenticate interactively using your institution's normal SSH/MFA procedure; the repository contains no credentials or SSH control sockets.

On Torch, before Python work:

```bash
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate py310
export PYTHONNOUSERSITE=1
python -c 'import sys, torch, numpy, scipy, h5py; print(sys.executable); print(torch.__version__)'
```

Historical scripts use `/scratch/xh2906/conda/envs/py310/bin/python`. For MATLAB stages also load `module load matlab/2025b`. Do not install packages into the system Python or replace a working CUDA environment just to match a new package release. `requirements.txt` lists imported libraries; it is not a recovered version lockfile.

Clone this repository into a code directory. Keep run outputs outside the checkout. For the commands below, set `REPO` to the absolute checkout path. The folder `Previous 5D training` contains spaces and must be quoted.

## Storage and scheduler

| Stage | Recorded resources / policy |
|---|---|
| all615 grid preflight | CPU, 1 CPU / 16 GB / 1 h |
| source pilot and simulation waves | MATLAB, 8 CPU / 24 GB / 48 h, `cs`; controller limits active source tasks |
| all615 HDF5 assembly | CPU, 8 CPU / 64 GB / 12 h; held until source verification passes |
| scratch S/C/I training | One GPU / 8 CPU / 64 GB / 24 h per target |
| dense tuning | MATLAB CPU array, 8 CPU / 32 GB / 1 h, at most 8 concurrent cases |
| representative maps | MATLAB CPU array; launcher records actual resource settings and indices |
| plot/export | CPU/MATLAB; no training GPU required |

These are recorded settings, not promises of current partition availability. Inspect the generated `sbatch` command before submission. Submit from the login node; do not run simulation/training there. Use `squeue -j ID1,ID2` and scoped `sacct -j ...` checks, not continuous polling.

Scratch reached its inode quota during these runs. Later training/post-training used dedicated Home output directories and run-local symlinks. Preserve existing symlinks. Check `myquota` before large writes; do not delete unrelated runs. Use node-local `$SLURM_TMPDIR` or `/tmp/$USER/...` for temporary MATLAB preferences and Matplotlib cache. Keep durable checkpoints, receipts, logs and outputs in the chosen run directory.

## External assets that must be present

| Asset | Historical location / purpose |
|---|---|
| MATLAB model tree | `/scratch/xh2906/NYU-Vision-2Drive-main`: `Utils`, `Data`, `MFpVLDEComput`, network and LGN routines |
| Network parameter file | `AllMFPixPara_Paper2TuneFig1V4D2.mat`, resolved through the MATLAB model/data path |
| LGN spike data | `Data/Paper2_NetworkTuning/Fig1V4/Paper3LGNSpikeData`; required by simulation |
| Original acquisition reference | `7D/all615/plan.json` → `reference_run`, `reference_split`; supplies recorded command/source hashes and normalization |
| Source reuse manifest | `/scratch/xh2906/training_source_equilibria_20260913/cases.json`; existing 128-point source blocks are reused, remaining native points simulated |
| Shared all615 split | `.../librarySCI_runs/h96_tfsf7d_h128x4_scratch_fixedbaseline_goodFRgt20_whitelist17_all615DS10_20260928_original_loss/data_split_whitelist17` |
| 7D runtime patch template | `prepare_submit_logtfsf_atlas102.py` → `PATCH_TEMPLATE`; fixed-point and high-L4E support routines |
| Extra50 full-library references | `/scratch/xh2906/librarySCI_runs/h96_tfsf7d_extra50_full_libequ70_20260905` |
| Representative condition manifest | `/scratch/xh2906/smooth7_representative192_20260919/cases.json` |
| Matched representative library maps | `/scratch/xh2906/representative_full_library_192_20260913/equilibrium_ready13` |
| Current 5D split and matched library70 | Listed in `Previous 5D training/README.md` |

The original MATLAB model assets and large reference data are not bundled here. A missing dependency must be supplied from the recorded source; substituting a different cache can silently change the science. `provenance/external_paths.json` inventories absolute path references in the imported code to help stage a different account/filesystem.

## Safe relocation

Copy historical sources into an isolated run-local `code` or `template_code` directory first. Rebind only paths/account/interpreter settings as needed, record a diff and new SHA manifest, and retain the imported originals. Several scripts use `Path(__file__).parents[1]` as their run root. Preserve that layout. Do not run acquisition `pipeline.py init` inside this Git checkout: it writes state and can submit dataset and training jobs.

Do not launch a second controller after an uncertain `sbatch` response. Reconcile `*_intent.json`, `submission*.json`, and actual job IDs before retrying. Administrative cancellation, node/quota errors and MATLAB shutdown errors are different from model failure. Saved results after an error are usable only after independent numerical/provenance checks.
