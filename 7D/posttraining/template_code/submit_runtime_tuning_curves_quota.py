#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path


ACCOUNT = "torch_pr_482_general"
CODE_ROOT = Path(__file__).resolve().parent
SOURCE_TUNING = Path(
    "/scratch/xh2906/librarySCI_runs/"
    "h96_tfsf7d_capacity_h128x4_logtfsf_nodistill_combined_20260727/"
    "tfsf_tuning_angle0_contr100"
)
TF_VALUES = (0.5, 1.0, 2.0, 4.0, 10.0, 16.0, 24.0, 32.0)
SF_VALUES = (0.5, 1.0, 2.0, 2.5, 3.0, 5.0, 10.0)


def submit(args: list[str]) -> str:
    return subprocess.check_output(args, text=True).strip().split(";", 1)[0]


def replace_paths(source: Path, destination: Path, root: Path, runtime: Path) -> None:
    text = source.read_text()
    text, root_count = re.subn(
        r"root\s*=\s*\[.*?\];", f"root = '{root}';", text, count=1, flags=re.DOTALL
    )
    text, runtime_count = re.subn(
        r"runtime\s*=\s*fullfile\(root,\s*'runtime_h96iter'\s*\);",
        f"runtime = '{runtime}';",
        text,
        count=1,
    )
    if root_count != 1 or runtime_count != 1:
        raise RuntimeError(
            f"Failed to patch TF/SF task paths: root={root_count}, runtime={runtime_count}"
        )
    destination.write_text(text)


def link_directory(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def copy_library_aliases_with_tf32(source: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        else:
            shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for item in source.iterdir():
        (destination / item.name).symlink_to(item.resolve())
    template = source / "Func200V4D2-L6RealCtrlL4-Ang0.0-SF2.5TF24-Small.mat"
    if not template.exists():
        raise FileNotFoundError(template)
    tf32_alias = destination / "Func200V4D2-L6RealCtrlL4-Ang0.0-SF2.5TF32-Small.mat"
    tf32_alias.symlink_to(template.resolve())
    return tf32_alias


def write_tfsf_cases(path: Path) -> int:
    rows = [{"task": task, "tf": tf, "sf": 2.5} for task, tf in enumerate(TF_VALUES)]
    for sf in SF_VALUES:
        if sf != 2.5:
            rows.append({"task": len(rows), "tf": 10.0, "sf": sf})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "tf", "sf"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--post-root", type=Path)
    parser.add_argument("--atlas-root", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--profile", default="round3qfinal")
    args = parser.parse_args()

    post_root = args.post_root or (args.run_root / "posttraining_7d")
    atlas_root = args.atlas_root or (post_root / "equilibrium_102")
    runtime = args.runtime or (args.run_root / "runtime_h96iter")
    orientation_root = post_root / "tuning_curves"
    tfsf_root = post_root / "tfsf_tuning_angle0_contr100"

    required = [
        atlas_root / "hcnorm_summary.json",
        atlas_root / "cases.tsv",
        runtime / "LocalResponse_6D_MLP_prefAngle.m",
        CODE_ROOT / "plot_h96_7d_orientation_contrast.m",
        CODE_ROOT / "plot_sftf_tuning_curves.m",
        SOURCE_TUNING / "cases.tsv",
        SOURCE_TUNING / "code/run_tfsf_nn_task.m",
        SOURCE_TUNING / "code/eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition_tfsf.m",
        SOURCE_TUNING / "cache_alias",
        SOURCE_TUNING / "library_alias",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    orientation_root.mkdir(parents=True, exist_ok=True)
    for path in (tfsf_root / "code", tfsf_root / "logs", tfsf_root / "equilibrium"):
        path.mkdir(parents=True, exist_ok=True)

    orientation_script = post_root / "submit_orientation_contrast.sh"
    orientation_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'module purge\nmodule load matlab/2025b\nexport TMPDIR="/tmp/xh2906_aug20_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\nexport MATLAB_PREFDIR="$TMPDIR/prefs"\nmkdir -p "$TMPDIR" "$MATLAB_PREFDIR"\n'
        f"matlab -batch \"addpath('{CODE_ROOT}'); "
        f"plot_h96_7d_orientation_contrast('{atlas_root}','{orientation_root}')\"\n"
    )
    orientation_script.chmod(0o755)
    orientation_job = submit(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            "--cpus-per-task=2", "--mem=8G", "--time=00:30:00",
            f"--job-name={args.profile}_oricon",
            f"--output={post_root}/orientation_contrast.%j.out",
            f"--error={post_root}/orientation_contrast.%j.err",
            str(orientation_script),
        ]
    )

    case_count = write_tfsf_cases(tfsf_root / "cases.tsv")
    if case_count != 14:
        raise RuntimeError(f"Expected 14 unique TF/SF cases, found {case_count}")
    shutil.copy2(
        SOURCE_TUNING / "code/eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition_tfsf.m",
        tfsf_root / "code/eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition_tfsf.m",
    )
    replace_paths(
        SOURCE_TUNING / "code/run_tfsf_nn_task.m",
        tfsf_root / "code/run_tfsf_nn_task.m",
        tfsf_root,
        runtime,
    )
    shutil.copy2(CODE_ROOT / "plot_sftf_tuning_curves.m", tfsf_root / "code")
    link_directory(SOURCE_TUNING / "cache_alias", tfsf_root / "cache_alias")
    tf32_library_alias = copy_library_aliases_with_tf32(
        SOURCE_TUNING / "library_alias", tfsf_root / "library_alias"
    )

    array_script = tfsf_root / "submit_equilibrium_array.sh"
    array_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'module purge\nmodule load matlab/2025b\nexport TMPDIR="/tmp/xh2906_aug20_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\nexport MATLAB_PREFDIR="$TMPDIR/prefs"\nmkdir -p "$TMPDIR" "$MATLAB_PREFDIR"\n'
        f"cd {tfsf_root / 'code'}\nmatlab -batch \"run_tfsf_nn_task\"\n"
    )
    array_script.chmod(0o755)
    array_job = submit(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            f"--array=0-{case_count - 1}%{case_count}", "--cpus-per-task=8", "--mem=32G", "--time=01:00:00",
            f"--job-name={args.profile}_sftf",
            f"--output={tfsf_root}/logs/equ.%A_%a.out",
            f"--error={tfsf_root}/logs/equ.%A_%a.err",
            str(array_script),
        ]
    )

    plot_script = tfsf_root / "submit_plot.sh"
    plot_script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'module purge\nmodule load matlab/2025b\nexport TMPDIR="/tmp/xh2906_aug20_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"\nexport MATLAB_PREFDIR="$TMPDIR/prefs"\nmkdir -p "$TMPDIR" "$MATLAB_PREFDIR"\n'
        f"matlab -batch \"addpath('{tfsf_root / 'code'}'); "
        f"plot_sftf_tuning_curves('{tfsf_root}',[5 10; 1 10])\"\n"
    )
    plot_script.chmod(0o755)
    plot_job = submit(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            "--cpus-per-task=2", "--mem=8G", "--time=00:30:00",
            f"--dependency=afterok:{array_job}", f"--job-name={args.profile}_sftfplot",
            f"--output={tfsf_root}/logs/plot.%j.out",
            f"--error={tfsf_root}/logs/plot.%j.err",
            str(plot_script),
        ]
    )

    verify = (
        "set -euo pipefail; "
        f"test $(find {tfsf_root}/equilibrium -name 'eig_16HC_*.mat' | wc -l) -eq 14; "
        f"test -s {orientation_root}/orientation_tuning_contr95_default3pixels.png; "
        f"test -s {orientation_root}/contrast_tuning_angle0_pixel5_10.png; "
        f"test -s {orientation_root}/contrast_tuning_angle22p5_pixel1_10.png; "
        f"test -s {tfsf_root}/plots/tf_tuning_angle0_contr100_pixel5_10.png; "
        f"test -s {tfsf_root}/plots/sf_tuning_angle0_contr100_pixel5_10.png; "
        f"printf 'complete\n' > {post_root}/tuning_curves_complete.txt"
    )
    verify_job = submit(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            "--cpus-per-task=1", "--mem=1G", "--time=00:05:00",
            f"--dependency=afterok:{orientation_job}:{plot_job}",
            f"--job-name={args.profile}_curvecheck",
            f"--output={post_root}/curvecheck.%j.out", f"--error={post_root}/curvecheck.%j.err",
            "--wrap", f"bash -lc {shlex.quote(verify)}",
        ]
    )

    state = {
        "run_root": str(args.run_root),
        "runtime": str(runtime),
        "orientation_contrast_job": orientation_job,
        "tfsf_array_job": array_job,
        "tfsf_plot_job": plot_job,
        "verification_job": verify_job,
        "tfsf_case_count": case_count,
        "tf_values": list(TF_VALUES),
        "sf_values": list(SF_VALUES),
        "tf32_library_alias": str(tf32_library_alias),
        "tf32_hcnorm_valid": False,
    }
    (post_root / "tuning_curve_submission.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
