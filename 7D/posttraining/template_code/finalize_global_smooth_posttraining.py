#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
from pathlib import Path


ACCOUNT = "torch_pr_482_general"
PYTHON = Path("/scratch/xh2906/conda/envs/py310/bin/python")
SCRIPT = Path(__file__).resolve()
TUNING_LAUNCHER = Path(__file__).with_name("submit_runtime_dense_tuning.py")
MAX_CHECKS = 96
ATLAS_NAME = "atlas_102"
TUNING_JOB_KEYS = (
    "orientation_contrast_job",
    "tfsf_array_job",
    "tfsf_plot_job",
    "verification_job",
)
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}


def submit_check(run_root: Path, profile: str, attempt: int) -> str:
    post_root = run_root / "posttraining_7d_tf32_globalall7"
    logs = post_root / "logs"
    command = (
        f"{PYTHON} {SCRIPT} --run-root {shlex.quote(str(run_root))} "
        f"--profile {shlex.quote(profile)} --attempt {attempt} --atlas-name {shlex.quote(ATLAS_NAME)}"
    )
    return subprocess.check_output(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:15:00",
            "--begin=now+15minutes", f"--job-name={profile}postcheck",
            f"--output={logs}/postcheck.%j.out", f"--error={logs}/postcheck.%j.err",
            "--wrap", command,
        ],
        text=True,
    ).strip()


def query_job_state(job_id: str) -> str:
    output = subprocess.check_output(
        [
            "sacct", "-X", "-n", "-P", "-j", str(job_id),
            "-o", "JobIDRaw,State",
        ],
        text=True,
    )
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] == str(job_id):
            return fields[1].split("+", 1)[0]
    return "UNKNOWN"


def tuning_job_states(path: Path) -> dict[str, str]:
    submission = json.loads(path.read_text())
    return {
        key: query_job_state(str(submission[key]))
        for key in TUNING_JOB_KEYS
    }


def archive_failed_submission(path: Path, attempt: int) -> Path:
    archived = path.with_name(f"tuning_curve_submission.retry{attempt:02d}.json")
    if archived.exists():
        archived.unlink()
    path.rename(archived)
    return archived


def main() -> None:
    global ATLAS_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--atlas-name", default="atlas_102")
    args = parser.parse_args()
    ATLAS_NAME = args.atlas_name

    post_root = args.run_root / "posttraining_7d_tf32_globalall7"
    atlas_root = post_root / ATLAS_NAME
    state_path = post_root / "posttraining_finalize_state.json"
    post_root.mkdir(parents=True, exist_ok=True)
    (post_root / "logs").mkdir(exist_ok=True)

    hcnorm_summary = atlas_root / "hcnorm_summary.json"
    if not hcnorm_summary.is_file():
        if args.attempt >= MAX_CHECKS:
            raise RuntimeError(f"equ102 did not complete after {args.attempt} checks: {atlas_root}")
        next_job = submit_check(args.run_root, args.profile, args.attempt + 1)
        state = {
            "status": "waiting_for_equ102",
            "attempt": args.attempt,
            "next_check_job": next_job,
            "atlas_root": str(atlas_root),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state, indent=2))
        return

    tuning_state = post_root / "tuning_curve_submission.json"
    if ATLAS_NAME == 'atlas_139':
        summary = json.loads(hcnorm_summary.read_text())
        if summary.get('count') != 139 or summary.get('groups', {}).get('extra50', {}).get('count') != 44:
            raise RuntimeError('Expected complete SF15-excluded equ139 with 44 extra cases')
    if tuning_state.is_file() and 'dense_tuning' not in json.loads(tuning_state.read_text()):
        subprocess.run([str(PYTHON), str(TUNING_LAUNCHER), '--run-root', str(args.run_root), '--post-root', str(post_root), '--atlas-root', str(atlas_root), '--runtime', str(atlas_root / 'runtime_h96iter'), '--profile', args.profile], check=True)
    complete_marker = post_root / "tuning_curves_complete.txt"
    if complete_marker.is_file() and (post_root / "dense_tuning/dense_tuning_complete.txt").is_file():
        subprocess.run([str(PYTHON), str(SCRIPT.with_name('submit_archive.py'))], check=True)
        state = {
            "status": "complete",
            "attempt": args.attempt,
            "atlas_root": str(atlas_root),
            "tuning_submission": str(tuning_state),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state, indent=2))
        return

    archived_submission = None
    prior_states = None
    should_submit = not tuning_state.is_file()
    if tuning_state.is_file():
        prior_states = tuning_job_states(tuning_state)
        if any(value in ACTIVE_STATES or value == "UNKNOWN" for value in prior_states.values()):
            if args.attempt >= MAX_CHECKS:
                raise RuntimeError(
                    f"Tuning curves did not complete after {args.attempt} checks: {prior_states}"
                )
            next_job = submit_check(args.run_root, args.profile, args.attempt + 1)
            state = {
                "status": "waiting_for_tuning_curves",
                "attempt": args.attempt,
                "next_check_job": next_job,
                "job_states": prior_states,
                "tuning_submission": str(tuning_state),
            }
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(json.dumps(state, indent=2))
            return
        archived_submission = archive_failed_submission(tuning_state, args.attempt)
        should_submit = True

    if should_submit:
        if args.attempt >= MAX_CHECKS:
            raise RuntimeError(
                f"Tuning curves failed after {args.attempt} checks: {prior_states}"
            )
        subprocess.run(
            [
                str(PYTHON), str(TUNING_LAUNCHER),
                "--run-root", str(args.run_root),
                "--post-root", str(post_root),
                "--atlas-root", str(atlas_root),
                "--runtime", str(atlas_root / "runtime_h96iter"),
                "--profile", args.profile,
            ],
            check=True,
        )
        next_job = submit_check(args.run_root, args.profile, args.attempt + 1)
        state = {
            "status": "tuning_submitted",
            "attempt": args.attempt,
            "next_check_job": next_job,
            "atlas_root": str(atlas_root),
            "hcnorm_summary": str(hcnorm_summary),
            "tuning_submission": str(tuning_state),
            "prior_job_states": prior_states,
            "archived_submission": str(archived_submission) if archived_submission else None,
        }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
