#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ACCOUNT = "torch_pr_482_general"
PYTHON = Path("/scratch/xh2906/conda/envs/py310/bin/python")
SCRIPT = Path(__file__).resolve()
SOURCE = Path(
    "/scratch/xh2906/librarySCI_runs/"
    "h96_tfsf7d_h128x3_full_library_fixedpoint_atlas_102_20260726"
)
ROOT: Path
RUNTIME: Path
PROFILE: str
SAVE_OPERATING_POINTS = False
MAX_ATTEMPTS = 4
EXTRA50_ROOT: Path | None = None


def output_name(angle: float, contrast: int) -> str:
    return f"eig_16HC_6DprefAngle_input_contr{contrast}_angle_{angle:.2f}.mat"


def load_cases() -> list[dict[str, str]]:
    with (SOURCE / "cases.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 102:
        raise RuntimeError(f"Expected 102 source cases, found {len(rows)}")
    for row in rows:
        source_name = Path(row["save_dir"]).name
        row["save_dir"] = str(ROOT / "results" / source_name)
    if EXTRA50_ROOT is not None:
        with (EXTRA50_ROOT / 'extra50_cases.tsv').open() as handle:
            extra = list(csv.DictReader(handle, delimiter='\t'))
        if len(extra) != 50 or {int(r['case']) for r in extra} != set(range(103, 153)):
            raise ValueError('Extra atlas must contain cases 103 through 152 exactly')
        seen = {tuple(float(r[k]) for k in ('angle', 'contrast', 'tf', 'sf')) for r in rows}
        for r in extra:
            if float(r['sf']) > 10.0:
                continue
            key = tuple(float(r[k]) for k in ('angle', 'contrast', 'tf', 'sf'))
            if key in seen:
                raise ValueError(f'Duplicate validation condition: {key}')
            seen.add(key)
            base = EXTRA50_ROOT / r['tag']
            cache = base / 'equilibrium_library' / (
                f"libonly_16HC_6DprefAngle_input_contr{int(r['contrast'])}_angle_{float(r['angle']):.2f}.mat")
            data = loadmat(cache, variable_names=['Angle', 'Contr', 'lgnTF', 'lgnSF', 'LDEEquv'])
            actual = tuple(float(np.asarray(data[k]).item()) for k in ('Angle', 'Contr', 'lgnTF', 'lgnSF'))
            if actual != key or 'LDEEquv' not in data:
                raise ValueError(f'Wrong library reference: {cache}')
            rows.append(dict(task=str(len(rows)+1), case=r['case'], angle=r['angle'],
                contrast=r['contrast'], tf=r['tf'], sf=r['sf'], library_dir=str(base/'library'),
                save_dir=str(ROOT/'results'/r['tag'])))
    rows = [r for r in rows if float(r['sf']) <= 10.0]
    for task, row in enumerate(rows, 1):
        row['task'] = str(task)
    return rows


def case_complete(row: dict[str, str]) -> bool:
    return (Path(row["save_dir"]) / output_name(float(row["angle"]), int(row["contrast"]))).is_file()


def write_cases(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ("task", "case", "angle", "contrast", "tf", "sf", "library_dir", "save_dir")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def add_operating_point_capture(iteration_path: Path, driver_path: Path) -> None:
    iteration = iteration_path.read_text()
    old_signature = (
        "function [LDERepFinal,LDEIL2Diff,L2DiffNormNeib,L2Diameter,LDEequv,FuncUseAll,NANFinal] = ..."
    )
    new_signature = (
        "function [LDERepFinal,LDEIL2Diff,L2DiffNormNeib,L2Diameter,LDEequv,FuncUseAll,NANFinal,OperatingPoint] = ..."
    )
    if old_signature not in iteration and new_signature not in iteration:
        raise RuntimeError(f"Missing iteration signature marker in {iteration_path}")
    iteration = iteration.replace(old_signature, new_signature, 1)
    op_marker = "if strcmpi(Outflag,'f(xn)')\n"
    op_block = """% Final NN fixed-point query inputs used by each population.
OperatingPoint = struct();
OperatingPoint.S_L4E = L4EUse_S;
OperatingPoint.S_L4I = L4IUse_S;
OperatingPoint.C_L4E = L4EUse_C;
OperatingPoint.C_L4I = L4IUse_C;
OperatingPoint.I_L4E = L4EUse_I;
OperatingPoint.I_L4I = L4IUse_I;
OperatingPoint.L6 = L6ELibInd;
OperatingPoint.PixLGNCtgr = PixLGNCtgr;

"""
    if "OperatingPoint.S_L4E" not in iteration:
        if op_marker not in iteration:
            raise RuntimeError(f"Missing operating-point insertion marker in {iteration_path}")
        iteration = iteration.replace(op_marker, op_block + op_marker, 1)
    iteration_path.write_text(iteration)

    driver = driver_path.read_text()
    old_call_outputs = (
        "[LDEOutAll{TestId},LDEIL2Diff{TestId},L2DiffNormNeib{TestId},~,~,~,...\n"
        "    NANFlag(TestId)] = ..."
    )
    new_call_outputs = (
        "[LDEOutAll{TestId},LDEIL2Diff{TestId},L2DiffNormNeib{TestId},~,~,~,...\n"
        "    NANFlag(TestId),OperatingPoint] = ..."
    )
    if old_call_outputs not in driver and new_call_outputs not in driver:
        raise RuntimeError(f"Missing NN iteration call marker in {driver_path}")
    driver = driver.replace(old_call_outputs, new_call_outputs, 1)

    target_marker = "    % LDEEquv = ICUse;  % this is to compare it with the precomputed NESS state, that is ZC's result.\n"
    target_block = """
% Evaluate the full library once at the final NN operating state. This is a
% matched local-response target, not the independently converged library state.
[OperatingTargetTrace,~,~,~,~,~,OperatingTargetNAN] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6(...
    PixLGNCtgr,L6Kernel,LDEsigmoid,1,L6pars{L6parId},1,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp,L4SIp,L4CEp,L4CIp,L4IEp,L4IIp,...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,...
    CtgrOrderReadout,angle,Contr,N_HCOutY,NPixX,NPixY,...
    Isaturation,'xn',EKpUse,IKpUse);
OperatingTarget = [];
OperatingTargetValid = ~OperatingTargetNAN;
if OperatingTargetValid
    OperatingTarget = OperatingTargetTrace{end};
    OperatingTargetValid = all(isfinite(OperatingTarget.S(:))) && ...
        all(isfinite(OperatingTarget.C(:))) && all(isfinite(OperatingTarget.I(:)));
end
if ~OperatingTargetValid
    OperatingTarget = [];
    warning('H96:OperatingTargetUnavailable', 'Optional library query at the final NN state is outside library support; no target is stored. HC uses the independently converged library state.');
end

% Capture the model-query inputs induced by the independently converged
% full-library state. The MLP output from this one-step call is ignored;
% only the pre-response L4E/L4I/L6 operating point is used for training.
[~,~,~,~,~,~,OperatingPointLibNAN,OperatingPointLib] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle(...
    PixLGNCtgr,L6Kernel,LDEEquv,1,L6pars{L6parId},1,...
    C_SS_meanU,C_CS_meanU,C_IS_mean,...
    C_SC_meanU,C_CC_meanU,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp,L4SIp,L4CEp,L4CIp,L4IEp,L4IIp,...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll,Contruse,Orientationuse,...
    N_HCOutY,NPixX,NPixY,Isaturation,'xn',EKpUse,IKpUse);
OperatingPointLibValid = ~OperatingPointLibNAN;
if ~OperatingPointLibValid
    OperatingPointLib = [];
    warning('H96:OperatingPointLibUnavailable', 'Optional library-state NN query capture unavailable; no diagnostic inputs are stored.');
end
OperatingTargetLib = LDEEquv;

"""
    if "OperatingTargetTrace" not in driver:
        if target_marker not in driver:
            raise RuntimeError(f"Missing operating-target insertion marker in {driver_path}")
        driver = driver.replace(target_marker, target_block + target_marker, 1)

    old_save_tail = (
        "'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib', 'err_ode_true_all', 'lgnTF', 'lgnSF', 'SaveToFolder');"
    )
    new_save_tail = (
        "'S_map_fulllib', 'C_map_fulllib', 'I_map_fulllib', 'err_ode_true_all', 'lgnTF', 'lgnSF', "
        "'SaveToFolder', 'OperatingPoint', 'OperatingTarget', 'OperatingPointLib', 'OperatingTargetLib', 'OperatingTargetNAN', 'OperatingTargetValid', 'OperatingPointLibNAN', 'OperatingPointLibValid');"
    )
    if old_save_tail not in driver and new_save_tail not in driver:
        raise RuntimeError(f"Missing save marker in {driver_path}")
    driver = driver.replace(old_save_tail, new_save_tail, 1)
    driver_path.write_text(driver)


def prepare() -> list[dict[str, str]]:
    for path in (SOURCE / "cases.tsv", SOURCE / "eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition.m", RUNTIME):
        if not path.exists():
            raise FileNotFoundError(path)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows = load_cases()
    write_cases(ROOT / "cases.tsv", rows)
    source_driver = SOURCE / "eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition.m"
    target_driver = ROOT / "eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition.m"
    if source_driver.resolve() != target_driver.resolve():
        shutil.copy2(source_driver, target_driver)
    if EXTRA50_ROOT is not None:
        driver = target_driver.read_text()
        marker = 'UseSpecialLowContrastLibCache = ~isempty(SpecialLowContrastLibCacheDir) && isfolder(SpecialLowContrastLibCacheDir);'
        if marker not in driver:
            raise RuntimeError('Missing special-cache guard marker')
        driver = driver.replace(marker, marker + '\nif exist(\'Extra50Case\',\'var\') && Extra50Case, UseSpecialLowContrastLibCache = false; end')
        target_driver.write_text(driver)
    if SAVE_OPERATING_POINTS:
        add_operating_point_capture(
            RUNTIME / "LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle.m",
            target_driver,
        )
    (ROOT / "run_case.m").write_text(
        f"""function run_case()
root = '{ROOT}';
runtime = '{RUNTIME}';
driver = fullfile(root, 'eigenvalue_7D_tfsf_loop_only_equilibrium_single_condition.m');
casesPath = getenv('H96_ATLAS_CASES');
task = str2double(getenv('SLURM_ARRAY_TASK_ID'));
T = readtable(casesPath, 'FileType', 'text', 'Delimiter', '\t', 'VariableNamingRule', 'preserve');
row = T(T.task == task, :);
if height(row) ~= 1, error('Expected one case for task %d', task); end
angle = row.angle(1);
contrast = row.contrast(1);
lgnTF = row.tf(1);
lgnSF = row.sf(1);
libDir = char(string(row.library_dir(1)));
saveDir = char(string(row.save_dir(1)));
if ~exist(saveDir, 'dir'), mkdir(saveDir); end
addpath(runtime, '-begin');
cd('/scratch/xh2906/NYU-Vision-2Drive-main');
AngleListOverride = {{sprintf('%.2f', angle)}};
ContrListOverride = contrast;
SaveBaseDirOverride = saveDir;
SaveFigOverride = false;
SavePngOverride = true;
RuntimeBundleDirOverride = runtime;
BGFlagOverride = false;
LibCacheDirOverride = '__NO_CACHE__';
LowContrastLibCacheDirOverride = '__NO_CACHE__';
Extra50Case = row.case(1) > 102;
if Extra50Case
    LibCacheDirOverride = fullfile(fileparts(libDir), 'equilibrium_library');
    if ~isfolder(LibCacheDirOverride), error('Missing extra50 library cache'); end
end
SaveToFolderOverride = libDir;
lgnTFOverride = lgnTF;
lgnSFOverride = lgnSF;
global H96_LGN_TF_USE H96_LGN_SF_USE
H96_LGN_TF_USE = lgnTF;
H96_LGN_SF_USE = lgnSF;
run(driver);
end
"""
    )
    (ROOT / "model_source.txt").write_text(
        f"Profile: {PROFILE}\n"
        f"Runtime: {RUNTIME}\n"
        "Architecture: separate S/C/I tanh MLPs with Softplus output\n"
    )
    return rows


def submit_rows(rows: list[dict[str, str]], attempt: int) -> dict[str, object]:
    pending = [row.copy() for row in rows if not case_complete(row)]
    if not pending:
        return summarize(rows)
    if attempt > MAX_ATTEMPTS:
        raise RuntimeError(f"Still missing {len(pending)} cases after {MAX_ATTEMPTS} attempts")
    for task, row in enumerate(pending, start=1):
        row["task"] = str(task)
        Path(row["save_dir"]).mkdir(parents=True, exist_ok=True)
    cases_path = ROOT / f"pending_attempt{attempt}.tsv"
    write_cases(cases_path, pending)
    cpus_per_task = int(os.environ.get("H96_ATLAS_CPUS", "4"))
    memory = os.environ.get("H96_ATLAS_MEM", "8G")
    if cpus_per_task < 1:
        raise ValueError("H96_ATLAS_CPUS must be positive")
    submit = ROOT / f"submit_attempt{attempt}.sh"
    submit.write_text(
        f"""#!/usr/bin/env bash
#SBATCH --job-name={PROFILE}atlas102
#SBATCH --account={ACCOUNT}
#SBATCH --partition=cpu_short,cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time=01:00:00
#SBATCH --array=1-{len(pending)}%64
#SBATCH --output={ROOT}/logs/attempt{attempt}.%A_%a.out
#SBATCH --error={ROOT}/logs/attempt{attempt}.%A_%a.err
set -euo pipefail
module purge
module load matlab/2025b
export H96_ATLAS_CASES='{cases_path}'
export MATLAB_PREFDIR="/tmp/xh2906_aug20_mp/${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
export TMPDIR="/tmp/xh2906_aug20_mt/${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$MATLAB_PREFDIR" "$TMPDIR"
cd "$TMPDIR"
matlab -batch "addpath('{ROOT}'); run_case"
"""
    )
    submit.chmod(0o755)
    array_job = subprocess.check_output(["sbatch", "--parsable", str(submit)], text=True).strip()
    followup = subprocess.check_output(
        [
            "sbatch", "--parsable", f"--account={ACCOUNT}", "--partition=cpu_short,cs",
            "--cpus-per-task=1", "--mem=4G", "--time=00:20:00",
            f"--dependency=afterany:{array_job}", f"--job-name={PROFILE}atlas_check",
            f"--output={ROOT}/logs/followup{attempt}.%j.out",
            f"--error={ROOT}/logs/followup{attempt}.%j.err",
            "--wrap", (
                f"{PYTHON} {shlex.quote(str(SCRIPT))} --profile {shlex.quote(PROFILE)} "
                f"--root {shlex.quote(str(ROOT))} --runtime {shlex.quote(str(RUNTIME))} --mode advance"
                + (" --save-operating-points" if SAVE_OPERATING_POINTS else "")
                + (f" --extra50-root {shlex.quote(str(EXTRA50_ROOT))}" if EXTRA50_ROOT else "")
            ),
        ],
        text=True,
    ).strip()
    state = {
        "status": "submitted",
        "attempt": attempt,
        "pending": len(pending),
        "array_job": array_job,
        "followup_job": followup,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
    }
    (ROOT / "submission_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))
    return state


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    values = []
    for row in rows:
        path = Path(row["save_dir"]) / output_name(float(row["angle"]), int(row["contrast"]))
        data = loadmat(path, variable_names=["err_ode_true_all"])
        err = np.asarray(data["err_ode_true_all"], dtype=float)
        if err.ndim != 2 or err.shape[1] <= 250 or not np.isfinite(err[-1, 250:]).all():
            raise ValueError(f'Invalid HCnorm trace: {path}')
        values.append(
            {
                "case": int(row["case"]),
                "group": "equ70" if int(row["case"]) <= 70 else ("heldout30" if int(row["case"]) <= 100 else ("matched_extreme" if int(row["case"]) <= 102 else "extra50")),
                "angle": float(row["angle"]),
                "contrast": int(row["contrast"]),
                "tf": float(row["tf"]),
                "sf": float(row["sf"]),
                "HCnorm": float(np.mean(err[-1, 250:])),
                "mat": str(path),
            }
        )
    values.sort(key=lambda row: row["HCnorm"], reverse=True)
    with (ROOT / f"hcnorm_all_{len(rows)}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=values[0].keys())
        writer.writeheader()
        writer.writerows(values)
    groups = {}
    for group in dict.fromkeys(row['group'] for row in values):
        selected = [row["HCnorm"] for row in values if row["group"] == group]
        groups[group] = {
            "count": len(selected),
            "mean": float(np.mean(selected)),
            "max": float(np.max(selected)),
            "min": float(np.min(selected)),
        }
    summary = {
        "count": len(values),
        "HC_gt_5": sum(row['HCnorm'] > 5 for row in values),
        "mean": float(np.mean([row["HCnorm"] for row in values])),
        "max": float(values[0]["HCnorm"]),
        "min": float(values[-1]["HCnorm"]),
        "groups": groups,
        "top10": values[:10],
    }
    (ROOT / "hcnorm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    state = {"status": "complete", "attempt": 0, "missing": 0, **summary}
    (ROOT / "submission_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    global PROFILE, ROOT, RUNTIME, SAVE_OPERATING_POINTS, EXTRA50_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("prepare", "submit", "advance", "summarize"),
        default="submit",
    )
    parser.add_argument("--save-operating-points", action="store_true")
    parser.add_argument("--extra50-root", type=Path)
    args = parser.parse_args()
    PROFILE = args.profile
    ROOT = args.root
    RUNTIME = args.runtime
    SAVE_OPERATING_POINTS = bool(args.save_operating_points)
    EXTRA50_ROOT = args.extra50_root
    rows = prepare()
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared", "cases": len(rows), "root": str(ROOT)}, indent=2))
        return
    if args.mode == "summarize":
        summarize(rows)
        return
    previous = json.loads((ROOT / "submission_state.json").read_text()) if (ROOT / "submission_state.json").is_file() else {}
    attempt = int(previous.get("attempt", 0)) + 1
    submit_rows(rows, attempt)


if __name__ == "__main__":
    main()
