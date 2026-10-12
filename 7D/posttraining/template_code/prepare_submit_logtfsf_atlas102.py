#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


PYTHON = Path("/scratch/xh2906/conda/envs/py310/bin/python")
CODE_ROOT = Path("/scratch/xh2906/h96_tfsf7d_log_variant_20260727")
EXPORTER = Path(__file__).with_name("export_pref6d_matlab.py")
ATLAS_SCRIPT = CODE_ROOT / "build_submit_logtfsf_atlas102.py"
PATCH_TEMPLATE = Path(
    "/scratch/xh2906/librarySCI_runs/"
    "h96_tfsf7d_newcolsonly_tfsfdefault_mid10_fromlong_20260704_195639/"
    "hpc_runtime_bundle_tfsf7d_export_mid10_20260705_0001_highl4eguard_h96iter"
)


def add_high_l4e_guard(path: Path) -> None:
    text = path.read_text()
    if "h96_high_l4e_guard" in text:
        return
    marker = "contrastEff = ContrastUse;\n"
    if marker not in text:
        raise RuntimeError(f"Missing high-L4E guard insertion marker in {path}")
    text = text.replace(
        marker,
        marker + "\n% Smoothly saturate only above the h96 training edge.\nL4EUse = h96_high_l4e_guard(L4EUse);\n",
        1,
    )
    text += """

function xg = h96_high_l4e_guard(x)
edge = 47500;
full = 55000;
tailSlope = 0.02;
u = (double(x) - edge) ./ (full - edge);
u = min(max(u, 0), 1);
g = u.^3 .* (10 - 15*u + 6*u.^2);
xSat = edge + tailSlope .* (double(x) - edge);
xg = (1 - g) .* double(x) + g .* xSat;
xg = cast(xg, 'like', x);
end
"""
    path.write_text(text)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_provenance(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "model_config": checkpoint["model_config"],
        "feature_norm": checkpoint["feature_norm"],
        "checkpoint_extra": json.loads(
            json.dumps(checkpoint.get("extra", {}), default=str)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--model-s", type=Path, required=True)
    parser.add_argument("--model-c", type=Path, required=True)
    parser.add_argument("--model-i", type=Path, required=True)
    parser.add_argument("--require-tf-max", type=float)
    parser.add_argument("--require-transform", choices=("identity", "log"), default="log")
    parser.add_argument("--save-operating-points", action="store_true")
    parser.add_argument("--extra50-root", type=Path)
    parser.add_argument("--mode", choices=("prepare", "submit"), default="submit")
    args = parser.parse_args()
    atlas_script = Path(__file__).with_name('build_submit_logtfsf_atlas102.py') if args.extra50_root else ATLAS_SCRIPT

    models = {"S": args.model_s, "C": args.model_c, "I": args.model_i}
    for path in (*models.values(), EXPORTER, atlas_script, PATCH_TEMPLATE):
        if not path.exists():
            raise FileNotFoundError(path)
    model_metadata = {target: checkpoint_provenance(path) for target, path in models.items()}
    for target, metadata in model_metadata.items():
        model_config = metadata["model_config"]
        feature_norm = metadata["feature_norm"]
        if int(model_config.get("input_dim", -1)) != 7 or model_config.get("feature_mode") != "tf_sf7d":
            raise RuntimeError(f"{target}: checkpoint is not a 7D TF/SF model")
        for feature in ("lgnTF", "lgnSF"):
            transform = str(feature_norm.get(feature, {}).get("transform", "identity")).lower()
            if transform != args.require_transform:
                raise RuntimeError(
                    f"{target}: {feature} transform={transform!r}, expected {args.require_transform!r}"
                )
        if args.require_tf_max is not None:
            observed = float(feature_norm.get("lgnTF", {}).get("domain_max", float("nan")))
            if abs(observed - args.require_tf_max) > 1e-8:
                raise RuntimeError(
                    f"{target}: lgnTF domain_max={observed}, expected {args.require_tf_max}"
                )

    args.atlas_root.mkdir(parents=True, exist_ok=True)
    runtime = args.atlas_root / "runtime_h96iter"
    runtime.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(PYTHON), str(EXPORTER),
            "--model_s", str(models["S"]),
            "--model_c", str(models["C"]),
            "--model_i", str(models["I"]),
            "--out_dir", str(runtime),
        ],
        check=True,
    )
    add_high_l4e_guard(runtime / "LocalResponse_6D_MLP_prefAngle.m")
    for name in (
        "LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle.m",
        "LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle.m",
    ):
        shutil.copy2(PATCH_TEMPLATE / name, runtime / name)

    # Pin the library geometry independently of MATLAB path precedence.
    library_name = "LDEIterFunc_Grating_135Func_RealLGNL6.m"
    library_source = Path("/scratch/xh2906/NYU-Vision-2Drive-main/Utils") / library_name
    if sha256(library_source) != "3b0c4b95a5edbfb32eaed138b47c78285c967a2e41fe5c6b78279a1d657b9c84":
        raise RuntimeError("Library channel mapping differs from geometric_all_contrasts_v1")
    shutil.copy2(library_source, runtime / library_name)

    predictor = (runtime / "predict_pref6D_S.m").read_text()
    log_lines = ("X(:, 6) = log(X(:, 6));", "X(:, 7) = log(X(:, 7));")
    if args.require_transform == "log" and not all(line in predictor for line in log_lines):
        raise RuntimeError("Exported predictor is missing log TF/SF transforms")
    if args.require_transform == "identity" and any(line in predictor for line in log_lines):
        raise RuntimeError("Identity-normalized predictor unexpectedly applies a log TF/SF transform")
    local_response = runtime / "LocalResponse_6D_MLP_prefAngle.m"
    if "h96_high_l4e_guard" not in local_response.read_text():
        raise RuntimeError("Corrected h96 high-L4E guard is missing")

    provenance = {
        "library_mapping_policy": "geometric_all_contrasts_v1",
        "library_response_sha256": sha256(runtime / library_name),
        "cached_library_maps_policy": "Historical reference maps retain original provenance; not recomputed by this code update",
        "profile": args.profile,
        "atlas_root": str(args.atlas_root),
        "runtime": str(runtime),
        "model_paths": {target: str(path) for target, path in models.items()},
        "models": model_metadata,
        "tfsf_transform": args.require_transform,
        "log_tfsf": args.require_transform == "log",
        "high_l4e_guard": True,
        "local_response_sha256": sha256(local_response),
        "hcnorm_definition": "mean(err_ode_true_all(end,251:end))",
        "case_count": 139 if args.extra50_root else 95,
        "excluded_sf": [15.0],
        "extra50_root": str(args.extra50_root) if args.extra50_root else None,
    }
    (args.atlas_root / "runtime_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    atlas_cmd = [
        str(PYTHON), str(atlas_script),
        "--profile", args.profile,
        "--root", str(args.atlas_root),
        "--runtime", str(runtime),
        "--mode", args.mode,
    ]
    if args.save_operating_points:
        atlas_cmd.append("--save-operating-points")
    if args.extra50_root:
        atlas_cmd.extend(['--extra50-root', str(args.extra50_root)])
    subprocess.run(atlas_cmd, check=True)


if __name__ == "__main__":
    main()
