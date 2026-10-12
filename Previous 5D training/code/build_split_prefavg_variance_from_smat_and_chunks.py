#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common import save_json
    from dataset_hdf5 import H5MatReader, compute_feature_norm_from_shard
else:
    from .common import save_json
    from .dataset_hdf5 import H5MatReader, compute_feature_norm_from_shard


SHARD_FIELDS = (
    ("OriCos2", np.float32),
    ("OriSin2", np.float32),
    ("Contrast", np.float32),
    ("L6", np.float32),
    ("L4E", np.float32),
    ("L4I", np.float32),
    ("FR_Response", np.float32),
    ("FR_ResponseVar", np.float32),
    ("FR_ResponseN", np.float32),
    ("Gamma", np.float32),
    ("LGNctgr", np.int16),
)
META_FIELDS = ("Alpha", "Contrast", "Gamma")
SPLIT_CODES = {"train": np.uint8(0), "val": np.uint8(1), "test": np.uint8(2)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build split shards from completed S MAT plus C/I direct chunk files.")
    ap.add_argument("--mat_s", required=True)
    ap.add_argument("--chunks_c_dir", required=True)
    ap.add_argument("--chunks_i_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=262144)
    ap.add_argument("--target_high_s", type=float, default=30.0)
    ap.add_argument("--target_high_c", type=float, default=100.0)
    ap.add_argument("--target_high_i", type=float, default=140.0)
    return ap.parse_args()


def check_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise RuntimeError("train/val/test ratios must all be positive")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise RuntimeError("train/val/test ratios must sum to 1")


def create_shards(shard_dir: Path, target: str, batch_size: int) -> tuple[dict[str, str], dict[str, h5py.File], dict[str, int]]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    handles: dict[str, h5py.File] = {}
    write_at: dict[str, int] = {}
    for split_name in SPLIT_CODES:
        path = shard_dir / f"{target}_{split_name}.h5"
        paths[split_name] = str(path)
        h5 = h5py.File(path, "w")
        handles[split_name] = h5
        write_at[split_name] = 0
        for field_name, dtype in SHARD_FIELDS:
            h5.create_dataset(
                field_name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.dtype(dtype),
                chunks=(max(int(batch_size), 1),),
            )
    return paths, handles, write_at


def append_to_shards(
    handles: dict[str, h5py.File],
    write_at: dict[str, int],
    batch: dict[str, np.ndarray],
    split_id: np.ndarray,
    target_high: float,
) -> dict[str, int]:
    response = np.asarray(batch["FR_Response"], dtype=np.float32).reshape(-1)
    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, split_code in SPLIT_CODES.items():
        keep = (split_id == split_code) & (response <= np.float32(target_high))
        if not np.any(keep):
            continue
        n = int(np.sum(keep))
        dest = slice(write_at[split_name], write_at[split_name] + n)
        h5 = handles[split_name]
        for field_name, _dtype in SHARD_FIELDS:
            values = np.asarray(batch[field_name]).reshape(-1)[keep]
            h5[field_name].resize((write_at[split_name] + n,))
            h5[field_name][dest] = values
        write_at[split_name] += n
        counts[split_name] += n
    return counts


def make_split_ids(rng: np.random.Generator, n: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    u = rng.random(int(n))
    out = np.empty(int(n), dtype=np.uint8)
    out[u < train_ratio] = SPLIT_CODES["train"]
    out[(u >= train_ratio) & (u < train_ratio + val_ratio)] = SPLIT_CODES["val"]
    out[u >= train_ratio + val_ratio] = SPLIT_CODES["test"]
    return out


def read_batch(reader: H5MatReader, rows: slice) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for field_name, dtype in SHARD_FIELDS:
        out[field_name] = reader.read_1d(field_name, rows, dtype)
    return out


def update_coverage(coverage: dict[str, dict], split_id: np.ndarray, meta: dict[str, np.ndarray]) -> None:
    for split_name, split_code in SPLIT_CODES.items():
        mask = split_id == split_code
        if not np.any(mask):
            continue
        cov = coverage[split_name]
        cov["n_rows_unfiltered"] += int(np.sum(mask))
        cov["alpha_values"].update(float(x) for x in np.unique(meta["Alpha"][mask]))
        cov["contrast_values"].update(int(round(float(x))) for x in np.unique(meta["Contrast"][mask]))
        cov["gamma_values"].update(float(x) for x in np.unique(meta["Gamma"][mask]))


def empty_coverage() -> dict[str, dict]:
    return {
        name: {"n_rows_unfiltered": 0, "alpha_values": set(), "contrast_values": set(), "gamma_values": set()}
        for name in SPLIT_CODES
    }


def freeze_coverage(cov: dict[str, dict]) -> dict[str, dict]:
    return {
        name: {
            "n_rows_unfiltered": int(item["n_rows_unfiltered"]),
            "alpha_values": sorted(item["alpha_values"]),
            "contrast_values": sorted(item["contrast_values"]),
            "gamma_values": sorted(item["gamma_values"]),
        }
        for name, item in cov.items()
    }


def build_from_mat(
    target: str,
    mat_path: Path,
    shard_dir: Path,
    seed: int,
    args: argparse.Namespace,
    target_high: float,
) -> tuple[dict[str, str], dict, dict]:
    paths, handles, write_at = create_shards(shard_dir, target, args.batch_size)
    rng = np.random.default_rng(seed)
    coverage = empty_coverage()
    split_counts = {"train": 0, "val": 0, "test": 0}
    with H5MatReader(mat_path) as reader:
        total = reader.length("FR_Response")
        for start in range(0, total, args.batch_size):
            rows = slice(start, min(start + args.batch_size, total))
            n = rows.stop - rows.start
            batch = read_batch(reader, rows)
            split_id = make_split_ids(rng, n, args.train_ratio, args.val_ratio)
            meta = {
                "Alpha": reader.read_1d("Alpha", rows, np.float32),
                "Contrast": batch["Contrast"],
                "Gamma": batch["Gamma"],
            }
            update_coverage(coverage, split_id, meta)
            add_counts = append_to_shards(handles, write_at, batch, split_id, target_high)
            for key, value in add_counts.items():
                split_counts[key] += value
    for h5 in handles.values():
        h5.close()
    split_counts["total_unfiltered"] = int(sum(item["n_rows_unfiltered"] for item in coverage.values()))
    split_counts["total_filtered"] = int(sum(split_counts[k] for k in ("train", "val", "test")))
    return paths, split_counts, freeze_coverage(coverage)


def chunk_files(chunks_dir: Path, target: str) -> list[Path]:
    files = sorted(chunks_dir.glob(f"{target}_case*.mat"))
    if len(files) != 147:
        raise RuntimeError(f"Expected 147 {target} chunk files in {chunks_dir}, found {len(files)}")
    return files


def build_from_chunks(
    target: str,
    chunks_dir: Path,
    shard_dir: Path,
    seed: int,
    args: argparse.Namespace,
    target_high: float,
) -> tuple[dict[str, str], dict, dict]:
    paths, handles, write_at = create_shards(shard_dir, target, args.batch_size)
    rng = np.random.default_rng(seed)
    coverage = empty_coverage()
    split_counts = {"train": 0, "val": 0, "test": 0}
    total = 0
    for path in chunk_files(chunks_dir, target):
        with H5MatReader(path) as reader:
            n_rows = reader.length("FR_Response")
            total += n_rows
            for start in range(0, n_rows, args.batch_size):
                rows = slice(start, min(start + args.batch_size, n_rows))
                n = rows.stop - rows.start
                batch = read_batch(reader, rows)
                split_id = make_split_ids(rng, n, args.train_ratio, args.val_ratio)
                meta = {
                    "Alpha": reader.read_1d("Alpha", rows, np.float32),
                    "Contrast": batch["Contrast"],
                    "Gamma": batch["Gamma"],
                }
                update_coverage(coverage, split_id, meta)
                add_counts = append_to_shards(handles, write_at, batch, split_id, target_high)
                for key, value in add_counts.items():
                    split_counts[key] += value
    for h5 in handles.values():
        h5.close()
    split_counts["total_unfiltered"] = int(total)
    split_counts["total_filtered"] = int(sum(split_counts[k] for k in ("train", "val", "test")))
    return paths, split_counts, freeze_coverage(coverage)


def validate_variance_shards(shard_paths: dict[str, dict[str, str]], batch_size: int) -> dict:
    out: dict[str, dict[str, dict]] = {}
    for target, split_paths in shard_paths.items():
        out[target] = {}
        for split_name, shard_path in split_paths.items():
            with H5MatReader(shard_path) as reader:
                total = reader.length("FR_Response")
                var_min = float("inf")
                var_max = float("-inf")
                n_min = float("inf")
                n_max = float("-inf")
                for start in range(0, total, batch_size):
                    rows = slice(start, min(start + batch_size, total))
                    response = reader.read_1d("FR_Response", rows, np.float64)
                    var = reader.read_1d("FR_ResponseVar", rows, np.float64)
                    repeat_n = reader.read_1d("FR_ResponseN", rows, np.float64)
                    if not np.all(np.isfinite(response)):
                        raise RuntimeError(f"Non-finite FR_Response in {shard_path}")
                    if not np.all(np.isfinite(var)) or np.any(var < 0):
                        raise RuntimeError(f"Invalid FR_ResponseVar in {shard_path}")
                    if not np.all(np.isfinite(repeat_n)) or np.any(repeat_n <= 0):
                        raise RuntimeError(f"Invalid FR_ResponseN in {shard_path}")
                    if var.size:
                        var_min = min(var_min, float(np.min(var)))
                        var_max = max(var_max, float(np.max(var)))
                        n_min = min(n_min, float(np.min(repeat_n)))
                        n_max = max(n_max, float(np.max(repeat_n)))
                out[target][split_name] = {
                    "n_rows": int(total),
                    "FR_ResponseVar_min": float(var_min if np.isfinite(var_min) else 0.0),
                    "FR_ResponseVar_max": float(var_max if np.isfinite(var_max) else 0.0),
                    "FR_ResponseN_min": float(n_min if np.isfinite(n_min) else 0.0),
                    "FR_ResponseN_max": float(n_max if np.isfinite(n_max) else 0.0),
                }
    return out


def main() -> None:
    args = parse_args()
    check_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    out_dir = Path(args.out_dir)
    shard_dir = out_dir / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: dict[str, dict[str, str]] = {}
    split_counts: dict[str, dict] = {}
    coverage: dict[str, dict] = {}

    target_specs = {
        "S": ("mat", Path(args.mat_s), float(args.target_high_s), 0),
        "C": ("chunks", Path(args.chunks_c_dir), float(args.target_high_c), 1),
        "I": ("chunks", Path(args.chunks_i_dir), float(args.target_high_i), 2),
    }
    for target, (kind, source, target_high, seed_offset) in target_specs.items():
        print(f"[{target}] source={kind} path={source}")
        if kind == "mat":
            paths, counts, cov = build_from_mat(target, source, shard_dir, args.seed + seed_offset, args, target_high)
        else:
            paths, counts, cov = build_from_chunks(target, source, shard_dir, args.seed + seed_offset, args, target_high)
        shard_paths[target] = paths
        split_counts[target] = counts
        coverage[target] = cov
        print(f"[{target}] filtered rows train={counts['train']} val={counts['val']} test={counts['test']}")

    variance_validation = validate_variance_shards(shard_paths, args.batch_size)
    feature_norm = compute_feature_norm_from_shard(shard_paths["S"]["train"], batch_size=args.batch_size)
    save_json(out_dir / "feature_norm.json", feature_norm)

    manifest = {
        "split_mode": "stream_random_rows_from_smat_and_condition_chunks",
        "seed": int(args.seed),
        "target_seed_offsets": {"S": 0, "C": 1, "I": 2},
        "sources": {
            "S": {"kind": "mat", "path": str(Path(args.mat_s))},
            "C": {"kind": "chunks", "path": str(Path(args.chunks_c_dir))},
            "I": {"kind": "chunks", "path": str(Path(args.chunks_i_dir))},
        },
        "shard_paths": shard_paths,
        "split_paths": {},
        "target_filter": {
            "low": None,
            "high": None,
            "high_by_target": {"S": args.target_high_s, "C": args.target_high_c, "I": args.target_high_i},
        },
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "split_rows_by_target": split_counts,
        "coverage_by_target": coverage,
        "feature_norm_source": "S_train shard",
        "variance_fields": {
            "FR_ResponseVar": "sample variance across repeats; defaults to 0 when absent in source",
            "FR_ResponseN": "repeat count estimate for the mean response; defaults to 1 when absent in source",
            "validation": variance_validation,
        },
    }
    save_json(out_dir / "split_manifest.json", manifest)
    print(f"[done] wrote split manifest to {out_dir}")


if __name__ == "__main__":
    main()
