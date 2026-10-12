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
OPTIONAL_SOURCE_FIELD_DEFAULTS = {
    "FR_ResponseVar": 0.0,
    "FR_ResponseN": 1.0,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build independent random row 80/10/10 split shards per S/C/I target, preserving variance fields."
    )
    ap.add_argument("--mat_s", required=True)
    ap.add_argument("--mat_c", required=True)
    ap.add_argument("--mat_i", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--norm_batch_size", type=int, default=65536)
    ap.add_argument("--shard_batch_size", type=int, default=65536)
    ap.add_argument("--target_low", type=float, default=None)
    ap.add_argument("--target_high", type=float, default=None)
    ap.add_argument("--target_high_s", type=float, default=None)
    ap.add_argument("--target_high_c", type=float, default=None)
    ap.add_argument("--target_high_i", type=float, default=None)
    return ap.parse_args()


def has_key(reader: H5MatReader, key: str) -> bool:
    return key in reader.h5


def assign_random_split_id(total_rows: int, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> np.ndarray:
    if total_rows <= 0:
        raise RuntimeError("total_rows must be positive")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise RuntimeError("train/val/test ratios must all be positive")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise RuntimeError("train/val/test ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_rows).astype(np.int64, copy=False)
    n_train = int(np.floor(total_rows * train_ratio))
    n_val = int(np.floor(total_rows * val_ratio))
    n_test = total_rows - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise RuntimeError(f"Invalid split sizes: train={n_train} val={n_val} test={n_test}")
    split_id = np.empty(total_rows, dtype=np.uint8)
    split_id[perm[:n_train]] = 0
    split_id[perm[n_train:n_train + n_val]] = 1
    split_id[perm[n_train + n_val:]] = 2
    return split_id


def read_source_field_or_default(reader: H5MatReader, field_name: str, batch_slice: slice, dtype) -> np.ndarray:
    if has_key(reader, field_name):
        return reader.read_1d(field_name, batch_slice, dtype)
    if field_name in OPTIONAL_SOURCE_FIELD_DEFAULTS:
        length = int(batch_slice.stop - batch_slice.start)
        return np.full(length, OPTIONAL_SOURCE_FIELD_DEFAULTS[field_name], dtype=np.dtype(dtype))
    return reader.read_1d(field_name, batch_slice, dtype)


def write_split_shards(
    mat_paths: dict[str, str],
    shard_dir: Path,
    split_id: np.ndarray,
    batch_size: int,
    target_low: float | None,
    target_high: float | None,
    target_high_by_target: dict[str, float | None] | None = None,
) -> dict[str, dict[str, str]]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: dict[str, dict[str, str]] = {}
    split_codes = {"train": np.uint8(0), "val": np.uint8(1), "test": np.uint8(2)}
    total_rows = int(split_id.size)
    target_high_by_target = target_high_by_target or {}

    for target, mat_path in mat_paths.items():
        shard_paths[target] = {}
        effective_target_high = target_high_by_target.get(target, target_high)
        with H5MatReader(mat_path) as reader:
            out_h5s: dict[str, h5py.File] = {}
            write_at: dict[str, int] = {}
            for split_name in split_codes:
                shard_path = shard_dir / f"{target}_{split_name}.h5"
                shard_paths[target][split_name] = str(shard_path)
                out_h5 = h5py.File(shard_path, "w")
                out_h5s[split_name] = out_h5
                write_at[split_name] = 0
                for field_name, dtype in SHARD_FIELDS:
                    out_h5.create_dataset(
                        field_name,
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.dtype(dtype),
                        chunks=(max(batch_size, 1),),
                    )
            try:
                for batch_start in range(0, total_rows, batch_size):
                    batch_stop = min(batch_start + batch_size, total_rows)
                    batch_slice = slice(batch_start, batch_stop)
                    chunk_split = split_id[batch_start:batch_stop]
                    response = reader.read_1d("FR_Response", batch_slice, np.float32)
                    batch_data: dict[str, np.ndarray] = {"FR_Response": response}
                    for field_name, dtype in SHARD_FIELDS:
                        if field_name == "FR_Response":
                            continue
                        batch_data[field_name] = read_source_field_or_default(reader, field_name, batch_slice, dtype)
                    for split_name, split_code in split_codes.items():
                        keep_mask = chunk_split == split_code
                        if target_low is not None:
                            keep_mask &= response >= np.float32(target_low)
                        if effective_target_high is not None:
                            keep_mask &= response <= np.float32(effective_target_high)
                        if not np.any(keep_mask):
                            continue
                        batch_len = int(np.sum(keep_mask))
                        dest_slice = slice(write_at[split_name], write_at[split_name] + batch_len)
                        out_h5 = out_h5s[split_name]
                        for field_name, _dtype in SHARD_FIELDS:
                            values = batch_data[field_name][keep_mask]
                            out_h5[field_name].resize((write_at[split_name] + batch_len,))
                            out_h5[field_name][dest_slice] = values
                        write_at[split_name] += batch_len
            finally:
                for out_h5 in out_h5s.values():
                    out_h5.close()
    return shard_paths


def coverage_from_split_id(mat_path: str, split_id: np.ndarray, batch_size: int) -> dict[str, dict]:
    split_codes = {"train": np.uint8(0), "val": np.uint8(1), "test": np.uint8(2)}
    coverage = {
        split_name: {
            "n_rows": int(np.sum(split_id == split_code)),
            "alpha_values": set(),
            "contrast_values": set(),
            "gamma_values": set(),
        }
        for split_name, split_code in split_codes.items()
    }
    with H5MatReader(mat_path) as reader:
        total_rows = int(split_id.size)
        for batch_start in range(0, total_rows, batch_size):
            batch_stop = min(batch_start + batch_size, total_rows)
            batch_slice = slice(batch_start, batch_stop)
            chunk_split = split_id[batch_start:batch_stop]
            alpha = reader.read_1d("Alpha", batch_slice, np.float32)
            contrast = reader.read_1d("Contrast", batch_slice, np.float32)
            gamma = reader.read_1d("Gamma", batch_slice, np.float32)
            for split_name, split_code in split_codes.items():
                mask = chunk_split == split_code
                if not np.any(mask):
                    continue
                cov = coverage[split_name]
                cov["alpha_values"].update(float(x) for x in np.unique(alpha[mask]))
                cov["contrast_values"].update(int(round(float(x))) for x in np.unique(contrast[mask]))
                cov["gamma_values"].update(float(x) for x in np.unique(gamma[mask]))
    return {
        split_name: {
            "n_rows": cov["n_rows"],
            "alpha_values": sorted(cov["alpha_values"]),
            "contrast_values": sorted(cov["contrast_values"]),
            "gamma_values": sorted(cov["gamma_values"]),
        }
        for split_name, cov in coverage.items()
    }


def validate_variance_shards(shard_paths: dict[str, dict[str, str]], batch_size: int) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for target, split_paths in shard_paths.items():
        out[target] = {}
        for split_name, shard_path in split_paths.items():
            n_rows = 0
            var_min = float("inf")
            var_max = float("-inf")
            n_min = float("inf")
            n_max = float("-inf")
            with H5MatReader(shard_path) as reader:
                for required in ("FR_Response", "FR_ResponseVar", "FR_ResponseN"):
                    if not has_key(reader, required):
                        raise RuntimeError(f"Missing {required} in shard {shard_path}")
                total_rows = reader.length("FR_Response")
                for batch_start in range(0, total_rows, batch_size):
                    batch_slice = slice(batch_start, min(batch_start + batch_size, total_rows))
                    response = reader.read_1d("FR_Response", batch_slice, np.float64)
                    var = reader.read_1d("FR_ResponseVar", batch_slice, np.float64)
                    repeat_n = reader.read_1d("FR_ResponseN", batch_slice, np.float64)
                    if not np.all(np.isfinite(response)):
                        raise RuntimeError(f"Non-finite FR_Response values in {shard_path}")
                    if not np.all(np.isfinite(var)) or np.any(var < 0.0):
                        raise RuntimeError(f"Invalid FR_ResponseVar values in {shard_path}")
                    if not np.all(np.isfinite(repeat_n)) or np.any(repeat_n <= 0.0):
                        raise RuntimeError(f"Invalid FR_ResponseN values in {shard_path}")
                    if var.size:
                        var_min = min(var_min, float(np.min(var)))
                        var_max = max(var_max, float(np.max(var)))
                        n_min = min(n_min, float(np.min(repeat_n)))
                        n_max = max(n_max, float(np.max(repeat_n)))
                    n_rows += int(response.size)
            out[target][split_name] = {
                "n_rows": int(n_rows),
                "FR_ResponseVar_min": float(var_min if np.isfinite(var_min) else 0.0),
                "FR_ResponseVar_max": float(var_max if np.isfinite(var_max) else 0.0),
                "FR_ResponseN_min": float(n_min if np.isfinite(n_min) else 0.0),
                "FR_ResponseN_max": float(n_max if np.isfinite(n_max) else 0.0),
            }
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mat_paths = {"S": args.mat_s, "C": args.mat_c, "I": args.mat_i}
    high_by_target = {"S": args.target_high_s, "C": args.target_high_c, "I": args.target_high_i}

    target_rows = {}
    split_counts = {}
    split_paths = {}
    shard_paths = {}
    coverage = {}

    for ti, (target, mat_path) in enumerate(mat_paths.items()):
        with H5MatReader(mat_path) as reader:
            total_rows = reader.length("FR_Response")
        target_rows[target] = int(total_rows)
        split_id = assign_random_split_id(
            total_rows=total_rows,
            seed=args.seed + ti,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        train_rows = np.flatnonzero(split_id == 0).astype(np.int64, copy=False)
        val_rows = np.flatnonzero(split_id == 1).astype(np.int64, copy=False)
        test_rows = np.flatnonzero(split_id == 2).astype(np.int64, copy=False)
        np.save(out_dir / f"split_id_{target}.npy", split_id)
        np.save(out_dir / f"train_rows_{target}.npy", train_rows)
        np.save(out_dir / f"val_rows_{target}.npy", val_rows)
        np.save(out_dir / f"test_rows_{target}.npy", test_rows)
        split_paths[target] = {
            "split_id": str(out_dir / f"split_id_{target}.npy"),
            "train_rows": str(out_dir / f"train_rows_{target}.npy"),
            "val_rows": str(out_dir / f"val_rows_{target}.npy"),
            "test_rows": str(out_dir / f"test_rows_{target}.npy"),
        }
        split_counts[target] = {
            "train": int(train_rows.size),
            "val": int(val_rows.size),
            "test": int(test_rows.size),
            "total": int(total_rows),
        }
        target_shards = write_split_shards(
            {target: mat_path},
            out_dir / "shards",
            split_id,
            batch_size=args.shard_batch_size,
            target_low=args.target_low,
            target_high=args.target_high,
            target_high_by_target={target: high_by_target[target]},
        )
        shard_paths.update(target_shards)
        coverage[target] = coverage_from_split_id(mat_path, split_id, args.norm_batch_size)
        print(f"[{target}] rows total={total_rows} train={train_rows.size} val={val_rows.size} test={test_rows.size}")

    variance_validation = validate_variance_shards(shard_paths, args.norm_batch_size)
    feature_norm = compute_feature_norm_from_shard(
        shard_paths["S"]["train"], batch_size=args.norm_batch_size
    )
    save_json(out_dir / "feature_norm.json", feature_norm)

    manifest = {
        "split_mode": "random_rows_independent_by_target",
        "seed": int(args.seed),
        "target_seed_offsets": {"S": 0, "C": 1, "I": 2},
        "mat_paths": mat_paths,
        "shard_paths": shard_paths,
        "split_paths": split_paths,
        "target_filter": {
            "low": args.target_low,
            "high": args.target_high,
            "high_by_target": high_by_target,
        },
        "ratios": {"train": float(args.train_ratio), "val": float(args.val_ratio), "test": float(args.test_ratio)},
        "total_rows_by_target": target_rows,
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
    print(f"[done] independent target split manifest written to {out_dir}")
    print(f"[shards] wrote split shards under {out_dir / 'shards'}")


if __name__ == "__main__":
    main()
