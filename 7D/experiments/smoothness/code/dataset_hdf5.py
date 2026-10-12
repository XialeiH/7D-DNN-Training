from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterator, Sequence

import h5py
import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common import FEATURE_NAMES, feature_names_for_mode, norm_feature_names_for_mode
else:
    from .common import FEATURE_NAMES, feature_names_for_mode, norm_feature_names_for_mode


TRAIN_FEATURE_FIELDS_BY_MODE = {
    "pref6d": ("OriCos2", "OriSin2", "Contrast", "L6", "L4E", "L4I"),
    "cosonly5d": ("OriCos2", "Contrast", "L6", "L4E", "L4I"),
    "tf_sf7d": ("OriCos2", "Contrast", "L6", "L4E", "L4I", "lgnTF", "lgnSF"),
}
OPTIONAL_META_FIELDS = {
    "Alpha": np.float32,
    "ThetaPref": np.float32,
    "OriCos2": np.float32,
    "OriSin2": np.float32,
    "Gamma": np.float32,
    "Contrast": np.float32,
    "LGNctgr": np.int16,
    "L6": np.float32,
    "L4E": np.float32,
    "L4I": np.float32,
    "FR_ResponseVar": np.float32,
    "FR_ResponseN": np.float32,
    "lgnTF": np.float32,
    "lgnSF": np.float32,
}
DEFAULT_META_FIELD_VALUES = {
    "FR_ResponseVar": 0.0,
    "FR_ResponseN": 1.0,
}


class H5MatReader:
    def __init__(self, mat_path: str | Path):
        self.mat_path = str(mat_path)
        self._h5: h5py.File | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.mat_path, "r")

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self) -> h5py.File:
        self.open()
        assert self._h5 is not None
        return self._h5

    def length(self, key: str) -> int:
        ds = self.h5[key]
        if ds.ndim == 1:
            return int(ds.shape[0])
        if ds.ndim != 2:
            raise ValueError(f"Expected 1D/2D dataset for {key}, got {ds.shape}")
        if ds.shape[0] == 1:
            return int(ds.shape[1])
        if ds.shape[1] == 1:
            return int(ds.shape[0])
        raise ValueError(f"Cannot infer 1D dataset length for {key}, got {ds.shape}")

    def has_key(self, key: str) -> bool:
        return key in self.h5

    def read_1d(self, key: str, idx: np.ndarray | slice | int | None = None, dtype=None) -> np.ndarray:
        ds = self.h5[key]
        if ds.ndim == 1:
            arr = ds[...] if idx is None else ds[idx]
        elif ds.ndim == 2:
            if idx is None:
                arr = ds[...]
            elif ds.shape[0] == 1:
                arr = ds[0, idx]
            elif ds.shape[1] == 1:
                arr = ds[idx, 0]
            else:
                raise ValueError(f"Cannot read 1D dataset for {key}, got {ds.shape}")
        else:
            raise ValueError(f"Expected 1D/2D dataset for {key}, got {ds.shape}")
        arr = np.asarray(arr).reshape(-1)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr


class SplitShardLoader:
    def __init__(
        self,
        shard_path: str | Path,
        feature_norm: Dict[str, Dict[str, float]],
        batch_size: int,
        shuffle: bool,
        seed: int,
        max_batches: int = 0,
        meta_fields: Sequence[str] = (),
        feature_mode: str = "pref6d",
        sampler_mode: str = "uniform",
        target_rate_bins: Sequence[float] = (0.0, 0.5, 2.0, 5.0, 15.0, float("inf")),
        strata_policy: str = "equal",
    ):
        self.shard_path = str(shard_path)
        self.feature_norm = feature_norm
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.max_batches = int(max_batches)
        self.meta_fields = tuple(meta_fields)
        self.feature_mode = str(feature_mode).lower()
        self.feature_fields = TRAIN_FEATURE_FIELDS_BY_MODE[self.feature_mode]
        self.feature_names = feature_names_for_mode(self.feature_mode)
        self.norm_feature_names = norm_feature_names_for_mode(self.feature_mode)
        self.sampler_mode = str(sampler_mode).lower()
        self.target_rate_bins = tuple(float(x) for x in target_rate_bins)
        self.strata_policy = str(strata_policy).lower()
        self._reader: H5MatReader | None = None
        self._epoch = 0
        self._strata_indices: list[np.ndarray] | None = None
        if self.sampler_mode not in {"uniform", "target_rate_stratified"}:
            raise ValueError(f"Unknown sampler_mode={sampler_mode!r}")
        if self.strata_policy != "equal":
            raise ValueError(f"Only strata_policy='equal' is supported, got {strata_policy!r}")

    @property
    def reader(self) -> H5MatReader:
        if self._reader is None:
            self._reader = H5MatReader(self.shard_path)
            self._reader.open()
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __len__(self) -> int:
        total_rows = self.reader.length("FR_Response")
        n_batches = math.ceil(total_rows / max(self.batch_size, 1))
        if self.max_batches > 0:
            n_batches = min(n_batches, self.max_batches)
        return n_batches

    def _normalize(self, feats: np.ndarray) -> np.ndarray:
        for name in self.norm_feature_names:
            col = self.feature_names.index(name)
            stat = self.feature_norm[name]
            values = feats[:, col]
            transform = str(stat.get("transform", "identity")).lower()
            if transform == "log":
                if np.any(values <= 0.0):
                    raise ValueError(f"Feature {name} contains nonpositive values required by log transform")
                values = np.log(values)
            elif transform != "identity":
                raise ValueError(f"Unsupported transform {transform!r} for feature {name}")
            mu = float(stat["mean"])
            sd = float(stat["std"])
            feats[:, col] = (values - mu) / sd
        return feats

    def _read_batch(self, rows: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, dict]:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        reader = self.reader

        feature_arrays = [
            reader.read_1d(field, unique_rows, np.float32)[inverse]
            for field in self.feature_fields
        ]
        target = reader.read_1d("FR_Response", unique_rows, np.float32)[inverse]

        feats = np.stack(feature_arrays, axis=1).astype(np.float32, copy=False)
        feats = self._normalize(feats)

        meta: dict[str, np.ndarray] = {}
        for field in self.meta_fields:
            dtype = OPTIONAL_META_FIELDS[field]
            if reader.has_key(field):
                meta[field] = reader.read_1d(field, unique_rows, dtype)[inverse]
            elif field in DEFAULT_META_FIELD_VALUES:
                meta[field] = np.full(rows.shape[0], DEFAULT_META_FIELD_VALUES[field], dtype=dtype)
            else:
                raise KeyError(f"Missing required metadata field {field!r} in {self.shard_path}")
        return torch.from_numpy(feats), torch.from_numpy(target.reshape(-1, 1)), meta

    def iter_sample(self, sample_size: int, seed: int) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict]]:
        if sample_size <= 0:
            return
        rng = np.random.default_rng(seed)
        total_rows = self.reader.length("FR_Response")
        rows = rng.integers(0, total_rows, size=int(sample_size), dtype=np.int64)
        yield self._read_batch(rows)


    def _get_strata_indices(self) -> list[np.ndarray]:
        if self._strata_indices is not None:
            return self._strata_indices
        bins = np.asarray(self.target_rate_bins, dtype=np.float64)
        if bins.ndim != 1 or bins.size < 2 or np.any(bins[:-1] >= bins[1:]):
            raise ValueError(f"target_rate_bins must be increasing, got {self.target_rate_bins!r}")
        y = self.reader.read_1d("FR_Response", None, np.float32)
        strata: list[np.ndarray] = []
        all_rows = np.arange(y.shape[0], dtype=np.int64)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y >= lo) & (y < hi)
            rows = all_rows[mask]
            if rows.size:
                strata.append(rows.astype(np.int64, copy=False))
        if not strata:
            strata = [all_rows]
        self._strata_indices = strata
        return strata

    def _sample_stratified_rows(self, rng: np.random.Generator, total_rows: int) -> np.ndarray:
        strata = self._get_strata_indices()
        per = max(1, int(math.ceil(self.batch_size / max(len(strata), 1))))
        parts: list[np.ndarray] = []
        for rows in strata:
            n = min(per, self.batch_size)
            parts.append(rng.choice(rows, size=n, replace=rows.size < n))
        out = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if out.size < self.batch_size:
            add = rng.integers(0, total_rows, size=self.batch_size - out.size, dtype=np.int64)
            out = np.concatenate([out, add])
        elif out.size > self.batch_size:
            out = rng.choice(out, size=self.batch_size, replace=False)
        rng.shuffle(out)
        return out.astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        total_rows = self.reader.length("FR_Response")

        if self.shuffle:
            n_batches = self.max_batches
            if n_batches <= 0:
                n_batches = math.ceil(total_rows / max(self.batch_size, 1))
            for _ in range(n_batches):
                if self.sampler_mode == "target_rate_stratified":
                    rows = self._sample_stratified_rows(rng, total_rows)
                else:
                    rows = rng.integers(0, total_rows, size=self.batch_size, dtype=np.int64)
                yield self._read_batch(rows)
            return

        batches_yielded = 0
        for batch_start in range(0, total_rows, self.batch_size):
            if self.max_batches > 0 and batches_yielded >= self.max_batches:
                return
            batch_rows = np.arange(batch_start, min(batch_start + self.batch_size, total_rows), dtype=np.int64)
            yield self._read_batch(batch_rows)
            batches_yielded += 1


def compute_feature_norm_from_shard(
    shard_path: str | Path,
    batch_size: int = 65536,
    feature_mode: str = "pref6d",
) -> Dict[str, Dict[str, float]]:
    norm_feature_names = norm_feature_names_for_mode(feature_mode)
    sums = {name: 0.0 for name in norm_feature_names}
    sumsqs = {name: 0.0 for name in norm_feature_names}
    count = 0
    with H5MatReader(shard_path) as reader:
        total_rows = reader.length("FR_Response")
        for batch_start in range(0, total_rows, batch_size):
            batch_rows = slice(batch_start, min(batch_start + batch_size, total_rows))
            batch = {
                name: reader.read_1d(name, batch_rows, np.float64)
                for name in norm_feature_names
            }
            bsz = len(batch["Contrast"])
            count += bsz
            for name, arr in batch.items():
                sums[name] += float(arr.sum())
                sumsqs[name] += float(np.square(arr).sum())

    active_feature_names = feature_names_for_mode(feature_mode)
    out: Dict[str, Dict[str, float]] = {
        "feature_names": {"order": list(active_feature_names)},
        "active_feature_names": {"order": list(active_feature_names)},
        "norm_feature_names": {"order": list(norm_feature_names)},
        "count": {"n_train_rows": int(count)},
    }
    for name in norm_feature_names:
        mean = sums[name] / max(count, 1)
        var = sumsqs[name] / max(count, 1) - mean * mean
        out[name] = {
            "mean": float(mean),
            "std": float(math.sqrt(max(var, 1e-12))),
        }
    return out
