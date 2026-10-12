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
        self._reader: H5MatReader | None = None
        self._epoch = 0

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
            mu = float(self.feature_norm[name]["mean"])
            sd = float(self.feature_norm[name]["std"])
            feats[:, col] = (feats[:, col] - mu) / sd
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

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, dict]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        total_rows = self.reader.length("FR_Response")

        if self.shuffle:
            n_batches = self.max_batches
            if n_batches <= 0:
                n_batches = math.ceil(total_rows / max(self.batch_size, 1))
            for _ in range(n_batches):
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
