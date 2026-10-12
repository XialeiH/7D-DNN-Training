#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import OrderedDict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from common import (
        ModelConfig,
        TrainConfig,
        build_model,
        clamp_raw_rate_array,
        clamp_raw_rate_tensor,
        diag_line,
        device_autocast,
        ensure_dir,
        feature_names_for_mode,
        input_dim_for_mode,
        load_json,
        save_json,
        set_seed,
        to_serializable_config,
    )
    from dataset_hdf5 import H5MatReader, SplitShardLoader
else:
    from .common import (
        ModelConfig,
        TrainConfig,
        build_model,
        clamp_raw_rate_array,
        clamp_raw_rate_tensor,
        diag_line,
        device_autocast,
        ensure_dir,
        feature_names_for_mode,
        input_dim_for_mode,
        load_json,
        save_json,
        set_seed,
        to_serializable_config,
    )
    from .dataset_hdf5 import H5MatReader, SplitShardLoader


DEFAULT_S_MAT = "/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/S_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part.mat"
DEFAULT_C_MAT = "/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/C_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part.mat"
DEFAULT_I_MAT = "/scratch/xh2906/NYU-Vision-2Drive-main/Data/Paper2_NetworkTuning/Fig1V4/6D_training_data/I_allLGN_prefAngleSymm_6D_allOrientation_allContrasts_part.mat"
DEFAULT_RUN_ROOT = "/scratch/xh2906/librarySCI_runs"
TARGET_TO_MAT = {"S": DEFAULT_S_MAT, "C": DEFAULT_C_MAT, "I": DEFAULT_I_MAT}
BAD_SLAB_WEIGHT_BY_PAIR = {
    (0.0, 52.0): 3.0,
    (3.75, 52.0): 3.0,
    (7.5, 52.0): 3.0,
    (11.25, 52.0): 3.0,
    (0.0, 95.0): 3.0,
    (7.5, 95.0): 3.0,
    (15.0, 95.0): 3.0,
    (22.5, 95.0): 3.0,
    (18.75, 71.0): 2.0,
    (22.5, 71.0): 2.0,
    (22.5, 14.0): 2.0,
}
BAD_HC_SELECTION_PAIRS = (
    (0.0, 42.0),
    (0.0, 52.0),
    (0.0, 61.0),
    (0.0, 95.0),
    (3.75, 52.0),
    (7.5, 52.0),
    (7.5, 95.0),
    (11.25, 52.0),
    (15.0, 95.0),
    (18.75, 71.0),
    (22.5, 14.0),
    (22.5, 71.0),
    (22.5, 95.0),
)
TARGET_TO_MASK_BOUNDS = {
    "S": (0.5, 30.0),
    "C": (0.5, 100.0),
    "I": (0.5, 140.0),
}
TFSF_POPULATION_NAMES = (
    "population_default",
    "population_tf_axis",
    "population_sf_axis",
    "population_joint",
)
TFSF_FOCUS_POPULATION_NAME = "population_focus"
ALL_TFSF_POPULATION_NAMES = (*TFSF_POPULATION_NAMES, TFSF_FOCUS_POPULATION_NAME)


def active_tfsf_population_names(cfg: TrainConfig) -> tuple[str, ...]:
    mode = str(getattr(cfg, "population_grad_mode", "none")).lower()
    if mode == "tfsf4":
        return TFSF_POPULATION_NAMES
    if mode == "tfsf4_focus":
        return ALL_TFSF_POPULATION_NAMES
    return ()


class MetricAccumulator:
    def __init__(
        self,
        loss_eps: float,
        rel_alpha: float,
        rmse_space: str,
        w_log: float,
        w_rel: float,
        raw_mse_lambda: float,
        low_penalty_lambda: float,
        low_penalty_tau: float,
        mono_pair_lambda: float,
        loss_mode: str = "base",
        mid_under_lambda: float = 0.0,
        edge_over_lambda: float = 0.0,
        corr_lambda: float = 0.0,
        corr_w_under_mid0: float = 1.0,
        corr_w_over_edge225: float = 1.0,
    ) -> None:
        self.loss_eps = float(loss_eps)
        self.rel_alpha = float(rel_alpha)
        self.rmse_space = str(rmse_space).lower()
        self.w_log = float(w_log)
        self.w_rel = float(w_rel)
        self.raw_mse_lambda = float(raw_mse_lambda)
        self.low_penalty_lambda = float(low_penalty_lambda)
        self.low_penalty_tau = float(low_penalty_tau)
        self.mono_pair_lambda = float(mono_pair_lambda)
        self.loss_mode = str(loss_mode)
        self.mid_under_lambda = float(mid_under_lambda)
        self.edge_over_lambda = float(edge_over_lambda)
        self.corr_lambda = float(corr_lambda)
        self.corr_w_under_mid0 = float(corr_w_under_mid0)
        self.corr_w_over_edge225 = float(corr_w_over_edge225)
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_log_sq = 0.0
        self.sum_rel_sq = 0.0
        self.sum_low_over_sq = 0.0
        self.sum_mono_weighted = 0.0
        self.sum_mid_under_weighted = 0.0
        self.sum_edge_over_weighted = 0.0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        mono_pair_loss: float = 0.0,
        mid_under_penalty: float = 0.0,
        edge_over_penalty: float = 0.0,
    ) -> None:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        self.n += int(y_true.size)
        self.sum_y += float(y_true.sum())
        self.sum_y2 += float(np.square(y_true).sum())
        self.sum_abs += float(np.abs(y_true - y_pred).sum())
        self.sum_sq += float(np.square(y_true - y_pred).sum())
        self.sum_log_sq += float(np.square(np.log1p(y_true) - np.log1p(np.clip(y_pred, a_min=0.0, a_max=None))).sum())
        self.sum_rel_sq += float(np.square((y_pred - y_true) / (y_true + self.rel_alpha)).sum())
        low_mask = y_true < self.low_penalty_tau
        over = np.maximum(y_pred - y_true, 0.0)
        self.sum_low_over_sq += float(np.square(over[low_mask]).sum())
        self.sum_mono_weighted += float(mono_pair_loss) * max(int(y_true.size), 1)
        self.sum_mid_under_weighted += float(mid_under_penalty) * max(int(y_true.size), 1)
        self.sum_edge_over_weighted += float(edge_over_penalty) * max(int(y_true.size), 1)

    def finalize(self) -> dict:
        n = max(self.n, 1)
        mean_y = self.sum_y / n
        ss_tot = self.sum_y2 - n * mean_y * mean_y
        rmse_log = float(np.sqrt(self.sum_log_sq / n + self.loss_eps))
        rmse_raw = float(np.sqrt(self.sum_sq / n + self.loss_eps))
        mse_raw = float(self.sum_sq / n)
        relrmse_raw = float(np.sqrt(self.sum_rel_sq / n + self.loss_eps))
        low_over_penalty = float(self.sum_low_over_sq / n)
        mono_pair_loss = float(self.sum_mono_weighted / n)
        mid_under_penalty = float(self.sum_mid_under_weighted / n)
        edge_over_penalty = float(self.sum_edge_over_weighted / n)
        rmse_term = rmse_log if self.rmse_space == "log" else rmse_raw
        if self.loss_mode == "legacy_gamma_contrast_asym":
            corr_penalty = self.corr_lambda * (self.corr_w_under_mid0 * mid_under_penalty + self.corr_w_over_edge225 * edge_over_penalty)
        else:
            corr_penalty = self.mid_under_lambda * mid_under_penalty + self.edge_over_lambda * edge_over_penalty
        return {
            "n": self.n,
            "rmse_log": rmse_log,
            "rmse_term": rmse_term,
            "relrmse_raw": relrmse_raw,
            "low_over_penalty": low_over_penalty,
            "mono_pair_loss": mono_pair_loss,
            "mid_under_penalty": mid_under_penalty,
            "edge_over_penalty": edge_over_penalty,
            "corr_penalty": corr_penalty,
            "total_loss": self.w_log * rmse_term + self.w_rel * relrmse_raw + self.raw_mse_lambda * mse_raw + self.low_penalty_lambda * low_over_penalty + self.mono_pair_lambda * mono_pair_loss + corr_penalty,
            "rmse_raw": rmse_raw,
            "mse_raw": mse_raw,
            "mae_raw": self.sum_abs / n,
            "r2_raw": 1.0 - self.sum_sq / (ss_tot + 1e-12),
        }


RATE_BANDS = (
    ("0_0p5", 0.0, 0.5),
    ("0p5_2", 0.5, 2.0),
    ("2_5", 2.0, 5.0),
    ("5_15", 5.0, 15.0),
    ("15_inf", 15.0, float("inf")),
)


class RateBandAccumulator:
    def __init__(self, rel_alpha: float, loss_eps: float) -> None:
        self.rel_alpha = float(rel_alpha)
        self.loss_eps = float(loss_eps)
        self.stats = {
            name: {"n": 0, "sum_sq": 0.0, "sum_abs": 0.0, "sum_rel_sq": 0.0}
            for name, _lo, _hi in RATE_BANDS
        }

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = y_true.reshape(-1).astype(np.float64, copy=False)
        y_pred = y_pred.reshape(-1).astype(np.float64, copy=False)
        err = y_pred - y_true
        for name, lo, hi in RATE_BANDS:
            if np.isfinite(hi):
                mask = (y_true >= lo) & (y_true < hi)
            else:
                mask = y_true >= lo
            if not np.any(mask):
                continue
            bucket = self.stats[name]
            bucket["n"] += int(np.sum(mask))
            bucket["sum_sq"] += float(np.square(err[mask]).sum())
            bucket["sum_abs"] += float(np.abs(err[mask]).sum())
            bucket["sum_rel_sq"] += float(np.square(err[mask] / (y_true[mask] + self.rel_alpha)).sum())

    def finalize(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for name, stats in self.stats.items():
            n = max(int(stats["n"]), 1)
            out[name] = {
                "n": int(stats["n"]),
                "rmse_raw": float(np.sqrt(stats["sum_sq"] / n + self.loss_eps)),
                "mae_raw": float(stats["sum_abs"] / n),
                "relrmse_raw": float(np.sqrt(stats["sum_rel_sq"] / n + self.loss_eps)),
            }
        return out


def numpy_variance_weights(meta: dict, target_safe: np.ndarray, cfg: TrainConfig, space: str) -> np.ndarray | None:
    if not variance_weight_enabled(cfg):
        return None
    if "FR_ResponseVar" not in meta or "FR_ResponseN" not in meta:
        return None
    var = np.asarray(meta["FR_ResponseVar"], dtype=np.float64).reshape(-1)
    repeat_n = np.asarray(meta["FR_ResponseN"], dtype=np.float64).reshape(-1)
    target_safe = np.asarray(target_safe, dtype=np.float64).reshape(-1)
    valid = np.isfinite(var) & np.isfinite(repeat_n) & (var >= 0.0) & (repeat_n > 0.0)
    sem = np.zeros_like(var, dtype=np.float64)
    sem[valid] = var[valid] / np.maximum(repeat_n[valid], 1.0)
    key = str(space).lower()
    if key == "log":
        sigma2 = sem / np.square(1.0 + target_safe)
        ref = float(getattr(cfg, "variance_ref_log", 1.0))
    elif key == "rel":
        sigma2 = sem / np.square(target_safe + float(cfg.rel_alpha))
        ref = float(getattr(cfg, "variance_ref_rel", 1.0))
    elif key == "raw":
        sigma2 = sem
        ref = float(getattr(cfg, "variance_ref_raw", 1.0))
    else:
        raise ValueError(f"Unknown variance weight space {space!r}")
    if not np.isfinite(ref) or ref <= 0.0:
        return np.ones_like(target_safe, dtype=np.float64)
    weights = 1.0 / (1.0 + sigma2 / max(ref, float(getattr(cfg, "variance_eps", 1e-8))))
    weights = np.clip(weights, float(getattr(cfg, "variance_min_weight", 0.25)), 1.0)
    weights[~valid] = 1.0
    return weights.astype(np.float64, copy=False)


class WeightedLossAccumulator:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.sum_w_log = 0.0
        self.sum_w_raw = 0.0
        self.sum_w_rel = 0.0
        self.sum_log_sq = 0.0
        self.sum_raw_sq = 0.0
        self.sum_rel_sq = 0.0
        self.sum_low_over_sq = 0.0
        self.sum_w_low = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, meta: dict) -> None:
        target_safe = np.clip(y_true.reshape(-1).astype(np.float64, copy=False), a_min=0.0, a_max=None)
        pred_safe = np.clip(y_pred.reshape(-1).astype(np.float64, copy=False), a_min=0.0, a_max=None)
        w_log = numpy_variance_weights(meta, target_safe, self.cfg, "log")
        w_raw = numpy_variance_weights(meta, target_safe, self.cfg, "raw")
        w_rel = numpy_variance_weights(meta, target_safe, self.cfg, "rel")
        if w_log is None:
            w_log = np.ones_like(target_safe)
        if w_raw is None:
            w_raw = np.ones_like(target_safe)
        if w_rel is None:
            w_rel = np.ones_like(target_safe)
        log_sq = np.square(np.log1p(pred_safe) - np.log1p(target_safe))
        raw_sq = np.square(pred_safe - target_safe)
        rel_sq = np.square((pred_safe - target_safe) / (target_safe + float(self.cfg.rel_alpha)))
        self.sum_log_sq += float(np.sum(w_log * log_sq))
        self.sum_raw_sq += float(np.sum(w_raw * raw_sq))
        self.sum_rel_sq += float(np.sum(w_rel * rel_sq))
        self.sum_w_log += float(np.sum(w_log))
        self.sum_w_raw += float(np.sum(w_raw))
        self.sum_w_rel += float(np.sum(w_rel))
        low_mask = target_safe < float(self.cfg.low_penalty_tau)
        if np.any(low_mask):
            low_over = np.maximum(pred_safe - target_safe, 0.0)
            self.sum_low_over_sq += float(np.sum(w_raw[low_mask] * np.square(low_over[low_mask])))
            self.sum_w_low += float(np.sum(w_raw[low_mask]))

    def finalize(self) -> dict[str, float]:
        rmse_log = float(np.sqrt(self.sum_log_sq / max(self.sum_w_log, 1e-12) + float(self.cfg.loss_eps)))
        rmse_raw = float(np.sqrt(self.sum_raw_sq / max(self.sum_w_raw, 1e-12) + float(self.cfg.loss_eps)))
        mse_raw = float(self.sum_raw_sq / max(self.sum_w_raw, 1e-12))
        relrmse_raw = float(np.sqrt(self.sum_rel_sq / max(self.sum_w_rel, 1e-12) + float(self.cfg.loss_eps)))
        low_over_penalty = float(self.sum_low_over_sq / max(self.sum_w_low, 1e-12)) if self.sum_w_low > 0.0 else 0.0
        rmse_term = rmse_log if str(self.cfg.rmse_space).lower() == "log" else rmse_raw
        total_loss = float(self.cfg.w_log) * rmse_term + float(self.cfg.w_rel) * relrmse_raw + float(getattr(self.cfg, "raw_mse_lambda", 0.0)) * mse_raw + float(self.cfg.low_penalty_lambda) * low_over_penalty
        return {
            "weighted_total_loss": total_loss,
            "weighted_rmse_log": rmse_log,
            "weighted_rmse_raw": rmse_raw,
            "weighted_mse_raw": mse_raw,
            "weighted_rmse_term": rmse_term,
            "weighted_relrmse_raw": relrmse_raw,
            "weighted_low_over_penalty": low_over_penalty,
        }


def parse_loss_exclude_condition_file(path_text: str) -> tuple[tuple[float, float], ...]:
    if not path_text:
        return ()
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"loss exclude condition file not found: {path}")
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            rows = data.get("bad_conditions", data.get("conditions", []))
        else:
            rows = data
        pairs = []
        for row in rows:
            if isinstance(row, dict):
                angle = row.get("angle", row.get("Gamma", row.get("gamma")))
                contrast = row.get("contrast", row.get("Contrast"))
            else:
                angle, contrast = row[:2]
            pairs.append((round(float(angle), 8), round(float(contrast), 8)))
        return tuple(sorted(set(pairs)))
    pairs = []
    with path.open(newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]
            if len(parts) < 2 or parts[0].lower() in {"angle", "gamma"}:
                continue
            pairs.append((round(float(parts[0]), 8), round(float(parts[1]), 8)))
    return tuple(sorted(set(pairs)))


def parse_population_focus_condition_file(path_text: str) -> tuple[tuple[float, ...], ...]:
    if not path_text:
        return ()
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"population focus condition file not found: {path}")
    if path.suffix.lower() == ".json":
        data = load_json(path)
        rows = data.get("conditions", []) if isinstance(data, dict) else data
    else:
        rows = []
        with path.open(newline="") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [part.strip() for part in line.replace(",", " ").split()]
                if len(parts) < 3 or parts[0].lower() == "contrast":
                    continue
                rows.append(parts[:4])
    conditions = []
    for row in rows:
        if isinstance(row, dict):
            contrast = row.get("contrast", row.get("Contrast"))
            tf = row.get("tf", row.get("lgnTF"))
            sf = row.get("sf", row.get("lgnSF"))
            angle = row.get("angle", row.get("Alpha"))
        else:
            contrast, tf, sf = row[:3]
            angle = row[3] if len(row) >= 4 else None
        condition = (round(float(contrast), 8), round(float(tf), 8), round(float(sf), 8))
        if angle is not None:
            condition = (*condition, round(float(angle), 8))
        conditions.append(condition)
    return tuple(sorted(set(conditions)))


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train one 6D preferred-angle single-output regressor")
    ap.add_argument("--target", required=True, choices=["S", "C", "I"])
    ap.add_argument("--train_mat", default=None)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--feature_mode", type=str, choices=["pref6d", "cosonly5d", "tf_sf7d"], default="pref6d")
    ap.add_argument("--model_family", type=str, choices=["single_mlp", "double_head_additive", "threshold_additive"], default="single_mlp")
    ap.add_argument("--hidden_dims", type=str, default="64,128,128")
    ap.add_argument("--activation_seq", type=str, default="")
    ap.add_argument("--shape_hidden_dims", type=str, default="96,96,96")
    ap.add_argument("--shape_activation_seq", type=str, default="tanh,tanh,tanh")
    ap.add_argument("--contrast_hidden_dims", type=str, default="32,32")
    ap.add_argument("--contrast_activation_seq", type=str, default="tanh,tanh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25, help="Alias for stage_epochs")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32768)
    ap.add_argument("--eval_batch_size", type=int, default=65536)
    ap.add_argument("--max_train_batches", type=int, default=10)
    ap.add_argument("--max_eval_batches", type=int, default=0)
    ap.add_argument("--stage_epochs", type=int, default=25)
    ap.add_argument("--max_stages", type=int, default=6)
    ap.add_argument("--lr_stage_values", type=str, default="5e-3,2e-3,9e-4,6e-4,2e-4")
    ap.add_argument("--min_improve", type=float, default=0.05)
    ap.add_argument("--min_best_rel_gain", type=float, default=0.002)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--loss_eps", type=float, default=1e-8)
    ap.add_argument("--rel_alpha", type=float, default=1.0)
    ap.add_argument("--rmse_space", type=str, choices=["log", "raw"], default="log")
    ap.add_argument("--w_log", type=float, default=0.5)
    ap.add_argument("--w_rel", type=float, default=0.5)
    ap.add_argument("--raw_mse_lambda", type=float, default=0.0, help="Weight for variance-aware raw firing-rate MSE")
    ap.add_argument("--low_penalty_lambda", type=float, default=0.0)
    ap.add_argument("--low_penalty_tau", type=float, default=3.0)
    ap.add_argument("--mono_pair_lambda", type=float, default=0.0)
    ap.add_argument("--mono_grad_lambda", type=float, default=0.0, help="Section 5.5 derivative monotonicity penalty weight")
    ap.add_argument("--mono_grad_max_points", type=int, default=2048, help="Max rows per batch used for derivative monotonicity penalty")
    ap.add_argument("--curvature_lambda", type=float, default=0.0, help="Central finite-difference curvature penalty weight")
    ap.add_argument("--curvature_max_points", type=int, default=2048)
    ap.add_argument("--curvature_oricos2_delta", type=float, default=0.05)
    ap.add_argument("--curvature_contrast_delta", type=float, default=5.0)
    ap.add_argument("--curvature_mode", choices=["local_oc_contrast", "global_all7"], default="local_oc_contrast")
    ap.add_argument("--global_curvature_delta_fraction", type=float, default=0.04, help="Finite-difference step as a fraction of each full input-domain width")
    ap.add_argument("--global_curvature_direction_count", type=int, default=4, help="Random multi-axis directions used to penalize mixed curvature")
    ap.add_argument("--global_curvature_output_scale", type=float, default=0.0, help="Rate scale for global curvature; <=0 uses the target upper bound")
    ap.add_argument("--global_curvature_balance_mode", choices=["none", "baseline_relative"], default="none", help="Normalize each axis and mixed-direction penalty by target-specific baseline curvature")
    ap.add_argument("--global_curvature_balance_path", type=str, default="", help="Global smoothness audit JSON used by baseline_relative balancing")
    ap.add_argument("--global_curvature_reduce_mode", choices=["mean", "smoothmax", "focus_axis"], default="mean", help="Reduce normalized axis/mixed penalties by their mean, a smooth maximum, or one named axis plus a global mean")
    ap.add_argument("--global_curvature_smoothmax_temperature", type=float, default=5.0)
    ap.add_argument("--global_curvature_focus_feature", type=str, default="")
    ap.add_argument("--global_curvature_focus_mean_weight", type=float, default=0.1)
    ap.add_argument("--curvature_grad_ratio", type=float, default=0.0, help="If >0, apply curvature after the Adam regression step and cap its direct parameter update to this fraction of the Adam update norm")
    ap.add_argument("--ckpt_global_curvature_beta", type=float, default=0.0, help="Deterministic global-curvature weight used only for checkpoint selection")
    ap.add_argument("--ckpt_global_curvature_points", type=int, default=512)
    ap.add_argument("--grad_lambda", type=float, default=0.0)
    ap.add_argument("--grad_max_points", type=int, default=128)
    ap.add_argument("--grad_interior_eps", type=float, default=0.05)
    ap.add_argument("--online_interp_enable", action="store_true")
    ap.add_argument("--online_interp_frac", type=float, default=0.10)
    ap.add_argument("--online_interp_eps", type=float, default=0.05)
    ap.add_argument("--online_interp_cache_slabs", type=int, default=8)
    ap.add_argument("--online_interp_badslab_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--task_uw_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--task_uw_shadow_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--task_uw_shadow_lr", type=float, default=1e-2)
    ap.add_argument("--task_uw_aux_warmup_frac", type=float, default=0.0)
    ap.add_argument("--task_uw_aux_warmup_scale", type=float, default=1.0)
    ap.add_argument("--task_pcgrad_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--task_cagrad_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cagrad_alpha", type=float, default=0.5)
    ap.add_argument("--cagrad_rescale", type=int, choices=[0, 1, 2], default=1)
    ap.add_argument("--population_grad_mode", choices=["none", "tfsf4", "tfsf4_focus"], default="none")
    ap.add_argument("--population_focus_condition_file", type=str, default="", help="JSON/CSV/TXT rows of Contrast, TF, SF for the population focus CAGrad task")
    ap.add_argument("--population_focus_raw_lambda", type=float, default=0.0, help="Dimensionless raw-RMSE weight used only by the focus population")
    ap.add_argument("--population_focus_raw_scale", type=float, default=0.0, help="Raw-rate scale for focus RMSE; <=0 uses the target upper range")
    ap.add_argument("--population_focus_group_cvar_fraction", type=float, default=0.0, help="If >0, average the worst fraction of per-condition focus losses")
    ap.add_argument("--ckpt_population_focus_beta", type=float, default=0.0, help="Checkpoint-selection weight for mixed loss on focus conditions")
    ap.add_argument("--subloss_norm_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--subloss_norm_init_batches", type=int, default=8)
    ap.add_argument("--gradcos_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gradcos_every", type=int, default=20)
    ap.add_argument("--diag_sample_size", type=int, default=200000)
    ap.add_argument("--ckpt_bad_subset_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ckpt_bad_subset_beta", type=float, default=1.0)
    ap.add_argument("--ckpt_bad_subset_gamma", type=float, default=0.5)
    ap.add_argument("--variance_weight_mode", type=str, choices=["none", "soft_inv_sem"], default="none")
    ap.add_argument("--variance_min_weight", type=float, default=0.25)
    ap.add_argument("--variance_ref_quantile", type=float, default=0.50)
    ap.add_argument("--variance_eps", type=float, default=1e-8)
    ap.add_argument("--sampler_mode", type=str, choices=["uniform", "target_rate_stratified"], default="uniform")
    ap.add_argument("--target_rate_bins", type=str, default="0,0.5,2,5,15,inf")
    ap.add_argument("--strata_policy", type=str, choices=["equal"], default="equal")
    ap.add_argument("--loss_exclude_contrasts", type=str, default="", help="Comma-separated Contrast values to omit from gradient loss, e.g. 2,5,10")
    ap.add_argument("--loss_exclude_condition_file", type=str, default="", help="JSON/CSV/TXT file listing angle/Gamma and Contrast pairs to omit from gradient loss")
    ap.add_argument("--loss_exclude_eval", action=argparse.BooleanOptionalAction, default=False, help="Also apply loss exclusion during validation metrics")
    ap.add_argument("--adv_enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--adv_epsilon", type=float, default=0.03, help="FGSM epsilon in normalized feature units")
    ap.add_argument("--adv_lambda", type=float, default=0.25, help="Weight for Huber loss on adversarial inputs")
    ap.add_argument("--adv_smooth_lambda", type=float, default=0.10, help="Weight for prediction smoothness between clean/adversarial inputs")
    ap.add_argument("--adv_huber_delta", type=float, default=1.0)
    ap.add_argument("--adv_clip_value", type=float, default=6.0, help="Clamp adversarial normalized features to [-value, value]; <=0 disables")
    ap.add_argument("--guide_uncertainty_mode", type=str, choices=["none", "guide83_count_weighted", "guide84_poisson"], default="none")
    ap.add_argument("--guide83_w_log", type=float, default=1.0)
    ap.add_argument("--guide83_w_raw", type=float, default=1.0)
    ap.add_argument("--guide83_c", type=float, default=1.0)
    ap.add_argument("--guide84_time_window", type=float, default=1.0)
    ap.add_argument("--guide84_eps", type=float, default=1e-6)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--loss_mode", type=str, choices=["base", "targeted_mid_edge_asym", "legacy_gamma_contrast_asym"], default="base")
    ap.add_argument("--corr_lambda", type=float, default=0.0)
    ap.add_argument("--corr_w_under_mid0", type=float, default=1.0)
    ap.add_argument("--corr_w_over_edge225", type=float, default=1.0)
    ap.add_argument("--mid_under_lambda", type=float, default=0.0)
    ap.add_argument("--edge_over_lambda", type=float, default=0.0)
    ap.add_argument("--mid_under_gamma_max", type=float, default=15.0)
    ap.add_argument("--edge_over_gamma_radius", type=float, default=10.0)
    ap.add_argument("--mid_under_contrasts", type=str, default="42,52,61")
    ap.add_argument("--edge_over_contrasts", type=str, default="14,95")
    ap.add_argument("--preset_name", type=str, default="")
    ap.add_argument("--init_checkpoint", type=str, default="", help="Optional checkpoint to load before training, used for widened 7D initialization")
    ap.add_argument("--train_stage_modes", type=str, default="", help="Comma-separated trainability stages: all, first, newcols")
    ap.add_argument("--weight_anchor_checkpoint", type=str, default="", help="Optional checkpoint used for L2 weight-change regularization")
    ap.add_argument("--weight_anchor_lambda", type=float, default=0.0)
    ap.add_argument("--weight_anchor_include_new_inputs", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--tfsf_smooth_lambda", type=float, default=0.0, help="Prediction smoothness penalty along normalized lgnTF/lgnSF inputs")
    ap.add_argument("--tfsf_smooth_delta", type=float, default=0.05)
    ap.add_argument("--tfsf_smooth_max_points", type=int, default=2048)
    ap.add_argument("--tf_unimodal_lambda", type=float, default=0.0, help="Soft rise-then-fall penalty along the observed TF grid")
    ap.add_argument("--sf_unimodal_lambda", type=float, default=0.0, help="Soft rise-then-fall penalty along the observed SF grid")
    ap.add_argument("--tfsf_unimodal_max_points", type=int, default=512)
    ap.add_argument("--tfsf_unimodal_prominence_fraction", type=float, default=0.02, help="Required interior-peak prominence as a fraction of the target firing-rate range")
    ap.add_argument("--distill_anchor_lambda", type=float, default=0.0, help="Old-function distillation weight")
    ap.add_argument(
        "--distill_anchor_mode",
        choices=["default_tfsf", "same_input_below_tf"],
        default="default_tfsf",
        help="Anchor at default TF/SF or at unchanged inputs below a TF cutoff",
    )
    ap.add_argument("--distill_default_tf", type=float, default=10.0)
    ap.add_argument("--distill_default_sf", type=float, default=2.5)
    ap.add_argument("--distill_max_tf_exclusive", type=float, default=32.0)
    ap.add_argument("--distill_max_points", type=int, default=2048)
    args = ap.parse_args()

    def parse_int_csv(raw: str) -> tuple[int, ...]:
        return tuple(int(x) for x in str(raw).split(",") if x.strip())

    def parse_str_csv(raw: str, default_len: int = 0, default_name: str = "relu") -> tuple[str, ...]:
        vals = tuple(x.strip() for x in str(raw).split(",") if x.strip())
        if vals:
            return vals
        return tuple(default_name for _ in range(default_len))

    def parse_float_csv(raw: str) -> tuple[float, ...]:
        vals: list[float] = []
        for item in str(raw).split(","):
            item = item.strip().lower()
            if not item:
                continue
            vals.append(float("inf") if item in ("inf", "+inf", "infinity") else float(item))
        return tuple(vals)

    def parse_stage_modes(raw: str) -> tuple[str, ...]:
        modes = tuple(x.strip().lower() for x in str(raw).split(",") if x.strip())
        allowed = {"all", "first", "newcols"}
        bad = [x for x in modes if x not in allowed]
        if bad:
            raise ValueError(f"Unknown train stage mode(s): {bad}; allowed={sorted(allowed)}")
        return modes

    stage_values = tuple(float(x) for x in args.lr_stage_values.split(",") if x.strip())
    hidden_dims = parse_int_csv(args.hidden_dims)
    if not hidden_dims:
        raise ValueError("hidden_dims must be non-empty")
    activation_seq = parse_str_csv(args.activation_seq, default_len=len(hidden_dims), default_name="relu")
    if len(hidden_dims) != len(activation_seq):
        raise ValueError("hidden_dims and activation_seq must have matching length")
    shape_hidden_dims = parse_int_csv(args.shape_hidden_dims)
    shape_activation_seq = parse_str_csv(args.shape_activation_seq, default_len=len(shape_hidden_dims), default_name="tanh")
    contrast_hidden_dims = parse_int_csv(args.contrast_hidden_dims)
    contrast_activation_seq = parse_str_csv(args.contrast_activation_seq, default_len=len(contrast_hidden_dims), default_name="tanh")
    if args.model_family == "double_head_additive":
        if args.feature_mode != "cosonly5d":
            raise ValueError("double_head_additive requires --feature_mode cosonly5d")
        if shape_hidden_dims != (96, 96, 96):
            raise ValueError("double_head_additive v1 requires --shape_hidden_dims 96,96,96")
        if shape_activation_seq != ("tanh", "tanh", "tanh"):
            raise ValueError("double_head_additive v1 requires --shape_activation_seq tanh,tanh,tanh")
        if contrast_hidden_dims != (32, 32):
            raise ValueError("double_head_additive v1 requires --contrast_hidden_dims 32,32")
        if contrast_activation_seq != ("tanh", "tanh"):
            raise ValueError("double_head_additive v1 requires --contrast_activation_seq tanh,tanh")
    if args.model_family == "threshold_additive" and args.feature_mode != "cosonly5d":
        raise ValueError("threshold_additive requires --feature_mode cosonly5d")
    target_rate_bins = parse_float_csv(args.target_rate_bins)
    if len(target_rate_bins) < 2:
        raise ValueError("--target_rate_bins must include at least two edges")
    if any(target_rate_bins[i] >= target_rate_bins[i + 1] for i in range(len(target_rate_bins) - 1)):
        raise ValueError("--target_rate_bins must be strictly increasing")
    if args.epochs != args.stage_epochs:
        raise ValueError("epochs is treated as an alias for stage_epochs and must match it")

    train_mat = args.train_mat or TARGET_TO_MAT[args.target]
    out_dir = args.out_dir or str(Path(DEFAULT_RUN_ROOT) / f"sixd_prefAngle_{args.target}")
    cfg = TrainConfig(
        train_mat=train_mat,
        split_dir=args.split_dir,
        out_dir=out_dir,
        target=args.target,
        device=args.device,
        feature_mode=args.feature_mode,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        stage_epochs=args.stage_epochs,
        max_stages=args.max_stages,
        lr_stage_values=stage_values,
        min_improve=args.min_improve,
        min_best_rel_gain=args.min_best_rel_gain,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        loss_eps=args.loss_eps,
        rel_alpha=args.rel_alpha,
        rmse_space=args.rmse_space,
        w_log=args.w_log,
        w_rel=args.w_rel,
        raw_mse_lambda=args.raw_mse_lambda,
        low_penalty_lambda=args.low_penalty_lambda,
        low_penalty_tau=args.low_penalty_tau,
        mono_pair_lambda=args.mono_pair_lambda,
        mono_grad_lambda=args.mono_grad_lambda,
        mono_grad_max_points=args.mono_grad_max_points,
        curvature_lambda=args.curvature_lambda,
        curvature_max_points=args.curvature_max_points,
        curvature_oricos2_delta=args.curvature_oricos2_delta,
        curvature_contrast_delta=args.curvature_contrast_delta,
        curvature_mode=args.curvature_mode,
        global_curvature_delta_fraction=args.global_curvature_delta_fraction,
        global_curvature_direction_count=args.global_curvature_direction_count,
        global_curvature_output_scale=args.global_curvature_output_scale,
        global_curvature_balance_mode=args.global_curvature_balance_mode,
        global_curvature_balance_path=args.global_curvature_balance_path,
        global_curvature_reduce_mode=args.global_curvature_reduce_mode,
        global_curvature_smoothmax_temperature=args.global_curvature_smoothmax_temperature,
        global_curvature_focus_feature=args.global_curvature_focus_feature,
        global_curvature_focus_mean_weight=args.global_curvature_focus_mean_weight,
        curvature_grad_ratio=args.curvature_grad_ratio,
        ckpt_global_curvature_beta=args.ckpt_global_curvature_beta,
        ckpt_global_curvature_points=args.ckpt_global_curvature_points,
        grad_lambda=args.grad_lambda,
        grad_max_points=args.grad_max_points,
        grad_interior_eps=args.grad_interior_eps,
        online_interp_enable=args.online_interp_enable,
        online_interp_frac=args.online_interp_frac,
        online_interp_eps=args.online_interp_eps,
        online_interp_cache_slabs=args.online_interp_cache_slabs,
        online_interp_badslab_only=args.online_interp_badslab_only,
        task_uw_enable=args.task_uw_enable,
        task_uw_shadow_enable=args.task_uw_shadow_enable,
        task_uw_shadow_lr=args.task_uw_shadow_lr,
        task_uw_aux_warmup_frac=args.task_uw_aux_warmup_frac,
        task_uw_aux_warmup_scale=args.task_uw_aux_warmup_scale,
        task_pcgrad_enable=args.task_pcgrad_enable,
        task_cagrad_enable=args.task_cagrad_enable,
        cagrad_alpha=args.cagrad_alpha,
        cagrad_rescale=args.cagrad_rescale,
        population_grad_mode=args.population_grad_mode,
        population_focus_conditions=parse_population_focus_condition_file(args.population_focus_condition_file),
        population_focus_raw_lambda=args.population_focus_raw_lambda,
        population_focus_raw_scale=args.population_focus_raw_scale,
        population_focus_group_cvar_fraction=args.population_focus_group_cvar_fraction,
        ckpt_population_focus_beta=args.ckpt_population_focus_beta,
        subloss_norm_enable=args.subloss_norm_enable,
        subloss_norm_init_batches=args.subloss_norm_init_batches,
        gradcos_enable=args.gradcos_enable,
        gradcos_every=args.gradcos_every,
        diag_sample_size=args.diag_sample_size,
        ckpt_bad_subset_enable=args.ckpt_bad_subset_enable,
        ckpt_bad_subset_beta=args.ckpt_bad_subset_beta,
        ckpt_bad_subset_gamma=args.ckpt_bad_subset_gamma,
        variance_weight_mode=args.variance_weight_mode,
        variance_min_weight=args.variance_min_weight,
        variance_ref_quantile=args.variance_ref_quantile,
        variance_eps=args.variance_eps,
        sampler_mode=args.sampler_mode,
        target_rate_bins=target_rate_bins,
        strata_policy=args.strata_policy,
        loss_exclude_contrasts=tuple(float(x) for x in args.loss_exclude_contrasts.split(",") if x.strip()),
        loss_exclude_conditions=parse_loss_exclude_condition_file(args.loss_exclude_condition_file),
        loss_exclude_eval=args.loss_exclude_eval,
        adv_enable=args.adv_enable,
        adv_epsilon=args.adv_epsilon,
        adv_lambda=args.adv_lambda,
        adv_smooth_lambda=args.adv_smooth_lambda,
        adv_huber_delta=args.adv_huber_delta,
        adv_clip_value=args.adv_clip_value,
        guide_uncertainty_mode=args.guide_uncertainty_mode,
        guide83_w_log=args.guide83_w_log,
        guide83_w_raw=args.guide83_w_raw,
        guide83_c=args.guide83_c,
        guide84_time_window=args.guide84_time_window,
        guide84_eps=args.guide84_eps,
        init_checkpoint=args.init_checkpoint,
        train_stage_modes=parse_stage_modes(args.train_stage_modes),
        weight_anchor_checkpoint=args.weight_anchor_checkpoint,
        weight_anchor_lambda=args.weight_anchor_lambda,
        weight_anchor_include_new_inputs=args.weight_anchor_include_new_inputs,
        tfsf_smooth_lambda=args.tfsf_smooth_lambda,
        tfsf_smooth_delta=args.tfsf_smooth_delta,
        tfsf_smooth_max_points=args.tfsf_smooth_max_points,
        tf_unimodal_lambda=args.tf_unimodal_lambda,
        sf_unimodal_lambda=args.sf_unimodal_lambda,
        tfsf_unimodal_max_points=args.tfsf_unimodal_max_points,
        tfsf_unimodal_prominence_fraction=args.tfsf_unimodal_prominence_fraction,
        distill_anchor_lambda=args.distill_anchor_lambda,
        distill_anchor_mode=args.distill_anchor_mode,
        distill_default_tf=args.distill_default_tf,
        distill_default_sf=args.distill_default_sf,
        distill_max_tf_exclusive=args.distill_max_tf_exclusive,
        distill_max_points=args.distill_max_points,
    )
    cfg.hidden_dims = hidden_dims
    cfg.activation_seq = activation_seq
    cfg.model_family = args.model_family
    cfg.shape_hidden_dims = shape_hidden_dims
    cfg.shape_activation_seq = shape_activation_seq
    cfg.contrast_hidden_dims = contrast_hidden_dims
    cfg.contrast_activation_seq = contrast_activation_seq
    cfg.loss_mode = args.loss_mode
    cfg.corr_lambda = args.corr_lambda
    cfg.corr_w_under_mid0 = args.corr_w_under_mid0
    cfg.corr_w_over_edge225 = args.corr_w_over_edge225
    cfg.mid_under_lambda = args.mid_under_lambda
    cfg.edge_over_lambda = args.edge_over_lambda
    cfg.mid_under_gamma_max = args.mid_under_gamma_max
    cfg.edge_over_gamma_radius = args.edge_over_gamma_radius
    cfg.mid_under_contrasts = tuple(float(x) for x in args.mid_under_contrasts.split(",") if x.strip())
    cfg.edge_over_contrasts = tuple(float(x) for x in args.edge_over_contrasts.split(",") if x.strip())
    cfg.preset_name = args.preset_name
    if not (0.0 < cfg.variance_min_weight <= 1.0):
        raise ValueError("--variance_min_weight must be in (0, 1]")
    if not (0.0 <= cfg.variance_ref_quantile <= 1.0):
        raise ValueError("--variance_ref_quantile must be in [0, 1]")
    if cfg.variance_eps <= 0.0:
        raise ValueError("--variance_eps must be positive")
    if float(getattr(cfg, "weight_anchor_lambda", 0.0)) < 0.0:
        raise ValueError("--weight_anchor_lambda must be nonnegative")
    if float(getattr(cfg, "raw_mse_lambda", 0.0)) < 0.0:
        raise ValueError("--raw_mse_lambda must be nonnegative")
    if float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) < 0.0:
        raise ValueError("--tfsf_smooth_lambda must be nonnegative")
    if float(getattr(cfg, "tf_unimodal_lambda", 0.0)) < 0.0:
        raise ValueError("--tf_unimodal_lambda must be nonnegative")
    if float(getattr(cfg, "sf_unimodal_lambda", 0.0)) < 0.0:
        raise ValueError("--sf_unimodal_lambda must be nonnegative")
    prominence = float(getattr(cfg, "tfsf_unimodal_prominence_fraction", 0.02))
    if prominence < 0.0:
        raise ValueError("--tfsf_unimodal_prominence_fraction must be nonnegative")
    if not (0.0 < float(getattr(cfg, "global_curvature_delta_fraction", 0.04)) < 0.5):
        raise ValueError("--global_curvature_delta_fraction must be in (0, 0.5)")
    if int(getattr(cfg, "global_curvature_direction_count", 4)) < 0:
        raise ValueError("--global_curvature_direction_count must be nonnegative")
    if str(getattr(cfg, "global_curvature_balance_mode", "none")) == "baseline_relative":
        balance_path = Path(str(getattr(cfg, "global_curvature_balance_path", "")))
        if not balance_path.is_file():
            raise FileNotFoundError(
                "--global_curvature_balance_path must be an existing audit JSON "
                "when --global_curvature_balance_mode baseline_relative"
            )
    reduce_mode = str(getattr(cfg, "global_curvature_reduce_mode", "mean")).lower()
    if reduce_mode == "smoothmax":
        if str(getattr(cfg, "global_curvature_balance_mode", "none")) != "baseline_relative":
            raise ValueError("--global_curvature_reduce_mode smoothmax requires baseline_relative balancing")
        if float(getattr(cfg, "global_curvature_smoothmax_temperature", 5.0)) <= 0.0:
            raise ValueError("--global_curvature_smoothmax_temperature must be positive")
    if reduce_mode == "focus_axis":
        if str(getattr(cfg, "global_curvature_balance_mode", "none")) != "baseline_relative":
            raise ValueError("--global_curvature_reduce_mode focus_axis requires baseline_relative balancing")
        feature = str(getattr(cfg, "global_curvature_focus_feature", ""))
        if feature not in feature_names_for_mode(cfg.feature_mode):
            raise ValueError(f"Unknown --global_curvature_focus_feature: {feature!r}")
        if float(getattr(cfg, "global_curvature_focus_mean_weight", 0.1)) < 0.0:
            raise ValueError("--global_curvature_focus_mean_weight must be nonnegative")
    if float(getattr(cfg, "curvature_grad_ratio", 0.0)) < 0.0:
        raise ValueError("--curvature_grad_ratio must be nonnegative")
    if float(getattr(cfg, "ckpt_global_curvature_beta", 0.0)) < 0.0:
        raise ValueError("--ckpt_global_curvature_beta must be nonnegative")
    if int(getattr(cfg, "ckpt_global_curvature_points", 512)) <= 0:
        raise ValueError("--ckpt_global_curvature_points must be positive")
    if float(getattr(cfg, "tfsf_smooth_delta", 0.0)) <= 0.0:
        raise ValueError("--tfsf_smooth_delta must be positive")
    if float(getattr(cfg, "distill_anchor_lambda", 0.0)) < 0.0:
        raise ValueError("--distill_anchor_lambda must be nonnegative")
    if (
        str(getattr(cfg, "distill_anchor_mode", "default_tfsf")) == "same_input_below_tf"
        and float(getattr(cfg, "distill_max_tf_exclusive", 32.0)) <= 0.0
    ):
        raise ValueError("--distill_max_tf_exclusive must be positive")
    return cfg


def load_split_metadata(split_dir: str | Path) -> tuple[dict, dict]:
    split_dir = Path(split_dir)
    manifest = load_json(split_dir / "split_manifest.json")
    feature_norm = load_json(split_dir / "feature_norm.json")
    if float(feature_norm.get("lgnSF", {}).get("domain_max", 10.0)) > 10.0:
        raise ValueError("SF>10 is no longer supported; use the noSF15 split.")
    return manifest, feature_norm


def estimate_variance_references(
    cfg: TrainConfig,
    manifest: dict,
    *,
    batch_size: int = 262144,
    max_samples: int = 1_000_000,
) -> dict[str, float | bool | int]:
    if not variance_weight_enabled(cfg):
        return {
            "has_fields": False,
            "ref_raw": 1.0,
            "ref_log": 1.0,
            "ref_rel": 1.0,
            "positive_sample_count": 0,
        }
    shard_path = resolve_shard_path(cfg, manifest, "train")
    raw_samples: list[np.ndarray] = []
    log_samples: list[np.ndarray] = []
    rel_samples: list[np.ndarray] = []
    rng = np.random.default_rng(int(cfg.seed) + 9091)
    with H5MatReader(shard_path) as reader:
        if not (reader.has_key("FR_ResponseVar") and reader.has_key("FR_ResponseN")):
            return {
                "has_fields": False,
                "ref_raw": 1.0,
                "ref_log": 1.0,
                "ref_rel": 1.0,
                "positive_sample_count": 0,
            }
        total_rows = reader.length("FR_Response")
        n_batches = max(1, int(math.ceil(total_rows / max(batch_size, 1))))
        batch_cap = max(1024, int(math.ceil(max_samples / n_batches)))
        for batch_start in range(0, total_rows, batch_size):
            rows = slice(batch_start, min(batch_start + batch_size, total_rows))
            y = np.clip(reader.read_1d("FR_Response", rows, np.float64), a_min=0.0, a_max=None)
            var = reader.read_1d("FR_ResponseVar", rows, np.float64)
            repeat_n = reader.read_1d("FR_ResponseN", rows, np.float64)
            valid = np.isfinite(y) & np.isfinite(var) & np.isfinite(repeat_n) & (var >= 0.0) & (repeat_n > 0.0)
            if not np.any(valid):
                continue
            sem = var[valid] / np.maximum(repeat_n[valid], 1.0)
            yv = y[valid]
            raw = sem
            log = sem / np.square(1.0 + yv)
            rel = sem / np.square(yv + float(cfg.rel_alpha))
            positive = np.isfinite(raw) & (raw > 0.0) & np.isfinite(log) & (log > 0.0) & np.isfinite(rel) & (rel > 0.0)
            if not np.any(positive):
                continue
            idx = np.flatnonzero(positive)
            if idx.size > batch_cap:
                idx = rng.choice(idx, size=batch_cap, replace=False)
            raw_samples.append(raw[idx].astype(np.float64, copy=False))
            log_samples.append(log[idx].astype(np.float64, copy=False))
            rel_samples.append(rel[idx].astype(np.float64, copy=False))

    if not raw_samples:
        return {
            "has_fields": True,
            "ref_raw": 1.0,
            "ref_log": 1.0,
            "ref_rel": 1.0,
            "positive_sample_count": 0,
        }
    raw_all = np.concatenate(raw_samples)
    log_all = np.concatenate(log_samples)
    rel_all = np.concatenate(rel_samples)
    q = float(cfg.variance_ref_quantile)

    def ref_value(values: np.ndarray) -> float:
        ref = float(np.quantile(values, q))
        if not np.isfinite(ref) or ref <= 0.0:
            return 1.0
        return ref

    return {
        "has_fields": True,
        "ref_raw": ref_value(raw_all),
        "ref_log": ref_value(log_all),
        "ref_rel": ref_value(rel_all),
        "positive_sample_count": int(raw_all.size),
    }


def _round_condition_value(value: float) -> float:
    return round(float(value), 4)


def _condition_key(gamma: float, contrast: float, lgnctgr: int) -> tuple[float, float, int]:
    return (_round_condition_value(gamma), _round_condition_value(contrast), int(lgnctgr))


def _pair_key(gamma: float, contrast: float) -> tuple[float, float]:
    return (_round_condition_value(gamma), _round_condition_value(contrast))


BAD_HC_SELECTION_PAIR_SET = frozenset(_pair_key(gamma, contrast) for gamma, contrast in BAD_HC_SELECTION_PAIRS)


def _target_mask_bounds(target: str) -> tuple[float, float]:
    return TARGET_TO_MASK_BOUNDS[str(target)]


VARIANCE_META_FIELDS = ("FR_ResponseVar", "FR_ResponseN")


def variance_weight_enabled(cfg: TrainConfig) -> bool:
    return str(getattr(cfg, "variance_weight_mode", "none")).lower() != "none"


def loss_exclusion_enabled(cfg: TrainConfig) -> bool:
    return (
        len(tuple(getattr(cfg, "loss_exclude_contrasts", ()) or ())) > 0
        or len(tuple(getattr(cfg, "loss_exclude_conditions", ()) or ())) > 0
    )


class OnlineTrilinearTeacher3D:
    def __init__(
        self,
        mat_path: str | Path,
        target: str,
        *,
        interp_eps: float,
        badslab_only: bool,
        cache_slabs: int,
        bad_pair_weights: dict[tuple[float, float], float] | None = None,
    ):
        self.mat_path = str(mat_path)
        self.target = str(target)
        self.interp_eps = float(interp_eps)
        self.badslab_only = bool(badslab_only)
        self.cache_slabs = max(int(cache_slabs), 1)
        self.target_low, self.target_high = _target_mask_bounds(target)
        pair_source = bad_pair_weights if bad_pair_weights is not None else BAD_SLAB_WEIGHT_BY_PAIR
        self.bad_pair_weights = {
            _pair_key(gamma, contrast): float(weight)
            for (gamma, contrast), weight in pair_source.items()
            if float(weight) > 0.0
        }
        self._slab_store = self._load_bad_slab_store()
        self._lru: OrderedDict[tuple[float, float, int], dict] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.retry_count = 0
        self.replace_success = 0
        self.replace_fallback = 0

    def _load_bad_slab_store(self) -> dict[tuple[float, float, int], dict]:
        selected: dict[tuple[float, float, int], list[list[np.ndarray]]] = {}
        chunk_size = 1_000_000
        with H5MatReader(self.mat_path) as reader:
            total_rows = reader.length("FR_Response")
            for start in range(0, total_rows, chunk_size):
                rows = slice(start, min(start + chunk_size, total_rows))
                gamma = reader.read_1d("Gamma", rows, np.float32)
                contrast = reader.read_1d("Contrast", rows, np.float32)
                pair_mask = np.zeros(gamma.shape[0], dtype=bool)
                for gamma_v, contrast_v in self.bad_pair_weights:
                    pair_mask |= np.isclose(gamma, gamma_v) & np.isclose(contrast, contrast_v)
                if not np.any(pair_mask):
                    continue
                lgn = reader.read_1d("LGNctgr", rows, np.int16)[pair_mask]
                l6 = reader.read_1d("L6", rows, np.float32)[pair_mask]
                l4e = reader.read_1d("L4E", rows, np.float32)[pair_mask]
                l4i = reader.read_1d("L4I", rows, np.float32)[pair_mask]
                y = reader.read_1d("FR_Response", rows, np.float32)[pair_mask]
                gamma_sel = gamma[pair_mask]
                contrast_sel = contrast[pair_mask]
                for g, c, lgn_v, l6_v, l4e_v, l4i_v, y_v in zip(gamma_sel, contrast_sel, lgn, l6, l4e, l4i, y):
                    key = _condition_key(g, c, int(lgn_v))
                    bucket = selected.setdefault(key, [[], [], [], []])
                    bucket[0].append(np.asarray([l6_v], dtype=np.float32))
                    bucket[1].append(np.asarray([l4e_v], dtype=np.float32))
                    bucket[2].append(np.asarray([l4i_v], dtype=np.float32))
                    bucket[3].append(np.asarray([y_v], dtype=np.float32))

        out: dict[tuple[float, float, int], dict] = {}
        for key, arrays in selected.items():
            l6 = np.concatenate(arrays[0]).reshape(-1)
            l4e = np.concatenate(arrays[1]).reshape(-1)
            l4i = np.concatenate(arrays[2]).reshape(-1)
            y = np.concatenate(arrays[3]).reshape(-1)
            l6_axis = np.unique(l6.astype(np.float32))
            l4e_axis = np.unique(l4e.astype(np.float32))
            l4i_axis = np.unique(l4i.astype(np.float32))
            values = np.full((l6_axis.size, l4e_axis.size, l4i_axis.size), np.nan, dtype=np.float32)
            l6_idx = np.searchsorted(l6_axis, l6)
            l4e_idx = np.searchsorted(l4e_axis, l4e)
            l4i_idx = np.searchsorted(l4i_axis, l4i)
            values[l6_idx, l4e_idx, l4i_idx] = y
            out[key] = {
                "l6_axis": l6_axis,
                "l4e_axis": l4e_axis,
                "l4i_axis": l4i_axis,
                "values": values,
            }
        return out

    def row_weight(self, gamma: float, contrast: float) -> float:
        key = _pair_key(gamma, contrast)
        weight = self.bad_pair_weights.get(key, 0.0 if self.badslab_only else 1.0)
        return float(weight)

    def _get_slab(self, gamma: float, contrast: float, lgnctgr: int) -> dict | None:
        key = _condition_key(gamma, contrast, lgnctgr)
        if key in self._lru:
            self.cache_hits += 1
            slab = self._lru.pop(key)
            self._lru[key] = slab
            return slab
        slab = self._slab_store.get(key)
        if slab is None:
            self.cache_misses += 1
            return None
        self.cache_misses += 1
        self._lru[key] = slab
        while len(self._lru) > self.cache_slabs:
            self._lru.popitem(last=False)
        return slab

    @staticmethod
    def _candidate_lows(axis: np.ndarray, raw_value: float) -> list[int]:
        idx = int(np.searchsorted(axis, np.float32(raw_value)))
        idx = max(0, min(idx, axis.size - 1))
        candidates: list[int] = []
        if axis.size < 2:
            return candidates
        if idx > 0:
            candidates.append(idx - 1)
        if idx < axis.size - 1:
            candidates.append(idx)
        ordered: list[int] = []
        for item in candidates:
            if item not in ordered:
                ordered.append(item)
        return ordered

    @staticmethod
    def _interp_from_cube(
        values: np.ndarray,
        l6_lo: int,
        l4e_lo: int,
        l4i_lo: int,
        u: float,
        v: float,
        w: float,
    ) -> float:
        f000 = values[l6_lo, l4e_lo, l4i_lo]
        f100 = values[l6_lo, l4e_lo + 1, l4i_lo]
        f010 = values[l6_lo, l4e_lo, l4i_lo + 1]
        f110 = values[l6_lo, l4e_lo + 1, l4i_lo + 1]
        f001 = values[l6_lo + 1, l4e_lo, l4i_lo]
        f101 = values[l6_lo + 1, l4e_lo + 1, l4i_lo]
        f011 = values[l6_lo + 1, l4e_lo, l4i_lo + 1]
        f111 = values[l6_lo + 1, l4e_lo + 1, l4i_lo + 1]
        one_u = 1.0 - u
        one_v = 1.0 - v
        one_w = 1.0 - w
        return float(
            one_u * one_v * one_w * f000
            + u * one_v * one_w * f100
            + one_u * v * one_w * f010
            + u * v * one_w * f110
            + one_u * one_v * w * f001
            + u * one_v * w * f101
            + one_u * v * w * f011
            + u * v * w * f111
        )

    def sample_replacement(
        self,
        *,
        gamma: float,
        contrast: float,
        lgnctgr: int,
        l6_raw: float,
        l4e_raw: float,
        l4i_raw: float,
        rng: np.random.Generator,
        max_retries: int = 8,
    ) -> tuple[float, float, float, float] | None:
        slab = self._get_slab(gamma, contrast, lgnctgr)
        if slab is None:
            self.replace_fallback += 1
            return None
        l6_axis = slab["l6_axis"]
        l4e_axis = slab["l4e_axis"]
        l4i_axis = slab["l4i_axis"]
        values = slab["values"]
        cand_l6 = self._candidate_lows(l6_axis, l6_raw)
        cand_l4e = self._candidate_lows(l4e_axis, l4e_raw)
        cand_l4i = self._candidate_lows(l4i_axis, l4i_raw)
        if not cand_l6 or not cand_l4e or not cand_l4i:
            self.replace_fallback += 1
            return None
        for _ in range(max_retries):
            self.retry_count += 1
            l6_lo = int(rng.choice(cand_l6))
            l4e_lo = int(rng.choice(cand_l4e))
            l4i_lo = int(rng.choice(cand_l4i))
            cube = values[l6_lo : l6_lo + 2, l4e_lo : l4e_lo + 2, l4i_lo : l4i_lo + 2]
            if cube.shape != (2, 2, 2) or not np.all(np.isfinite(cube)):
                continue
            u = float(rng.uniform(self.interp_eps, 1.0 - self.interp_eps))
            v = float(rng.uniform(self.interp_eps, 1.0 - self.interp_eps))
            w = float(rng.uniform(self.interp_eps, 1.0 - self.interp_eps))
            l6_q = float(l6_axis[l6_lo] + w * (l6_axis[l6_lo + 1] - l6_axis[l6_lo]))
            l4e_q = float(l4e_axis[l4e_lo] + u * (l4e_axis[l4e_lo + 1] - l4e_axis[l4e_lo]))
            l4i_q = float(l4i_axis[l4i_lo] + v * (l4i_axis[l4i_lo + 1] - l4i_axis[l4i_lo]))
            teacher_y = self._interp_from_cube(values, l6_lo, l4e_lo, l4i_lo, u, v, w)
            if not (self.target_low <= teacher_y <= self.target_high):
                continue
            self.replace_success += 1
            return l6_q, l4e_q, l4i_q, teacher_y
        self.replace_fallback += 1
        return None


def resolve_shard_path(cfg: TrainConfig, manifest: dict, split_name: str) -> str:
    shard_paths = manifest.get("shard_paths", {})
    target_paths = shard_paths.get(cfg.target, {})
    shard_path = target_paths.get(split_name)
    if not shard_path:
        raise RuntimeError(f"Missing shard path for target={cfg.target} split={split_name}")
    return str(shard_path)


def make_sample_eval_loader(
    cfg: TrainConfig,
    split_name: str,
    manifest: dict,
    feature_norm: dict,
    max_batches: int,
) -> SplitShardLoader:
    meta_fields = ("Gamma", "Contrast", "LGNctgr")
    if tuple(getattr(cfg, "population_focus_conditions", ()) or ()):
        meta_fields = tuple(dict.fromkeys((*meta_fields, "lgnTF", "lgnSF")))
    if cfg.mono_pair_lambda > 0:
        meta_fields = ("Gamma", "Contrast", "LGNctgr", "L6", "L4E", "L4I")
    if variance_weight_enabled(cfg):
        meta_fields = tuple(dict.fromkeys((*meta_fields, *VARIANCE_META_FIELDS)))
    return SplitShardLoader(
        shard_path=resolve_shard_path(cfg, manifest, split_name),
        feature_norm=feature_norm,
        batch_size=cfg.eval_batch_size,
        shuffle=True,
        seed=cfg.seed + 7000,
        max_batches=max_batches,
        meta_fields=meta_fields,
        feature_mode=cfg.feature_mode,
        sampler_mode="uniform",
        target_rate_bins=getattr(cfg, "target_rate_bins", (0.0, 0.5, 2.0, 5.0, 15.0, float("inf"))),
        strata_policy=getattr(cfg, "strata_policy", "equal"),
    )


def make_loader(cfg: TrainConfig, split_name: str, manifest: dict, feature_norm: dict, shuffle: bool) -> SplitShardLoader:
    meta_fields = ()
    if str(getattr(cfg, "population_grad_mode", "none")).lower() in ("tfsf4", "tfsf4_focus"):
        meta_fields = ("lgnTF", "lgnSF")
    if str(getattr(cfg, "population_grad_mode", "none")).lower() == "tfsf4_focus":
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Contrast", "Gamma", "LGNctgr")))
    if getattr(cfg, "loss_mode", "base") in ("targeted_mid_edge_asym", "legacy_gamma_contrast_asym"):
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Gamma", "Contrast")))
    if cfg.mono_pair_lambda > 0:
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Gamma", "Contrast", "L6", "L4E", "L4I")))
    if cfg.grad_lambda > 0:
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Gamma", "Contrast", "LGNctgr", "L6", "L4E", "L4I")))
    if shuffle and getattr(cfg, "online_interp_enable", False):
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Gamma", "Contrast", "LGNctgr", "L6", "L4E", "L4I")))
    if shuffle and loss_exclusion_enabled(cfg):
        meta_fields = tuple(dict.fromkeys((*meta_fields, "Contrast")))
    if variance_weight_enabled(cfg):
        meta_fields = tuple(dict.fromkeys((*meta_fields, *VARIANCE_META_FIELDS)))
    return SplitShardLoader(
        shard_path=resolve_shard_path(cfg, manifest, split_name),
        feature_norm=feature_norm,
        batch_size=cfg.batch_size if shuffle else cfg.eval_batch_size,
        shuffle=shuffle,
        seed=cfg.seed,
        max_batches=cfg.max_train_batches if shuffle else cfg.max_eval_batches,
        meta_fields=meta_fields,
        feature_mode=cfg.feature_mode,
        sampler_mode=getattr(cfg, "sampler_mode", "uniform") if shuffle else "uniform",
        target_rate_bins=getattr(cfg, "target_rate_bins", (0.0, 0.5, 2.0, 5.0, 15.0, float("inf"))),
        strata_policy=getattr(cfg, "strata_policy", "equal"),
    )


class TrilinearTeacher3D:
    def __init__(self, mat_path: str | Path):
        with H5MatReader(mat_path) as reader:
            gamma = reader.read_1d("Gamma", None, np.float32)
            contrast = reader.read_1d("Contrast", None, np.float32)
            lgn = reader.read_1d("LGNctgr", None, np.int16)
            l6 = reader.read_1d("L6", None, np.float32)
            l4e = reader.read_1d("L4E", None, np.float32)
            l4i = reader.read_1d("L4I", None, np.float32)
            y = reader.read_1d("FR_Response", None, np.float32)

        self.gamma_axis = np.unique(gamma.astype(np.float32))
        self.contrast_axis = np.unique(contrast.astype(np.float32))
        self.lgn_axis = np.unique(lgn.astype(np.int16))
        self.l6_axis = np.unique(l6.astype(np.float32))
        self.l4e_axis = np.unique(l4e.astype(np.float32))
        self.l4i_axis = np.unique(l4i.astype(np.float32))

        shape = (
            self.gamma_axis.size,
            self.contrast_axis.size,
            self.lgn_axis.size,
            self.l6_axis.size,
            self.l4e_axis.size,
            self.l4i_axis.size,
        )
        self.values = np.empty(shape, dtype=np.float32)

        g_idx = np.searchsorted(self.gamma_axis, gamma)
        c_idx = np.searchsorted(self.contrast_axis, contrast)
        lgn_idx = np.searchsorted(self.lgn_axis, lgn)
        l6_idx = np.searchsorted(self.l6_axis, l6)
        l4e_idx = np.searchsorted(self.l4e_axis, l4e)
        l4i_idx = np.searchsorted(self.l4i_axis, l4i)
        self.values[g_idx, c_idx, lgn_idx, l6_idx, l4e_idx, l4i_idx] = y

    def _exact_index(self, axis: np.ndarray, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(axis, values)
        idx = np.clip(idx, 0, axis.size - 1)
        return idx.astype(np.int64, copy=False)

    def sample_value_and_gradients(
        self,
        gamma: np.ndarray,
        contrast: np.ndarray,
        lgnctgr: np.ndarray,
        l6_raw: np.ndarray,
        l4e_raw: np.ndarray,
        l4i_raw: np.ndarray,
        *,
        eps: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gamma = np.asarray(gamma, dtype=np.float32).reshape(-1)
        contrast = np.asarray(contrast, dtype=np.float32).reshape(-1)
        lgnctgr = np.asarray(lgnctgr, dtype=np.int16).reshape(-1)
        l6_raw = np.asarray(l6_raw, dtype=np.float32).reshape(-1)
        l4e_raw = np.asarray(l4e_raw, dtype=np.float32).reshape(-1)
        l4i_raw = np.asarray(l4i_raw, dtype=np.float32).reshape(-1)

        g_idx = self._exact_index(self.gamma_axis, gamma)
        c_idx = self._exact_index(self.contrast_axis, contrast)
        lgn_idx = self._exact_index(self.lgn_axis, lgnctgr)
        l6_idx = self._exact_index(self.l6_axis, l6_raw)
        l4e_idx = self._exact_index(self.l4e_axis, l4e_raw)
        l4i_idx = self._exact_index(self.l4i_axis, l4i_raw)

        l6_lo = np.clip(l6_idx, 0, self.l6_axis.size - 2)
        l4e_lo = np.clip(l4e_idx, 0, self.l4e_axis.size - 2)
        l4i_lo = np.clip(l4i_idx, 0, self.l4i_axis.size - 2)
        l6_hi = l6_lo + 1
        l4e_hi = l4e_lo + 1
        l4i_hi = l4i_lo + 1

        u = rng.uniform(eps, 1.0 - eps, size=gamma.shape[0]).astype(np.float32)
        v = rng.uniform(eps, 1.0 - eps, size=gamma.shape[0]).astype(np.float32)
        w = rng.uniform(eps, 1.0 - eps, size=gamma.shape[0]).astype(np.float32)

        e0 = self.l4e_axis[l4e_lo]
        e1 = self.l4e_axis[l4e_hi]
        i0 = self.l4i_axis[l4i_lo]
        i1 = self.l4i_axis[l4i_hi]
        l0 = self.l6_axis[l6_lo]
        l1 = self.l6_axis[l6_hi]

        de = np.maximum(e1 - e0, 1e-12)
        di = np.maximum(i1 - i0, 1e-12)
        dl = np.maximum(l1 - l0, 1e-12)

        e_q = e0 + u * de
        i_q = i0 + v * di
        l_q = l0 + w * dl

        vals = self.values
        f000 = vals[g_idx, c_idx, lgn_idx, l6_lo, l4e_lo, l4i_lo]
        f100 = vals[g_idx, c_idx, lgn_idx, l6_lo, l4e_hi, l4i_lo]
        f010 = vals[g_idx, c_idx, lgn_idx, l6_lo, l4e_lo, l4i_hi]
        f110 = vals[g_idx, c_idx, lgn_idx, l6_lo, l4e_hi, l4i_hi]
        f001 = vals[g_idx, c_idx, lgn_idx, l6_hi, l4e_lo, l4i_lo]
        f101 = vals[g_idx, c_idx, lgn_idx, l6_hi, l4e_hi, l4i_lo]
        f011 = vals[g_idx, c_idx, lgn_idx, l6_hi, l4e_lo, l4i_hi]
        f111 = vals[g_idx, c_idx, lgn_idx, l6_hi, l4e_hi, l4i_hi]

        one_u = 1.0 - u
        one_v = 1.0 - v
        one_w = 1.0 - w

        teacher_y = (
            one_u * one_v * one_w * f000
            + u * one_v * one_w * f100
            + one_u * v * one_w * f010
            + u * v * one_w * f110
            + one_u * one_v * w * f001
            + u * one_v * w * f101
            + one_u * v * w * f011
            + u * v * w * f111
        )
        d_du = (
            one_v * one_w * (f100 - f000)
            + v * one_w * (f110 - f010)
            + one_v * w * (f101 - f001)
            + v * w * (f111 - f011)
        )
        d_dv = (
            one_u * one_w * (f010 - f000)
            + u * one_w * (f110 - f100)
            + one_u * w * (f011 - f001)
            + u * w * (f111 - f101)
        )
        d_dw = (
            one_u * one_v * (f001 - f000)
            + u * one_v * (f101 - f100)
            + one_u * v * (f011 - f010)
            + u * v * (f111 - f110)
        )
        grad_l4e = d_du / de
        grad_l4i = d_dv / di
        grad_l6 = d_dw / dl
        return e_q, i_q, l_q, teacher_y, grad_l4e, grad_l4i, grad_l6


def interpolation_grad_loss(
    model,
    xb: torch.Tensor,
    meta: dict,
    feature_norm: dict,
    teacher: TrilinearTeacher3D,
    *,
    grad_max_points: int,
    grad_interior_eps: float,
    device: torch.device,
) -> torch.Tensor:
    batch_n = xb.shape[0]
    if batch_n <= 0:
        return xb.new_zeros(())
    use_n = min(int(grad_max_points), int(batch_n))
    if use_n <= 0:
        return xb.new_zeros(())

    if use_n < batch_n:
        choose = np.random.choice(batch_n, size=use_n, replace=False)
    else:
        choose = np.arange(batch_n, dtype=np.int64)

    choose_t = torch.from_numpy(choose.astype(np.int64, copy=False)).to(device=device)
    xb_sub = xb[choose_t].detach().clone()

    gamma = np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1)[choose]
    contrast = np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1)[choose]
    lgn = np.asarray(meta["LGNctgr"], dtype=np.int16).reshape(-1)[choose]
    l6_raw = np.asarray(meta["L6"], dtype=np.float32).reshape(-1)[choose]
    l4e_raw = np.asarray(meta["L4E"], dtype=np.float32).reshape(-1)[choose]
    l4i_raw = np.asarray(meta["L4I"], dtype=np.float32).reshape(-1)[choose]

    rng = np.random.default_rng()
    e_q, i_q, l_q, _teacher_y, grad_e_t, grad_i_t, grad_l_t = teacher.sample_value_and_gradients(
        gamma,
        contrast,
        lgn,
        l6_raw,
        l4e_raw,
        l4i_raw,
        eps=grad_interior_eps,
        rng=rng,
    )

    xb_sub[:, 2] = torch.from_numpy((l_q - float(feature_norm["L6"]["mean"])) / float(feature_norm["L6"]["std"])).to(device=device)
    xb_sub[:, 3] = torch.from_numpy((e_q - float(feature_norm["L4E"]["mean"])) / float(feature_norm["L4E"]["std"])).to(device=device)
    xb_sub[:, 4] = torch.from_numpy((i_q - float(feature_norm["L4I"]["mean"])) / float(feature_norm["L4I"]["std"])).to(device=device)
    xb_sub.requires_grad_(True)

    pred_q = model(xb_sub)
    grads = torch.autograd.grad(pred_q.sum(), xb_sub, create_graph=True)[0]

    pred_grad_l6 = grads[:, 2] / float(feature_norm["L6"]["std"])
    pred_grad_l4e = grads[:, 3] / float(feature_norm["L4E"]["std"])
    pred_grad_l4i = grads[:, 4] / float(feature_norm["L4I"]["std"])

    teacher_grad_l6 = torch.from_numpy(grad_l_t).to(device=device, dtype=pred_grad_l6.dtype)
    teacher_grad_l4e = torch.from_numpy(grad_e_t).to(device=device, dtype=pred_grad_l4e.dtype)
    teacher_grad_l4i = torch.from_numpy(grad_i_t).to(device=device, dtype=pred_grad_l4i.dtype)

    return (
        torch.mean(torch.square(pred_grad_l4e - teacher_grad_l4e))
        + torch.mean(torch.square(pred_grad_l4i - teacher_grad_l4i))
        + torch.mean(torch.square(pred_grad_l6 - teacher_grad_l6))
    )


def mono_pair_loss_from_meta(pred_raw: torch.Tensor, meta: dict, device: torch.device) -> torch.Tensor:
    pred_raw = clamp_raw_rate_tensor(pred_raw)
    if "Gamma" not in meta:
        return pred_raw.new_zeros(())
    gamma = np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1)
    if gamma.size < 2:
        return pred_raw.new_zeros(())
    group_cols = np.stack(
        [
            np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1),
            np.asarray(meta["L6"], dtype=np.float32).reshape(-1),
            np.asarray(meta["L4E"], dtype=np.float32).reshape(-1),
            np.asarray(meta["L4I"], dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    if group_cols.shape[0] < 2:
        return pred_raw.new_zeros(())
    key_view = np.ascontiguousarray(group_cols).view(
        np.dtype((np.void, group_cols.dtype.itemsize * group_cols.shape[1]))
    )
    _, inv = np.unique(key_view, return_inverse=True)
    left_idx = []
    right_idx = []
    for gid in np.unique(inv):
        idx = np.flatnonzero(inv == gid)
        if idx.size < 2:
            continue
        order = np.argsort(gamma[idx], kind="stable")
        idx = idx[order]
        g = gamma[idx]
        mask = g[:, None] < g[None, :]
        if not np.any(mask):
            continue
        left, right = np.nonzero(mask)
        left_idx.append(idx[left])
        right_idx.append(idx[right])
    if not left_idx:
        return pred_raw.new_zeros(())
    left_idx = np.concatenate(left_idx).astype(np.int64, copy=False)
    right_idx = np.concatenate(right_idx).astype(np.int64, copy=False)
    i_t = torch.from_numpy(left_idx).to(device=device)
    j_t = torch.from_numpy(right_idx).to(device=device)
    viol = pred_raw[j_t] - pred_raw[i_t]
    return torch.mean(torch.square(torch.clamp(viol, min=0.0)))


def weighted_mean_tensor(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    return torch.sum(weights * values) / torch.clamp(torch.sum(weights), min=1e-12)


def weighted_huber_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    delta: float,
) -> torch.Tensor:
    pred_safe = torch.clamp_min(pred_raw, 0.0)
    target_safe = torch.clamp_min(target_raw, 0.0)
    err = pred_safe - target_safe
    beta = max(float(delta), 1e-12)
    abs_err = torch.abs(err)
    loss = torch.where(abs_err < beta, 0.5 * torch.square(err) / beta, abs_err - 0.5 * beta)
    return weighted_mean_tensor(loss, weights)


def guide83_count_weighted_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    sample_weights: dict[str, torch.Tensor | None],
    cfg: TrainConfig,
) -> torch.Tensor:
    pred_safe = torch.clamp_min(pred_raw, 0.0)
    target_safe = torch.clamp_min(target_raw, 0.0)
    log_sq = torch.square(torch.log1p(pred_safe) - torch.log1p(target_safe))
    raw_sq_count_weighted = torch.square(pred_safe - target_safe) / (target_safe + float(getattr(cfg, "guide83_c", 1.0)))
    return (
        float(getattr(cfg, "guide83_w_log", 1.0)) * weighted_mean_tensor(log_sq, sample_weights.get("log"))
        + float(getattr(cfg, "guide83_w_raw", 1.0)) * weighted_mean_tensor(raw_sq_count_weighted, sample_weights.get("raw"))
    )


def guide84_poisson_rate_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    sample_weights: dict[str, torch.Tensor | None],
    cfg: TrainConfig,
) -> torch.Tensor:
    pred_rate = torch.clamp_min(pred_raw, float(getattr(cfg, "guide84_eps", 1e-6)))
    target_rate = torch.clamp_min(target_raw, 0.0)
    time_window = max(float(getattr(cfg, "guide84_time_window", 1.0)), float(getattr(cfg, "guide84_eps", 1e-6)))
    # Poisson deviance is the NLL up to target-only constants, using pseudo-count N=T*y.
    ratio = (target_rate + float(getattr(cfg, "guide84_eps", 1e-6))) / pred_rate
    deviance = 2.0 * time_window * (pred_rate - target_rate + target_rate * torch.log(ratio))
    return weighted_mean_tensor(deviance, sample_weights.get("raw"))


def guide55_mono_grad_loss(
    model,
    xb: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Soft derivative signs from guide section 5.5.

    Expected drive signs are dF/dL4E >= 0, dF/dL6 >= 0, dF/dL4I <= 0.
    Feature normalization preserves these signs because all std values are positive.
    """
    feature_names = feature_names_for_mode(cfg.feature_mode)
    required = ("L4E", "L6", "L4I")
    if any(name not in feature_names for name in required):
        raise RuntimeError(f"mono_grad penalty requires {required} in feature mode {cfg.feature_mode!r}")
    n_rows = int(xb.shape[0])
    max_points = int(getattr(cfg, "mono_grad_max_points", 2048))
    if max_points > 0 and n_rows > max_points:
        idx = torch.randperm(n_rows, device=xb.device)[:max_points]
        x_mono = xb.index_select(0, idx).detach().clone().requires_grad_(True)
    else:
        x_mono = xb.detach().clone().requires_grad_(True)
    pred_mono = model(x_mono)
    grad_x = torch.autograd.grad(
        pred_mono,
        x_mono,
        grad_outputs=torch.ones_like(pred_mono),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]
    l4e_grad = grad_x[:, feature_names.index("L4E")]
    l6_grad = grad_x[:, feature_names.index("L6")]
    l4i_grad = grad_x[:, feature_names.index("L4I")]
    penalty = (
        torch.square(torch.clamp(-l4e_grad, min=0.0))
        + torch.square(torch.clamp(-l6_grad, min=0.0))
        + torch.square(torch.clamp(l4i_grad, min=0.0))
    )
    return torch.mean(penalty)


def central_curvature_loss(
    model,
    xb: torch.Tensor,
    cfg: TrainConfig,
    feature_norm: dict,
) -> torch.Tensor:
    if str(getattr(cfg, "curvature_mode", "local_oc_contrast")).lower() == "global_all7":
        return global_all_input_curvature_loss(model, xb, cfg, feature_norm)

    feature_names = feature_names_for_mode(cfg.feature_mode)
    active = []
    if "OriCos2" in feature_names and float(getattr(cfg, "curvature_oricos2_delta", 0.0)) > 0.0:
        active.append(("OriCos2", float(getattr(cfg, "curvature_oricos2_delta", 0.05)), -1.0, 1.0))
    if "Contrast" in feature_names and float(getattr(cfg, "curvature_contrast_delta", 0.0)) > 0.0:
        active.append(("Contrast", float(getattr(cfg, "curvature_contrast_delta", 5.0)), 0.0, 100.0))
    if not active:
        return xb.new_zeros(())

    n_rows = int(xb.shape[0])
    max_points = int(getattr(cfg, "curvature_max_points", 2048))
    if max_points > 0 and n_rows > max_points:
        idx = torch.randperm(n_rows, device=xb.device)[:max_points]
        x0 = xb.index_select(0, idx).detach()
    else:
        x0 = xb.detach()
    pred0 = model(x0)
    penalties = []
    for name, delta_raw, raw_min, raw_max in active:
        col = feature_names.index(name)
        stat = feature_norm.get(name, {})
        if "mean" in stat and "std" in stat and float(stat.get("std", 0.0)) > 0.0:
            mean = float(stat["mean"])
            std = float(stat["std"])
            raw = x0[:, col] * std + mean
            plus_col = (torch.clamp(raw + delta_raw, min=raw_min, max=raw_max) - mean) / std
            minus_col = (torch.clamp(raw - delta_raw, min=raw_min, max=raw_max) - mean) / std
        else:
            plus_col = torch.clamp(x0[:, col] + delta_raw, min=raw_min, max=raw_max)
            minus_col = torch.clamp(x0[:, col] - delta_raw, min=raw_min, max=raw_max)
        x_plus = x0.clone()
        x_minus = x0.clone()
        x_plus[:, col] = plus_col
        x_minus[:, col] = minus_col
        curv = model(x_minus) - 2.0 * pred0 + model(x_plus)
        penalties.append(torch.mean(torch.square(curv)))
    return torch.stack(penalties).mean()


def _normalized_global_domain(feature_norm: dict, feature_names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    raw_defaults = {
        "OriCos2": (-1.0, 1.0),
        "Contrast": (0.0, 95.0),
        "L6": (1.0, 40.0),
        "L4E": (0.0, 48000.0),
        "L4I": (0.0, 35606.1796875),
        "lgnTF": (0.5, 32.0),
        "lgnSF": (0.5, 10.0),
    }
    lows: list[float] = []
    highs: list[float] = []
    for name in feature_names:
        stat = feature_norm.get(name, {}) if isinstance(feature_norm.get(name, {}), dict) else {}
        raw_lo, raw_hi = raw_defaults[name]
        raw_lo = float(stat.get("domain_min", raw_lo))
        raw_hi = float(stat.get("domain_max", raw_hi))
        transform = str(stat.get("transform", "identity")).lower()
        if transform == "log":
            if raw_lo <= 0.0:
                raise ValueError(f"Global curvature domain for {name} must be positive before log transform")
            raw_lo, raw_hi = math.log(raw_lo), math.log(raw_hi)
        elif transform != "identity":
            raise ValueError(f"Unsupported normalization transform {transform!r} for {name}")
        if "mean" in stat and "std" in stat:
            std = float(stat["std"])
            if std <= 0.0:
                raise ValueError(f"Feature {name} has nonpositive normalization std")
            raw_lo = (raw_lo - float(stat["mean"])) / std
            raw_hi = (raw_hi - float(stat["mean"])) / std
        if not raw_hi > raw_lo:
            raise ValueError(f"Invalid global curvature domain for {name}: {raw_lo}, {raw_hi}")
        lows.append(raw_lo)
        highs.append(raw_hi)
    return torch.tensor(lows, dtype=torch.float32), torch.tensor(highs, dtype=torch.float32)


_GLOBAL_CURVATURE_BALANCE_CACHE: dict[tuple[str, str, tuple[str, ...]], tuple[list[float], float]] = {}


def _global_curvature_balance_scales(
    cfg: TrainConfig,
    feature_names: tuple[str, ...],
) -> tuple[list[float], float] | None:
    mode = str(getattr(cfg, "global_curvature_balance_mode", "none")).lower()
    if mode == "none":
        return None
    if mode != "baseline_relative":
        raise ValueError(f"Unsupported global curvature balance mode: {mode}")
    path = str(Path(str(cfg.global_curvature_balance_path)).resolve())
    key = (path, str(cfg.target), tuple(feature_names))
    if key not in _GLOBAL_CURVATURE_BALANCE_CACHE:
        report = load_json(path)
        target_report = report["targets"][str(cfg.target)]
        axis_report = target_report["axis_second_derivative"]
        axis_scales = [float(axis_report[name]["mean_square"]) for name in feature_names]
        mixed_scale = float(
            target_report["mixed_direction_second_derivative"]["mean_square"]
        )
        scales = [*axis_scales, mixed_scale]
        if any((not math.isfinite(value)) or value <= 0.0 for value in scales):
            raise ValueError(
                f"Invalid baseline curvature scales for target {cfg.target}: {scales}"
            )
        _GLOBAL_CURVATURE_BALANCE_CACHE[key] = (axis_scales, mixed_scale)
    return _GLOBAL_CURVATURE_BALANCE_CACHE[key]


def global_all_input_curvature_loss(
    model,
    xb: torch.Tensor,
    cfg: TrainConfig,
    feature_norm: dict,
    generator: torch.Generator | None = None,
    max_points_override: int | None = None,
) -> torch.Tensor:
    feature_names = feature_names_for_mode(cfg.feature_mode)
    if tuple(feature_names) != ("OriCos2", "Contrast", "L6", "L4E", "L4I", "lgnTF", "lgnSF"):
        raise RuntimeError("global_all7 curvature requires --feature_mode tf_sf7d")

    n = int(xb.shape[0])
    max_points = (
        int(max_points_override)
        if max_points_override is not None
        else int(getattr(cfg, "curvature_max_points", 2048))
    )
    if max_points > 0:
        n = min(n, max_points)
    if n <= 0:
        return xb.new_zeros(())

    # Second differences are vulnerable to cancellation under AMP, so this
    # regularizer always evaluates the model in float32.
    with torch.autocast(device_type=xb.device.type, enabled=False):
        lo_cpu, hi_cpu = _normalized_global_domain(feature_norm, feature_names)
        lo = lo_cpu.to(device=xb.device, dtype=torch.float32)
        hi = hi_cpu.to(device=xb.device, dtype=torch.float32)
        width = hi - lo
        delta = float(getattr(cfg, "global_curvature_delta_fraction", 0.04))

        # Uniform anchors make the penalty global rather than concentrated where rows are dense.
        unit = delta + (1.0 - 2.0 * delta) * torch.rand(
            (n, len(feature_names)), device=xb.device, dtype=torch.float32,
            generator=generator,
        )
        x0 = lo + unit * width
        pred0 = model(x0)
        output_scale = float(getattr(cfg, "global_curvature_output_scale", 0.0))
        if output_scale <= 0.0:
            output_scale = float(TARGET_TO_MASK_BOUNDS[str(cfg.target)][1])
        output_scale = max(output_scale, 1e-8)
        balance_scales = _global_curvature_balance_scales(cfg, feature_names)

        penalties: list[torch.Tensor] = []
        for col in range(len(feature_names)):
            step = delta * width[col]
            x_minus = x0.clone()
            x_plus = x0.clone()
            x_minus[:, col] -= step
            x_plus[:, col] += step
            second_unit = (model(x_minus) - 2.0 * pred0 + model(x_plus)) / (delta * delta)
            penalty = torch.mean(torch.square(second_unit / output_scale))
            if balance_scales is not None:
                penalty = penalty / balance_scales[0][col]
            penalties.append(penalty)

        direction_count = int(getattr(cfg, "global_curvature_direction_count", 4))
        direction_scale = delta / math.sqrt(float(len(feature_names)))
        for _ in range(direction_count):
            signs = torch.where(
                torch.rand(
                    (len(feature_names),), device=xb.device, generator=generator
                ) < 0.5,
                -torch.ones((len(feature_names),), device=xb.device, dtype=torch.float32),
                torch.ones((len(feature_names),), device=xb.device, dtype=torch.float32),
            )
            step = direction_scale * width * signs
            second_direction = (model(x0 - step) - 2.0 * pred0 + model(x0 + step)) / (delta * delta)
            penalty = torch.mean(torch.square(second_direction / output_scale))
            if balance_scales is not None:
                penalty = penalty / balance_scales[1]
            penalties.append(penalty)
        stacked = torch.stack(penalties)
        reduce_mode = str(getattr(cfg, "global_curvature_reduce_mode", "mean")).lower()
        if reduce_mode == "mean":
            return stacked.mean()
        if reduce_mode == "smoothmax":
            temperature = float(getattr(cfg, "global_curvature_smoothmax_temperature", 5.0))
            return (
                torch.logsumexp(temperature * stacked, dim=0)
                - math.log(float(stacked.numel()))
            ) / temperature
        if reduce_mode == "focus_axis":
            focus_feature = str(getattr(cfg, "global_curvature_focus_feature", ""))
            focus_index = feature_names.index(focus_feature)
            mean_weight = float(getattr(cfg, "global_curvature_focus_mean_weight", 0.1))
            return stacked[focus_index] + mean_weight * stacked.mean()
        raise ValueError(f"Unsupported global curvature reduce mode: {reduce_mode}")


def deterministic_global_curvature_penalty(
    model,
    cfg: TrainConfig,
    feature_norm: dict,
    device: torch.device,
) -> float:
    if str(getattr(cfg, "curvature_mode", "local_oc_contrast")).lower() != "global_all7":
        return 0.0
    points = int(getattr(cfg, "ckpt_global_curvature_points", 512))
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 20260805)
    dummy = torch.empty((points, 7), device=device, dtype=torch.float32)
    with torch.no_grad():
        value = global_all_input_curvature_loss(
            model,
            dummy,
            cfg,
            feature_norm,
            generator=generator,
            max_points_override=points,
        )
    if was_training:
        model.train()
    return float(value.detach().cpu().item())


def apply_gradient_ratio_regularized_step(
    model,
    optimizer,
    base_loss: torch.Tensor,
    regularizer_loss: torch.Tensor,
    max_ratio: float,
) -> dict[str, float]:
    params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    before = [param.detach().clone() for param in params]
    base_grads = torch.autograd.grad(
        base_loss, params, retain_graph=True, allow_unused=True
    )
    regularizer_grads = torch.autograd.grad(
        regularizer_loss, params, allow_unused=True
    )

    def grad_norm(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
        terms = [torch.sum(torch.square(grad.float())) for grad in grads if grad is not None]
        if not terms:
            return torch.zeros((), device=base_loss.device, dtype=torch.float32)
        return torch.sqrt(torch.stack(terms).sum())

    regularizer_grad_norm = grad_norm(regularizer_grads)
    optimizer.zero_grad(set_to_none=True)
    for param, base_grad in zip(params, base_grads):
        param.grad = base_grad
    optimizer.step()

    # Adam rescales coordinates independently, so clipping the curvature gradient
    # before Adam does not bound its actual parameter displacement. Apply Adam only
    # to the regression objective, then add a direct curvature step whose global
    # norm is explicitly bounded relative to the realized Adam update.
    base_update_terms = [
        torch.sum(torch.square((param.detach() - old).float()))
        for param, old in zip(params, before)
    ]
    base_update_norm = torch.sqrt(torch.stack(base_update_terms).sum())
    eps = torch.finfo(torch.float32).eps
    allowed_update = float(max_ratio) * base_update_norm
    scale = allowed_update / torch.clamp(regularizer_grad_norm, min=eps)
    with torch.no_grad():
        for param, regularizer_grad in zip(params, regularizer_grads):
            if regularizer_grad is not None:
                param.add_(regularizer_grad, alpha=-float(scale.detach().cpu().item()))
    applied_update_norm = regularizer_grad_norm * scale
    base_value = float(base_update_norm.detach().cpu().item())
    return {
        "base_norm": base_value,
        "regularizer_norm": float(applied_update_norm.detach().cpu().item()),
        "applied_scale": float(scale.detach().cpu().item()),
        "applied_ratio": float(applied_update_norm.detach().cpu().item()) / max(base_value, float(eps)),
    }


def variance_sigma2_from_meta(
    meta: dict,
    target_safe: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    space: str,
) -> torch.Tensor | None:
    if not variance_weight_enabled(cfg):
        return None
    if "FR_ResponseVar" not in meta or "FR_ResponseN" not in meta:
        return None
    var = torch.as_tensor(np.asarray(meta["FR_ResponseVar"], dtype=np.float32).reshape(-1), device=device, dtype=target_safe.dtype)
    repeat_n = torch.as_tensor(np.asarray(meta["FR_ResponseN"], dtype=np.float32).reshape(-1), device=device, dtype=target_safe.dtype)
    valid = torch.isfinite(var) & torch.isfinite(repeat_n) & (var >= 0.0) & (repeat_n > 0.0)
    sem = torch.where(valid, var / torch.clamp(repeat_n, min=1.0), torch.zeros_like(var))
    key = str(space).lower()
    if key == "log":
        return sem / torch.square(1.0 + target_safe)
    if key == "rel":
        return sem / torch.square(target_safe + float(cfg.rel_alpha))
    if key == "raw":
        return sem
    raise ValueError(f"Unknown variance weight space {space!r}")


def variance_weights_from_sigma2(sigma2: torch.Tensor | None, cfg: TrainConfig, ref_value: float) -> torch.Tensor | None:
    if sigma2 is None:
        return None
    ref = float(ref_value)
    if not np.isfinite(ref) or ref <= 0.0:
        return torch.ones_like(sigma2)
    weights = 1.0 / (1.0 + sigma2 / max(ref, float(getattr(cfg, "variance_eps", 1e-8))))
    weights = torch.clamp(weights, min=float(getattr(cfg, "variance_min_weight", 0.25)), max=1.0)
    return weights / torch.clamp(torch.mean(weights), min=1e-12)


def variance_loss_weights(meta: dict, target_safe: torch.Tensor, cfg: TrainConfig, device: torch.device) -> dict[str, torch.Tensor | None]:
    if not variance_weight_enabled(cfg):
        return {"raw": None, "log": None, "rel": None}
    return {
        "raw": variance_weights_from_sigma2(
            variance_sigma2_from_meta(meta, target_safe, cfg, device, "raw"),
            cfg,
            float(getattr(cfg, "variance_ref_raw", 1.0)),
        ),
        "log": variance_weights_from_sigma2(
            variance_sigma2_from_meta(meta, target_safe, cfg, device, "log"),
            cfg,
            float(getattr(cfg, "variance_ref_log", 1.0)),
        ),
        "rel": variance_weights_from_sigma2(
            variance_sigma2_from_meta(meta, target_safe, cfg, device, "rel"),
            cfg,
            float(getattr(cfg, "variance_ref_rel", 1.0)),
        ),
    }


def loss_exclusion_mask(meta: dict, cfg: TrainConfig, device: torch.device, n_rows: int) -> torch.Tensor | None:
    contrasts = tuple(float(x) for x in (getattr(cfg, "loss_exclude_contrasts", ()) or ()))
    conditions = tuple((float(a), float(c)) for a, c in (getattr(cfg, "loss_exclude_conditions", ()) or ()))
    if not contrasts and not conditions:
        return None
    if "Contrast" not in meta:
        raise RuntimeError("Loss exclusion requires Contrast metadata in the training shard")
    contrast = torch.as_tensor(np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1), device=device)
    if int(contrast.numel()) != int(n_rows):
        raise RuntimeError(f"Contrast metadata length {contrast.numel()} does not match batch size {n_rows}")
    keep = torch.ones_like(contrast, dtype=torch.bool)
    for value in contrasts:
        keep &= torch.abs(contrast - float(value)) > 1e-6
    if conditions:
        if "Gamma" not in meta:
            raise RuntimeError("--loss_exclude_condition_file requires Gamma metadata in the training shard")
        gamma = torch.as_tensor(np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1), device=device)
        if int(gamma.numel()) != int(n_rows):
            raise RuntimeError(f"Gamma metadata length {gamma.numel()} does not match batch size {n_rows}")
        drop = torch.zeros_like(contrast, dtype=torch.bool)
        for gamma_value, contrast_value in conditions:
            drop |= (torch.abs(gamma - float(gamma_value)) <= 1e-5) & (torch.abs(contrast - float(contrast_value)) <= 1e-5)
        keep &= ~drop
    mask = keep.to(dtype=torch.float32)
    kept = torch.sum(mask)
    if float(kept.detach().cpu().item()) <= 0.0:
        return mask
    return mask / torch.clamp(torch.mean(mask), min=1e-12)


def apply_loss_exclusion_weights(
    sample_weights: dict[str, torch.Tensor | None],
    meta: dict,
    cfg: TrainConfig,
    device: torch.device,
    n_rows: int,
    *,
    enabled: bool,
) -> dict[str, torch.Tensor | None]:
    if not enabled or not loss_exclusion_enabled(cfg):
        return sample_weights
    mask = loss_exclusion_mask(meta, cfg, device, n_rows)
    if mask is None:
        return sample_weights
    out: dict[str, torch.Tensor | None] = {}
    for key in ("raw", "log", "rel"):
        base = sample_weights.get(key)
        out[key] = mask if base is None else base * mask
    return out


def mixed_total_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    *,
    rel_alpha: float,
    loss_eps: float,
    rmse_space: str,
    w_log: float,
    w_rel: float,
    low_penalty_lambda: float,
    low_penalty_tau: float,
    sample_weights: dict[str, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_safe = torch.clamp_min(pred_raw, 0.0)
    target_safe = torch.clamp_min(target_raw, 0.0)
    sample_weights = sample_weights or {"raw": None, "log": None, "rel": None}
    rmse_log = torch.sqrt(
        weighted_mean_tensor(torch.square(torch.log1p(pred_safe) - torch.log1p(target_safe)), sample_weights.get("log")) + loss_eps
    )
    rmse_raw = torch.sqrt(weighted_mean_tensor(torch.square(pred_safe - target_safe), sample_weights.get("raw")) + loss_eps)
    relrmse_raw = torch.sqrt(
        weighted_mean_tensor(torch.square((pred_safe - target_safe) / (target_safe + rel_alpha)), sample_weights.get("rel")) + loss_eps
    )
    low_mask = target_safe < low_penalty_tau
    low_over = torch.clamp(pred_safe - target_safe, min=0.0)
    if torch.any(low_mask):
        raw_weights = sample_weights.get("raw")
        low_weights = raw_weights[low_mask] if raw_weights is not None else None
        low_over_penalty = weighted_mean_tensor((low_over[low_mask]) ** 2, low_weights)
    else:
        low_over_penalty = torch.zeros((), dtype=pred_safe.dtype, device=pred_safe.device)
    rmse_term = rmse_log if str(rmse_space).lower() == "log" else rmse_raw
    total = w_log * rmse_term + w_rel * relrmse_raw + low_penalty_lambda * low_over_penalty
    return total, rmse_log, rmse_raw, relrmse_raw, low_over_penalty


def tfsf_smoothness_loss(model, xb: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if str(getattr(cfg, "feature_mode", "")).lower() != "tf_sf7d":
        raise RuntimeError("TF/SF smoothness requires --feature_mode tf_sf7d")
    n = int(xb.shape[0])
    if n <= 0:
        return xb.new_zeros(())
    max_points = int(getattr(cfg, "tfsf_smooth_max_points", 2048))
    if max_points > 0 and n > max_points:
        idx = torch.randperm(n, device=xb.device)[:max_points]
        x0 = xb.index_select(0, idx).detach()
    else:
        x0 = xb.detach()
    delta = float(getattr(cfg, "tfsf_smooth_delta", 0.05))
    base = torch.clamp_min(model(x0), 0.0)
    losses = []
    for col in (5, 6):
        for sign in (-1.0, 1.0):
            xp = x0.clone()
            xp[:, col] = torch.clamp(xp[:, col] + sign * delta, min=-6.0, max=6.0)
            pred = torch.clamp_min(model(xp), 0.0)
            losses.append(torch.mean(torch.square(pred - base)))
    return torch.stack(losses).mean() if losses else xb.new_zeros(())


def tfsf_unimodality_losses(
    model,
    xb: torch.Tensor,
    cfg: TrainConfig,
    feature_norm: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize TF/SF curves that cannot rise then fall around an interior peak."""
    if str(getattr(cfg, "feature_mode", "")).lower() != "tf_sf7d":
        raise RuntimeError("TF/SF unimodality requires --feature_mode tf_sf7d")
    from whitelist_pairs import ALLOWED_PAIRS

    n = int(xb.shape[0])
    if n <= 0:
        zero = xb.new_zeros(())
        return zero, zero
    max_points = int(getattr(cfg, "tfsf_unimodal_max_points", 512))
    if max_points > 0 and n > max_points:
        idx = torch.randperm(n, device=xb.device)[:max_points]
        x0 = xb.index_select(0, idx).detach()
    else:
        x0 = xb.detach()

    output_scale = float(TARGET_TO_MASK_BOUNDS[str(cfg.target)][1])
    prominence = float(getattr(cfg, "tfsf_unimodal_prominence_fraction", 0.02))

    def axis_loss(column: int, feature: str, raw_grid: tuple[float, ...], anchors: torch.Tensor) -> torch.Tensor:
        if len(raw_grid) < 3:
            return xb.new_zeros(())  # No interior peak exists for a two-point sweep.
        normalized_grid = torch.tensor(
            [normalized_feature_value(feature_norm, feature, value) for value in raw_grid],
            device=xb.device,
            dtype=anchors.dtype,
        )
        expanded = anchors[:, None, :].expand(-1, len(raw_grid), -1).clone()
        expanded[:, :, column] = normalized_grid[None, :]
        curve = torch.clamp_min(model(expanded.reshape(-1, anchors.shape[1])), 0.0).reshape(
            anchors.shape[0], len(raw_grid)
        )
        curve = curve / max(output_scale, 1e-8)
        differences = curve[:, 1:] - curve[:, :-1]
        candidates = []
        for peak in range(1, len(raw_grid) - 1):
            rise_violation = torch.mean(torch.square(torch.clamp(-differences[:, :peak], min=0.0)), dim=1)
            fall_violation = torch.mean(torch.square(torch.clamp(differences[:, peak:], min=0.0)), dim=1)
            peak_value = curve[:, peak]
            endpoint_violation = (
                torch.square(torch.clamp(prominence - (peak_value - curve[:, 0]), min=0.0))
                + torch.square(torch.clamp(prominence - (peak_value - curve[:, -1]), min=0.0))
            )
            candidates.append(rise_violation + fall_violation + endpoint_violation)
        return torch.mean(torch.min(torch.stack(candidates, dim=1), dim=1).values)

    allowed = xb.new_tensor([[normalized_feature_value(feature_norm, "lgnTF", tf),
                              normalized_feature_value(feature_norm, "lgnSF", sf)]
                             for tf, sf in ALLOWED_PAIRS])
    valid = torch.isclose(x0[:, None, 5:7], allowed[None, :, :], rtol=0, atol=1e-6).all(dim=2).any(dim=1)
    if not bool(valid.all()):
        raise ValueError("Loss anchor outside TF/SF pair whitelist")

    def conditional_axis_loss(axis: int) -> torch.Tensor:
        other = 1 - axis
        column, feature = (5, "lgnTF") if axis == 0 else (6, "lgnSF")
        other_column, other_feature = (6, "lgnSF") if axis == 0 else (5, "lgnTF")
        loss = xb.new_zeros(())
        for value in sorted({pair[other] for pair in ALLOWED_PAIRS}):
            mask = torch.isclose(x0[:, other_column], x0.new_tensor(normalized_feature_value(feature_norm, other_feature, value)), rtol=0, atol=1e-6)
            count = int(mask.sum())
            if count:
                grid = tuple(sorted({pair[axis] for pair in ALLOWED_PAIRS if pair[other] == value}))
                loss = loss + axis_loss(column, feature, grid, x0[mask]) * (count / x0.shape[0])
        return loss

    tf_loss = conditional_axis_loss(0)
    sf_loss = conditional_axis_loss(1)
    return tf_loss, sf_loss


def normalized_feature_value(feature_norm: dict, name: str, raw_value: float) -> float:
    norm_fields = feature_norm.get("norm_feature_names", {}).get("order") or feature_norm.get("norm_feature_order") or ()
    if name not in norm_fields:
        raise RuntimeError(f"Feature {name!r} is not normalized in feature_norm")
    stat = feature_norm.get(name, {})
    if isinstance(stat, dict) and "mean" in stat and "std" in stat:
        value = float(raw_value)
        transform = str(stat.get("transform", "identity")).lower()
        if transform == "log":
            if value <= 0.0:
                raise ValueError(f"Feature {name} requires a positive value for log normalization")
            value = math.log(value)
        elif transform != "identity":
            raise ValueError(f"Unsupported transform {transform!r} for feature {name}")
        mean = float(stat["mean"])
        std = float(stat["std"])
        if std <= 0.0:
            std = 1.0
        return (value - mean) / std
    mean_values = feature_norm.get("mean", [])
    std_values = feature_norm.get("std", [])
    idx = list(norm_fields).index(name)
    std = float(std_values[idx]) if idx < len(std_values) else 1.0
    if std <= 0.0:
        std = 1.0
    mean = float(mean_values[idx]) if idx < len(mean_values) else 0.0
    return (float(raw_value) - mean) / std


def distill_anchor_loss(
    model,
    teacher_model,
    xb: torch.Tensor,
    cfg: TrainConfig,
    feature_norm: dict,
) -> torch.Tensor:
    if teacher_model is None or float(getattr(cfg, "distill_anchor_lambda", 0.0)) <= 0.0:
        return xb.new_zeros(())
    if str(getattr(cfg, "feature_mode", "")).lower() != "tf_sf7d":
        raise RuntimeError("Distillation at default TF/SF requires --feature_mode tf_sf7d")
    n = int(xb.shape[0])
    if n <= 0:
        return xb.new_zeros(())
    mode = str(getattr(cfg, "distill_anchor_mode", "default_tfsf")).lower()
    if mode == "default_tfsf":
        x_ref = xb.detach().clone()
        x_ref[:, 5] = normalized_feature_value(
            feature_norm, "lgnTF", float(getattr(cfg, "distill_default_tf", 10.0))
        )
        x_ref[:, 6] = normalized_feature_value(
            feature_norm, "lgnSF", float(getattr(cfg, "distill_default_sf", 2.5))
        )
    elif mode == "same_input_below_tf":
        tf_cutoff = normalized_feature_value(
            feature_norm,
            "lgnTF",
            float(getattr(cfg, "distill_max_tf_exclusive", 32.0)),
        )
        keep = xb[:, 5] < tf_cutoff - 1e-6
        if not torch.any(keep):
            return xb.new_zeros(())
        x_ref = xb[keep].detach().clone()
    else:
        raise RuntimeError(f"Unknown distill_anchor_mode {mode!r}")
    max_points = int(getattr(cfg, "distill_max_points", 2048))
    if max_points > 0 and int(x_ref.shape[0]) > max_points:
        idx = torch.randperm(int(x_ref.shape[0]), device=xb.device)[:max_points]
        x_ref = x_ref.index_select(0, idx)
    with torch.no_grad():
        y_teacher = torch.clamp_min(teacher_model(x_ref), 0.0)
    y_student = torch.clamp_min(model(x_ref), 0.0)
    return torch.mean(torch.square(torch.log1p(y_student) - torch.log1p(y_teacher)))


def focus_angle_mask_torch(
    meta: dict,
    angle: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if "Gamma" not in meta or "LGNctgr" not in meta:
        raise RuntimeError("Angle-aware population focus requires Gamma and LGNctgr metadata")
    gamma = torch.as_tensor(
        np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1),
        device=device,
        dtype=dtype,
    )
    lgn = torch.as_tensor(
        np.asarray(meta["LGNctgr"], dtype=np.int64).reshape(-1),
        device=device,
        dtype=torch.long,
    )
    theta_map = torch.tensor([0.0, 0.0, 135.0, 90.0, 45.0], device=device, dtype=dtype)
    foreground = (lgn >= 1) & (lgn <= 4)
    safe_lgn = torch.clamp(lgn, 0, 4)
    theta_pref = theta_map[safe_lgn]
    wrapped = torch.remainder(float(angle) - theta_pref + 90.0, 180.0) - 90.0
    expected_gamma = torch.abs(wrapped)
    return foreground & (torch.abs(gamma - expected_gamma) <= 1e-4)


def focus_angle_mask_numpy(meta: dict, angle: float, n_rows: int) -> np.ndarray:
    if "Gamma" not in meta or "LGNctgr" not in meta:
        raise RuntimeError("Angle-aware population focus requires Gamma and LGNctgr metadata")
    gamma = np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1)
    lgn = np.asarray(meta["LGNctgr"], dtype=np.int64).reshape(-1)
    if gamma.size != int(n_rows) or lgn.size != int(n_rows):
        raise RuntimeError("Angle-aware focus metadata length does not match batch size")
    theta_map = np.asarray([0.0, 0.0, 135.0, 90.0, 45.0], dtype=np.float32)
    foreground = (lgn >= 1) & (lgn <= 4)
    safe_lgn = np.clip(lgn, 0, 4)
    wrapped = np.mod(float(angle) - theta_map[safe_lgn] + 90.0, 180.0) - 90.0
    expected_gamma = np.abs(wrapped)
    return foreground & np.isclose(gamma, expected_gamma, atol=1e-4, rtol=0.0)


def tfsf_population_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    meta: dict,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if "lgnTF" not in meta or "lgnSF" not in meta:
        raise RuntimeError("TF/SF population gradients require lgnTF and lgnSF metadata")
    pred_safe = torch.clamp_min(pred, 0.0)
    target_safe = torch.clamp_min(target, 0.0)
    tf = torch.as_tensor(np.asarray(meta["lgnTF"], dtype=np.float32).reshape(-1), device=device, dtype=target_safe.dtype)
    sf = torch.as_tensor(np.asarray(meta["lgnSF"], dtype=np.float32).reshape(-1), device=device, dtype=target_safe.dtype)
    default_tf = torch.abs(tf - 10.0) <= 1e-6
    default_sf = torch.abs(sf - 2.5) <= 1e-6
    masks = {
        "population_default": default_tf & default_sf,
        "population_tf_axis": (~default_tf) & default_sf,
        "population_sf_axis": default_tf & (~default_sf),
        "population_joint": (~default_tf) & (~default_sf),
    }
    focus_group_masks: list[torch.Tensor] = []
    if str(getattr(cfg, "population_grad_mode", "none")).lower() == "tfsf4_focus":
        conditions = tuple(getattr(cfg, "population_focus_conditions", ()) or ())
        if not conditions:
            raise RuntimeError("tfsf4_focus requires at least one population focus condition")
        if "Contrast" not in meta:
            raise RuntimeError("TF/SF focus population requires Contrast metadata")
        contrast = torch.as_tensor(
            np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1),
            device=device,
            dtype=target_safe.dtype,
        )
        focus_mask = torch.zeros_like(default_tf)
        for condition in conditions:
            contrast_value, tf_value, sf_value = condition[:3]
            condition_mask = (
                (torch.abs(contrast - float(contrast_value)) <= 1e-6)
                & (torch.abs(tf - float(tf_value)) <= 1e-6)
                & (torch.abs(sf - float(sf_value)) <= 1e-6)
            )
            if len(condition) >= 4:
                condition_mask &= focus_angle_mask_torch(meta, float(condition[3]), device, target_safe.dtype)
            focus_group_masks.append(condition_mask)
            focus_mask |= condition_mask
        masks[TFSF_FOCUS_POPULATION_NAME] = focus_mask
    sample_weights = variance_loss_weights(meta, target_safe, cfg, device)
    log_sq = torch.square(torch.log1p(pred_safe) - torch.log1p(target_safe))
    raw_sq = torch.square(pred_safe - target_safe)
    rel_sq = torch.square((pred_safe - target_safe) / (target_safe + float(cfg.rel_alpha)))
    def masked_loss(mask: torch.Tensor, include_focus_raw: bool) -> torch.Tensor:
        if not torch.any(mask):
            raise RuntimeError("Cannot compute a population loss from an empty mask")
        mask_float = mask.to(dtype=target_safe.dtype)
        log_weights = mask_float if sample_weights["log"] is None else mask_float * sample_weights["log"]
        rel_weights = mask_float if sample_weights["rel"] is None else mask_float * sample_weights["rel"]
        rmse_log = torch.sqrt(weighted_mean_tensor(log_sq, log_weights) + float(cfg.loss_eps))
        relrmse = torch.sqrt(weighted_mean_tensor(rel_sq, rel_weights) + float(cfg.loss_eps))
        loss = float(cfg.w_log) * rmse_log + float(cfg.w_rel) * relrmse
        if include_focus_raw and float(getattr(cfg, "population_focus_raw_lambda", 0.0)) > 0.0:
            raw_weights = mask_float if sample_weights["raw"] is None else mask_float * sample_weights["raw"]
            rmse_raw = torch.sqrt(weighted_mean_tensor(raw_sq, raw_weights) + float(cfg.loss_eps))
            raw_scale = float(getattr(cfg, "population_focus_raw_scale", 0.0))
            if raw_scale <= 0.0:
                raw_scale = {"S": 30.0, "C": 100.0, "I": 140.0}[str(cfg.target)]
            loss = loss + float(cfg.population_focus_raw_lambda) * rmse_raw / raw_scale
        return loss

    out: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        cvar_fraction = float(getattr(cfg, "population_focus_group_cvar_fraction", 0.0))
        if name == TFSF_FOCUS_POPULATION_NAME and cvar_fraction > 0.0:
            group_losses = [masked_loss(group_mask, True) for group_mask in focus_group_masks if torch.any(group_mask)]
            if not group_losses:
                raise RuntimeError("Batch has no rows for any angle-aware focus group")
            stacked = torch.stack(group_losses)
            top_count = max(1, int(math.ceil(min(cvar_fraction, 1.0) * int(stacked.numel()))))
            out[name] = torch.topk(stacked, k=top_count, largest=True).values.mean()
        else:
            out[name] = masked_loss(mask, name == TFSF_FOCUS_POPULATION_NAME)
    return out


def population_focus_condition_mask_numpy(meta: dict, condition: tuple[float, ...], n_rows: int) -> np.ndarray:
    for field in ("Contrast", "lgnTF", "lgnSF"):
        if field not in meta:
            raise RuntimeError(f"Population focus metrics require {field} metadata")
    contrast = np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1)
    tf = np.asarray(meta["lgnTF"], dtype=np.float32).reshape(-1)
    sf = np.asarray(meta["lgnSF"], dtype=np.float32).reshape(-1)
    if contrast.size != int(n_rows) or tf.size != int(n_rows) or sf.size != int(n_rows):
        raise RuntimeError("Population focus metadata length does not match batch size")
    contrast_value, tf_value, sf_value = condition[:3]
    mask = (
        np.isclose(contrast, contrast_value)
        & np.isclose(tf, tf_value)
        & np.isclose(sf, sf_value)
    )
    if len(condition) >= 4:
        mask &= focus_angle_mask_numpy(meta, float(condition[3]), n_rows)
    return mask


def population_focus_mask_numpy(meta: dict, cfg: TrainConfig, n_rows: int) -> np.ndarray:
    conditions = tuple(getattr(cfg, "population_focus_conditions", ()) or ())
    mask = np.zeros(int(n_rows), dtype=bool)
    for condition in conditions:
        mask |= population_focus_condition_mask_numpy(meta, condition, n_rows)
    return mask


def compute_loss_terms(
    model,
    xb: torch.Tensor,
    yb: torch.Tensor,
    meta: dict,
    cfg: TrainConfig,
    device: torch.device,
    grad_teacher: TrilinearTeacher3D | None,
    feature_norm: dict,
    teacher_model=None,
    apply_loss_exclusion: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = model(xb)
    target_safe = torch.clamp_min(yb, 0.0)
    sample_weights = variance_loss_weights(meta, target_safe, cfg, device)
    sample_weights = apply_loss_exclusion_weights(
        sample_weights,
        meta,
        cfg,
        device,
        n_rows=int(target_safe.numel()),
        enabled=apply_loss_exclusion,
    )
    _base_total, rmse_log, rmse_raw, relrmse_raw, low_over_penalty = mixed_total_loss(
        pred,
        yb,
        rel_alpha=cfg.rel_alpha,
        loss_eps=cfg.loss_eps,
        rmse_space=cfg.rmse_space,
        w_log=cfg.w_log,
        w_rel=cfg.w_rel,
        low_penalty_lambda=cfg.low_penalty_lambda,
        low_penalty_tau=cfg.low_penalty_tau,
        sample_weights=sample_weights,
    )
    rmse_term = rmse_log if str(cfg.rmse_space).lower() == "log" else rmse_raw
    pred_safe = torch.clamp_min(pred, 0.0)
    raw_mse = weighted_mean_tensor(
        torch.square(pred_safe - target_safe),
        sample_weights.get("raw"),
    )
    mono_pair_loss = pred.new_zeros(())
    mono_grad_loss = pred.new_zeros(())
    curvature_loss = pred.new_zeros(())
    corr_penalty = pred.new_zeros(())
    grad_loss = pred.new_zeros(())
    adv_huber_loss = pred.new_zeros(())
    adv_smooth_loss = pred.new_zeros(())
    guide83_loss = pred.new_zeros(())
    guide84_poisson_loss = pred.new_zeros(())
    tfsf_smooth_loss = pred.new_zeros(())
    tf_unimodal_loss = pred.new_zeros(())
    sf_unimodal_loss = pred.new_zeros(())
    distill_loss = pred.new_zeros(())
    population_losses = {name: pred.new_zeros(()) for name in ALL_TFSF_POPULATION_NAMES}

    if active_tfsf_population_names(cfg):
        population_losses.update(tfsf_population_losses(pred, yb, meta, cfg, device))

    if getattr(cfg, "loss_mode", "base") == "targeted_mid_edge_asym":
        corr_penalty, _, _ = targeted_mid_edge_penalty(pred, yb, meta, cfg, device)
    elif getattr(cfg, "loss_mode", "base") == "legacy_gamma_contrast_asym":
        corr_penalty, _, _ = legacy_gamma_contrast_asym_penalty(pred, yb, meta, cfg, device)

    if cfg.mono_pair_lambda > 0:
        mono_pair_loss = mono_pair_loss_from_meta(pred, meta, device)

    if float(getattr(cfg, "mono_grad_lambda", 0.0)) > 0.0:
        mono_grad_loss = guide55_mono_grad_loss(model, xb, cfg)

    if float(getattr(cfg, "curvature_lambda", 0.0)) > 0.0:
        curvature_loss = central_curvature_loss(model, xb, cfg, feature_norm)

    if float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) > 0.0:
        tfsf_smooth_loss = tfsf_smoothness_loss(model, xb, cfg)

    if (
        float(getattr(cfg, "tf_unimodal_lambda", 0.0)) > 0.0
        or float(getattr(cfg, "sf_unimodal_lambda", 0.0)) > 0.0
    ):
        tf_unimodal_loss, sf_unimodal_loss = tfsf_unimodality_losses(
            model, xb, cfg, feature_norm
        )

    if float(getattr(cfg, "distill_anchor_lambda", 0.0)) > 0.0:
        distill_loss = distill_anchor_loss(model, teacher_model, xb, cfg, feature_norm)

    guide_mode = str(getattr(cfg, "guide_uncertainty_mode", "none")).lower()
    if guide_mode == "guide83_count_weighted":
        guide83_loss = guide83_count_weighted_loss(pred, yb, sample_weights, cfg)
    elif guide_mode == "guide84_poisson":
        guide84_poisson_loss = guide84_poisson_rate_loss(pred, yb, sample_weights, cfg)
    elif guide_mode != "none":
        raise RuntimeError(f"Unknown guide_uncertainty_mode {guide_mode!r}")

    if cfg.grad_lambda > 0.0:
        if grad_teacher is None:
            raise RuntimeError("Gradient teacher is required when grad_lambda > 0")
        grad_loss = interpolation_grad_loss(
            model,
            xb,
            meta,
            feature_norm,
            grad_teacher,
            grad_max_points=cfg.grad_max_points,
            grad_interior_eps=cfg.grad_interior_eps,
            device=device,
        )

    if bool(getattr(cfg, "adv_enable", False)) and (
        float(getattr(cfg, "adv_lambda", 0.0)) > 0.0 or float(getattr(cfg, "adv_smooth_lambda", 0.0)) > 0.0
    ):
        eps = float(getattr(cfg, "adv_epsilon", 0.0))
        if eps > 0.0:
            with torch.enable_grad():
                x_probe = xb.detach().clone().requires_grad_(True)
                pred_probe = model(x_probe)
                raw_weights = sample_weights.get("raw")
                adv_source_loss = weighted_huber_loss(
                    pred_probe,
                    yb,
                    raw_weights,
                    delta=float(getattr(cfg, "adv_huber_delta", 1.0)),
                )
                grad_x = torch.autograd.grad(
                    adv_source_loss,
                    x_probe,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
            delta = eps * torch.sign(torch.nan_to_num(grad_x.detach(), nan=0.0, posinf=0.0, neginf=0.0))
            x_adv = xb.detach() + delta
            clip_value = float(getattr(cfg, "adv_clip_value", 6.0))
            if clip_value > 0.0:
                x_adv = torch.clamp(x_adv, min=-clip_value, max=clip_value)
            pred_adv = model(x_adv)
            raw_weights = sample_weights.get("raw")
            adv_huber_loss = weighted_huber_loss(
                pred_adv,
                yb,
                raw_weights,
                delta=float(getattr(cfg, "adv_huber_delta", 1.0)),
            )
            adv_smooth_loss = weighted_mean_tensor(
                torch.square(torch.clamp_min(pred_adv, 0.0) - torch.clamp_min(pred, 0.0)),
                raw_weights,
            )

    terms = {
        "rmse_term": rmse_term,
        "raw_mse": raw_mse,
        "relrmse_raw": relrmse_raw,
        "low_over_penalty": low_over_penalty,
        "mono_pair_loss": mono_pair_loss,
        "mono_grad_loss": mono_grad_loss,
        "curvature_loss": curvature_loss,
        "corr_penalty": corr_penalty,
        "grad_loss": grad_loss,
        "adv_huber_loss": adv_huber_loss,
        "adv_smooth_loss": adv_smooth_loss,
        "guide83_loss": guide83_loss,
        "guide84_poisson_loss": guide84_poisson_loss,
        "tfsf_smooth_loss": tfsf_smooth_loss,
        "tf_unimodal_loss": tf_unimodal_loss,
        "sf_unimodal_loss": sf_unimodal_loss,
        "distill_loss": distill_loss,
        **population_losses,
        "rmse_log": rmse_log,
        "rmse_raw": rmse_raw,
        "variance_weight_raw_mean": torch.mean(sample_weights["raw"]) if sample_weights["raw"] is not None else pred.new_tensor(1.0),
        "variance_weight_raw_min": torch.min(sample_weights["raw"]) if sample_weights["raw"] is not None else pred.new_tensor(1.0),
        "variance_weight_raw_max": torch.max(sample_weights["raw"]) if sample_weights["raw"] is not None else pred.new_tensor(1.0),
        "loss_exclusion_keep_frac": torch.mean((sample_weights["raw"] > 0).to(torch.float32)) if sample_weights["raw"] is not None and loss_exclusion_enabled(cfg) and apply_loss_exclusion else pred.new_tensor(1.0),
    }
    return pred, terms


def default_subloss_scales() -> dict[str, float]:
    return {
        "rmse_term": 1.0,
        "raw_mse": 1.0,
        "relrmse_raw": 1.0,
        "low_over_penalty": 1.0,
        "mono_pair_loss": 1.0,
        "mono_grad_loss": 1.0,
        "curvature_loss": 1.0,
        "corr_penalty": 1.0,
        "grad_loss": 1.0,
        "adv_huber_loss": 1.0,
        "adv_smooth_loss": 1.0,
        "guide83_loss": 1.0,
        "guide84_poisson_loss": 1.0,
        "tfsf_smooth_loss": 1.0,
        "tf_unimodal_loss": 1.0,
        "sf_unimodal_loss": 1.0,
        "distill_loss": 1.0,
        "population_default": 1.0,
        "population_tf_axis": 1.0,
        "population_sf_axis": 1.0,
        "population_joint": 1.0,
        "population_focus": 1.0,
    }


def compute_aux_warmup_factor(cfg: TrainConfig, current_step: int, total_steps: int) -> float:
    warmup_frac = float(getattr(cfg, "task_uw_aux_warmup_frac", 0.0))
    if warmup_frac <= 0.0 or total_steps <= 0:
        return 1.0
    warmup_scale = float(getattr(cfg, "task_uw_aux_warmup_scale", 1.0))
    warmup_scale = min(max(warmup_scale, 0.0), 1.0)
    warmup_steps = max(1, int(round(float(total_steps) * warmup_frac)))
    if current_step >= warmup_steps:
        return 1.0
    alpha = float(current_step) / float(warmup_steps)
    return warmup_scale + (1.0 - warmup_scale) * alpha


def build_total_loss_from_terms(
    terms: dict[str, torch.Tensor],
    cfg: TrainConfig,
    subloss_scales: dict[str, float] | None = None,
    *,
    uw_log_vars: torch.nn.Parameter | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    scales = subloss_scales or default_subloss_scales()

    def norm(name: str) -> torch.Tensor:
        return terms[name] / float(scales.get(name, 1.0))

    normalized = {
        "rmse_term": norm("rmse_term"),
        "raw_mse": norm("raw_mse"),
        "relrmse_raw": norm("relrmse_raw"),
        "low_over_penalty": norm("low_over_penalty"),
        "mono_pair_loss": norm("mono_pair_loss"),
        "mono_grad_loss": norm("mono_grad_loss"),
        "curvature_loss": norm("curvature_loss"),
        "corr_penalty": norm("corr_penalty"),
        "grad_loss": norm("grad_loss"),
        "adv_huber_loss": norm("adv_huber_loss"),
        "adv_smooth_loss": norm("adv_smooth_loss"),
        "guide83_loss": norm("guide83_loss"),
        "guide84_poisson_loss": norm("guide84_poisson_loss"),
        "tfsf_smooth_loss": norm("tfsf_smooth_loss"),
        "tf_unimodal_loss": norm("tf_unimodal_loss"),
        "sf_unimodal_loss": norm("sf_unimodal_loss"),
        "distill_loss": norm("distill_loss"),
        "population_default": norm("population_default"),
        "population_tf_axis": norm("population_tf_axis"),
        "population_sf_axis": norm("population_sf_axis"),
        "population_joint": norm("population_joint"),
        "population_focus": norm("population_focus"),
    }
    guide_mode = str(getattr(cfg, "guide_uncertainty_mode", "none")).lower()
    if guide_mode == "guide83_count_weighted":
        normalized["main_task"] = normalized["guide83_loss"]
    elif guide_mode == "guide84_poisson":
        normalized["main_task"] = normalized["guide84_poisson_loss"]
    else:
        normalized["main_task"] = (
            float(cfg.w_log) * normalized["rmse_term"]
            + float(cfg.w_rel) * normalized["relrmse_raw"]
            + float(getattr(cfg, "raw_mse_lambda", 0.0)) * normalized["raw_mse"]
        )
    normalized["aux_task"] = float(cfg.low_penalty_lambda) * normalized["low_over_penalty"]

    aux_warmup_factor = 1.0
    if bool(getattr(cfg, "task_uw_enable", False)):
        if uw_log_vars is None:
            raise RuntimeError("uw_log_vars is required when task_uw_enable is set")
        aux_warmup_factor = compute_aux_warmup_factor(cfg, int(current_step or 0), int(total_steps or 0))
        total = (
            torch.exp(-uw_log_vars[0]) * normalized["main_task"]
            + uw_log_vars[0]
            + torch.exp(-uw_log_vars[1]) * (normalized["aux_task"] * float(aux_warmup_factor))
            + uw_log_vars[1]
            + float(cfg.mono_pair_lambda) * normalized["mono_pair_loss"]
            + float(getattr(cfg, "mono_grad_lambda", 0.0)) * normalized["mono_grad_loss"]
            + float(getattr(cfg, "curvature_lambda", 0.0)) * normalized["curvature_loss"]
            + normalized["corr_penalty"]
            + float(cfg.grad_lambda) * normalized["grad_loss"]
            + float(getattr(cfg, "adv_lambda", 0.0)) * normalized["adv_huber_loss"]
            + float(getattr(cfg, "adv_smooth_lambda", 0.0)) * normalized["adv_smooth_loss"]
            + float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) * normalized["tfsf_smooth_loss"]
            + float(getattr(cfg, "tf_unimodal_lambda", 0.0)) * normalized["tf_unimodal_loss"]
            + float(getattr(cfg, "sf_unimodal_lambda", 0.0)) * normalized["sf_unimodal_loss"]
            + float(getattr(cfg, "distill_anchor_lambda", 0.0)) * normalized["distill_loss"]
        )
    else:
        total = (
            normalized["main_task"]
            + normalized["aux_task"]
            + float(cfg.mono_pair_lambda) * normalized["mono_pair_loss"]
            + float(getattr(cfg, "mono_grad_lambda", 0.0)) * normalized["mono_grad_loss"]
            + float(getattr(cfg, "curvature_lambda", 0.0)) * normalized["curvature_loss"]
            + normalized["corr_penalty"]
            + float(cfg.grad_lambda) * normalized["grad_loss"]
            + float(getattr(cfg, "adv_lambda", 0.0)) * normalized["adv_huber_loss"]
            + float(getattr(cfg, "adv_smooth_lambda", 0.0)) * normalized["adv_smooth_loss"]
            + float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) * normalized["tfsf_smooth_loss"]
            + float(getattr(cfg, "tf_unimodal_lambda", 0.0)) * normalized["tf_unimodal_loss"]
            + float(getattr(cfg, "sf_unimodal_lambda", 0.0)) * normalized["sf_unimodal_loss"]
            + float(getattr(cfg, "distill_anchor_lambda", 0.0)) * normalized["distill_loss"]
        )
    diag = {
        "aux_warmup_factor": float(aux_warmup_factor),
        "uw_s_main": float(uw_log_vars[0].detach().cpu().item()) if uw_log_vars is not None else 0.0,
        "uw_s_aux": float(uw_log_vars[1].detach().cpu().item()) if uw_log_vars is not None else 0.0,
        "uw_weight_main": float(torch.exp(-uw_log_vars[0]).detach().cpu().item()) if uw_log_vars is not None else 1.0,
        "uw_weight_aux": float(torch.exp(-uw_log_vars[1]).detach().cpu().item()) if uw_log_vars is not None else 1.0,
    }
    return total, normalized, diag


def estimate_initial_subloss_scales(
    model,
    loader: SplitShardLoader,
    device: torch.device,
    cfg: TrainConfig,
    grad_teacher: TrilinearTeacher3D | None,
    online_interp_teacher: OnlineTrilinearTeacher3D | None,
    feature_norm: dict,
    init_batches: int,
) -> dict[str, float]:
    sums = {name: 0.0 for name in default_subloss_scales()}
    n_batches = 0
    model.eval()
    with torch.no_grad():
        pass
    for xb, yb, meta in loader:
        if n_batches >= int(init_batches):
            break
        if getattr(cfg, "online_interp_enable", False):
            if online_interp_teacher is None:
                raise RuntimeError("Online interpolation teacher is required when online_interp_enable is set")
            xb, yb, meta = apply_online_interp_augmentation(xb, yb.squeeze(-1), meta, cfg, online_interp_teacher, feature_norm)
            yb = yb.unsqueeze(-1)
        xb = xb.to(device=device, non_blocking=True)
        yb = yb.to(device=device, non_blocking=True).squeeze(-1)
        with device_autocast(device, cfg.use_amp):
            _, terms = compute_loss_terms(model, xb, yb, meta, cfg, device, grad_teacher, feature_norm, apply_loss_exclusion=True)
        for name in sums:
            sums[name] += float(terms[name].detach().cpu().item())
        n_batches += 1
    if n_batches <= 0:
        return default_subloss_scales()
    return {name: max(value / n_batches, 1e-8) for name, value in sums.items()}


def _flatten_grads(grads: list[torch.Tensor | None], params: list[torch.nn.Parameter], device: torch.device) -> torch.Tensor:
    flat_parts: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=device, dtype=torch.float32))
        else:
            flat_parts.append(grad.detach().reshape(-1).to(device=device, dtype=torch.float32))
    if not flat_parts:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.cat(flat_parts)


def compute_gradcos_metrics(
    model,
    component_terms: dict[str, torch.Tensor],
    active_names: list[str],
    coefficients: dict[str, float] | None = None,
) -> dict[str, float]:
    params = [param for param in model.parameters() if param.requires_grad]
    out: dict[str, float] = {}
    if not params:
        return out
    valid_names = [
        name
        for name in active_names
        if name in component_terms and bool(getattr(component_terms[name], "requires_grad", False))
    ]
    if not valid_names:
        return out
    grad_vecs: dict[str, torch.Tensor] = {}
    ref_device = params[0].device
    for idx, name in enumerate(valid_names):
        retain_graph = idx < len(valid_names) - 1
        grads = torch.autograd.grad(component_terms[name], params, retain_graph=retain_graph, allow_unused=True)
        grad_vecs[name] = _flatten_grads(list(grads), params, ref_device)
        norm = float(torch.linalg.norm(grad_vecs[name]).item())
        coefficient = float((coefficients or {}).get(name, 1.0))
        out[f"{name}__grad_norm"] = norm
        out[f"{name}__weighted_grad_norm"] = abs(coefficient) * norm
    for left, right in combinations(valid_names, 2):
        lv = grad_vecs[left]
        rv = grad_vecs[right]
        left_norm = float(torch.linalg.norm(lv).detach().cpu().item())
        right_norm = float(torch.linalg.norm(rv).detach().cpu().item())
        key = f"{left}__vs__{right}"
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            out[key] = float("nan")
        else:
            out[key] = float(torch.nn.functional.cosine_similarity(lv, rv, dim=0).detach().cpu().item())
    return out


def _compute_term_grad_list(
    term: torch.Tensor,
    params: list[torch.nn.Parameter],
    scaler,
    *,
    retain_graph: bool,
) -> list[torch.Tensor | None]:
    if not bool(getattr(term, "requires_grad", False)):
        return [None for _ in params]
    scale = 1.0
    grad_term = term
    if scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
        scale = float(scaler.get_scale())
        grad_term = scaler.scale(term)
    grads = torch.autograd.grad(grad_term, params, retain_graph=retain_graph, allow_unused=True)
    out: list[torch.Tensor | None] = []
    for grad in grads:
        if grad is None:
            out.append(None)
        else:
            out.append((grad.detach() / scale).to(dtype=torch.float32))
    return out


def _unflatten_grad_vector(flat: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    offset = 0
    for param in params:
        n = int(param.numel())
        out.append(flat[offset:offset + n].view_as(param))
        offset += n
    return out


def apply_pcgrad_two_task_step(
    model,
    optimizer,
    scaler,
    main_term: torch.Tensor,
    aux_term: torch.Tensor,
    extra_term: torch.Tensor,
) -> None:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        optimizer.step()
        return

    ref_device = params[0].device
    extra_requires = bool(getattr(extra_term, "requires_grad", False))
    main_grads = _compute_term_grad_list(main_term, params, scaler, retain_graph=True)
    aux_grads = _compute_term_grad_list(aux_term, params, scaler, retain_graph=extra_requires)
    if extra_requires:
        extra_grads = _compute_term_grad_list(extra_term, params, scaler, retain_graph=False)
    else:
        extra_grads = [None for _ in params]

    main_vec = _flatten_grads(main_grads, params, ref_device)
    aux_vec = _flatten_grads(aux_grads, params, ref_device)
    main_proj = main_vec.clone()
    aux_proj = aux_vec.clone()

    dot = float(torch.dot(main_vec, aux_vec).detach().cpu().item())
    main_norm_sq = float(torch.dot(main_vec, main_vec).detach().cpu().item())
    aux_norm_sq = float(torch.dot(aux_vec, aux_vec).detach().cpu().item())
    if dot < 0.0:
        if aux_norm_sq > 1e-12:
            main_proj = main_vec - (dot / aux_norm_sq) * aux_vec
        if main_norm_sq > 1e-12:
            aux_proj = aux_vec - (dot / main_norm_sq) * main_vec

    final_vec = main_proj + aux_proj + _flatten_grads(extra_grads, params, ref_device)
    final_grads = _unflatten_grad_vector(final_vec, params)
    for param, grad in zip(params, final_grads):
        param.grad = grad.to(device=param.device, dtype=param.dtype).clone()
    optimizer.step()


def _project_simplex(values: torch.Tensor) -> torch.Tensor:
    ordered, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(ordered, dim=0) - 1.0
    indices = torch.arange(1, values.numel() + 1, dtype=values.dtype, device=values.device)
    valid = ordered - cumulative / indices > 0
    rho = int(torch.nonzero(valid, as_tuple=False)[-1].item())
    theta = cumulative[rho] / float(rho + 1)
    return torch.clamp(values - theta, min=0.0)


def _cagrad_multi_weights(grad_matrix: torch.Tensor, alpha: float) -> torch.Tensor:
    task_count = int(grad_matrix.shape[0])
    gram = (grad_matrix @ grad_matrix.T).detach().to(device="cpu", dtype=torch.float64)
    uniform = torch.full((task_count,), 1.0 / task_count, dtype=torch.float64)
    g0_norm = torch.linalg.norm(torch.mean(grad_matrix, dim=0)).detach().to(device="cpu", dtype=torch.float64)
    c = float(alpha) * g0_norm

    def objective(weights: torch.Tensor) -> torch.Tensor:
        return torch.dot(weights, gram @ uniform) + c * torch.sqrt(torch.clamp(torch.dot(weights, gram @ weights), min=1e-16))

    weights = uniform.clone()
    step = 1.0 / max(float(torch.linalg.norm(gram, ord=2).item()), 1e-8)
    for _ in range(64):
        quad_root = torch.sqrt(torch.clamp(torch.dot(weights, gram @ weights), min=1e-16))
        gradient = gram @ uniform + c * (gram @ weights) / quad_root
        current = objective(weights)
        trial_step = step
        candidate = weights
        accepted = False
        for _ in range(20):
            candidate = _project_simplex(weights - trial_step * gradient)
            if objective(candidate) <= current + 1e-12:
                accepted = True
                break
            trial_step *= 0.5
        if not accepted or torch.linalg.norm(candidate - weights) <= 1e-8:
            break
        weights = candidate
        step = trial_step * 1.2
    return weights.to(device=grad_matrix.device, dtype=grad_matrix.dtype)


def apply_cagrad_multi_task_step(
    model,
    optimizer,
    task_terms: list[torch.Tensor],
    extra_term: torch.Tensor,
    *,
    alpha: float,
    rescale: int,
) -> dict[str, float]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        optimizer.step()
        return {"weights": []}
    if len(task_terms) < 2:
        raise RuntimeError("CAGrad requires at least two population tasks")

    ref_device = params[0].device
    extra_requires = bool(getattr(extra_term, "requires_grad", False))
    task_vectors = []
    for index, term in enumerate(task_terms):
        retain_graph = index < len(task_terms) - 1 or extra_requires
        grad_list = _compute_term_grad_list(term, params, None, retain_graph=retain_graph)
        task_vectors.append(_flatten_grads(grad_list, params, ref_device))
    extra_list = _compute_term_grad_list(extra_term, params, None, retain_graph=False) if extra_requires else [None for _ in params]
    grad_matrix = torch.stack(task_vectors, dim=0)
    extra = _flatten_grads(extra_list, params, ref_device)
    g0 = torch.mean(grad_matrix, dim=0)
    weights = _cagrad_multi_weights(grad_matrix, alpha)
    gw = weights @ grad_matrix
    c = float(alpha) * torch.linalg.norm(g0)
    combined = g0 + (c / (torch.linalg.norm(gw) + 1e-4)) * gw
    if int(rescale) == 1:
        combined = combined / (1.0 + float(alpha) ** 2)
    elif int(rescale) == 2:
        combined = combined / (1.0 + float(alpha))
    combined = combined + extra
    for param, grad in zip(params, _unflatten_grad_vector(combined, params)):
        param.grad = grad.to(device=param.device, dtype=param.dtype).clone()
    optimizer.step()
    return {"weights": [float(value) for value in weights.detach().cpu().tolist()]}


def targeted_mid_edge_penalty(pred_raw: torch.Tensor, target_raw: torch.Tensor, meta: dict, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_raw = clamp_raw_rate_tensor(pred_raw)
    target_raw = clamp_raw_rate_tensor(target_raw)
    gamma = torch.as_tensor(np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1), device=device)
    contrast = torch.as_tensor(np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1), device=device)
    over = torch.clamp(pred_raw - target_raw, min=0.0)
    under = torch.clamp(target_raw - pred_raw, min=0.0)
    mid_vals = torch.as_tensor(np.asarray(getattr(cfg, "mid_under_contrasts", (42.0, 52.0, 61.0)), dtype=np.float32), device=device)
    edge_vals = torch.as_tensor(np.asarray(getattr(cfg, "edge_over_contrasts", (14.0, 95.0)), dtype=np.float32), device=device)
    mid_mask = (gamma < float(getattr(cfg, "mid_under_gamma_max", 15.0))) & torch.isin(contrast, mid_vals)
    edge_mask = (torch.abs(contrast.new_tensor(22.5) - gamma) < float(getattr(cfg, "edge_over_gamma_radius", 10.0))) & torch.isin(contrast, edge_vals)
    if torch.any(mid_mask):
        mid_under = torch.mean(under[mid_mask] ** 2)
    else:
        mid_under = torch.zeros((), dtype=pred_raw.dtype, device=device)
    if torch.any(edge_mask):
        edge_over = torch.mean(over[edge_mask] ** 2)
    else:
        edge_over = torch.zeros((), dtype=pred_raw.dtype, device=device)
    corr = float(getattr(cfg, "mid_under_lambda", 0.0)) * mid_under + float(getattr(cfg, "edge_over_lambda", 0.0)) * edge_over
    return corr, mid_under, edge_over


def legacy_gamma_contrast_asym_penalty(pred_raw: torch.Tensor, target_raw: torch.Tensor, meta: dict, cfg: TrainConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_raw = clamp_raw_rate_tensor(pred_raw)
    target_raw = clamp_raw_rate_tensor(target_raw)
    gamma = torch.as_tensor(np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1), device=device)
    contrast = torch.as_tensor(np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1), device=device)
    over = torch.clamp(pred_raw - target_raw, min=0.0)
    under = torch.clamp(target_raw - pred_raw, min=0.0)
    mask_mid_under = (gamma < 15.0) & (contrast >= 42.0) & (contrast <= 61.0)
    mask_edge_over = (torch.abs(gamma - 22.5) < 7.5) & ((contrast <= 14.0) | (contrast >= 71.0))
    if torch.any(mask_mid_under):
        mid_under = torch.mean(under[mask_mid_under] ** 2)
    else:
        mid_under = torch.zeros((), dtype=pred_raw.dtype, device=device)
    if torch.any(mask_edge_over):
        edge_over = torch.mean(over[mask_edge_over] ** 2)
    else:
        edge_over = torch.zeros((), dtype=pred_raw.dtype, device=device)
    corr = float(getattr(cfg, "corr_lambda", 0.0)) * (
        float(getattr(cfg, "corr_w_under_mid0", 1.0)) * mid_under
        + float(getattr(cfg, "corr_w_over_edge225", 1.0)) * edge_over
    )
    return corr, mid_under, edge_over


def apply_online_interp_augmentation(
    xb: torch.Tensor,
    yb: torch.Tensor,
    meta: dict,
    cfg: TrainConfig,
    teacher: OnlineTrilinearTeacher3D,
    feature_norm: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    batch_n = int(xb.shape[0])
    if batch_n <= 0:
        return xb, yb, meta

    gamma = np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1)
    contrast = np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1)
    lgn = np.asarray(meta["LGNctgr"], dtype=np.int16).reshape(-1)
    l6 = np.asarray(meta["L6"], dtype=np.float32).reshape(-1)
    l4e = np.asarray(meta["L4E"], dtype=np.float32).reshape(-1)
    l4i = np.asarray(meta["L4I"], dtype=np.float32).reshape(-1)

    weights = np.asarray([teacher.row_weight(g, c) for g, c in zip(gamma, contrast)], dtype=np.float64)
    eligible = np.flatnonzero(weights > 0)
    if eligible.size == 0:
        return xb, yb, meta

    use_n = max(1, int(round(batch_n * float(getattr(cfg, "online_interp_frac", 0.10)))))
    use_n = min(use_n, int(eligible.size))
    if use_n <= 0:
        return xb, yb, meta

    probs = weights[eligible]
    probs = probs / np.sum(probs)
    rng = np.random.default_rng()
    chosen = rng.choice(eligible, size=use_n, replace=False, p=probs)
    xb_mod = xb.clone()
    yb_mod = yb.clone()
    meta_mod = {key: np.array(val, copy=True) for key, val in meta.items()}
    feature_names = feature_names_for_mode(cfg.feature_mode)

    def set_norm_col(row_idx: int, feature_name: str, raw_value: float) -> None:
        col = feature_names.index(feature_name)
        if feature_name in feature_norm:
            mu = float(feature_norm[feature_name]["mean"])
            sd = float(feature_norm[feature_name]["std"])
            xb_mod[row_idx, col] = torch.tensor((float(raw_value) - mu) / sd, dtype=xb_mod.dtype)
        else:
            xb_mod[row_idx, col] = torch.tensor(float(raw_value), dtype=xb_mod.dtype)

    for row_idx in chosen:
        repl = teacher.sample_replacement(
            gamma=float(gamma[row_idx]),
            contrast=float(contrast[row_idx]),
            lgnctgr=int(lgn[row_idx]),
            l6_raw=float(l6[row_idx]),
            l4e_raw=float(l4e[row_idx]),
            l4i_raw=float(l4i[row_idx]),
            rng=rng,
        )
        if repl is None:
            continue
        l6_q, l4e_q, l4i_q, teacher_y = repl
        set_norm_col(int(row_idx), "Contrast", float(contrast[row_idx]))
        set_norm_col(int(row_idx), "L6", l6_q)
        set_norm_col(int(row_idx), "L4E", l4e_q)
        set_norm_col(int(row_idx), "L4I", l4i_q)
        yb_mod[int(row_idx)] = torch.tensor(float(teacher_y), dtype=yb_mod.dtype)
        meta_mod["L6"][int(row_idx)] = np.float32(l6_q)
        meta_mod["L4E"][int(row_idx)] = np.float32(l4e_q)
        meta_mod["L4I"][int(row_idx)] = np.float32(l4i_q)
        if "FR_ResponseVar" in meta_mod:
            meta_mod["FR_ResponseVar"][int(row_idx)] = np.float32(0.0)
        if "FR_ResponseN" in meta_mod:
            meta_mod["FR_ResponseN"][int(row_idx)] = np.float32(1.0)
    return xb_mod, yb_mod, meta_mod


def run_epoch(
    model,
    loader: SplitShardLoader,
    optimizer,
    scaler,
    device: torch.device,
    cfg: TrainConfig,
    grad_teacher: TrilinearTeacher3D | None,
    online_interp_teacher: OnlineTrilinearTeacher3D | None,
    feature_norm: dict,
    subloss_scales: dict[str, float] | None,
    batch_writer: csv.DictWriter | None,
    gradcos_writer: csv.DictWriter | None,
    global_step_start: int,
    total_steps_planned: int,
    uw_log_vars: torch.nn.Parameter | None,
    shadow_uw_log_vars: torch.nn.Parameter | None,
    shadow_uw_optimizer,
    weight_anchor_state: dict[str, torch.Tensor] | None = None,
    teacher_model=None,
) -> tuple[float, int, float, float, float, float, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_grad_loss = 0.0
    total_curvature_grad_scale = 0.0
    total_curvature_base_norm = 0.0
    total_curvature_regularizer_norm = 0.0
    total_curvature_applied_ratio = 0.0
    for xb, yb, meta in loader:
        if getattr(cfg, "online_interp_enable", False):
            if online_interp_teacher is None:
                raise RuntimeError("Online interpolation teacher is required when online_interp_enable is set")
            xb, yb, meta = apply_online_interp_augmentation(xb, yb.squeeze(-1), meta, cfg, online_interp_teacher, feature_norm)
            yb = yb.unsqueeze(-1)
        xb = xb.to(device=device, non_blocking=True)
        yb = yb.to(device=device, non_blocking=True).squeeze(-1)
        optimizer.zero_grad(set_to_none=True)
        batch_index = n_batches + 1
        global_step = int(global_step_start + batch_index)
        with device_autocast(device, cfg.use_amp):
            pred, terms = compute_loss_terms(model, xb, yb, meta, cfg, device, grad_teacher, feature_norm, teacher_model=teacher_model, apply_loss_exclusion=True)
            loss, normalized_terms, loss_diag = build_total_loss_from_terms(
                terms,
                cfg,
                subloss_scales if getattr(cfg, "subloss_norm_enable", False) else None,
                uw_log_vars=uw_log_vars,
                current_step=global_step,
                total_steps=total_steps_planned,
            )
            grad_loss = terms["grad_loss"]
            if float(getattr(cfg, "weight_anchor_lambda", 0.0)) > 0.0:
                loss = loss + float(cfg.weight_anchor_lambda) * model_weight_anchor_loss(model, weight_anchor_state, cfg)
        if shadow_uw_log_vars is not None and shadow_uw_optimizer is not None:
            shadow_aux_warmup = compute_aux_warmup_factor(cfg, global_step, total_steps_planned)
            shadow_main = (
                float(cfg.w_log) * terms["rmse_term"].detach()
                + float(cfg.w_rel) * terms["relrmse_raw"].detach()
                + float(getattr(cfg, "raw_mse_lambda", 0.0)) * terms["raw_mse"].detach()
            )
            shadow_aux = float(cfg.low_penalty_lambda) * terms["low_over_penalty"].detach()
            shadow_loss = (
                torch.exp(-shadow_uw_log_vars[0]) * shadow_main
                + shadow_uw_log_vars[0]
                + torch.exp(-shadow_uw_log_vars[1]) * (shadow_aux * float(shadow_aux_warmup))
                + shadow_uw_log_vars[1]
            )
            shadow_uw_optimizer.zero_grad(set_to_none=True)
            shadow_loss.backward()
            shadow_uw_optimizer.step()
            loss_diag = {
                "aux_warmup_factor": float(shadow_aux_warmup),
                "uw_s_main": float(shadow_uw_log_vars[0].detach().cpu().item()),
                "uw_s_aux": float(shadow_uw_log_vars[1].detach().cpu().item()),
                "uw_weight_main": float(torch.exp(-shadow_uw_log_vars[0]).detach().cpu().item()),
                "uw_weight_aux": float(torch.exp(-shadow_uw_log_vars[1]).detach().cpu().item()),
            }
        if batch_writer is not None:
            batch_writer.writerow(
                {
                    "global_step": global_step,
                    "batch_index": batch_index,
                    "raw_rmse_term": float(terms["rmse_term"].detach().cpu().item()),
                    "raw_raw_mse": float(terms["raw_mse"].detach().cpu().item()),
                    "raw_relrmse_raw": float(terms["relrmse_raw"].detach().cpu().item()),
                    "raw_low_over_penalty": float(terms["low_over_penalty"].detach().cpu().item()),
                    "raw_mono_pair_loss": float(terms["mono_pair_loss"].detach().cpu().item()),
                    "raw_mono_grad_loss": float(terms["mono_grad_loss"].detach().cpu().item()),
                    "raw_curvature_loss": float(terms["curvature_loss"].detach().cpu().item()),
                    "raw_corr_penalty": float(terms["corr_penalty"].detach().cpu().item()),
                    "raw_grad_loss": float(terms["grad_loss"].detach().cpu().item()),
                    "raw_adv_huber_loss": float(terms["adv_huber_loss"].detach().cpu().item()),
                    "raw_adv_smooth_loss": float(terms["adv_smooth_loss"].detach().cpu().item()),
                    "raw_guide83_loss": float(terms["guide83_loss"].detach().cpu().item()),
                    "raw_guide84_poisson_loss": float(terms["guide84_poisson_loss"].detach().cpu().item()),
                    "raw_tfsf_smooth_loss": float(terms["tfsf_smooth_loss"].detach().cpu().item()),
                    "raw_tf_unimodal_loss": float(terms["tf_unimodal_loss"].detach().cpu().item()),
                    "raw_sf_unimodal_loss": float(terms["sf_unimodal_loss"].detach().cpu().item()),
                    "raw_distill_loss": float(terms["distill_loss"].detach().cpu().item()),
                    "raw_main_task": float((float(cfg.w_log) * terms["rmse_term"] + float(cfg.w_rel) * terms["relrmse_raw"] + float(getattr(cfg, "raw_mse_lambda", 0.0)) * terms["raw_mse"]).detach().cpu().item()),
                    "raw_aux_task": float((float(cfg.low_penalty_lambda) * terms["low_over_penalty"]).detach().cpu().item()),
                    "norm_rmse_term": float(normalized_terms["rmse_term"].detach().cpu().item()),
                    "norm_raw_mse": float(normalized_terms["raw_mse"].detach().cpu().item()),
                    "norm_relrmse_raw": float(normalized_terms["relrmse_raw"].detach().cpu().item()),
                    "norm_low_over_penalty": float(normalized_terms["low_over_penalty"].detach().cpu().item()),
                    "norm_mono_pair_loss": float(normalized_terms["mono_pair_loss"].detach().cpu().item()),
                    "norm_mono_grad_loss": float(normalized_terms["mono_grad_loss"].detach().cpu().item()),
                    "norm_curvature_loss": float(normalized_terms["curvature_loss"].detach().cpu().item()),
                    "norm_corr_penalty": float(normalized_terms["corr_penalty"].detach().cpu().item()),
                    "norm_grad_loss": float(normalized_terms["grad_loss"].detach().cpu().item()),
                    "norm_adv_huber_loss": float(normalized_terms["adv_huber_loss"].detach().cpu().item()),
                    "norm_adv_smooth_loss": float(normalized_terms["adv_smooth_loss"].detach().cpu().item()),
                    "norm_guide83_loss": float(normalized_terms["guide83_loss"].detach().cpu().item()),
                    "norm_guide84_poisson_loss": float(normalized_terms["guide84_poisson_loss"].detach().cpu().item()),
                    "norm_tfsf_smooth_loss": float(normalized_terms["tfsf_smooth_loss"].detach().cpu().item()),
                    "norm_tf_unimodal_loss": float(normalized_terms["tf_unimodal_loss"].detach().cpu().item()),
                    "norm_sf_unimodal_loss": float(normalized_terms["sf_unimodal_loss"].detach().cpu().item()),
                    "norm_distill_loss": float(normalized_terms["distill_loss"].detach().cpu().item()),
                    "norm_main_task": float(normalized_terms["main_task"].detach().cpu().item()),
                    "norm_aux_task": float(normalized_terms["aux_task"].detach().cpu().item()),
                    "variance_weight_raw_mean": float(terms["variance_weight_raw_mean"].detach().cpu().item()),
                    "variance_weight_raw_min": float(terms["variance_weight_raw_min"].detach().cpu().item()),
                    "variance_weight_raw_max": float(terms["variance_weight_raw_max"].detach().cpu().item()),
                    "loss_exclusion_keep_frac": float(terms["loss_exclusion_keep_frac"].detach().cpu().item()),
                    "aux_warmup_factor": loss_diag["aux_warmup_factor"],
                    "uw_s_main": loss_diag["uw_s_main"],
                    "uw_s_aux": loss_diag["uw_s_aux"],
                    "uw_weight_main": loss_diag["uw_weight_main"],
                    "uw_weight_aux": loss_diag["uw_weight_aux"],
                    "train_total_loss": float(loss.detach().cpu().item()),
                }
            )
        if gradcos_writer is not None and getattr(cfg, "gradcos_enable", False) and int(getattr(cfg, "gradcos_every", 20)) > 0:
            if ((global_step - 1) % int(getattr(cfg, "gradcos_every", 20))) == 0:
                grad_coefficients = {}
                if active_tfsf_population_names(cfg):
                    active_grad_terms = list(active_tfsf_population_names(cfg))
                elif bool(getattr(cfg, "task_uw_enable", False)) or bool(getattr(cfg, "task_uw_shadow_enable", False)):
                    active_grad_terms = ["main_task", "aux_task"]
                    if float(cfg.mono_pair_lambda) > 0.0:
                        active_grad_terms.append("mono_pair_loss")
                    if float(getattr(cfg, "mono_grad_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("mono_grad_loss")
                    if float(getattr(cfg, "curvature_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("curvature_loss")
                    active_grad_terms.append("corr_penalty")
                    if float(cfg.grad_lambda) > 0.0:
                        active_grad_terms.append("grad_loss")
                    if float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("tfsf_smooth_loss")
                    if float(getattr(cfg, "tf_unimodal_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("tf_unimodal_loss")
                    if float(getattr(cfg, "sf_unimodal_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("sf_unimodal_loss")
                    if float(getattr(cfg, "distill_anchor_lambda", 0.0)) > 0.0:
                        active_grad_terms.append("distill_loss")
                else:
                    active_grad_terms = []
                    for name in ("rmse_term", "raw_mse", "relrmse_raw", "low_over_penalty", "mono_pair_loss", "mono_grad_loss", "curvature_loss", "corr_penalty", "grad_loss", "adv_huber_loss", "adv_smooth_loss", "guide83_loss", "guide84_poisson_loss", "tfsf_smooth_loss", "tf_unimodal_loss", "sf_unimodal_loss", "distill_loss"):
                        coeff = {
                            "rmse_term": float(cfg.w_log),
                            "raw_mse": float(getattr(cfg, "raw_mse_lambda", 0.0)),
                            "relrmse_raw": float(cfg.w_rel),
                            "low_over_penalty": float(cfg.low_penalty_lambda),
                            "mono_pair_loss": float(cfg.mono_pair_lambda),
                            "mono_grad_loss": float(getattr(cfg, "mono_grad_lambda", 0.0)),
                            "curvature_loss": float(getattr(cfg, "curvature_lambda", 0.0)),
                            "corr_penalty": 1.0,
                            "grad_loss": float(cfg.grad_lambda),
                            "adv_huber_loss": float(getattr(cfg, "adv_lambda", 0.0)),
                            "adv_smooth_loss": float(getattr(cfg, "adv_smooth_lambda", 0.0)),
                            "guide83_loss": 1.0 if str(getattr(cfg, "guide_uncertainty_mode", "none")).lower() == "guide83_count_weighted" else 0.0,
                            "guide84_poisson_loss": 1.0 if str(getattr(cfg, "guide_uncertainty_mode", "none")).lower() == "guide84_poisson" else 0.0,
                            "tfsf_smooth_loss": float(getattr(cfg, "tfsf_smooth_lambda", 0.0)),
                            "tf_unimodal_loss": float(getattr(cfg, "tf_unimodal_lambda", 0.0)),
                            "sf_unimodal_loss": float(getattr(cfg, "sf_unimodal_lambda", 0.0)),
                            "distill_loss": float(getattr(cfg, "distill_anchor_lambda", 0.0)),
                        }[name]
                        if coeff > 0.0:
                            active_grad_terms.append(name)
                            grad_coefficients[name] = coeff
                with device_autocast(device, cfg.use_amp):
                    _probe_pred, probe_terms = compute_loss_terms(model, xb, yb, meta, cfg, device, grad_teacher, feature_norm, teacher_model=teacher_model, apply_loss_exclusion=True)
                    _probe_loss, probe_normalized_terms, _probe_diag = build_total_loss_from_terms(
                        probe_terms,
                        cfg,
                        subloss_scales if getattr(cfg, "subloss_norm_enable", False) else None,
                        uw_log_vars=uw_log_vars,
                        current_step=global_step,
                        total_steps=total_steps_planned,
                    )
                gradcos_metrics = compute_gradcos_metrics(model, probe_normalized_terms, active_grad_terms, grad_coefficients)
                row = {"global_step": global_step, "batch_index": batch_index}
                row.update(gradcos_metrics)
                gradcos_writer.writerow(row)
        curvature_grad_diag = {
            "base_norm": 0.0,
            "regularizer_norm": 0.0,
            "applied_scale": 0.0,
            "applied_ratio": 0.0,
        }
        if bool(getattr(cfg, "task_cagrad_enable", False)):
            weight_anchor_term = pred.new_zeros(())
            if float(getattr(cfg, "weight_anchor_lambda", 0.0)) > 0.0:
                weight_anchor_term = float(cfg.weight_anchor_lambda) * model_weight_anchor_loss(model, weight_anchor_state, cfg)
            extra_term = (
                float(getattr(cfg, "raw_mse_lambda", 0.0)) * normalized_terms["raw_mse"]
                + float(cfg.low_penalty_lambda) * normalized_terms["low_over_penalty"]
                + float(cfg.mono_pair_lambda) * normalized_terms["mono_pair_loss"]
                + float(getattr(cfg, "mono_grad_lambda", 0.0)) * normalized_terms["mono_grad_loss"]
                + float(getattr(cfg, "curvature_lambda", 0.0)) * normalized_terms["curvature_loss"]
                + normalized_terms["corr_penalty"]
                + float(cfg.grad_lambda) * normalized_terms["grad_loss"]
                + float(getattr(cfg, "adv_lambda", 0.0)) * normalized_terms["adv_huber_loss"]
                + float(getattr(cfg, "adv_smooth_lambda", 0.0)) * normalized_terms["adv_smooth_loss"]
                + float(getattr(cfg, "tfsf_smooth_lambda", 0.0)) * normalized_terms["tfsf_smooth_loss"]
                + float(getattr(cfg, "tf_unimodal_lambda", 0.0)) * normalized_terms["tf_unimodal_loss"]
                + float(getattr(cfg, "sf_unimodal_lambda", 0.0)) * normalized_terms["sf_unimodal_loss"]
                + float(getattr(cfg, "distill_anchor_lambda", 0.0)) * normalized_terms["distill_loss"]
                + weight_anchor_term
            )
            apply_cagrad_multi_task_step(
                model,
                optimizer,
                [normalized_terms[name] for name in active_tfsf_population_names(cfg)],
                extra_term,
                alpha=float(getattr(cfg, "cagrad_alpha", 0.5)),
                rescale=int(getattr(cfg, "cagrad_rescale", 1)),
            )
        elif bool(getattr(cfg, "task_pcgrad_enable", False)):
            extra_term = (
                float(cfg.mono_pair_lambda) * normalized_terms["mono_pair_loss"]
                + float(getattr(cfg, "mono_grad_lambda", 0.0)) * normalized_terms["mono_grad_loss"]
                + float(getattr(cfg, "curvature_lambda", 0.0)) * normalized_terms["curvature_loss"]
                + normalized_terms["corr_penalty"]
                + float(cfg.grad_lambda) * normalized_terms["grad_loss"]
                + float(getattr(cfg, "tf_unimodal_lambda", 0.0)) * normalized_terms["tf_unimodal_loss"]
                + float(getattr(cfg, "sf_unimodal_lambda", 0.0)) * normalized_terms["sf_unimodal_loss"]
            )
            apply_pcgrad_two_task_step(
                model,
                optimizer,
                scaler,
                normalized_terms["main_task"],
                normalized_terms["aux_task"],
                extra_term,
            )
        elif (
            float(getattr(cfg, "curvature_grad_ratio", 0.0)) > 0.0
            and float(getattr(cfg, "curvature_lambda", 0.0)) > 0.0
        ):
            curvature_contribution = (
                float(cfg.curvature_lambda) * normalized_terms["curvature_loss"]
            )
            curvature_grad_diag = apply_gradient_ratio_regularized_step(
                model,
                optimizer,
                loss - curvature_contribution,
                curvature_contribution,
                float(cfg.curvature_grad_ratio),
            )
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.detach().cpu().item())
        total_grad_loss += float(grad_loss.detach().cpu().item())
        total_curvature_grad_scale += curvature_grad_diag["applied_scale"]
        total_curvature_base_norm += curvature_grad_diag["base_norm"]
        total_curvature_regularizer_norm += curvature_grad_diag["regularizer_norm"]
        total_curvature_applied_ratio += curvature_grad_diag["applied_ratio"]
        n_batches += 1
    denom = max(n_batches, 1)
    return (
        total_loss / denom,
        n_batches,
        total_grad_loss / denom,
        total_curvature_grad_scale / denom,
        total_curvature_base_norm / denom,
        total_curvature_regularizer_norm / denom,
        total_curvature_applied_ratio / denom,
    )


def evaluate(model, loader: SplitShardLoader, device: torch.device, cfg: TrainConfig) -> dict:
    model.eval()
    acc = MetricAccumulator(
        cfg.loss_eps,
        cfg.rel_alpha,
        cfg.rmse_space,
        cfg.w_log,
        cfg.w_rel,
        getattr(cfg, "raw_mse_lambda", 0.0),
        cfg.low_penalty_lambda,
        cfg.low_penalty_tau,
        cfg.mono_pair_lambda,
        getattr(cfg, "loss_mode", "base"),
        getattr(cfg, "mid_under_lambda", 0.0),
        getattr(cfg, "edge_over_lambda", 0.0),
        getattr(cfg, "corr_lambda", 0.0),
        getattr(cfg, "corr_w_under_mid0", 1.0),
        getattr(cfg, "corr_w_over_edge225", 1.0),
    )
    bad_pair_stats: dict[tuple[float, float], list[float]] | None = None
    if getattr(cfg, "ckpt_bad_subset_enable", False):
        bad_pair_stats = {pair: [0.0, 0.0] for pair in BAD_HC_SELECTION_PAIR_SET}
    band_acc = RateBandAccumulator(cfg.rel_alpha, cfg.loss_eps)
    weighted_acc = WeightedLossAccumulator(cfg) if variance_weight_enabled(cfg) else None
    original5d_acc = None
    focus_acc = None
    focus_group_accs = None
    if tuple(getattr(cfg, "population_focus_conditions", ()) or ()):
        focus_acc = MetricAccumulator(
            cfg.loss_eps,
            cfg.rel_alpha,
            cfg.rmse_space,
            cfg.w_log,
            cfg.w_rel,
            getattr(cfg, "raw_mse_lambda", 0.0),
            cfg.low_penalty_lambda,
            cfg.low_penalty_tau,
            0.0,
            "base",
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
        )
        if float(getattr(cfg, "population_focus_group_cvar_fraction", 0.0)) > 0.0:
            focus_group_accs = [
                MetricAccumulator(
                    cfg.loss_eps,
                    cfg.rel_alpha,
                    cfg.rmse_space,
                    cfg.w_log,
                    cfg.w_rel,
                    0.0,
                    0.0,
                    cfg.low_penalty_tau,
                    0.0,
                    "base",
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                )
                for _ in cfg.population_focus_conditions
            ]
    if str(getattr(cfg, "feature_mode", "")).lower() == "tf_sf7d":
        original5d_acc = MetricAccumulator(
            cfg.loss_eps,
            cfg.rel_alpha,
            cfg.rmse_space,
            cfg.w_log,
            cfg.w_rel,
            getattr(cfg, "raw_mse_lambda", 0.0),
            cfg.low_penalty_lambda,
            cfg.low_penalty_tau,
            cfg.mono_pair_lambda,
            getattr(cfg, "loss_mode", "base"),
            getattr(cfg, "mid_under_lambda", 0.0),
            getattr(cfg, "edge_over_lambda", 0.0),
            getattr(cfg, "corr_lambda", 0.0),
            getattr(cfg, "corr_w_under_mid0", 1.0),
            getattr(cfg, "corr_w_over_edge225", 1.0),
        )
    with torch.no_grad():
        for xb, yb_raw, meta in loader:
            xb = xb.to(device=device, non_blocking=True)
            with device_autocast(device, cfg.use_amp):
                pred_raw = model(xb)
            mono_pair_loss = 0.0
            mid_under_penalty = 0.0
            edge_over_penalty = 0.0
            if getattr(cfg, "loss_mode", "base") == "targeted_mid_edge_asym":
                _, mid_under_t, edge_over_t = targeted_mid_edge_penalty(pred_raw, torch.as_tensor(yb_raw.numpy(), device=device).squeeze(-1), meta, cfg, device)
                mid_under_penalty = float(mid_under_t.detach().cpu().item())
                edge_over_penalty = float(edge_over_t.detach().cpu().item())
            elif getattr(cfg, "loss_mode", "base") == "legacy_gamma_contrast_asym":
                _, mid_under_t, edge_over_t = legacy_gamma_contrast_asym_penalty(pred_raw, torch.as_tensor(yb_raw.numpy(), device=device).squeeze(-1), meta, cfg, device)
                mid_under_penalty = float(mid_under_t.detach().cpu().item())
                edge_over_penalty = float(edge_over_t.detach().cpu().item())
            if cfg.mono_pair_lambda > 0:
                mono_pair_loss = float(mono_pair_loss_from_meta(pred_raw, meta, device).detach().cpu().item())
            y_true_np = yb_raw.numpy().astype(np.float64)
            y_pred_np = clamp_raw_rate_array(pred_raw.detach().cpu().numpy().astype(np.float64))
            acc.update(
                y_true_np,
                y_pred_np,
                mono_pair_loss=mono_pair_loss,
                mid_under_penalty=mid_under_penalty,
                edge_over_penalty=edge_over_penalty,
            )
            if original5d_acc is not None:
                # The loader consumes TF/SF as active features, so they are not
                # duplicated in meta. This split centers TF=10 and SF=2.5 at 0.
                xb_np = xb.detach().cpu().numpy()
                original5d_mask = np.isclose(xb_np[:, 5], 0.0) & np.isclose(xb_np[:, 6], 0.0)
                if np.any(original5d_mask):
                    original5d_acc.update(
                        y_true_np.reshape(-1)[original5d_mask],
                        y_pred_np.reshape(-1)[original5d_mask],
                    )
            if focus_acc is not None:
                focus_mask = population_focus_mask_numpy(meta, cfg, y_true_np.size)
                if np.any(focus_mask):
                    focus_acc.update(
                        y_true_np.reshape(-1)[focus_mask],
                        y_pred_np.reshape(-1)[focus_mask],
                    )
                if focus_group_accs is not None:
                    for condition, group_acc in zip(cfg.population_focus_conditions, focus_group_accs):
                        group_mask = population_focus_condition_mask_numpy(meta, condition, y_true_np.size)
                        if np.any(group_mask):
                            group_acc.update(
                                y_true_np.reshape(-1)[group_mask],
                                y_pred_np.reshape(-1)[group_mask],
                            )
            band_acc.update(y_true_np, y_pred_np)
            if weighted_acc is not None:
                weighted_acc.update(y_true_np, y_pred_np, meta)
            if bad_pair_stats is not None:
                gamma = np.asarray(meta["Gamma"], dtype=np.float32).reshape(-1)
                contrast = np.asarray(meta["Contrast"], dtype=np.float32).reshape(-1)
                sq_err = np.square(y_true_np.reshape(-1) - y_pred_np.reshape(-1))
                for pair in BAD_HC_SELECTION_PAIR_SET:
                    mask = np.isclose(gamma, pair[0]) & np.isclose(contrast, pair[1])
                    if np.any(mask):
                        stats = bad_pair_stats[pair]
                        stats[0] += float(sq_err[mask].sum())
                        stats[1] += float(mask.sum())
    metrics = acc.finalize()
    if original5d_acc is not None and original5d_acc.n > 0:
        original5d_metrics = original5d_acc.finalize()
        metrics["original5d_n"] = int(original5d_metrics["n"])
        metrics["original5d_rmse_raw"] = float(original5d_metrics["rmse_raw"])
        metrics["original5d_rmse_log"] = float(original5d_metrics["rmse_log"])
        metrics["original5d_relrmse_raw"] = float(original5d_metrics["relrmse_raw"])
    else:
        metrics["original5d_n"] = 0
        metrics["original5d_rmse_raw"] = 0.0
        metrics["original5d_rmse_log"] = 0.0
        metrics["original5d_relrmse_raw"] = 0.0
    if focus_acc is not None and focus_acc.n > 0:
        focus_metrics = focus_acc.finalize()
        metrics["population_focus_n"] = int(focus_metrics["n"])
        metrics["population_focus_total_loss"] = float(focus_metrics["total_loss"])
        metrics["population_focus_rmse_raw"] = float(focus_metrics["rmse_raw"])
        metrics["population_focus_rmse_log"] = float(focus_metrics["rmse_log"])
        metrics["population_focus_relrmse_raw"] = float(focus_metrics["relrmse_raw"])
        if focus_group_accs is not None:
            group_losses = []
            raw_scale = float(getattr(cfg, "population_focus_raw_scale", 0.0))
            if raw_scale <= 0.0:
                raw_scale = {"S": 30.0, "C": 100.0, "I": 140.0}[str(cfg.target)]
            for group_acc in focus_group_accs:
                if group_acc.n <= 0:
                    continue
                group_metrics = group_acc.finalize()
                group_loss = float(group_metrics["total_loss"])
                group_loss += (
                    float(getattr(cfg, "population_focus_raw_lambda", 0.0))
                    * float(group_metrics["rmse_raw"])
                    / raw_scale
                )
                group_losses.append(group_loss)
            if group_losses:
                fraction = min(float(cfg.population_focus_group_cvar_fraction), 1.0)
                top_count = max(1, int(math.ceil(fraction * len(group_losses))))
                metrics["population_focus_group_cvar_loss"] = float(
                    np.mean(np.sort(np.asarray(group_losses, dtype=np.float64))[-top_count:])
                )
                metrics["population_focus_group_count"] = int(len(group_losses))
    else:
        metrics["population_focus_n"] = 0
        metrics["population_focus_total_loss"] = 0.0
        metrics["population_focus_rmse_raw"] = 0.0
        metrics["population_focus_rmse_log"] = 0.0
        metrics["population_focus_relrmse_raw"] = 0.0
    metrics["rate_bands"] = band_acc.finalize()
    if weighted_acc is not None:
        metrics.update(weighted_acc.finalize())
    if bad_pair_stats is not None:
        pair_rmses = [float(np.sqrt(sum_sq / count + cfg.loss_eps)) for sum_sq, count in bad_pair_stats.values() if count > 0]
        if pair_rmses:
            metrics["bad_subset_mean_rmse_raw"] = float(np.mean(pair_rmses))
            metrics["bad_subset_max_rmse_raw"] = float(np.max(pair_rmses))
            metrics["bad_subset_n_pairs"] = int(len(pair_rmses))
        else:
            metrics["bad_subset_mean_rmse_raw"] = 0.0
            metrics["bad_subset_max_rmse_raw"] = 0.0
            metrics["bad_subset_n_pairs"] = 0
    else:
        metrics["bad_subset_mean_rmse_raw"] = 0.0
        metrics["bad_subset_max_rmse_raw"] = 0.0
        metrics["bad_subset_n_pairs"] = 0
    return metrics


def checkpoint_selection_score(val_metrics: dict, cfg: TrainConfig) -> float:
    score = float(val_metrics.get("weighted_total_loss", val_metrics["total_loss"])) if variance_weight_enabled(cfg) else float(val_metrics["total_loss"])
    focus_metric = val_metrics.get("population_focus_total_loss", 0.0)
    if float(getattr(cfg, "population_focus_group_cvar_fraction", 0.0)) > 0.0:
        focus_metric = val_metrics.get("population_focus_group_cvar_loss", focus_metric)
    score += float(getattr(cfg, "ckpt_population_focus_beta", 0.0)) * float(focus_metric)
    if getattr(cfg, "ckpt_bad_subset_enable", False):
        score += float(getattr(cfg, "ckpt_bad_subset_beta", 1.0)) * float(val_metrics.get("bad_subset_mean_rmse_raw", 0.0))
        score += float(getattr(cfg, "ckpt_bad_subset_gamma", 0.5)) * float(val_metrics.get("bad_subset_max_rmse_raw", 0.0))
    return score


def save_checkpoint(path: Path, model, model_cfg: ModelConfig, train_cfg: TrainConfig, feature_norm: dict, extra: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": to_serializable_config(model_cfg),
            "train_config": to_serializable_config(train_cfg),
            "feature_norm": feature_norm,
            "extra": extra,
        },
        path,
    )


def set_optimizer_lr(optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr_value)


def checkpoint_model_state(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint at {path} does not contain a model state dict")
    return {str(k): v.detach().to(device=device) for k, v in state.items() if torch.is_tensor(v)}


def load_init_checkpoint(model, path: str | Path, device: torch.device) -> None:
    state = checkpoint_model_state(path, device)
    model.load_state_dict(state, strict=True)


def first_linear_layer(model) -> torch.nn.Linear:
    net = getattr(model, "net", None)
    if net is None:
        raise RuntimeError("Stage-freeze modes require a single_mlp model with .net")
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            return layer
    raise RuntimeError("Could not find first Linear layer for stage-freeze mode")


def train_stage_mode_for_index(cfg: TrainConfig, stage_index: int) -> str:
    modes = tuple(getattr(cfg, "train_stage_modes", ()) or ())
    if not modes:
        return "all"
    return str(modes[min(max(stage_index - 1, 0), len(modes) - 1)]).lower()


def apply_train_stage_mode(model, cfg: TrainConfig, stage_mode: str) -> tuple[list, float]:
    mode = str(stage_mode).lower()
    handles: list = []
    for param in model.parameters():
        param.requires_grad_(False)
    if mode == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return handles, float(cfg.weight_decay)
    first = first_linear_layer(model)
    if mode == "first":
        for param in first.parameters():
            param.requires_grad_(True)
        return handles, float(cfg.weight_decay)
    if mode == "newcols":
        if first.weight.shape[1] <= 5:
            raise RuntimeError("newcols stage requires an input layer wider than the original 5D inputs")
        first.weight.requires_grad_(True)
        mask = torch.zeros_like(first.weight)
        mask[:, 5:] = 1.0
        handles.append(first.weight.register_hook(lambda grad, mask=mask: grad * mask))
        return handles, 0.0
    raise RuntimeError(f"Unknown train stage mode {stage_mode!r}")


def make_stage_optimizer(model, cfg: TrainConfig, lr_value: float, uw_log_vars: torch.nn.Parameter | None, weight_decay: float):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable model parameters for this stage")
    groups: list[dict[str, object]] = [{"params": params}]
    if uw_log_vars is not None:
        groups.append({"params": [uw_log_vars], "weight_decay": 0.0})
    return optim.AdamW(groups, lr=float(lr_value), weight_decay=float(weight_decay))


def model_weight_anchor_loss(model, anchor_state: dict[str, torch.Tensor] | None, cfg: TrainConfig) -> torch.Tensor:
    params = list(model.parameters())
    if not params:
        raise RuntimeError("Model has no parameters")
    out = params[0].new_zeros(())
    if anchor_state is None or float(getattr(cfg, "weight_anchor_lambda", 0.0)) <= 0.0:
        return out
    terms = []
    include_new = bool(getattr(cfg, "weight_anchor_include_new_inputs", False))
    for name, param in model.named_parameters():
        anchor = anchor_state.get(name)
        if anchor is None or tuple(anchor.shape) != tuple(param.shape):
            continue
        anchor = anchor.to(device=param.device, dtype=param.dtype)
        if name == "net.0.weight" and param.ndim == 2 and param.shape[1] > 5 and not include_new:
            terms.append(torch.mean(torch.square(param[:, :5] - anchor[:, :5])))
        else:
            terms.append(torch.mean(torch.square(param - anchor)))
    if not terms:
        return out
    return torch.stack(terms).mean()


def stage_improve(val_total_history: list[float]) -> float:
    if not val_total_history:
        return 0.0
    win = max(1, len(val_total_history) // 2)
    head = float(np.mean(val_total_history[:win]))
    tail = float(np.mean(val_total_history[-win:]))
    if head <= 0:
        return 0.0
    return (head - tail) / head


def collect_diag_sample(model, loader: SplitShardLoader, device: torch.device, cfg: TrainConfig, stage_index: int) -> dict[str, np.ndarray]:
    model.eval()
    xb_cpu, yb_cpu, meta = next(loader.iter_sample(cfg.diag_sample_size, seed=cfg.seed + 1000 + stage_index))
    with torch.no_grad():
        xb = xb_cpu.to(device=device, non_blocking=True)
        with device_autocast(device, cfg.use_amp):
            pred = model(xb)
    y_true = yb_cpu.numpy().astype(np.float64).reshape(-1)
    y_pred = clamp_raw_rate_array(pred.detach().cpu().numpy().astype(np.float64).reshape(-1))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "Gamma": meta["Gamma"].astype(np.float64, copy=False),
        "Contrast": meta["Contrast"].astype(np.float64, copy=False),
        "LGNctgr": meta["LGNctgr"].astype(np.int64, copy=False),
    }


def save_stage_diagnostics(out_dir: Path, stage_index: int, sample: dict[str, np.ndarray], target: str) -> None:
    y_true = sample["y_true"]
    y_pred = sample["y_pred"]
    residual = sample["residual"]
    gamma = sample["Gamma"]
    contrast = sample["Contrast"]
    lgn = sample["LGNctgr"]

    finite_mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(residual)
        & np.isfinite(gamma)
        & np.isfinite(contrast)
        & np.isfinite(lgn)
    )
    if not np.any(finite_mask):
        return
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]
    residual = residual[finite_mask]
    gamma = gamma[finite_mask]
    contrast = contrast[finite_mask]
    lgn = lgn[finite_mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    lim_max = float(max(np.quantile(y_true, 0.995), np.quantile(y_pred, 0.995), 1.0))
    hb = ax.hexbin(y_true, y_pred, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    ax.plot([0, lim_max], [0, lim_max], color="red", linewidth=1.5)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("Pred vs True")
    fig.colorbar(hb, ax=ax, label="count")

    ax = axes[0, 1]
    ax.scatter(gamma, residual, s=3, alpha=0.08, rasterized=True)
    gx, gy = diag_line(gamma, residual, n_bins=24)
    if gx.size:
        ax.plot(gx, gy, color="red", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Gamma")

    ax = axes[1, 0]
    unique_contrast = np.unique(contrast)
    contrast_data = [residual[contrast == c] for c in unique_contrast]
    ax.boxplot(contrast_data, positions=np.arange(len(unique_contrast)), widths=0.6, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(unique_contrast)))
    ax.set_xticklabels([f"{c:g}" for c in unique_contrast], rotation=45, ha="right")
    ax.set_xlabel("Contrast")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs Contrast")

    ax = axes[1, 1]
    unique_lgn = np.unique(lgn)
    lgn_data = [residual[lgn == c] for c in unique_lgn]
    ax.boxplot(lgn_data, positions=np.arange(len(unique_lgn)), widths=0.6, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(unique_lgn)))
    ax.set_xticklabels([str(int(c)) for c in unique_lgn])
    ax.set_xlabel("LGNctgr")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs LGNctgr")

    fig.suptitle(f"{target} Stage {stage_index} Val Diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / f"stage_{stage_index:02d}_val_diagnostics.png", dpi=180)
    plt.close(fig)

    rows = []
    for split_name, values in (
        ("gamma_bin", gamma),
        ("contrast", contrast),
        ("LGNctgr", lgn.astype(np.float64)),
    ):
        if split_name == "gamma_bin":
            bins = np.round(np.linspace(0.0, 90.0, 13), 2)
            idx = np.digitize(values, bins[1:-1], right=False)
            labels = bins[np.clip(idx, 0, len(bins) - 1)]
        else:
            labels = values
        for key in np.unique(labels):
            mask = labels == key
            rows.append(
                {
                    "group_name": split_name,
                    "group_value": float(key),
                    "count": int(np.sum(mask)),
                    "mean_residual": float(np.mean(residual[mask])),
                    "mae": float(np.mean(np.abs(residual[mask]))),
                }
            )
    diag_csv = out_dir / f"stage_{stage_index:02d}_val_diagnostics.csv"
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_name", "group_value", "count", "mean_residual", "mae"])
        writer.writeheader()
        writer.writerows(rows)


def save_stage_diagnostics_at_step(
    out_dir: Path,
    stage_index: int,
    global_step: int,
    sample: dict[str, np.ndarray],
    target: str,
) -> None:
    save_stage_diagnostics(out_dir, stage_index, sample, target)

    png_src = out_dir / f"stage_{stage_index:02d}_val_diagnostics.png"
    csv_src = out_dir / f"stage_{stage_index:02d}_val_diagnostics.csv"
    png_dst = out_dir / f"step_{global_step:05d}_val_diagnostics.png"
    csv_dst = out_dir / f"step_{global_step:05d}_val_diagnostics.csv"
    png_dst.write_bytes(png_src.read_bytes())
    csv_dst.write_bytes(csv_src.read_bytes())


def save_loss_curve(out_dir: Path, log_path: Path, target: str) -> None:
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    if data.size == 0:
        return
    if data.ndim == 0:
        steps = np.asarray([data["global_step"]], dtype=np.float64) if "global_step" in data.dtype.names else np.asarray([data["epoch"]], dtype=np.float64)
        train_loss = np.asarray([data["train_loss"]], dtype=np.float64)
        val_total = np.asarray([data["val_total_loss"]], dtype=np.float64)
    else:
        steps = data["global_step"] if "global_step" in data.dtype.names else data["epoch"]
        train_loss = data["train_loss"]
        val_total = data["val_total_loss"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_loss, label="train_loss")
    ax.plot(steps, val_total, label="val_total_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{target} Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=180)
    plt.close(fig)


def save_subloss_curves(out_dir: Path, log_path: Path, target: str) -> None:
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    if data.size == 0:
        return
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    steps = data["global_step"].astype(np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    raw_names = [
        "raw_rmse_term",
        "raw_raw_mse",
        "raw_relrmse_raw",
        "raw_low_over_penalty",
        "raw_mono_pair_loss",
        "raw_corr_penalty",
        "raw_grad_loss",
        "raw_tfsf_smooth_loss",
        "raw_tf_unimodal_loss",
        "raw_sf_unimodal_loss",
        "raw_distill_loss",
        "raw_main_task",
        "raw_aux_task",
    ]
    norm_names = [
        "norm_rmse_term",
        "norm_raw_mse",
        "norm_relrmse_raw",
        "norm_low_over_penalty",
        "norm_mono_pair_loss",
        "norm_corr_penalty",
        "norm_grad_loss",
        "norm_tfsf_smooth_loss",
        "norm_tf_unimodal_loss",
        "norm_sf_unimodal_loss",
        "norm_distill_loss",
        "norm_main_task",
        "norm_aux_task",
    ]
    for name in raw_names:
        if name in data.dtype.names:
            axes[0].plot(steps, data[name], label=name)
    axes[0].set_title(f"{target} Raw Subloss Curves")
    axes[0].set_ylabel("Raw value")
    axes[0].legend(fontsize=8)
    for name in norm_names:
        if name in data.dtype.names:
            axes[1].plot(steps, data[name], label=name)
    axes[1].set_title(f"{target} Normalized Subloss Curves")
    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "subloss_curves.png", dpi=180)
    plt.close(fig)


def save_gradcos_curves(out_dir: Path, log_path: Path, target: str) -> None:
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    if data.size == 0:
        return
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    pair_names = [name for name in data.dtype.names if name not in ("global_step", "batch_index")]
    if not pair_names:
        return
    steps = data["global_step"].astype(np.float64)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in pair_names:
        ax.plot(steps, data[name], label=name)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"{target} Subloss Gradient Cosines")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "gradcos_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    if bool(getattr(cfg, "task_pcgrad_enable", False)) and bool(getattr(cfg, "task_uw_enable", False)):
        raise ValueError("task_pcgrad_enable does not support task_uw_enable; use shadow UW or disable UW")
    if bool(getattr(cfg, "task_cagrad_enable", False)) and bool(getattr(cfg, "task_uw_enable", False)):
        raise ValueError("task_cagrad_enable does not support task_uw_enable")
    if bool(getattr(cfg, "task_cagrad_enable", False)) and bool(getattr(cfg, "task_pcgrad_enable", False)):
        raise ValueError("Enable only one of task_cagrad_enable and task_pcgrad_enable")
    if float(getattr(cfg, "curvature_grad_ratio", 0.0)) > 0.0 and (
        bool(getattr(cfg, "task_cagrad_enable", False))
        or bool(getattr(cfg, "task_pcgrad_enable", False))
    ):
        raise ValueError("curvature_grad_ratio cannot be combined with task CAGrad or PCGrad")
    if bool(getattr(cfg, "task_cagrad_enable", False)) and not active_tfsf_population_names(cfg):
        raise ValueError("task_cagrad_enable requires a TF/SF population gradient mode")
    if str(getattr(cfg, "population_grad_mode", "none")).lower() == "tfsf4_focus" and not tuple(getattr(cfg, "population_focus_conditions", ()) or ()):
        raise ValueError("--population_grad_mode tfsf4_focus requires --population_focus_condition_file")
    if float(getattr(cfg, "cagrad_alpha", 0.5)) < 0.0:
        raise ValueError("--cagrad_alpha must be nonnegative")
    if bool(getattr(cfg, "task_pcgrad_enable", False)) and bool(getattr(cfg, "use_amp", False)):
        cfg.use_amp = False
    if bool(getattr(cfg, "task_cagrad_enable", False)) and bool(getattr(cfg, "use_amp", False)):
        cfg.use_amp = False
    if float(getattr(cfg, "curvature_grad_ratio", 0.0)) > 0.0 and bool(getattr(cfg, "use_amp", False)):
        cfg.use_amp = False
    out_dir = ensure_dir(cfg.out_dir)
    manifest, feature_norm = load_split_metadata(cfg.split_dir)
    variance_refs = estimate_variance_references(cfg, manifest)
    cfg.variance_has_fields = bool(variance_refs["has_fields"])
    cfg.variance_ref_raw = float(variance_refs["ref_raw"])
    cfg.variance_ref_log = float(variance_refs["ref_log"])
    cfg.variance_ref_rel = float(variance_refs["ref_rel"])
    if variance_weight_enabled(cfg):
        save_json(
            out_dir / "variance_weighting.json",
            {
                "mode": cfg.variance_weight_mode,
                "min_weight": cfg.variance_min_weight,
                "ref_quantile": cfg.variance_ref_quantile,
                "eps": cfg.variance_eps,
                "has_fields": cfg.variance_has_fields,
                "ref_raw": cfg.variance_ref_raw,
                "ref_log": cfg.variance_ref_log,
                "ref_rel": cfg.variance_ref_rel,
                "positive_sample_count": int(variance_refs["positive_sample_count"]),
            },
        )
        print(
            "[variance] "
            f"mode={cfg.variance_weight_mode} has_fields={cfg.variance_has_fields} "
            f"refs raw/log/rel={cfg.variance_ref_raw:.6g}/{cfg.variance_ref_log:.6g}/{cfg.variance_ref_rel:.6g}"
        )
    device = torch.device(cfg.device)

    model_family = getattr(cfg, "model_family", "single_mlp")
    softplus_raw_family = model_family in ("single_mlp", "threshold_additive")
    model_cfg = ModelConfig(
        target=cfg.target,
        model_family=model_family,
        feature_mode=cfg.feature_mode,
        input_dim=input_dim_for_mode(cfg.feature_mode),
        hidden_dims=tuple(cfg.hidden_dims),
        activation_seq=tuple(cfg.activation_seq),
        shape_hidden_dims=tuple(getattr(cfg, "shape_hidden_dims", ())),
        shape_activation_seq=tuple(getattr(cfg, "shape_activation_seq", ())),
        contrast_hidden_dims=tuple(getattr(cfg, "contrast_hidden_dims", ())),
        contrast_activation_seq=tuple(getattr(cfg, "contrast_activation_seq", ())),
        softplus_out=softplus_raw_family,
        output_semantics=(
            "raw_rate_softplus"
            if softplus_raw_family
            else "raw_rate_expm1_sum_heads"
        ),
    )
    from common import constant_baseline_rule
    model_cfg.constant_baseline_rule = constant_baseline_rule(feature_norm, cfg.target)
    model = build_model(model_cfg).to(device)
    if str(getattr(cfg, "init_checkpoint", "") or ""):
        load_init_checkpoint(model, cfg.init_checkpoint, device)
        print(f"[init] loaded {cfg.init_checkpoint}")
    teacher_model = None
    if float(getattr(cfg, "distill_anchor_lambda", 0.0)) > 0.0:
        anchor_path = str(getattr(cfg, "weight_anchor_checkpoint", "") or getattr(cfg, "init_checkpoint", "") or "")
        if not anchor_path:
            raise RuntimeError("--distill_anchor_lambda > 0 requires --weight_anchor_checkpoint or --init_checkpoint")
        teacher_model = build_model(model_cfg).to(device)
        load_init_checkpoint(teacher_model, anchor_path, device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        print(
            f"[distill] teacher={anchor_path} lambda={float(cfg.distill_anchor_lambda):.6g} "
            f"mode={str(cfg.distill_anchor_mode)} default_tf={float(cfg.distill_default_tf):g} "
            f"default_sf={float(cfg.distill_default_sf):g} "
            f"max_tf_exclusive={float(cfg.distill_max_tf_exclusive):g}"
        )
    weight_anchor_state = None
    if float(getattr(cfg, "weight_anchor_lambda", 0.0)) > 0.0:
        anchor_path = str(getattr(cfg, "weight_anchor_checkpoint", "") or getattr(cfg, "init_checkpoint", "") or "")
        if not anchor_path:
            raise RuntimeError("--weight_anchor_lambda > 0 requires --weight_anchor_checkpoint or --init_checkpoint")
        weight_anchor_state = checkpoint_model_state(anchor_path, device)
        print(f"[anchor] loaded {anchor_path} lambda={float(cfg.weight_anchor_lambda):.6g}")
    uw_log_vars = None
    if bool(getattr(cfg, "task_uw_enable", False)):
        uw_log_vars = torch.nn.Parameter(torch.zeros(2, device=device, dtype=torch.float32))
    shadow_uw_log_vars = None
    shadow_uw_optimizer = None
    if bool(getattr(cfg, "task_uw_shadow_enable", False)):
        shadow_uw_log_vars = torch.nn.Parameter(torch.zeros(2, device=device, dtype=torch.float32))
        shadow_uw_optimizer = optim.AdamW([shadow_uw_log_vars], lr=float(getattr(cfg, "task_uw_shadow_lr", 1e-2)), weight_decay=0.0)
    optimizer = None
    stage_hook_handles: list = []
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))
    grad_teacher = None
    if cfg.grad_lambda > 0.0:
        grad_teacher = TrilinearTeacher3D(cfg.train_mat)
    online_interp_teacher = None
    if getattr(cfg, "online_interp_enable", False):
        online_interp_teacher = OnlineTrilinearTeacher3D(
            cfg.train_mat,
            cfg.target,
            interp_eps=float(getattr(cfg, "online_interp_eps", 0.05)),
            badslab_only=bool(getattr(cfg, "online_interp_badslab_only", True)),
            cache_slabs=int(getattr(cfg, "online_interp_cache_slabs", 8)),
        )

    train_loader = make_loader(cfg, "train", manifest, feature_norm, shuffle=True)
    val_loader = make_loader(cfg, "val", manifest, feature_norm, shuffle=False)
    test_loader = make_loader(cfg, "test", manifest, feature_norm, shuffle=False)
    val_diag_loader = make_sample_eval_loader(cfg, "val", manifest, feature_norm, max_batches=1)

    initial_val_metrics = evaluate(model, val_loader, device, cfg)
    initial_original5d_rmse = float(initial_val_metrics.get("original5d_rmse_raw", 0.0))
    initial_global_curvature_penalty = deterministic_global_curvature_penalty(
        model, cfg, feature_norm, device
    )
    initial_val_metrics["global_curvature_penalty"] = initial_global_curvature_penalty
    save_json(out_dir / "initial_validation_metrics.json", initial_val_metrics)
    print(
        f"[initial validation] full_rmse_raw={float(initial_val_metrics['rmse_raw']):.6f} "
        f"original5d_rmse_raw={initial_original5d_rmse:.6f} "
        f"original5d_n={int(initial_val_metrics.get('original5d_n', 0))}"
    )

    subloss_scales = default_subloss_scales()
    if getattr(cfg, "subloss_norm_enable", False):
        subloss_scales = estimate_initial_subloss_scales(
            model,
            train_loader,
            device,
            cfg,
            grad_teacher,
            online_interp_teacher,
            feature_norm,
            init_batches=max(1, int(getattr(cfg, "subloss_norm_init_batches", 8))),
        )
    save_json(out_dir / "subloss_scales.json", subloss_scales)

    log_path = out_dir / "train_log.csv"
    stage_summary_path = out_dir / "stage_summary.csv"
    subloss_log_path = out_dir / "subloss_batch_log.csv"
    gradcos_log_path = out_dir / "gradcos_log.csv"
    gradcos_fieldnames = ["global_step", "batch_index"]
    if bool(getattr(cfg, "task_uw_enable", False)) or bool(getattr(cfg, "task_uw_shadow_enable", False)):
        grad_components = ["main_task", "aux_task", "mono_pair_loss", "mono_grad_loss", "curvature_loss", "corr_penalty", "grad_loss", "adv_huber_loss", "adv_smooth_loss", "guide83_loss", "guide84_poisson_loss"]
    else:
        grad_components = ["rmse_term", "raw_mse", "relrmse_raw", "low_over_penalty", "mono_pair_loss", "mono_grad_loss", "curvature_loss", "corr_penalty", "grad_loss", "adv_huber_loss", "adv_smooth_loss", "guide83_loss", "guide84_poisson_loss"]
    grad_components.extend(["tfsf_smooth_loss", "tf_unimodal_loss", "sf_unimodal_loss", "distill_loss"])
    grad_components.extend(ALL_TFSF_POPULATION_NAMES)
    gradcos_fieldnames.extend(f"{name}__{suffix}" for name in grad_components for suffix in ("grad_norm", "weighted_grad_norm"))
    gradcos_fieldnames.extend([f"{left}__vs__{right}" for left, right in combinations(grad_components, 2)])
    total_steps_planned = int(max(1, cfg.max_stages * cfg.stage_epochs * max(1, cfg.max_train_batches)))

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "global_step",
                "stage_index",
                "stage_epoch",
                "lr",
                "train_loss",
                "val_total_loss",
                "val_weighted_total_loss",
                "val_rmse_log",
                "val_rmse_term",
                "val_rmse_raw",
                "val_mse_raw",
                "val_relrmse_raw",
                "val_weighted_rmse_log",
                "val_weighted_rmse_raw",
                "val_weighted_mse_raw",
                "val_weighted_relrmse_raw",
                "val_low_over_penalty",
                "val_mono_pair_loss",
                "val_bad_subset_mean_rmse_raw",
                "val_bad_subset_max_rmse_raw",
                "val_original5d_n",
                "val_original5d_rmse_raw",
                "val_original5d_rmse_delta",
                "val_original5d_rmse_ratio",
                "val_population_focus_n",
                "val_population_focus_total_loss",
                "val_population_focus_rmse_raw",
                "ckpt_selection_score",
                "val_mae_raw",
                "val_r2_raw",
                "train_grad_loss",
                "train_curvature_grad_scale",
                "train_curvature_base_update_norm",
                "train_curvature_regularizer_update_norm",
                "train_curvature_applied_update_ratio",
                "val_global_curvature_penalty",
                "uw_s_main",
                "uw_s_aux",
                "uw_weight_main",
                "uw_weight_aux",
                "aux_warmup_factor",
                "n_train_batches",
            ],
        )
        writer.writeheader()
        with open(stage_summary_path, "w", newline="", encoding="utf-8") as sf, open(
            subloss_log_path, "w", newline="", encoding="utf-8"
        ) as subf, open(gradcos_log_path, "w", newline="", encoding="utf-8") as gradf:
            stage_writer = csv.DictWriter(
                sf,
                fieldnames=[
                    "stage_index",
                    "end_step",
                    "lr",
                    "stage_train_mode",
                    "stage_best_val_total_loss",
                    "stage_best_ckpt_selection_score",
                    "stage_imp",
                    "stage_rel_gain",
                    "decision",
                ],
            )
            stage_writer.writeheader()
            subloss_writer = csv.DictWriter(
                subf,
                fieldnames=[
                    "global_step",
                    "batch_index",
                    "raw_rmse_term",
                    "raw_raw_mse",
                    "raw_relrmse_raw",
                    "raw_low_over_penalty",
                    "raw_mono_pair_loss",
                    "raw_mono_grad_loss",
                    "raw_curvature_loss",
                    "raw_corr_penalty",
                    "raw_grad_loss",
                    "raw_adv_huber_loss",
                    "raw_adv_smooth_loss",
                    "raw_guide83_loss",
                    "raw_guide84_poisson_loss",
                    "raw_tfsf_smooth_loss",
                    "raw_tf_unimodal_loss",
                    "raw_sf_unimodal_loss",
                    "raw_distill_loss",
                    "raw_main_task",
                    "raw_aux_task",
                    "norm_rmse_term",
                    "norm_raw_mse",
                    "norm_relrmse_raw",
                    "norm_low_over_penalty",
                    "norm_mono_pair_loss",
                    "norm_mono_grad_loss",
                    "norm_curvature_loss",
                    "norm_corr_penalty",
                    "norm_grad_loss",
                    "norm_adv_huber_loss",
                    "norm_adv_smooth_loss",
                    "norm_guide83_loss",
                    "norm_guide84_poisson_loss",
                    "norm_tfsf_smooth_loss",
                    "norm_tf_unimodal_loss",
                    "norm_sf_unimodal_loss",
                    "norm_distill_loss",
                    "norm_main_task",
                    "norm_aux_task",
                    "variance_weight_raw_mean",
                    "variance_weight_raw_min",
                    "variance_weight_raw_max",
                    "loss_exclusion_keep_frac",
                    "aux_warmup_factor",
                    "uw_s_main",
                    "uw_s_aux",
                    "uw_weight_main",
                    "uw_weight_aux",
                    "train_total_loss",
                ],
            )
            subloss_writer.writeheader()
            gradcos_writer = csv.DictWriter(gradf, fieldnames=gradcos_fieldnames)
            gradcos_writer.writeheader()

            best_val = float(initial_val_metrics["total_loss"])
            best_score = float(checkpoint_selection_score(initial_val_metrics, cfg)) + (
                float(getattr(cfg, "ckpt_global_curvature_beta", 0.0))
                * initial_global_curvature_penalty
            )
            best_global_curvature_penalty = initial_global_curvature_penalty
            best_epoch = 0
            save_checkpoint(
                out_dir / "best.pt",
                model,
                model_cfg,
                cfg,
                feature_norm,
                {
                    "best_epoch": best_epoch,
                    "best_val_total_loss": best_val,
                    "best_ckpt_selection_score": best_score,
                    "best_global_curvature_penalty": best_global_curvature_penalty,
                    "best_bad_subset_mean_rmse_raw": float(initial_val_metrics.get("bad_subset_mean_rmse_raw", 0.0)),
                    "best_bad_subset_max_rmse_raw": float(initial_val_metrics.get("bad_subset_max_rmse_raw", 0.0)),
                    "best_original5d_rmse_raw": float(initial_val_metrics.get("original5d_rmse_raw", 0.0)),
                    "best_population_focus_total_loss": float(initial_val_metrics.get("population_focus_total_loss", 0.0)),
                    "best_population_focus_rmse_raw": float(initial_val_metrics.get("population_focus_rmse_raw", 0.0)),
                    "initial_original5d_rmse_raw": initial_original5d_rmse,
                    "uw_log_vars": [0.0, 0.0],
                },
            )
            global_epoch = 0
            global_step = 0
            lr_idx = 0
            prev_stage_best_total = None
            executed_stages = 0
            while executed_stages < cfg.max_stages:
                executed_stages += 1
                stage_lr = float(cfg.lr_stage_values[lr_idx])
                for handle in stage_hook_handles:
                    handle.remove()
                stage_mode = train_stage_mode_for_index(cfg, executed_stages)
                stage_hook_handles, stage_weight_decay = apply_train_stage_mode(model, cfg, stage_mode)
                optimizer = make_stage_optimizer(model, cfg, stage_lr, uw_log_vars, stage_weight_decay)
                print(f"[stage {executed_stages}] mode={stage_mode} lr={stage_lr:.6g} weight_decay={stage_weight_decay:.6g}")
                val_total_history: list[float] = []
                stage_best_total = float("inf")
                stage_best_score = float("inf")

                for stage_epoch in range(1, cfg.stage_epochs + 1):
                    global_epoch += 1
                    (
                        train_loss,
                        n_train_batches,
                        train_grad_loss,
                        train_curvature_grad_scale,
                        train_curvature_base_grad_norm,
                        train_curvature_regularizer_grad_norm,
                        train_curvature_applied_grad_ratio,
                    ) = run_epoch(
                        model,
                        train_loader,
                        optimizer,
                        scaler,
                        device,
                        cfg,
                        grad_teacher,
                        online_interp_teacher,
                        feature_norm,
                        subloss_scales,
                        subloss_writer,
                        gradcos_writer if getattr(cfg, "gradcos_enable", False) else None,
                        global_step,
                        total_steps_planned,
                        uw_log_vars,
                        shadow_uw_log_vars,
                        shadow_uw_optimizer,
                        weight_anchor_state,
                        teacher_model,
                    )
                    global_step += int(n_train_batches)
                    subf.flush()
                    gradf.flush()
                    val_metrics = evaluate(model, val_loader, device, cfg)
                    val_global_curvature_penalty = deterministic_global_curvature_penalty(
                        model, cfg, feature_norm, device
                    )
                    ckpt_score = checkpoint_selection_score(val_metrics, cfg) + (
                        float(getattr(cfg, "ckpt_global_curvature_beta", 0.0))
                        * val_global_curvature_penalty
                    )
                    val_total_history.append(float(val_metrics["total_loss"]))
                    stage_best_total = min(stage_best_total, float(val_metrics["total_loss"]))
                    stage_best_score = min(stage_best_score, float(ckpt_score))

                    writer.writerow(
                        {
                            "epoch": global_epoch,
                            "global_step": global_step,
                            "stage_index": executed_stages,
                            "stage_epoch": stage_epoch,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "train_loss": train_loss,
                            "val_total_loss": val_metrics["total_loss"],
                            "val_weighted_total_loss": val_metrics.get("weighted_total_loss", 0.0),
                            "val_rmse_log": val_metrics["rmse_log"],
                            "val_rmse_term": val_metrics["rmse_term"],
                            "val_rmse_raw": val_metrics["rmse_raw"],
                            "val_mse_raw": val_metrics["mse_raw"],
                            "val_relrmse_raw": val_metrics["relrmse_raw"],
                            "val_weighted_rmse_log": val_metrics.get("weighted_rmse_log", 0.0),
                            "val_weighted_rmse_raw": val_metrics.get("weighted_rmse_raw", 0.0),
                            "val_weighted_mse_raw": val_metrics.get("weighted_mse_raw", 0.0),
                            "val_weighted_relrmse_raw": val_metrics.get("weighted_relrmse_raw", 0.0),
                            "val_low_over_penalty": val_metrics["low_over_penalty"],
                            "val_mono_pair_loss": val_metrics["mono_pair_loss"],
                            "val_bad_subset_mean_rmse_raw": val_metrics.get("bad_subset_mean_rmse_raw", 0.0),
                            "val_bad_subset_max_rmse_raw": val_metrics.get("bad_subset_max_rmse_raw", 0.0),
                            "val_original5d_n": val_metrics.get("original5d_n", 0),
                            "val_original5d_rmse_raw": val_metrics.get("original5d_rmse_raw", 0.0),
                            "val_original5d_rmse_delta": float(val_metrics.get("original5d_rmse_raw", 0.0)) - initial_original5d_rmse,
                            "val_original5d_rmse_ratio": float(val_metrics.get("original5d_rmse_raw", 0.0)) / max(initial_original5d_rmse, 1e-12),
                            "val_population_focus_n": val_metrics.get("population_focus_n", 0),
                            "val_population_focus_total_loss": val_metrics.get("population_focus_total_loss", 0.0),
                            "val_population_focus_rmse_raw": val_metrics.get("population_focus_rmse_raw", 0.0),
                            "ckpt_selection_score": ckpt_score,
                            "val_mae_raw": val_metrics["mae_raw"],
                            "val_r2_raw": val_metrics["r2_raw"],
                            "train_grad_loss": train_grad_loss,
                            "train_curvature_grad_scale": train_curvature_grad_scale,
                            "train_curvature_base_update_norm": train_curvature_base_grad_norm,
                            "train_curvature_regularizer_update_norm": train_curvature_regularizer_grad_norm,
                            "train_curvature_applied_update_ratio": train_curvature_applied_grad_ratio,
                            "val_global_curvature_penalty": val_global_curvature_penalty,
                            "uw_s_main": float((uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars)[0].detach().cpu().item()) if (uw_log_vars is not None or shadow_uw_log_vars is not None) else 0.0,
                            "uw_s_aux": float((uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars)[1].detach().cpu().item()) if (uw_log_vars is not None or shadow_uw_log_vars is not None) else 0.0,
                            "uw_weight_main": float(torch.exp(-(uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars)[0]).detach().cpu().item()) if (uw_log_vars is not None or shadow_uw_log_vars is not None) else 1.0,
                            "uw_weight_aux": float(torch.exp(-(uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars)[1]).detach().cpu().item()) if (uw_log_vars is not None or shadow_uw_log_vars is not None) else 1.0,
                            "aux_warmup_factor": compute_aux_warmup_factor(cfg, global_step, total_steps_planned),
                            "n_train_batches": n_train_batches,
                        }
                    )
                    f.flush()

                    if ckpt_score < best_score - 1e-9:
                        best_score = float(ckpt_score)
                        best_val = float(val_metrics["total_loss"])
                        best_epoch = global_epoch
                        best_global_curvature_penalty = val_global_curvature_penalty
                        save_checkpoint(
                            out_dir / "best.pt",
                            model,
                            model_cfg,
                            cfg,
                            feature_norm,
                            {
                                "best_epoch": best_epoch,
                                "best_val_total_loss": best_val,
                                "best_ckpt_selection_score": best_score,
                                "best_global_curvature_penalty": best_global_curvature_penalty,
                                "best_bad_subset_mean_rmse_raw": float(val_metrics.get("bad_subset_mean_rmse_raw", 0.0)),
                                "best_bad_subset_max_rmse_raw": float(val_metrics.get("bad_subset_max_rmse_raw", 0.0)),
                                "best_original5d_rmse_raw": float(val_metrics.get("original5d_rmse_raw", 0.0)),
                                "best_population_focus_total_loss": float(val_metrics.get("population_focus_total_loss", 0.0)),
                                "best_population_focus_rmse_raw": float(val_metrics.get("population_focus_rmse_raw", 0.0)),
                                "initial_original5d_rmse_raw": initial_original5d_rmse,
                                "uw_log_vars": [float(v) for v in ((uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars).detach().cpu().tolist() if (uw_log_vars is not None or shadow_uw_log_vars is not None) else [0.0, 0.0])],
                            },
                        )

                    print(
                        f"[ep {global_epoch:04d}][stage {executed_stages}:{stage_epoch:03d}] "
                        f"lr={float(optimizer.param_groups[0]['lr']):.6g} train_loss={train_loss:.6f} "
                        f"val_total_loss={val_metrics['total_loss']:.6f} "
                        f"val_weighted_total_loss={val_metrics.get('weighted_total_loss', 0.0):.6f} "
                        f"val_rmse_log={val_metrics['rmse_log']:.6f} "
                        f"val_rmse_raw={val_metrics['rmse_raw']:.6f} "
                        f"val_relrmse_raw={val_metrics['relrmse_raw']:.6f} "
                        f"val_low_over_penalty={val_metrics['low_over_penalty']:.6f} "
                        f"val_mono_pair_loss={val_metrics['mono_pair_loss']:.6f} "
                        f"val_bad_subset_mean_rmse_raw={val_metrics.get('bad_subset_mean_rmse_raw', 0.0):.6f} "
                        f"val_bad_subset_max_rmse_raw={val_metrics.get('bad_subset_max_rmse_raw', 0.0):.6f} "
                        f"val_original5d_rmse_raw={val_metrics.get('original5d_rmse_raw', 0.0):.6f} "
                        f"val_original5d_rmse_delta={float(val_metrics.get('original5d_rmse_raw', 0.0)) - initial_original5d_rmse:+.6f} "
                        f"val_population_focus_loss={val_metrics.get('population_focus_total_loss', 0.0):.6f} "
                        f"val_population_focus_rmse_raw={val_metrics.get('population_focus_rmse_raw', 0.0):.6f} "
                        f"ckpt_selection_score={ckpt_score:.6f} "
                        f"train_grad_loss={train_grad_loss:.6f}"
                    )

                diag_sample = collect_diag_sample(model, val_diag_loader, device, cfg, executed_stages)
                save_stage_diagnostics_at_step(out_dir, executed_stages, global_step, diag_sample, cfg.target)

                imp = stage_improve(val_total_history)
                rel_gain = None
                if prev_stage_best_total is not None and prev_stage_best_total > 0:
                    rel_gain = (prev_stage_best_total - stage_best_total) / prev_stage_best_total

                slow = imp < cfg.min_improve
                if rel_gain is not None:
                    slow = slow or (rel_gain < cfg.min_best_rel_gain)

                decision = "keep_lr"
                if slow:
                    if lr_idx < len(cfg.lr_stage_values) - 1:
                        decision = "drop_lr"
                        lr_idx += 1
                    else:
                        decision = "stop_lowest_lr"

                stage_writer.writerow(
                    {
                        "stage_index": executed_stages,
                        "end_step": global_step,
                        "lr": stage_lr,
                        "stage_train_mode": stage_mode,
                        "stage_best_val_total_loss": stage_best_total,
                        "stage_best_ckpt_selection_score": stage_best_score,
                        "stage_imp": imp,
                        "stage_rel_gain": rel_gain,
                        "decision": decision,
                    }
                )
                sf.flush()
                prev_stage_best_total = stage_best_total
                if decision == "stop_lowest_lr":
                    break

    save_checkpoint(
        out_dir / "last.pt",
        model,
        model_cfg,
        cfg,
        feature_norm,
        {
            "last_epoch": global_epoch,
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
            "best_ckpt_selection_score": best_score,
            "best_global_curvature_penalty": best_global_curvature_penalty,
            "executed_stages": executed_stages,
            "final_lr": float(cfg.lr_stage_values[lr_idx]),
            "uw_log_vars": [float(v) for v in ((uw_log_vars if uw_log_vars is not None else shadow_uw_log_vars).detach().cpu().tolist() if (uw_log_vars is not None or shadow_uw_log_vars is not None) else [0.0, 0.0])],
        },
    )

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=True)
    train_eval_loader = make_sample_eval_loader(cfg, "train", manifest, feature_norm, max_batches=64)
    train_metrics = evaluate(model, train_eval_loader, device, cfg)
    val_metrics = evaluate(model, val_loader, device, cfg)
    test_metrics = evaluate(model, test_loader, device, cfg)
    save_json(
        out_dir / "metrics_summary.json",
        {
            "target": cfg.target,
            "best_epoch": best_epoch,
            "best_val_total_loss": best_val,
            "best_ckpt_selection_score": best_score,
            "best_global_curvature_penalty": best_global_curvature_penalty,
            "variance_weighting": {
                "mode": cfg.variance_weight_mode,
                "has_fields": cfg.variance_has_fields,
                "min_weight": cfg.variance_min_weight,
                "ref_quantile": cfg.variance_ref_quantile,
                "ref_raw": cfg.variance_ref_raw,
                "ref_log": cfg.variance_ref_log,
                "ref_rel": cfg.variance_ref_rel,
            },
            "train_metrics_note": "sampled_train_eval_64_batches",
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
    )
    save_loss_curve(out_dir, log_path, cfg.target)
    save_subloss_curves(out_dir, subloss_log_path, cfg.target)
    if getattr(cfg, "gradcos_enable", False):
        save_gradcos_curves(out_dir, gradcos_log_path, cfg.target)

    train_eval_loader.close()
    train_loader.close()
    val_loader.close()
    val_diag_loader.close()
    test_loader.close()
    print(f"[done] target={cfg.target} out_dir={out_dir}")


if __name__ == "__main__":
    main()
