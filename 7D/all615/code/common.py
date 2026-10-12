from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

FEATURE_NAMES = (
    "OriCos2",
    "OriSin2",
    "Contrast",
    "L6",
    "L4E",
    "L4I",
)

FEATURE_NAMES_TF_SF7D = (
    "OriCos2",
    "Contrast",
    "L6",
    "L4E",
    "L4I",
    "lgnTF",
    "lgnSF",
)

NORM_FEATURE_NAMES = (
    "Contrast",
    "L6",
    "L4E",
    "L4I",
)

NORM_FEATURE_NAMES_TF_SF7D = (
    "Contrast",
    "L6",
    "L4E",
    "L4I",
    "lgnTF",
    "lgnSF",
)

TARGET_FIELD = "FR_Response"


@dataclass
class ModelConfig:
    target: str
    model_family: str = "single_mlp"
    feature_mode: str = "pref6d"
    input_dim: int = 6
    hidden_dims: tuple[int, ...] = (64, 128, 128)
    activation_seq: tuple[str, ...] = ("relu", "relu", "relu")
    shape_hidden_dims: tuple[int, ...] = ()
    shape_activation_seq: tuple[str, ...] = ()
    contrast_hidden_dims: tuple[int, ...] = ()
    contrast_activation_seq: tuple[str, ...] = ()
    softplus_out: bool = True
    output_semantics: str = "raw_rate_softplus"
    constant_baseline_rule: dict | None = None


@dataclass
class TrainConfig:
    train_mat: str
    split_dir: str
    out_dir: str
    target: str
    device: str
    feature_mode: str = "pref6d"
    seed: int = 42
    epochs: int = 25
    lr: float = 5e-3
    weight_decay: float = 1e-5
    batch_size: int = 32768
    eval_batch_size: int = 65536
    max_train_batches: int = 10
    max_eval_batches: int = 0
    stage_epochs: int = 25
    max_stages: int = 6
    lr_stage_values: tuple[float, ...] = (5e-3, 2e-3, 9e-4, 6e-4, 2e-4)
    min_improve: float = 0.05
    min_best_rel_gain: float = 0.002
    use_amp: bool = True
    num_workers: int = 0
    loss_eps: float = 1e-8
    rel_alpha: float = 1.0
    rmse_space: str = "log"
    w_log: float = 0.5
    w_rel: float = 0.5
    raw_mse_lambda: float = 0.0
    low_penalty_lambda: float = 0.0
    low_penalty_tau: float = 3.0
    mono_pair_lambda: float = 0.0
    mono_grad_lambda: float = 0.0
    mono_grad_max_points: int = 2048
    curvature_lambda: float = 0.0
    curvature_max_points: int = 2048
    curvature_oricos2_delta: float = 0.05
    curvature_contrast_delta: float = 5.0
    curvature_mode: str = "local_oc_contrast"
    global_curvature_delta_fraction: float = 0.04
    global_curvature_direction_count: int = 4
    global_curvature_output_scale: float = 0.0
    global_curvature_balance_mode: str = "none"
    global_curvature_balance_path: str = ""
    global_curvature_reduce_mode: str = "mean"
    global_curvature_smoothmax_temperature: float = 5.0
    global_curvature_focus_feature: str = ""
    global_curvature_focus_mean_weight: float = 0.1
    curvature_grad_ratio: float = 0.0
    ckpt_global_curvature_beta: float = 0.0
    ckpt_global_curvature_points: int = 512
    grad_lambda: float = 0.0
    grad_max_points: int = 128
    grad_interior_eps: float = 0.05
    online_interp_enable: bool = False
    online_interp_frac: float = 0.10
    online_interp_eps: float = 0.05
    online_interp_cache_slabs: int = 8
    online_interp_badslab_only: bool = True
    task_uw_enable: bool = False
    task_uw_shadow_enable: bool = False
    task_uw_shadow_lr: float = 1e-2
    task_uw_aux_warmup_frac: float = 0.0
    task_uw_aux_warmup_scale: float = 1.0
    task_pcgrad_enable: bool = False
    task_cagrad_enable: bool = False
    cagrad_alpha: float = 0.5
    cagrad_rescale: int = 1
    population_grad_mode: str = "none"
    population_focus_conditions: tuple[tuple[float, ...], ...] = ()
    population_focus_raw_lambda: float = 0.0
    population_focus_raw_scale: float = 0.0
    population_focus_group_cvar_fraction: float = 0.0
    ckpt_population_focus_beta: float = 0.0
    subloss_norm_enable: bool = False
    subloss_norm_init_batches: int = 8
    gradcos_enable: bool = False
    gradcos_every: int = 20
    diag_sample_size: int = 200000
    ckpt_bad_subset_enable: bool = False
    ckpt_bad_subset_beta: float = 1.0
    ckpt_bad_subset_gamma: float = 0.5
    variance_weight_mode: str = "none"
    variance_min_weight: float = 0.25
    variance_ref_quantile: float = 0.50
    variance_eps: float = 1e-8
    variance_ref_raw: float = 1.0
    variance_ref_log: float = 1.0
    variance_ref_rel: float = 1.0
    variance_has_fields: bool = False
    sampler_mode: str = "uniform"
    target_rate_bins: tuple[float, ...] = (0.0, 0.5, 2.0, 5.0, 15.0, float("inf"))
    strata_policy: str = "equal"
    loss_exclude_contrasts: tuple[float, ...] = ()
    loss_exclude_conditions: tuple[tuple[float, float], ...] = ()
    loss_exclude_eval: bool = False
    adv_enable: bool = False
    adv_epsilon: float = 0.03
    adv_lambda: float = 0.25
    adv_smooth_lambda: float = 0.10
    adv_huber_delta: float = 1.0
    adv_clip_value: float = 6.0
    guide_uncertainty_mode: str = "none"
    guide83_w_log: float = 1.0
    guide83_w_raw: float = 1.0
    guide83_c: float = 1.0
    guide84_time_window: float = 1.0
    guide84_eps: float = 1e-6
    init_checkpoint: str = ""
    train_stage_modes: tuple[str, ...] = ()
    weight_anchor_checkpoint: str = ""
    weight_anchor_lambda: float = 0.0
    weight_anchor_include_new_inputs: bool = False
    tfsf_smooth_lambda: float = 0.0
    tfsf_smooth_delta: float = 0.05
    tfsf_smooth_max_points: int = 2048
    tf_unimodal_lambda: float = 0.0
    sf_unimodal_lambda: float = 0.0
    tfsf_unimodal_max_points: int = 512
    tfsf_unimodal_prominence_fraction: float = 0.02
    distill_anchor_lambda: float = 0.0
    distill_anchor_mode: str = "default_tfsf"
    distill_default_tf: float = 10.0
    distill_default_sf: float = 2.5
    distill_max_tf_exclusive: float = 32.0
    distill_max_points: int = 2048


def feature_names_for_mode(mode: str) -> tuple[str, ...]:
    key = str(mode).lower()
    if key == "pref6d":
        return FEATURE_NAMES
    if key == "cosonly5d":
        return (
            "OriCos2",
            "Contrast",
            "L6",
            "L4E",
            "L4I",
        )
    if key == "tf_sf7d":
        return FEATURE_NAMES_TF_SF7D
    raise ValueError(f"Unknown feature_mode {mode!r}")


def input_dim_for_mode(mode: str) -> int:
    return len(feature_names_for_mode(mode))


def norm_feature_names_for_mode(mode: str) -> tuple[str, ...]:
    key = str(mode).lower()
    if key == "tf_sf7d":
        return NORM_FEATURE_NAMES_TF_SF7D
    feature_names_for_mode(mode)
    return NORM_FEATURE_NAMES


def constant_baseline_rule(feature_norm: dict, target: str) -> dict:
    thresholds = []
    for name, value in [("Contrast", 0), ("lgnTF", 0), ("lgnSF", 0),
                        ("lgnTF", 50), ("lgnSF", 15)]:
        stat = feature_norm[name]
        if str(stat.get("transform", "identity")) != "identity":
            raise ValueError("Fixed baseline requires the linear TF/SF normalization")
        # Match the dataset loader's float32 arithmetic exactly at boundaries.
        raw = np.array([value], dtype=np.float32)
        thresholds.append(float(((raw - float(stat["mean"])) / float(stat["std"]))[0]))
    return {"policy": "contrast==0 OR TF==0 OR SF==0 OR TF>50 OR SF>15",
            "rate": {"S": 2.5, "C": 4.5, "I": 12.5}[target],
            "normalized_thresholds": thresholds,
            "source": "User-specified uniform baseline assumption; not measured fit"}


class SingleOutputMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.activation_seq)
        self.softplus_out = bool(cfg.softplus_out)
        self.softplus = nn.Softplus()
        self.constant_baseline_rule = cfg.constant_baseline_rule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rule = self.constant_baseline_rule
        if rule is not None:
            threshold = x.new_tensor(rule["normalized_thresholds"])
            fixed = ((x[:, 1] == threshold[0]) | (x[:, 5] == threshold[1])
                     | (x[:, 6] == threshold[2]) | (x[:, 5] > threshold[3])
                     | (x[:, 6] > threshold[4]))
            # Fixed rows must not depend on even extreme values in other inputs.
            x = torch.where(fixed[:, None], torch.zeros_like(x), x)
        y = self.net(x).squeeze(-1)
        if self.softplus_out:
            y = self.softplus(y)
        if rule is not None:
            y = torch.where(fixed, y.new_tensor(rule["rate"]), y)
        return y


class DoubleHeadAdditiveMLP(nn.Module):
    SHAPE_COLS = (0, 3, 4, 2)
    CONTRAST_COLS = (1, 0)

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if str(cfg.feature_mode).lower() != "cosonly5d":
            raise ValueError("DoubleHeadAdditiveMLP requires feature_mode='cosonly5d'")
        if int(cfg.input_dim) != 5:
            raise ValueError("DoubleHeadAdditiveMLP requires input_dim=5")
        if len(cfg.shape_hidden_dims) != len(cfg.shape_activation_seq):
            raise ValueError("shape_hidden_dims and shape_activation_seq must have matching length")
        if len(cfg.contrast_hidden_dims) != len(cfg.contrast_activation_seq):
            raise ValueError("contrast_hidden_dims and contrast_activation_seq must have matching length")
        self.shape_head = build_mlp(4, cfg.shape_hidden_dims, cfg.shape_activation_seq)
        self.contrast_head = build_mlp(2, cfg.contrast_hidden_dims, cfg.contrast_activation_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_x = x[:, self.SHAPE_COLS]
        contrast_x = x[:, self.CONTRAST_COLS]
        z_shape = self.shape_head(shape_x)
        z_contr = self.contrast_head(contrast_x)
        z_total = z_shape + z_contr
        return torch.expm1(z_total).squeeze(-1)


class PositiveLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.nn.functional.softplus(self.weight_raw)
        return torch.nn.functional.linear(x, weight, self.bias)


def build_positive_mlp(input_dim: int, hidden_dims: tuple[int, ...], activation_seq: tuple[str, ...]) -> nn.Sequential:
    if len(hidden_dims) != len(activation_seq):
        raise ValueError("hidden_dims and activation_seq must have matching length")
    dims = [int(input_dim), *hidden_dims]
    layers: list[nn.Module] = []
    for i, act_name in enumerate(activation_seq):
        layers.append(PositiveLinear(dims[i], dims[i + 1]))
        layers.append(build_activation(act_name))
    layers.append(PositiveLinear(dims[-1], 1))
    return nn.Sequential(*layers)


class ThresholdAdditiveMLP(nn.Module):
    CONTRAST_COLS = (0, 1)
    L6_COL = 2
    L4E_COL = 3
    L4I_COL = 4

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if str(cfg.feature_mode).lower() != "cosonly5d":
            raise ValueError("ThresholdAdditiveMLP requires feature_mode='cosonly5d'")
        if int(cfg.input_dim) != 5:
            raise ValueError("ThresholdAdditiveMLP requires input_dim=5")
        if len(cfg.shape_hidden_dims) != len(cfg.shape_activation_seq):
            raise ValueError("shape_hidden_dims and shape_activation_seq must have matching length")
        if len(cfg.contrast_hidden_dims) != len(cfg.contrast_activation_seq):
            raise ValueError("contrast_hidden_dims and contrast_activation_seq must have matching length")
        self.base_oc = build_mlp(2, cfg.contrast_hidden_dims, cfg.contrast_activation_seq)
        self.m_e = build_positive_mlp(1, cfg.shape_hidden_dims, cfg.shape_activation_seq)
        self.m_6 = build_positive_mlp(1, cfg.shape_hidden_dims, cfg.shape_activation_seq)
        self.m_i = build_positive_mlp(1, cfg.shape_hidden_dims, cfg.shape_activation_seq)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (
            self.base_oc(x[:, self.CONTRAST_COLS])
            + self.m_e(x[:, self.L4E_COL : self.L4E_COL + 1])
            + self.m_6(x[:, self.L6_COL : self.L6_COL + 1])
            - self.m_i(x[:, self.L4I_COL : self.L4I_COL + 1])
        )
        return self.softplus(z).squeeze(-1)


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...], activation_seq: tuple[str, ...]) -> nn.Sequential:
    if len(hidden_dims) != len(activation_seq):
        raise ValueError("hidden_dims and activation_seq must have matching length")
    dims = [int(input_dim), *hidden_dims]
    layers: list[nn.Module] = []
    for i, act_name in enumerate(activation_seq):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(build_activation(act_name))
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


def build_model(cfg: ModelConfig) -> nn.Module:
    family = str(getattr(cfg, "model_family", "single_mlp")).lower()
    if family == "single_mlp":
        return SingleOutputMLP(cfg)
    if family == "double_head_additive":
        return DoubleHeadAdditiveMLP(cfg)
    if family == "threshold_additive":
        return ThresholdAdditiveMLP(cfg)
    raise ValueError(f"Unknown model_family {family!r}")


def clamp_raw_rate_tensor(y: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(y, 0.0)


def clamp_raw_rate_array(y: np.ndarray) -> np.ndarray:
    return np.clip(y, a_min=0.0, a_max=None)


def build_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU()
    if key == "softplus":
        return nn.Softplus()
    if key == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation {name!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inverse_target(y_hat: np.ndarray | torch.Tensor):
    return y_hat


def to_serializable_config(cfg) -> Dict:
    if hasattr(cfg, "__dataclass_fields__"):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Unsupported config type: {type(cfg)}")


def device_autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type=device.type, enabled=False)


def diag_line(x: np.ndarray, y: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if math.isclose(xmin, xmax):
        return np.array([xmin], dtype=np.float64), np.array([float(np.mean(y))], dtype=np.float64)
    edges = np.linspace(xmin, xmax, n_bins + 1)
    bins = np.digitize(x, edges[1:-1], right=False)
    xs = []
    ys = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        xs.append(float(np.mean(x[mask])))
        ys.append(float(np.mean(y[mask])))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
