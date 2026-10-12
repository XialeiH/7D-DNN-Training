#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import asdict

import torch
import torch.nn as nn

import train_sixd_single as base


class RationalActivation(nn.Module):
    """Small learnable rational activation: P3(x) / (1 + |Q2(x)|)."""

    def __init__(self) -> None:
        super().__init__()
        self.numerator = nn.Parameter(torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32))
        self.denominator = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a0, a1, a2, a3 = self.numerator
        b1, b2 = self.denominator
        numerator = a0 + a1 * x + a2 * x.square() + a3 * x.square() * x
        denominator = 1.0 + torch.abs(b1 * x + b2 * x.square())
        return numerator / denominator


def build_activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "rational":
        return RationalActivation()
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


class ResidualEveryTwoMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], activation_seq: tuple[str, ...]) -> None:
        super().__init__()
        if len(hidden_dims) != len(activation_seq):
            raise ValueError("hidden_dims and activation_seq must have matching length")
        if len(hidden_dims) % 2 != 0:
            raise ValueError("residual_every_two requires an even number of hidden layers")

        self.blocks = nn.ModuleList()
        prev_dim = int(input_dim)
        for i in range(0, len(hidden_dims), 2):
            h1 = int(hidden_dims[i])
            h2 = int(hidden_dims[i + 1])
            block = nn.ModuleDict(
                {
                    "linear1": nn.Linear(prev_dim, h1),
                    "act1": build_activation(activation_seq[i]),
                    "linear2": nn.Linear(h1, h2),
                    "act2": build_activation(activation_seq[i + 1]),
                    "skip": nn.Identity() if prev_dim == h2 else nn.Linear(prev_dim, h2),
                }
            )
            self.blocks.append(block)
            prev_dim = h2
        self.out = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            skip = block["skip"](h)
            z = block["act1"](block["linear1"](h))
            z = block["act2"](block["linear2"](z))
            h = z + skip
        return self.out(h)


class SingleOutputArchVariant(nn.Module):
    def __init__(self, cfg: base.ModelConfig) -> None:
        super().__init__()
        residual_skip_every = int(os.environ.get("SIXD_RESIDUAL_SKIP_EVERY", "0") or "0")
        if residual_skip_every == 2:
            self.net = ResidualEveryTwoMLP(cfg.input_dim, cfg.hidden_dims, cfg.activation_seq)
        elif residual_skip_every == 0:
            self.net = build_plain_mlp(cfg.input_dim, cfg.hidden_dims, cfg.activation_seq)
        else:
            raise ValueError("Only SIXD_RESIDUAL_SKIP_EVERY=0 or 2 is supported")
        self.softplus_out = bool(cfg.softplus_out)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x).squeeze(-1)
        if self.softplus_out:
            y = self.softplus(y)
        return y


def build_plain_mlp(input_dim: int, hidden_dims: tuple[int, ...], activation_seq: tuple[str, ...]) -> nn.Sequential:
    if len(hidden_dims) != len(activation_seq):
        raise ValueError("hidden_dims and activation_seq must have matching length")
    dims = [int(input_dim), *hidden_dims]
    layers: list[nn.Module] = []
    for i, act_name in enumerate(activation_seq):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(build_activation(act_name))
    layers.append(nn.Linear(dims[-1], 1))
    return nn.Sequential(*layers)


def build_model(cfg: base.ModelConfig) -> nn.Module:
    family = str(getattr(cfg, "model_family", "single_mlp")).lower()
    if family == "single_mlp":
        return SingleOutputArchVariant(cfg)
    return base.build_model(cfg)


def to_serializable_config(cfg):
    if hasattr(cfg, "__dataclass_fields__"):
        data = asdict(cfg)
    elif isinstance(cfg, dict):
        data = dict(cfg)
    else:
        raise TypeError(f"Unsupported config type: {type(cfg)}")
    data["arch_variant"] = os.environ.get("SIXD_ARCH_VARIANT", "")
    data["residual_skip_every"] = int(os.environ.get("SIXD_RESIDUAL_SKIP_EVERY", "0") or "0")
    data["rational_activation"] = "learned_p3_over_1_abs_q2"
    return data


base.build_model = build_model
base.to_serializable_config = to_serializable_config


if __name__ == "__main__":
    base.main()
