#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def matlab_num(x: float) -> str:
    return f"{float(x):.10g}"


def matlab_matrix(t: torch.Tensor, indent: str = "") -> str:
    a = t.detach().cpu().numpy()
    if a.ndim == 1:
        body = " ".join(matlab_num(v) for v in a.tolist())
        return f"[{body}]"
    rows = []
    for r in a:
        rows.append(" ".join(matlab_num(v) for v in r.tolist()))
    if len(rows) == 1:
        return f"[{rows[0]}]"
    sep = ";\n" + indent
    return "[" + sep.join(rows) + "]"


def load_ckpt(path: Path) -> Dict:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "state": obj["model"],
        "feature_norm": obj["feature_norm"],
        "model_config": obj["model_config"],
        "meta": obj,
        "path": str(path),
    }


def extract_linear_layers(state: Dict[str, torch.Tensor]) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    keys = [
        k for k in state
        if k.startswith("net.") and k.endswith(".weight") and k.split(".")[1].isdigit()
    ]
    keys = sorted(keys, key=lambda x: int(x.split(".")[1]))
    out = []
    for wk in keys:
        bk = wk.replace(".weight", ".bias")
        out.append((wk.split(".")[1], state[wk], state[bk]))
    return out


def has_residual_every_two_state(state: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("net.blocks.") for k in state)


def extract_prefixed_linear_layers(state: Dict[str, torch.Tensor], prefix: str) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    prefix = str(prefix)
    keys = [
        k for k in state
        if k.startswith(prefix) and k.endswith(".weight")
    ]
    def layer_idx(key: str) -> int:
        suffix = key[len(prefix):]
        return int(suffix.split(".", 1)[0])
    keys = sorted(keys, key=layer_idx)
    out = []
    for wk in keys:
        bk = wk.replace(".weight", ".bias")
        out.append((str(layer_idx(wk)), state[wk], state[bk]))
    return out


def feature_contract(feature_mode: str) -> Dict[str, object]:
    mode = str(feature_mode).lower()
    if mode == "pref6d":
        return {
            "signature": "L4EUse, L4IUse, L6Use, ContrastEff, OriCos2, OriSin2",
            "physical_order_comment": "[L4E, L4I, L6, ContrastEff, OriCos2, OriSin2]",
            "internal_order_comment": "[OriCos2, OriSin2, Contrast, L6, L4E, L4I]",
            "expand_fields": ("L4IUse", "L6Use", "ContrastEff", "OriCos2", "OriSin2"),
            "x_expr": "[OriCos2(:), OriSin2(:), ContrastEff(:), L6Use(:), L4EUse(:), L4IUse(:)]",
            "local_response_comment": "% Translate runtime variables into the 6D training contract:",
            "local_response_contract": "% [L4E, L4I, L6, ContrastEff, OriCos2, OriSin2].",
            "predict_call_args": "L4EUse, L4IUse, L6Use, contrastEff, oriCos2, oriSin2",
            "bg_extra": "    oriSin2(bgMask) = sind(2 .* gammaUse(bgMask));",
            "fg_extra": "    oriSin2(fgMask) = sind(2 .* gammaUse(fgMask));",
            "need_orisin": True,
            "need_tfsf": False,
        }
    if mode == "cosonly5d":
        return {
            "signature": "L4EUse, L4IUse, L6Use, ContrastEff, OriCos2",
            "physical_order_comment": "[L4E, L4I, L6, ContrastEff, OriCos2]",
            "internal_order_comment": "[OriCos2, Contrast, L6, L4E, L4I]",
            "expand_fields": ("L4IUse", "L6Use", "ContrastEff", "OriCos2"),
            "x_expr": "[OriCos2(:), ContrastEff(:), L6Use(:), L4EUse(:), L4IUse(:)]",
            "local_response_comment": "% Translate runtime variables into the 5D cos-only training contract:",
            "local_response_contract": "% [L4E, L4I, L6, ContrastEff, OriCos2].",
            "predict_call_args": "L4EUse, L4IUse, L6Use, contrastEff, oriCos2",
            "bg_extra": "",
            "fg_extra": "",
            "need_orisin": False,
            "need_tfsf": False,
        }
    if mode == "tf_sf7d":
        return {
            "signature": "L4EUse, L4IUse, L6Use, ContrastEff, OriCos2, lgnTFUse, lgnSFUse",
            "physical_order_comment": "[L4E, L4I, L6, ContrastEff, OriCos2, lgnTF, lgnSF]",
            "internal_order_comment": "[OriCos2, Contrast, L6, L4E, L4I, lgnTF, lgnSF]",
            "expand_fields": ("L4IUse", "L6Use", "ContrastEff", "OriCos2", "lgnTFUse", "lgnSFUse"),
            "x_expr": "[OriCos2(:), ContrastEff(:), L6Use(:), L4EUse(:), L4IUse(:), lgnTFUse(:), lgnSFUse(:)]",
            "local_response_comment": "% Translate runtime variables into the 7D TF/SF training contract:",
            "local_response_contract": "% [L4E, L4I, L6, ContrastEff, OriCos2, lgnTF, lgnSF].",
            "predict_call_args": "L4EUse, L4IUse, L6Use, contrastEff, oriCos2, lgnTFUse, lgnSFUse",
            "bg_extra": "",
            "fg_extra": "",
            "need_orisin": False,
            "need_tfsf": True,
        }
    raise ValueError(f"Unsupported feature mode {feature_mode!r}")


def feature_norm_vectors(norm: Dict[str, Dict[str, float]], feature_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    contract = feature_contract(feature_mode)
    if contract["need_orisin"]:
        mu = torch.tensor([
            0.0,
            0.0,
            norm["Contrast"]["mean"],
            norm["L6"]["mean"],
            norm["L4E"]["mean"],
            norm["L4I"]["mean"],
        ], dtype=torch.float32)
        sd = torch.tensor([
            1.0,
            1.0,
            norm["Contrast"]["std"],
            norm["L6"]["std"],
            norm["L4E"]["std"],
            norm["L4I"]["std"],
        ], dtype=torch.float32)
    elif contract.get("need_tfsf", False):
        mu = torch.tensor([
            0.0,
            norm["Contrast"]["mean"],
            norm["L6"]["mean"],
            norm["L4E"]["mean"],
            norm["L4I"]["mean"],
            norm["lgnTF"]["mean"],
            norm["lgnSF"]["mean"],
        ], dtype=torch.float32)
        sd = torch.tensor([
            1.0,
            norm["Contrast"]["std"],
            norm["L6"]["std"],
            norm["L4E"]["std"],
            norm["L4I"]["std"],
            norm["lgnTF"]["std"],
            norm["lgnSF"]["std"],
        ], dtype=torch.float32)
    else:
        mu = torch.tensor([
            0.0,
            norm["Contrast"]["mean"],
            norm["L6"]["mean"],
            norm["L4E"]["mean"],
            norm["L4I"]["mean"],
        ], dtype=torch.float32)
        sd = torch.tensor([
            1.0,
            norm["Contrast"]["std"],
            norm["L6"]["std"],
            norm["L4E"]["std"],
            norm["L4I"]["std"],
        ], dtype=torch.float32)
    return mu, sd


def is_soft_moe_family(model_family: str) -> bool:
    return str(model_family).lower().startswith("soft_moe")


def is_residual_moe_family(model_family: str) -> bool:
    return str(model_family).lower() == "residual_moe3"


def is_patch77_rbf_family(model_family: str) -> bool:
    return str(model_family).lower() == "patch77_rbf_single"


def is_patch77_gated_residual_family(model_family: str) -> bool:
    return str(model_family).lower() == "patch77_gated_residual_single"


def needs_lgn_gate_arg(model_family: str) -> bool:
    return is_soft_moe_family(model_family) or is_residual_moe_family(model_family)


def activation_fn_name(act: str, default: str) -> str:
    return {
        "relu": "act_relu",
        "tanh": "act_tanh",
        "gelu": "act_gelu",
        "silu": "act_silu",
        "rational": "act_rational",
    }.get(str(act).lower(), default)


def emit_activation_assignment(
    lines: List[str],
    out_name: str,
    act: str,
    z_name: str,
    state: Dict[str, torch.Tensor],
    param_prefix: str,
) -> None:
    act_key = str(act).lower()
    act_fn = activation_fn_name(act, "act_relu")
    if act_key == "rational":
        numerator = state[f"{param_prefix}.numerator"]
        denominator = state[f"{param_prefix}.denominator"]
        lines.extend([
            f"{out_name}_num = {matlab_matrix(numerator)};",
            f"{out_name}_den = {matlab_matrix(denominator)};",
            f"{out_name} = {act_fn}({z_name}, {out_name}_num, {out_name}_den);",
        ])
    else:
        lines.append(f"{out_name} = {act_fn}({z_name});")


def emit_single_mlp_layers(lines: List[str], payload: Dict, model_cfg: Dict) -> str:
    layers = extract_linear_layers(payload["state"])
    activation_seq = list(model_cfg.get("activation_seq", []))
    prev = "Xn"
    for li, (layer_name, W, b) in enumerate(layers[:-1], start=1):
        act = activation_seq[li - 1] if li - 1 < len(activation_seq) else "relu"
        lines.extend([
            f"W{li} = {matlab_matrix(W, indent='     ')};",
            f"b{li} = {matlab_matrix(b)};",
            f"Z{li} = {prev} * W{li}.' + b{li};",
        ])
        emit_activation_assignment(lines, f"H{li}", act, f"Z{li}", payload["state"], f"net.{int(layer_name) + 1}")
        prev = f"H{li}"
    last_i = len(layers)
    _, Wl, bl = layers[-1]
    lines.extend([
        f"W{last_i} = {matlab_matrix(Wl, indent='     ')};",
        f"b{last_i} = {matlab_matrix(bl)};",
        f"Z{last_i} = {prev} * W{last_i}.' + b{last_i};",
    ])
    return f"Z{last_i}"


def emit_residual_every_two_layers(lines: List[str], payload: Dict, model_cfg: Dict) -> str:
    state = payload["state"]
    activation_seq = list(model_cfg.get("activation_seq", []))
    block_ids = sorted({
        int(k.split(".")[2])
        for k in state
        if k.startswith("net.blocks.") and k.endswith(".linear1.weight")
    })
    prev = "Xn"
    for bi in block_ids:
        block_name = f"B{bi + 1}"
        prefix = f"net.blocks.{bi}"
        W1 = state[f"{prefix}.linear1.weight"]
        b1 = state[f"{prefix}.linear1.bias"]
        W2 = state[f"{prefix}.linear2.weight"]
        b2 = state[f"{prefix}.linear2.bias"]
        lines.extend([
            f"{block_name}W1 = {matlab_matrix(W1, indent='     ')};",
            f"{block_name}b1 = {matlab_matrix(b1)};",
            f"{block_name}Z1 = {prev} * {block_name}W1.' + {block_name}b1;",
        ])
        act1 = activation_seq[2 * bi] if 2 * bi < len(activation_seq) else "relu"
        emit_activation_assignment(lines, f"{block_name}H1", act1, f"{block_name}Z1", state, f"{prefix}.act1")
        lines.extend([
            f"{block_name}W2 = {matlab_matrix(W2, indent='     ')};",
            f"{block_name}b2 = {matlab_matrix(b2)};",
            f"{block_name}Z2 = {block_name}H1 * {block_name}W2.' + {block_name}b2;",
        ])
        act2 = activation_seq[2 * bi + 1] if 2 * bi + 1 < len(activation_seq) else "relu"
        emit_activation_assignment(lines, f"{block_name}H2", act2, f"{block_name}Z2", state, f"{prefix}.act2")
        skip_w_key = f"{prefix}.skip.weight"
        if skip_w_key in state:
            Wskip = state[skip_w_key]
            bskip = state[f"{prefix}.skip.bias"]
            lines.extend([
                f"{block_name}Wskip = {matlab_matrix(Wskip, indent='     ')};",
                f"{block_name}bskip = {matlab_matrix(bskip)};",
                f"{block_name}Skip = {prev} * {block_name}Wskip.' + {block_name}bskip;",
            ])
        else:
            lines.append(f"{block_name}Skip = {prev};")
        lines.append(f"{block_name}Out = {block_name}H2 + {block_name}Skip;")
        prev = f"{block_name}Out"
    Wout = state["net.out.weight"]
    bout = state["net.out.bias"]
    lines.extend([
        f"Wout = {matlab_matrix(Wout, indent='     ')};",
        f"bout = {matlab_matrix(bout)};",
        f"Zout = {prev} * Wout.' + bout;",
    ])
    return "Zout"


def emit_predictor(target: str, payload: Dict) -> str:
    model_cfg = payload["model_config"]
    model_family = str(model_cfg.get("model_family", "single_mlp")).lower()
    soft_moe = is_soft_moe_family(model_family)
    residual_moe = is_residual_moe_family(model_family)
    n_experts = int(model_cfg.get("n_experts", 3))
    feature_mode = str(model_cfg.get("feature_mode", "pref6d")).lower()
    contract = feature_contract(feature_mode)
    mu, sd = feature_norm_vectors(payload["feature_norm"], feature_mode)
    fn = f"predict_pref6D_{target}"
    signature = contract["signature"]
    if soft_moe or residual_moe:
        signature = f"{signature}, LGNctgrUse"
    lines = [
        f"function y = {fn}({signature}) %#codegen",
        f"% Auto-generated {feature_mode} pref-angle predictor.",
        f"% Physical input order is {contract['physical_order_comment']}.",
        "% Internal feature order matches training exactly:",
        f"% {contract['internal_order_comment']}.",
        "origSize = size(L4EUse);",
    ]
    for field in contract["expand_fields"]:
        lines.append(f"{field} = expand_to_size({field}, origSize, '{field}');")
    if soft_moe or residual_moe:
        lines.append("LGNctgrUse = expand_to_size(LGNctgrUse, origSize, 'LGNctgrUse');")
    lines.append(f"X = {contract['x_expr']};")
    rule = model_cfg.get("constant_baseline_rule")
    if rule is not None:
        if feature_mode != "tf_sf7d":
            raise ValueError("Constant baseline rule requires 7D input")
        lines.extend([
            "% User-assumed constant baseline; compare physical inputs before normalization.",
            "fixedBaseline = X(:,2)==0 | X(:,6)==0 | X(:,7)==0 | X(:,6)>50 | X(:,7)>15;",
            "X(fixedBaseline,:) = 0;",
        ])
    if contract.get("need_tfsf", False):
        for column, name in ((6, "lgnTF"), (7, "lgnSF")):
            transform = str(payload["feature_norm"].get(name, {}).get("transform", "identity")).lower()
            if transform == "log":
                lines.append(f"X(:, {column}) = log(X(:, {column}));")
            elif transform != "identity":
                raise ValueError(f"Unsupported transform {transform!r} for feature {name}")
    lines.extend([
        f"mu = {matlab_matrix(mu)};",
        f"sd = {matlab_matrix(sd)};",
        "Xn = (X - mu) ./ sd;",
    ])
    if model_family == "single_mlp":
        softplus_out = bool(model_cfg.get("softplus_out", False))
        output_semantics = str(model_cfg.get("output_semantics", "")).lower()
        final_z = (
            emit_residual_every_two_layers(lines, payload, model_cfg)
            if has_residual_every_two_state(payload["state"])
            else emit_single_mlp_layers(lines, payload, model_cfg)
        )
        if softplus_out and output_semantics == "raw_rate_softplus":
            lines.extend([
                f"Y = softplus({final_z});",
                *( [f"Y(fixedBaseline) = {float(rule['rate']):.17g};"] if rule is not None else [] ),
                "y = reshape(Y, origSize);",
                "end",
            ])
        else:
            lines.extend([
                f"Y = {final_z};",
                *( [f"Y(fixedBaseline) = {float(rule['rate']):.17g};"] if rule is not None else [] ),
                "y = reshape(Y, origSize);",
                "end",
            ])
    elif is_patch77_rbf_family(model_family) or is_patch77_gated_residual_family(model_family):
        layers = extract_prefixed_linear_layers(payload["state"], "parent.net.")
        activation_seq = list(model_cfg.get("activation_seq", []))
        softplus_out = bool(model_cfg.get("softplus_out", True))
        output_semantics = str(model_cfg.get("output_semantics", "raw_rate_softplus")).lower()
        if not softplus_out or output_semantics != "raw_rate_softplus":
            raise ValueError("patch77_rbf_single export expects raw_rate_softplus parent output")
        patch_centers = payload["state"].get("patch.patch_centers")
        patch_delta = payload["state"].get("patch.delta")
        sigma_alpha = payload["state"].get("patch.sigma_alpha")
        sigma_contrast = payload["state"].get("patch.sigma_contrast")
        if patch_centers is None:
            patch_centers = torch.tensor(model_cfg["patch_centers"], dtype=torch.float32)
        if patch_delta is None and is_patch77_rbf_family(model_family):
            raise ValueError("patch77_rbf_single checkpoint is missing patch.delta")
        if patch_delta is None:
            patch_delta = torch.zeros(patch_centers.shape[0], dtype=torch.float32)
        if sigma_alpha is None:
            sigma_alpha = torch.tensor(float(model_cfg.get("sigma_alpha", 4.0)))
        if sigma_contrast is None:
            sigma_contrast = torch.tensor(float(model_cfg.get("sigma_contrast", 12.0)))
        prev = "Xn"
        for li, (_, W, b) in enumerate(layers[:-1], start=1):
            act = activation_seq[li - 1] if li - 1 < len(activation_seq) else "relu"
            act_fn = activation_fn_name(act, "act_relu")
            lines.extend([
                f"W{li} = {matlab_matrix(W, indent='     ')};",
                f"b{li} = {matlab_matrix(b)};",
                f"Z{li} = {prev} * W{li}.' + b{li};",
                f"H{li} = {act_fn}(Z{li});",
            ])
            prev = f"H{li}"
        last_i = len(layers)
        _, Wl, bl = layers[-1]
        lines.extend([
            f"W{last_i} = {matlab_matrix(Wl, indent='     ')};",
            f"b{last_i} = {matlab_matrix(bl)};",
            f"Z{last_i} = {prev} * W{last_i}.' + b{last_i};",
            f"Yparent = softplus(Z{last_i});",
            f"patchCenters = {matlab_matrix(patch_centers, indent='               ')};",
            f"patchDelta = {matlab_matrix(patch_delta)};",
            f"sigmaAlpha = {matlab_num(float(sigma_alpha.detach().cpu().item()))};",
            f"sigmaContrast = {matlab_num(float(sigma_contrast.detach().cpu().item()))};",
            "gammaPatch = 0.5 .* acosd(max(min(OriCos2(:), 1), -1));",
            "contrastPatch = ContrastEff(:);",
            "Gpatch = zeros(size(Xn,1), size(patchCenters,1));",
            "for jj = 1:size(patchCenters,1)",
            "    da = (gammaPatch - patchCenters(jj,1)) ./ sigmaAlpha;",
            "    dc = (contrastPatch - patchCenters(jj,2)) ./ sigmaContrast;",
            "    Gpatch(:,jj) = exp(-0.5 .* da.^2 - 0.5 .* dc.^2);",
            "end",
        ])
        if is_patch77_rbf_family(model_family):
            lines.extend([
                "Y = max(Yparent + Gpatch * patchDelta(:), 0);",
                "y = reshape(Y, origSize);",
                "end",
            ])
        else:
            residual_layers = extract_prefixed_linear_layers(payload["state"], "residual.")
            lines.extend([
                "Gsum = min(sum(Gpatch, 2), 1);",
                "Xadapter = [Xn, Yparent, Gpatch, Gsum];",
            ])
            prev = "Xadapter"
            for li, (_, W, b) in enumerate(residual_layers[:-1], start=1):
                lines.extend([
                    f"Wr{li} = {matlab_matrix(W, indent='      ')};",
                    f"br{li} = {matlab_matrix(b)};",
                    f"Zr{li} = {prev} * Wr{li}.' + br{li};",
                    f"Hr{li} = act_tanh(Zr{li});",
                ])
                prev = f"Hr{li}"
            _, Wrl, brl = residual_layers[-1]
            lines.extend([
                f"Wr{len(residual_layers)} = {matlab_matrix(Wrl, indent='      ')};",
                f"br{len(residual_layers)} = {matlab_matrix(brl)};",
                f"DeltaRaw = {prev} * Wr{len(residual_layers)}.' + br{len(residual_layers)};",
                "Y = max(Yparent + Gsum .* DeltaRaw, 0);",
                "y = reshape(Y, origSize);",
                "end",
            ])
    elif model_family == "double_head_additive":
        shape_layers = extract_prefixed_linear_layers(payload["state"], "shape_head.")
        contrast_layers = extract_prefixed_linear_layers(payload["state"], "contrast_head.")
        shape_act_seq = list(model_cfg.get("shape_activation_seq", []))
        contrast_act_seq = list(model_cfg.get("contrast_activation_seq", []))
        lines.extend([
            "Xshape = Xn(:, [1, 4, 5, 3]);",
            "Xcontr = Xn(:, [2, 1]);",
        ])

        prev = "Xshape"
        for li, (_, W, b) in enumerate(shape_layers[:-1], start=1):
            act = shape_act_seq[li - 1] if li - 1 < len(shape_act_seq) else "tanh"
            act_fn = activation_fn_name(act, "act_relu")
            lines.extend([
                f"Ws{li} = {matlab_matrix(W, indent='      ')};",
                f"bs{li} = {matlab_matrix(b)};",
                f"Zs{li} = {prev} * Ws{li}.' + bs{li};",
                f"Hs{li} = {act_fn}(Zs{li});",
            ])
            prev = f"Hs{li}"
        _, Wsl, bsl = shape_layers[-1]
        lines.extend([
            f"Ws{len(shape_layers)} = {matlab_matrix(Wsl, indent='      ')};",
            f"bs{len(shape_layers)} = {matlab_matrix(bsl)};",
            f"Zshape = {prev} * Ws{len(shape_layers)}.' + bs{len(shape_layers)};",
        ])

        prev = "Xcontr"
        for li, (_, W, b) in enumerate(contrast_layers[:-1], start=1):
            act = contrast_act_seq[li - 1] if li - 1 < len(contrast_act_seq) else "tanh"
            act_fn = activation_fn_name(act, "act_relu")
            lines.extend([
                f"Wc{li} = {matlab_matrix(W, indent='      ')};",
                f"bc{li} = {matlab_matrix(b)};",
                f"Zc{li} = {prev} * Wc{li}.' + bc{li};",
                f"Hc{li} = {act_fn}(Zc{li});",
            ])
            prev = f"Hc{li}"
        _, Wcl, bcl = contrast_layers[-1]
        lines.extend([
            f"Wc{len(contrast_layers)} = {matlab_matrix(Wcl, indent='      ')};",
            f"bc{len(contrast_layers)} = {matlab_matrix(bcl)};",
            f"Zcontr = {prev} * Wc{len(contrast_layers)}.' + bc{len(contrast_layers)};",
            "Y = max(expm1_safe(Zshape + Zcontr), 0);",
            "y = reshape(Y, origSize);",
            "end",
        ])
    elif residual_moe:
        base_layers = extract_prefixed_linear_layers(payload["state"], "base_net.net.")
        base_hidden = list(model_cfg.get("base_activation_seq", model_cfg.get("expert_activation_seq", model_cfg.get("activation_seq", []))))
        expert_hidden = list(model_cfg.get("expert_activation_seq", model_cfg.get("activation_seq", [])))
        lines.extend([
            "gateOriCos2 = Xn(:,1);",
        ])
        if contract["need_orisin"]:
            lines.append("gateOriSin2 = Xn(:,2);")
        else:
            lines.extend([
                "gateOriCos2Raw = max(min(OriCos2(:), 1), -1);",
                "gateOriSin2 = sqrt(max(0, 1 - gateOriCos2Raw.^2));",
            ])
        contrast_col = 3 if contract["need_orisin"] else 2
        lines.extend([
            f"gateContrast = Xn(:,{contrast_col});",
            "gateLGN = zeros(size(Xn,1), 5);",
            "gateIdx = max(min(round(LGNctgrUse(:)), 5), 1);",
            "for row = 1:size(Xn,1)",
            "    gateLGN(row, gateIdx(row)) = 1;",
            "end",
            "Xgate = [gateOriCos2, gateOriSin2, gateContrast, gateLGN];",
        ])
        gate_layers = extract_prefixed_linear_layers(payload["state"], "gate.net.")
        prev = "Xgate"
        for li, (_, W, b) in enumerate(gate_layers[:-1], start=1):
            act_fn = activation_fn_name("tanh", "act_tanh")
            lines.extend([
                f"Wg{li} = {matlab_matrix(W, indent='      ')};",
                f"bg{li} = {matlab_matrix(b)};",
                f"Zg{li} = {prev} * Wg{li}.' + bg{li};",
                f"Hg{li} = {act_fn}(Zg{li});",
            ])
            prev = f"Hg{li}"
        _, Wgl, bgl = gate_layers[-1]
        lines.extend([
            f"Wg{len(gate_layers)} = {matlab_matrix(Wgl, indent='      ')};",
            f"bg{len(gate_layers)} = {matlab_matrix(bgl)};",
            f"Zgate = {prev} * Wg{len(gate_layers)}.' + bg{len(gate_layers)};",
            "G = softmax_rows(Zgate);",
        ])

        prev = "Xn"
        for li, (_, W, b) in enumerate(base_layers[:-1], start=1):
            act = base_hidden[li - 1] if li - 1 < len(base_hidden) else "tanh"
            act_fn = activation_fn_name(act, "act_tanh")
            lines.extend([
                f"Wb{li} = {matlab_matrix(W, indent='      ')};",
                f"bb{li} = {matlab_matrix(b)};",
                f"Zb{li} = {prev} * Wb{li}.' + bb{li};",
                f"Hb{li} = {act_fn}(Zb{li});",
            ])
            prev = f"Hb{li}"
        _, Wbl, bbl = base_layers[-1]
        lines.extend([
            f"Wb{len(base_layers)} = {matlab_matrix(Wbl, indent='      ')};",
            f"bb{len(base_layers)} = {matlab_matrix(bbl)};",
            f"Zbase = {prev} * Wb{len(base_layers)}.' + bb{len(base_layers)};",
        ])

        residual_terms = []
        for expert_idx in range(n_experts):
            expert_layers = extract_prefixed_linear_layers(payload["state"], f"experts.{expert_idx}.net.")
            prev = "Xn"
            for li, (_, W, b) in enumerate(expert_layers[:-1], start=1):
                act = expert_hidden[li - 1] if li - 1 < len(expert_hidden) else "tanh"
                act_fn = activation_fn_name(act, "act_tanh")
                lines.extend([
                    f"We{expert_idx+1}_{li} = {matlab_matrix(W, indent='        ')};",
                    f"be{expert_idx+1}_{li} = {matlab_matrix(b)};",
                    f"Ze{expert_idx+1}_{li} = {prev} * We{expert_idx+1}_{li}.' + be{expert_idx+1}_{li};",
                    f"He{expert_idx+1}_{li} = {act_fn}(Ze{expert_idx+1}_{li});",
                ])
                prev = f"He{expert_idx+1}_{li}"
            _, Wel, bel = expert_layers[-1]
            lines.extend([
                f"We{expert_idx+1}_{len(expert_layers)} = {matlab_matrix(Wel, indent='        ')};",
                f"be{expert_idx+1}_{len(expert_layers)} = {matlab_matrix(bel)};",
                f"R{expert_idx+1} = {prev} * We{expert_idx+1}_{len(expert_layers)}.' + be{expert_idx+1}_{len(expert_layers)};",
            ])
            residual_terms.append(f"G(:,{expert_idx+1}).*R{expert_idx+1}")
        mix_terms = " + ".join(residual_terms)
        lines.extend([
            f"Y = softplus(Zbase + {mix_terms});",
            "y = reshape(Y, origSize);",
            "end",
        ])
    elif soft_moe:
        expert_hidden = list(model_cfg.get("expert_activation_seq", model_cfg.get("activation_seq", [])))
        lines.extend([
            "gateOriCos2 = Xn(:,1);",
        ])
        if contract["need_orisin"]:
            lines.append("gateOriSin2 = Xn(:,2);")
        else:
            lines.extend([
                "gateOriCos2Raw = max(min(OriCos2(:), 1), -1);",
                "gateOriSin2 = sqrt(max(0, 1 - gateOriCos2Raw.^2));",
            ])
        contrast_col = 3 if contract["need_orisin"] else 2
        lines.extend([
            f"gateContrast = Xn(:,{contrast_col});",
            "gateLGN = zeros(size(Xn,1), 5);",
            "gateIdx = max(min(round(LGNctgrUse(:)), 5), 1);",
            "for row = 1:size(Xn,1)",
            "    gateLGN(row, gateIdx(row)) = 1;",
            "end",
            "Xgate = [gateOriCos2, gateOriSin2, gateContrast, gateLGN];",
        ])
        gate_layers = extract_prefixed_linear_layers(payload["state"], "gate.net.")
        prev = "Xgate"
        for li, (_, W, b) in enumerate(gate_layers[:-1], start=1):
            act_fn = activation_fn_name("tanh", "act_tanh")
            lines.extend([
                f"Wg{li} = {matlab_matrix(W, indent='      ')};",
                f"bg{li} = {matlab_matrix(b)};",
                f"Zg{li} = {prev} * Wg{li}.' + bg{li};",
                f"Hg{li} = {act_fn}(Zg{li});",
            ])
            prev = f"Hg{li}"
        _, Wgl, bgl = gate_layers[-1]
        lines.extend([
            f"Wg{len(gate_layers)} = {matlab_matrix(Wgl, indent='      ')};",
            f"bg{len(gate_layers)} = {matlab_matrix(bgl)};",
            f"Zgate = {prev} * Wg{len(gate_layers)}.' + bg{len(gate_layers)};",
            "G = softmax_rows(Zgate);",
        ])

        expert_preds = []
        for expert_idx in range(n_experts):
            expert_layers = extract_prefixed_linear_layers(payload["state"], f"experts.{expert_idx}.net.")
            prev = "Xn"
            for li, (_, W, b) in enumerate(expert_layers[:-1], start=1):
                act = expert_hidden[li - 1] if li - 1 < len(expert_hidden) else "tanh"
                act_fn = activation_fn_name(act, "act_tanh")
                lines.extend([
                    f"We{expert_idx+1}_{li} = {matlab_matrix(W, indent='        ')};",
                    f"be{expert_idx+1}_{li} = {matlab_matrix(b)};",
                    f"Ze{expert_idx+1}_{li} = {prev} * We{expert_idx+1}_{li}.' + be{expert_idx+1}_{li};",
                    f"He{expert_idx+1}_{li} = {act_fn}(Ze{expert_idx+1}_{li});",
                ])
                prev = f"He{expert_idx+1}_{li}"
            _, Wel, bel = expert_layers[-1]
            lines.extend([
                f"We{expert_idx+1}_{len(expert_layers)} = {matlab_matrix(Wel, indent='        ')};",
                f"be{expert_idx+1}_{len(expert_layers)} = {matlab_matrix(bel)};",
                f"Ze{expert_idx+1} = {prev} * We{expert_idx+1}_{len(expert_layers)}.' + be{expert_idx+1}_{len(expert_layers)};",
                f"Ye{expert_idx+1} = softplus(Ze{expert_idx+1});",
            ])
            expert_preds.append(f"Ye{expert_idx+1}")
        mix_terms = " + ".join([f"G(:,{idx+1}).*{name}" for idx, name in enumerate(expert_preds)])
        lines.extend([
            f"Y = {mix_terms};",
            "y = reshape(Y, origSize);",
            "end",
        ])
    else:
        raise ValueError(f"MATLAB export does not support model_family={model_family!r}.")
    lines.extend([
        "",
        "function out = expand_to_size(x, targetSize, name)",
        "if isscalar(x)",
        "    out = repmat(x, targetSize);",
        "elseif isequal(size(x), targetSize)",
        "    out = x;",
        "else",
        "    error('%s must be scalar or size-matched to L4EUse.', name);",
        "end",
        "end",
        "",
        "function y = act_relu(x)",
        "y = max(x, 0);",
        "end",
        "",
        "function y = act_tanh(x)",
        "y = builtin('tanh', x);",
        "end",
        "",
        "function y = act_gelu(x)",
        "y = 0.5 .* x .* (1 + erf(x ./ sqrt(2)));",
        "end",
        "",
        "function y = act_silu(x)",
        "y = x ./ (1 + exp(-x));",
        "end",
        "",
        "function y = act_rational(x, numerator, denominator)",
        "y = (numerator(1) + numerator(2).*x + numerator(3).*x.^2 + numerator(4).*x.^3) ./ ...",
        "    (1 + abs(denominator(1).*x + denominator(2).*x.^2));",
        "end",
        "",
        "function y = softplus(x)",
        "y = log1p(exp(-abs(x))) + max(x, 0);",
        "end",
        "",
        "function y = softmax_rows(x)",
        "xShift = x - max(x, [], 2);",
        "ex = exp(xShift);",
        "y = ex ./ sum(ex, 2);",
        "end",
        "",
        "function y = expm1_safe(x)",
        "y = exp(x) - 1;",
        "end",
    ])
    return "\n".join(lines) + "\n"


def emit_bundle(feature_mode: str, model_family: str) -> str:
    contract = feature_contract(feature_mode)
    sig = contract["signature"]
    if needs_lgn_gate_arg(model_family):
        sig = f"{sig}, LGNctgrUse"
    call = contract["predict_call_args"].replace("contrastEff", "ContrastEff").replace("oriCos2", "OriCos2")
    if contract["need_orisin"]:
        call = call.replace("oriSin2", "OriSin2")
    if needs_lgn_gate_arg(model_family):
        call = f"{call}, LGNctgrUse"
    return f"""function [rate_S, rate_C, rate_I] = predict_pref6D_bundle({sig}) %#codegen
rate_S = predict_pref6D_S({call});
rate_C = predict_pref6D_C({call});
rate_I = predict_pref6D_I({call});
end
"""


def emit_local_response(feature_mode: str, model_family: str) -> str:
    contract = feature_contract(feature_mode)
    maybe_orisin_init = "oriSin2 = zeros(size(alphaUse));\n" if contract["need_orisin"] else ""
    maybe_orisin_fg = contract["fg_extra"] + "\n" if contract["fg_extra"] else ""
    maybe_orisin_bg = contract["bg_extra"] + "\n" if contract["bg_extra"] else ""
    predict_call_args = contract["predict_call_args"]
    if needs_lgn_gate_arg(model_family):
        predict_call_args = f"{predict_call_args}, LGNctgrUse"
    maybe_tfsf_init = ""
    if contract.get("need_tfsf", False):
        maybe_tfsf_init = "global H96_LGN_TF_USE H96_LGN_SF_USE;\nif isempty(H96_LGN_TF_USE), H96_LGN_TF_USE = 10; end\nif isempty(H96_LGN_SF_USE), H96_LGN_SF_USE = 2.5; end\nlgnTFUse = H96_LGN_TF_USE .* ones(size(alphaUse));\nlgnSFUse = H96_LGN_SF_USE .* ones(size(alphaUse));\n"
    return f"""function LDEOutLIBy = LocalResponse_6D_MLP_prefAngle(celltype, alphaUse, LGNctgrUse, L4EUse, L4IUse, L6Use, ContrastUse) %#codegen
{contract['local_response_comment']}
{contract['local_response_contract']}
if ~isequal(size(alphaUse), size(LGNctgrUse), size(L4EUse), size(L4IUse), size(L6Use), size(ContrastUse))
    error('All LocalResponse inputs must share shape.');
end

thetaPref = zeros(size(alphaUse));
gammaUse = zeros(size(alphaUse));
oriCos2 = ones(size(alphaUse));
{maybe_orisin_init}{maybe_tfsf_init}contrastEff = ContrastUse;

fgMask = (LGNctgrUse >= 1) & (LGNctgrUse <= 4);
if any(fgMask(:))
    thetaMap = [0, 135, 90, 45];
    thetaPref(fgMask) = thetaMap(double(LGNctgrUse(fgMask)));
    gammaUse(fgMask) = abs(wrapTo90_pref(double(alphaUse(fgMask)) - thetaPref(fgMask)));
    oriCos2(fgMask) = cosd(2 .* gammaUse(fgMask));
{maybe_orisin_fg}end

bgMask = (LGNctgrUse == 5);
if any(bgMask(:))
    contrastEff(bgMask) = 0;
    gammaUse(bgMask) = abs(wrapTo90_pref(double(alphaUse(bgMask)) - thetaPref(bgMask)));
    oriCos2(bgMask) = cosd(2 .* gammaUse(bgMask));
{maybe_orisin_bg}end

switch celltype
    case 'S'
        LDEOutLIBy = predict_pref6D_S({predict_call_args});
    case 'C'
        LDEOutLIBy = predict_pref6D_C({predict_call_args});
    case 'I'
        LDEOutLIBy = predict_pref6D_I({predict_call_args});
    otherwise
        error('Unknown celltype');
end
end

function thetaWrapped = wrapTo90_pref(thetaDeg)
thetaWrapped = mod(thetaDeg + 90, 180) - 90;
end
"""


def emit_iterfunc() -> str:
    return """function LDEOutS = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
    celltype, L4EUse, L4IUse, ContrastUse, AlphaUse, PixLGNCtgr, L6AxisUse)

if ~isequal(size(L4EUse), size(L4IUse), size(ContrastUse), size(AlphaUse))
    error('L4EUse, L4IUse, ContrastUse, AlphaUse must have the same shape.');
end
if size(PixLGNCtgr, 1) ~= numel(L4EUse) || size(PixLGNCtgr, 2) ~= 5
    error('PixLGNCtgr must be N-by-5 where N = numel(L4EUse).');
end

LibyAll = zeros(length(L4EUse),5);
for LGNInd = 1:5
    LGNctgrUse = LGNInd * ones(size(AlphaUse));
    LibyAll(:,LGNInd) = LocalResponse_6D_MLP_prefAngle(...
        celltype, AlphaUse, LGNctgrUse, L4EUse, L4IUse, L6AxisUse, ContrastUse);
end
LDEOutS = sum(PixLGNCtgr .* LibyAll, 2);
end
"""


def emit_iteration_main() -> str:
    return """function [LDERepFinal,LDEIL2Diff,L2DiffNormNeib,L2Diameter,LDEequv,FuncUseAll,NANFinal] = ...
    LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle(...
    PixLGNCtgr,L6Kernel,LDEIni,p,L6pars,Epoc,...
    C_SS_mean,C_CS_mean,C_IS_mean,...
    C_SC_mean,C_CC_mean,C_IC_mean,...
    C_SI_mean,C_CI_mean,C_II_mean,...
    L4SEp, L4SIp, ...
    L4CEp, L4CIp, ...
    L4IEp, L4IIp, ...
    L4EmeshXAll,L4ImeshYAll,LDEFrfuncAll, ...
    ContrastUse, AlphaUse, varargin)

% Compatibility note:
% L4EmeshXAll, L4ImeshYAll, and LDEFrfuncAll are retained to preserve the
% old call signature, but they are unused because response lookup now comes
% from the exported 6D pref-angle neural networks.

NANFinal = false;

if ~isequal(size(ContrastUse), size(AlphaUse), size(LDEIni.S))
    error('ContrastUse, AlphaUse, and LDE fields must share shape.');
end

if ~isempty(varargin)
    N_HCOutY = varargin{1};
    NPixX = varargin{2};
    NPixY = varargin{3};
else
    N_HCOutY = 4; NPixX = 10; NPixY = 10;
end

if length(varargin)>3
    InhKillFlag = varargin{4};
else
    InhKillFlag = true;
end

if length(varargin)>4
   Outflag = varargin{5};
else
   Outflag = 'xn';
end

if length(varargin)>5
    EKp = varargin{6};
    IKp = varargin{7};
else
    IKp.Thrsld = 60; IKp.Highist = 120; IKp.Slope = 0.9;
end

LDEItr = cell(Epoc+1,1);LDEItr{1} = LDEIni;
LDEEpoOut = cell(Epoc+1,1); LDEEpoOut{1} = LDEIni;
LDEoutVec = zeros(size(LDEIni.I,1)*3,Epoc+1);
LDEoutVec(:,1) = [LDEIni.S;LDEIni.C;LDEIni.I];
FuncUseAll = ones(Epoc,1);

NANGlobalFlag = false;
for Epc = 1:Epoc
    if NANGlobalFlag
        continue
    end

    LDEInpt = LDEItr{Epc};
    LDEUse = LDEInpt;
    if InhKillFlag
        LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;
        LDEUseEAdj = L6Convert(LDEUse.E,EKp)./LDEUse.E;
        LDEUse.S = LDEUse.S .* LDEUseEAdj;
        LDEUse.C = LDEUse.C .* LDEUseEAdj;
        LDEUse.I = InhMulp(LDEUse.I, IKp);
    end
    LDEUse.E = LDEUse.S * (1-0.3077) + LDEUse.C * 0.3077;

    % Reconstruct one shared L4E/L4I pair, matching the 6D training table.
    L4SEUse = C_SS_mean*LDEUse.S + C_SC_mean*LDEUse.C;
    L4CEUse = C_CS_mean*LDEUse.S + C_CC_mean*LDEUse.C;
    L4IEUse = C_IS_mean*LDEUse.S + C_IC_mean*LDEUse.C;

    L4SIUse = C_SI_mean*LDEUse.I;
    L4CIUse = C_CI_mean*LDEUse.I;
    L4IIUse = C_II_mean*LDEUse.I;

    L4EUse_shared = L4SEUse + L4CEUse + L4IEUse;
    L4IUse_shared = L4SIUse + L4CIUse + L4IIUse;

    FieldRow = N_HCOutY * NPixY;
    FieldCol = floor(length(LDEUse.E)/FieldRow);
    L4Efield = reshape(LDEUse.E,FieldRow,FieldCol);
    L4Efield_padded = padarray(L4Efield, [1, 1], 'circular');
    C = conv2(L4Efield_padded, L6Kernel, 'same');
    L6EUse = L6Convert(C(2:end-1, 2:end-1),L6pars);

    L6EUse(L6EUse>120) = 120; L6EUse(L6EUse<3) = 3;
    L6AxisUse = L6EUse(:)/3;
    L6AxisUse(L6AxisUse<1) = 1; L6AxisUse(L6AxisUse>40) = 40;
    L6AxisUse = single(L6AxisUse);

    LDEOut = struct('S',[],'C',[],'I',[]);
    LDEOut.S = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
        'S',L4EUse_shared,L4IUse_shared,ContrastUse,AlphaUse,PixLGNCtgr,L6AxisUse);
    LDEOut.C = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
        'C',L4EUse_shared,L4IUse_shared,ContrastUse,AlphaUse,PixLGNCtgr,L6AxisUse);
    LDEOut.I = LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle(...
        'I',L4EUse_shared,L4IUse_shared,ContrastUse,AlphaUse,PixLGNCtgr,L6AxisUse);

    nanFlag = any(isnan([LDEOut.S,LDEOut.C,LDEOut.I]),'all');
    if nanFlag
        fprintf('***Warning! NAN results in the %d epoch. Returning...\\n',Epc)
        NANGlobalFlag = true;
    end

    LDENext = struct(...
        'S',LDEOut.S*p + LDEInpt.S*(1-p),...
        'C',LDEOut.C*p + LDEInpt.C*(1-p),...
        'I',LDEOut.I*p + LDEInpt.I*(1-p));
    LDEItr{Epc+1} = LDENext; LDEEpoOut{Epc+1} = LDEOut;
    LDEoutVec(:,Epc+1) = [LDEOut.S; LDEOut.C; LDEOut.I];
end

if strcmpi(Outflag,'f(xn)')
   LDERepFinal = LDEoutVec;
elseif strcmpi(Outflag,'xn')
   LDERepFinal = LDEItr;
else
   LDERepFinal = LDEItr;
end

L2DiffNormNeib = zeros(Epoc,1);
L2Diameter = zeros(Epoc,1);
LDEequv = [];
LDEIL2Diff = [];
NANFinal = NANGlobalFlag;
if ~NANGlobalFlag
    LDEequv = mean(LDEoutVec(:,floor((Epc+1)*2/3):end),2);
    LDEIL2Diff = sqrt(sum((LDEoutVec - repmat(LDEequv,1,Epoc+1)).^2, 1));
end
end
"""


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 6D pref-angle checkpoints to MATLAB functions")
    ap.add_argument("--model_s", required=True)
    ap.add_argument("--model_c", required=True)
    ap.add_argument("--model_i", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        "S": load_ckpt(Path(args.model_s)),
        "C": load_ckpt(Path(args.model_c)),
        "I": load_ckpt(Path(args.model_i)),
    }
    feature_mode = str(payloads["S"]["model_config"].get("feature_mode", "pref6d")).lower()
    model_family = str(payloads["S"]["model_config"].get("model_family", "single_mlp")).lower()

    for target, payload in payloads.items():
        write_text(out_dir / f"predict_pref6D_{target}.m", emit_predictor(target, payload))
    write_text(out_dir / "predict_pref6D_bundle.m", emit_bundle(feature_mode, model_family))
    write_text(out_dir / "LocalResponse_6D_MLP_prefAngle.m", emit_local_response(feature_mode, model_family))
    write_text(out_dir / "LDEIterFunc_Grating_135Func_RealLGNL6_MLP6D_prefAngle.m", emit_iterfunc())
    write_text(out_dir / "LDEIteration_135FuncMain_CombDom_RealLGNL6_MLP6D_prefAngle.m", emit_iteration_main())
    print(f"[done] MATLAB export written to {out_dir}")


if __name__ == "__main__":
    main()
