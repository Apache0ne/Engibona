#!/usr/bin/env python3
"""Test whether released low-bit matrices preserve paired/block operators.

Raw weight MSE cannot distinguish arbitrary coordinate damage from useful
cross-matrix compensation. This experiment compares released and naive low-bit
weights on Q/K attention scores, V/O composition, and full SwiGLU MLP outputs
for selected public Qwen3-1.7B layers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot import streaming forensic helpers")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


SUFFIXES = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def target_names(layers: list[int]) -> dict[str, tuple[int, str]]:
    result = {}
    for layer in layers:
        for short, suffix in SUFFIXES.items():
            name = f"model.layers.{layer}.{suffix}"
            result[name] = (layer, short)
    return result


def collect_full(repo_id: str, work: Path, layers: list[int]):
    targets = target_names(layers)
    files, weight_map = core.repo_layout(repo_id, work)
    by_file: dict[str, list[str]] = defaultdict(list)
    if weight_map is not None:
        for name in targets:
            filename = weight_map.get(name)
            if filename is None:
                raise RuntimeError(f"{repo_id}: missing {name}")
            by_file[filename].append(name)
    else:
        by_file[files[0]] = list(targets)

    output: dict[int, dict[str, torch.Tensor]] = {
        layer: {} for layer in layers
    }
    for filename in files:
        names = by_file.get(filename, [])
        if not names:
            continue
        local = core.download_file(repo_id, filename, work / Path(filename).name)
        with safe_open(str(local), framework="pt", device="cpu") as handle:
            for name in names:
                layer, short = targets[name]
                output[layer][short] = handle.get_tensor(name).half().contiguous()
        local.unlink(missing_ok=True)
    for layer in layers:
        missing = set(SUFFIXES) - set(output[layer])
        if missing:
            raise RuntimeError(f"{repo_id}: layer {layer} missing {sorted(missing)}")
    return output


def quantize_matrix(weight: torch.Tensor, mode: str, group_size: int):
    shape = weight.shape
    groups = weight.float().reshape(-1, group_size)
    if mode == "binary":
        _, _, quantized = core.naive_binary(groups)
    elif mode == "ternary":
        _, _, quantized = core.naive_ternary(groups)
    else:
        raise ValueError(mode)
    return quantized.reshape(shape)


def cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(
            reference.float().reshape(-1),
            candidate.float().reshape(-1),
            dim=0,
        )
    )


def nmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(
        (candidate.float() - reference.float()).square().mean()
        / reference.float().square().mean().clamp_min(1e-20)
    )


def scalar_aligned_nmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    scale = (candidate * reference).sum() / candidate.square().sum().clamp_min(1e-20)
    return float(
        (scale * candidate - reference).square().mean()
        / reference.square().mean().clamp_min(1e-20)
    )


def metrics(reference: torch.Tensor, candidate: torch.Tensor):
    return {
        "nmse": nmse(reference, candidate),
        "scalar_aligned_nmse": scalar_aligned_nmse(reference, candidate),
        "cosine": cosine(reference, candidate),
    }


def attention_scores(
    weights: dict[str, torch.Tensor],
    xq: torch.Tensor,
    xk: torch.Tensor,
    heads: int,
    kv_heads: int,
    head_dim: int,
):
    q = xq @ weights["q"].float().T
    k = xk @ weights["k"].float().T
    q = q.reshape(xq.shape[0], heads, head_dim)
    k = k.reshape(xk.shape[0], kv_heads, head_dim)
    k = k.repeat_interleave(heads // kv_heads, dim=1)
    return torch.einsum("thd,shd->hts", q, k) / math.sqrt(head_dim)


def attention_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    result = metrics(reference, candidate)
    teacher_probability = reference.float().softmax(dim=-1)
    result["softmax_kl"] = float(
        F.kl_div(
            candidate.float().log_softmax(dim=-1),
            teacher_probability,
            reduction="batchmean",
        )
    )
    return result


def vo_output(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    heads: int,
    kv_heads: int,
    head_dim: int,
):
    value = x @ weights["v"].float().T
    value = value.reshape(x.shape[0], kv_heads, head_dim)
    value = value.repeat_interleave(heads // kv_heads, dim=1)
    return value.reshape(x.shape[0], heads * head_dim) @ weights["o"].float().T


def mlp_output(weights: dict[str, torch.Tensor], x: torch.Tensor):
    gate = x @ weights["gate"].float().T
    up = x @ weights["up"].float().T
    hidden = F.silu(gate) * up
    return hidden @ weights["down"].float().T


def evaluate_layer(
    base: dict[str, torch.Tensor],
    release: dict[str, torch.Tensor],
    mode: str,
    group_size: int,
    probes: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    input_dim = base["q"].shape[1]
    xq = torch.randn(probes, input_dim, generator=generator)
    xk = torch.randn(probes, input_dim, generator=generator)
    xv = torch.randn(probes, input_dim, generator=generator)
    xm = torch.randn(probes, input_dim, generator=generator)

    naive = {
        name: quantize_matrix(weight, mode, group_size)
        for name, weight in base.items()
    }
    base_float = {name: value.float() for name, value in base.items()}
    release_float = {name: value.float() for name, value in release.items()}

    base_scores = attention_scores(base_float, xq, xk, heads, kv_heads, head_dim)
    base_vo = vo_output(base_float, xv, heads, kv_heads, head_dim)
    base_mlp = mlp_output(base_float, xm)

    output = {}
    for label, weights in (("released", release_float), ("naive", naive)):
        output[label] = {
            "qk_attention": attention_metrics(
                base_scores,
                attention_scores(weights, xq, xk, heads, kv_heads, head_dim),
            ),
            "vo_composition": metrics(
                base_vo,
                vo_output(weights, xv, heads, kv_heads, head_dim),
            ),
            "swiglu_mlp": metrics(base_mlp, mlp_output(weights, xm)),
            "individual_weight_nmse": {
                name: nmse(base_float[name], weights[name]) for name in SUFFIXES
            },
        }
    for operator in ("qk_attention", "vo_composition", "swiglu_mlp"):
        released = output["released"][operator]["nmse"]
        naive_value = output["naive"][operator]["nmse"]
        output[f"released_over_naive_{operator}_nmse"] = released / max(naive_value, 1e-20)
        released_aligned = output["released"][operator]["scalar_aligned_nmse"]
        naive_aligned = output["naive"][operator]["scalar_aligned_nmse"]
        output[f"released_over_naive_{operator}_aligned_nmse"] = (
            released_aligned / max(naive_aligned, 1e-20)
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 13, 27])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--output", default="paired_operator_forensics.json")
    args = parser.parse_args()
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="paired-bonsai-") as temporary:
        work = Path(temporary)
        base = collect_full(args.base, work / "base", args.layers)
        binary = collect_full(args.binary, work / "binary", args.layers)
        ternary = collect_full(args.ternary, work / "ternary", args.layers)

        result: dict[str, Any] = {
            "repositories": {
                "base": args.base,
                "binary": args.binary,
                "ternary": args.ternary,
            },
            "layers": args.layers,
            "probes": args.probes,
            "binary": {},
            "ternary": {},
        }
        for layer in args.layers:
            result["binary"][str(layer)] = evaluate_layer(
                base[layer],
                binary[layer],
                "binary",
                args.group_size,
                args.probes,
                args.heads,
                args.kv_heads,
                args.head_dim,
                args.seed + layer,
            )
            result["ternary"][str(layer)] = evaluate_layer(
                base[layer],
                ternary[layer],
                "ternary",
                args.group_size,
                args.probes,
                args.heads,
                args.kv_heads,
                args.head_dim,
                args.seed + 1000 + layer,
            )
        result["seconds"] = time.time() - started
        Path(args.output).write_text(
            json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
