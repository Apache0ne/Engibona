#!/usr/bin/env python3
"""Full-model functional comparison of public Qwen and Bonsai 1.7B.

Models are loaded sequentially on CPU to keep memory bounded:

1. original Qwen teacher;
2. naive binary projection of Qwen;
3. naive ternary projection of Qwen;
4. public unpacked binary Bonsai;
5. public unpacked ternary Bonsai.

The experiment compares prompt-token logits and all decoder hidden states. It is
the direct functional complement to released-weight geometry forensics.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version


DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in three precise sentences.",
    "Compute 37 times 48 and provide a compact derivation.",
    "Write a Python function that returns the longest increasing subsequence length.",
    "Return only JSON with keys action and arguments for a weather lookup in Tokyo.",
]


def load_model(repo_id: str, cache_dir: Path) -> nn.Module:
    common = dict(
        pretrained_model_name_or_path=repo_id,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            dtype=torch.float32,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            torch_dtype=torch.float32,
        )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


def clear_model(model: nn.Module | None, cache_dir: Path | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)
    gc.collect()


def grouped_binary(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    if weight.shape[-1] % group_size:
        raise ValueError(f"binary last dimension {weight.shape[-1]} not divisible by {group_size}")
    groups = weight.float().reshape(-1, group_size)
    codes = torch.where(groups >= 0, 1.0, -1.0)
    scales = groups.abs().mean(dim=1).clamp_min(1e-12)
    return (codes * scales[:, None]).reshape_as(weight)


def grouped_ternary(
    weight: torch.Tensor,
    group_size: int,
    iterations: int = 16,
) -> torch.Tensor:
    if weight.shape[-1] % group_size:
        raise ValueError(f"ternary last dimension {weight.shape[-1]} not divisible by {group_size}")
    groups = weight.float().reshape(-1, group_size)
    scales = groups.abs().mean(dim=1).clamp_min(1e-12)
    for _ in range(iterations):
        active = groups.abs() > 0.5 * scales[:, None]
        scales = (
            (groups.abs() * active).sum(dim=1)
            / active.sum(dim=1).clamp_min(1)
        ).clamp_min(1e-12)
    active = groups.abs() > 0.5 * scales[:, None]
    codes = torch.sign(groups) * active.float()
    return (codes * scales[:, None]).reshape_as(weight)


@torch.no_grad()
def quantize_model_in_place(
    model: nn.Module,
    mode: str,
    group_size: int,
) -> dict[str, Any]:
    seen: set[int] = set()
    tensors = 0
    weights = 0
    skipped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        parameter = module.weight
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if parameter.ndim != 2:
            skipped.append(name)
            continue
        if parameter.shape[-1] % group_size:
            skipped.append(name)
            continue
        quantized = (
            grouped_binary(parameter, group_size)
            if mode == "binary"
            else grouped_ternary(parameter, group_size)
        )
        parameter.copy_(quantized.to(parameter.dtype))
        tensors += 1
        weights += parameter.numel()
        del quantized
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return {
        "mode": mode,
        "tensors": tensors,
        "weights": weights,
        "skipped": skipped,
    }


def tokenize_prompts(tokenizer, prompts: list[str], max_length: int):
    batches = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        batches.append(
            {
                "prompt": prompt,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded.get(
                    "attention_mask", torch.ones_like(encoded["input_ids"])
                ),
            }
        )
    return batches


@torch.no_grad()
def capture_signature(model: nn.Module, batches) -> list[dict[str, Any]]:
    signatures = []
    for batch in batches:
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        signatures.append(
            {
                "prompt": batch["prompt"],
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
                "logits": output.logits.detach().cpu().to(torch.float16),
                "hidden_states": [
                    state.detach().cpu().to(torch.float16)
                    for state in hidden_states
                ],
            }
        )
        del output
        gc.collect()
    return signatures


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1, left.shape[-1])
    right = right.float().reshape(-1, right.shape[-1])
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = cross.square().sum()
    left_norm = (left.T @ left).square().sum().sqrt()
    right_norm = (right.T @ right).square().sum().sqrt()
    return float(numerator / (left_norm * right_norm).clamp_min(1e-20))


def compare_signatures(
    teacher: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_rows = []
    layer_cosines: list[list[float]] = []
    layer_ckas: list[list[float]] = []
    layer_norm_ratios: list[list[float]] = []

    for reference, current in zip(teacher, candidate):
        teacher_logits = reference["logits"].float()
        candidate_logits = current["logits"].float()
        teacher_probability = teacher_logits.softmax(dim=-1)
        token_kl = (
            teacher_probability
            * (
                teacher_probability.clamp_min(1e-12).log()
                - candidate_logits.log_softmax(dim=-1)
            )
        ).sum(dim=-1)
        top1 = (
            teacher_logits.argmax(dim=-1)
            == candidate_logits.argmax(dim=-1)
        ).float()
        logit_cosine = F.cosine_similarity(
            teacher_logits,
            candidate_logits,
            dim=-1,
        )
        teacher_centered = teacher_logits - teacher_logits.mean(dim=-1, keepdim=True)
        candidate_centered = candidate_logits - candidate_logits.mean(dim=-1, keepdim=True)
        centered_rmse = (
            (teacher_centered - candidate_centered).square().mean(dim=-1).sqrt()
        )
        teacher_scale = teacher_centered.square().mean(dim=-1).sqrt().clamp_min(1e-12)

        prompt_layer_cosine = []
        prompt_layer_cka = []
        prompt_layer_norm_ratio = []
        for teacher_hidden, candidate_hidden in zip(
            reference["hidden_states"], current["hidden_states"]
        ):
            teacher_hidden = teacher_hidden.float()
            candidate_hidden = candidate_hidden.float()
            prompt_layer_cosine.append(
                float(
                    F.cosine_similarity(
                        teacher_hidden.reshape(-1, teacher_hidden.shape[-1]),
                        candidate_hidden.reshape(-1, candidate_hidden.shape[-1]),
                        dim=-1,
                    ).mean()
                )
            )
            prompt_layer_cka.append(linear_cka(teacher_hidden, candidate_hidden))
            prompt_layer_norm_ratio.append(
                float(
                    candidate_hidden.square().mean().sqrt()
                    / teacher_hidden.square().mean().sqrt().clamp_min(1e-12)
                )
            )

        layer_cosines.append(prompt_layer_cosine)
        layer_ckas.append(prompt_layer_cka)
        layer_norm_ratios.append(prompt_layer_norm_ratio)
        prompt_rows.append(
            {
                "prompt": reference["prompt"],
                "tokens": int(teacher_logits.shape[1]),
                "token_kl_mean": float(token_kl.mean()),
                "last_token_kl": float(token_kl[:, -1].mean()),
                "top1_agreement": float(top1.mean()),
                "last_token_top1_agreement": float(top1[:, -1].mean()),
                "logit_cosine": float(logit_cosine.mean()),
                "centered_logit_relative_rmse": float(
                    (centered_rmse / teacher_scale).mean()
                ),
                "hidden_cosine_mean": float(torch.tensor(prompt_layer_cosine).mean()),
                "hidden_cka_mean": float(torch.tensor(prompt_layer_cka).mean()),
            }
        )

    layer_cosine_tensor = torch.tensor(layer_cosines)
    layer_cka_tensor = torch.tensor(layer_ckas)
    layer_norm_tensor = torch.tensor(layer_norm_ratios)
    return {
        "prompts": prompt_rows,
        "aggregate": {
            key: float(torch.tensor([row[key] for row in prompt_rows]).mean())
            for key in (
                "token_kl_mean",
                "last_token_kl",
                "top1_agreement",
                "last_token_top1_agreement",
                "logit_cosine",
                "centered_logit_relative_rmse",
                "hidden_cosine_mean",
                "hidden_cka_mean",
            )
        },
        "per_layer": {
            "hidden_cosine": layer_cosine_tensor.mean(dim=0).tolist(),
            "hidden_cka": layer_cka_tensor.mean(dim=0).tolist(),
            "hidden_norm_ratio": layer_norm_tensor.mean(dim=0).tolist(),
        },
    }


def run_candidate(
    name: str,
    repo_id: str,
    cache_dir: Path,
    batches,
    teacher_signature,
    naive_mode: str | None,
    group_size: int,
) -> tuple[dict[str, Any], nn.Module | None]:
    started = time.time()
    model = load_model(repo_id, cache_dir)
    quantization = None
    if naive_mode is not None:
        quantization = quantize_model_in_place(model, naive_mode, group_size)
    signature = capture_signature(model, batches)
    metrics = compare_signatures(teacher_signature, signature)
    result = {
        "name": name,
        "repository": repo_id,
        "quantization": quantization,
        "metrics": metrics,
        "seconds": time.time() - started,
    }
    del signature
    return result, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--output", default="full_model_functional_forensics.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    root = Path("functional_forensics_cache")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    tokenizer_cache = root / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        cache_dir=str(tokenizer_cache),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = tokenize_prompts(tokenizer, DEFAULT_PROMPTS, args.max_length)

    base_cache = root / "base"
    teacher_model = load_model(args.base, base_cache)
    teacher_parameter_count = sum(parameter.numel() for parameter in teacher_model.parameters())
    teacher_signature = capture_signature(teacher_model, batches)
    clear_model(teacher_model)
    teacher_model = None

    results = {}
    naive_binary, model = run_candidate(
        "naive_binary",
        args.base,
        base_cache,
        batches,
        teacher_signature,
        "binary",
        args.group_size,
    )
    results["naive_binary"] = naive_binary
    clear_model(model)

    naive_ternary, model = run_candidate(
        "naive_ternary",
        args.base,
        base_cache,
        batches,
        teacher_signature,
        "ternary",
        args.group_size,
    )
    results["naive_ternary"] = naive_ternary
    clear_model(model, base_cache)

    actual_binary, model = run_candidate(
        "released_binary",
        args.binary,
        root / "binary",
        batches,
        teacher_signature,
        None,
        args.group_size,
    )
    results["released_binary"] = actual_binary
    clear_model(model, root / "binary")

    actual_ternary, model = run_candidate(
        "released_ternary",
        args.ternary,
        root / "ternary",
        batches,
        teacher_signature,
        None,
        args.group_size,
    )
    results["released_ternary"] = actual_ternary
    clear_model(model, root / "ternary")

    comparisons = {}
    for mode in ("binary", "ternary"):
        naive = results[f"naive_{mode}"]["metrics"]["aggregate"]
        released = results[f"released_{mode}"]["metrics"]["aggregate"]
        comparisons[mode] = {
            "released_over_naive_token_kl": (
                released["token_kl_mean"] / max(naive["token_kl_mean"], 1e-20)
            ),
            "released_over_naive_last_token_kl": (
                released["last_token_kl"] / max(naive["last_token_kl"], 1e-20)
            ),
            "released_minus_naive_top1_agreement": (
                released["top1_agreement"] - naive["top1_agreement"]
            ),
            "released_minus_naive_hidden_cosine": (
                released["hidden_cosine_mean"] - naive["hidden_cosine_mean"]
            ),
            "released_minus_naive_hidden_cka": (
                released["hidden_cka_mean"] - naive["hidden_cka_mean"]
            ),
        }

    payload = {
        "repositories": {
            "base": args.base,
            "binary": args.binary,
            "ternary": args.ternary,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "teacher_parameter_count": teacher_parameter_count,
        "group_size": args.group_size,
        "prompts": DEFAULT_PROMPTS,
        "results": results,
        "comparisons": comparisons,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        name: value["metrics"]["aggregate"] for name, value in results.items()
    }, indent=2))
    print(json.dumps(comparisons, indent=2))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
