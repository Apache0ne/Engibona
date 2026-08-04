#!/usr/bin/env python3
"""Memory-bounded BF16 entry point for full-model functional forensics.

The original FP32 attempt exceeded the standard runner resource envelope. This
entry point keeps one 1.7B model in BF16 and quantizes matrices in row chunks.
The public Bonsai checkpoints serialize the unpadded tokenizer vocabulary while
the original Qwen checkpoint includes padded output rows, so logit comparisons
are performed on the shared vocabulary and the excluded probability mass is
reported explicitly.
"""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "full_functional_core",
    HERE / "run_full_model_functional_forensics.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load full functional forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


PROMPTS = [
    "Explain why the sky appears blue in three precise sentences.",
    "Compute 37 times 48 and provide a compact derivation.",
    "Return only JSON with keys action and arguments for a weather lookup in Tokyo.",
]


def load_model_bf16(repo_id: str, cache_dir: Path) -> nn.Module:
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
            dtype=torch.bfloat16,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            **common,
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@torch.no_grad()
def quantize_model_chunked(
    model: nn.Module,
    mode: str,
    group_size: int,
    row_chunk: int = 64,
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
        if parameter.ndim != 2 or parameter.shape[-1] % group_size:
            skipped.append(name)
            continue
        rows = parameter.shape[0]
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            source = parameter[start:end].float()
            quantized = (
                core.grouped_binary(source, group_size)
                if mode == "binary"
                else core.grouped_ternary(source, group_size)
            )
            parameter[start:end].copy_(quantized.to(parameter.dtype))
            del source, quantized
        tensors += 1
        weights += parameter.numel()
        gc.collect()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return {
        "mode": mode,
        "tensors": tensors,
        "weights": weights,
        "skipped": skipped,
        "row_chunk": row_chunk,
        "storage_dtype": "bfloat16",
    }


def compare_signatures_vocab_aligned(
    teacher: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_rows = []
    layer_cosines: list[list[float]] = []
    layer_ckas: list[list[float]] = []
    layer_norm_ratios: list[list[float]] = []

    for reference, current in zip(teacher, candidate):
        teacher_full = reference["logits"].float()
        candidate_full = current["logits"].float()
        shared_vocabulary = min(teacher_full.shape[-1], candidate_full.shape[-1])
        teacher_logits = teacher_full[..., :shared_vocabulary]
        candidate_logits = candidate_full[..., :shared_vocabulary]

        teacher_full_probability = teacher_full.softmax(dim=-1)
        candidate_full_probability = candidate_full.softmax(dim=-1)
        teacher_shared_mass = teacher_full_probability[..., :shared_vocabulary].sum(dim=-1)
        candidate_shared_mass = candidate_full_probability[..., :shared_vocabulary].sum(dim=-1)

        teacher_log_probability = teacher_logits.log_softmax(dim=-1)
        candidate_log_probability = candidate_logits.log_softmax(dim=-1)
        teacher_probability = teacher_log_probability.exp()
        token_kl = (
            teacher_probability
            * (teacher_log_probability - candidate_log_probability)
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
            prompt_layer_cka.append(core.linear_cka(teacher_hidden, candidate_hidden))
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
                "teacher_vocabulary_size": int(teacher_full.shape[-1]),
                "candidate_vocabulary_size": int(candidate_full.shape[-1]),
                "shared_vocabulary_size": int(shared_vocabulary),
                "teacher_shared_probability_mass": float(teacher_shared_mass.mean()),
                "candidate_shared_probability_mass": float(candidate_shared_mass.mean()),
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
    aggregate_keys = (
        "teacher_shared_probability_mass",
        "candidate_shared_probability_mass",
        "token_kl_mean",
        "last_token_kl",
        "top1_agreement",
        "last_token_top1_agreement",
        "logit_cosine",
        "centered_logit_relative_rmse",
        "hidden_cosine_mean",
        "hidden_cka_mean",
    )
    return {
        "prompts": prompt_rows,
        "aggregate": {
            key: float(torch.tensor([row[key] for row in prompt_rows]).mean())
            for key in aggregate_keys
        },
        "vocabulary": {
            "teacher_size": int(prompt_rows[0]["teacher_vocabulary_size"]),
            "candidate_size": int(prompt_rows[0]["candidate_vocabulary_size"]),
            "shared_size": int(prompt_rows[0]["shared_vocabulary_size"]),
            "comparison": "renormalized shared-prefix vocabulary",
        },
        "per_layer": {
            "hidden_cosine": layer_cosine_tensor.mean(dim=0).tolist(),
            "hidden_cka": layer_cka_tensor.mean(dim=0).tolist(),
            "hidden_norm_ratio": layer_norm_tensor.mean(dim=0).tolist(),
        },
    }


def main() -> None:
    core.DEFAULT_PROMPTS = PROMPTS
    core.load_model = load_model_bf16
    core.quantize_model_in_place = quantize_model_chunked
    core.compare_signatures = compare_signatures_vocab_aligned
    core.main()


if __name__ == "__main__":
    main()
