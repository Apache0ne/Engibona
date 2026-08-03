#!/usr/bin/env python3
"""Memory-bounded BF16 entry point for full-model functional forensics.

The original FP32 attempt exceeded the standard runner resource envelope. This
entry point keeps one 1.7B model in BF16 and quantizes matrices in row chunks,
while reusing the same signature and comparison implementation.
"""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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


def main() -> None:
    core.DEFAULT_PROMPTS = PROMPTS
    core.load_model = load_model_bf16
    core.quantize_model_in_place = quantize_model_chunked
    core.main()


if __name__ == "__main__":
    main()
