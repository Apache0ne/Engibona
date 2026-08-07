from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import EngibonaConfig, QuantMode
from .modules import GroupQuantizedEmbedding, GroupQuantizedLinear
from .modules_tied import TiedGroupQuantizedLMHead
from .packing import pack_binary, pack_ternary_2bit
from .projection import metric_project


@torch.no_grad()
def export_packed(
    model: torch.nn.Module,
    path: str | Path,
    config: EngibonaConfig,
) -> dict[str, Any]:
    """Export exact packed codes and FP16 scales."""
    tensors: dict[str, Any] = {}
    supported = (
        GroupQuantizedLinear,
        GroupQuantizedEmbedding,
        TiedGroupQuantizedLMHead,
    )
    for name, module in model.named_modules():
        if not isinstance(module, supported):
            continue
        if config.export_strategy == "trained" or isinstance(
            module, TiedGroupQuantizedLMHead
        ):
            codes, scales, _ = module.hard_codes_and_scales()
        else:
            result = metric_project(
                module.latent_weight.detach(),
                config.mode,
                group_size=config.group_size,
                metric=module.projection_metric,
                refine_steps=config.code_refine_steps,
                tolerance=config.code_refine_tolerance,
            )
            codes, scales = result.codes, result.scales

        codes_cpu = codes.cpu()
        if config.mode == QuantMode.BINARY:
            packed, pad = pack_binary(codes_cpu)
            encoding = "binary_1bit_lsb"
        else:
            packed, pad = pack_ternary_2bit(codes_cpu)
            encoding = "ternary_2bit_slots_lsb"
        tensors[name] = {
            "shape": list(codes_cpu.shape),
            "group_size": int(config.group_size),
            "encoding": encoding,
            "padding_symbols": int(pad),
            "packed_codes": packed.cpu(),
            "scales_fp16": scales.cpu(),
        }

    payload = {
        "format": "ENGIBONA_G128_RESEARCH_V2",
        "config": asdict(config),
        "tensors": tensors,
    }
    torch.save(payload, Path(path))
    return payload
