from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import EngibonaConfig, QuantMode
from .modules import GroupQuantizedLinear
from .packing import pack_binary, pack_ternary_2bit
from .projection import metric_project


@torch.no_grad()
def export_packed(
    model: torch.nn.Module,
    path: str | Path,
    config: EngibonaConfig,
) -> dict[str, Any]:
    """Export exact packed codes plus FP16 scales to a transparent research file.

    This is not GGUF. It is an auditable intermediate container suitable for
    validating the transformation before a runtime-specific serializer is added.
    """
    tensors: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, GroupQuantizedLinear):
            continue
        result = metric_project(
            module.latent_weight.detach(),
            config.mode,
            group_size=config.group_size,
            metric=module.projection_metric,
            refine_steps=config.code_refine_steps,
            tolerance=config.code_refine_tolerance,
        )
        codes = result.codes.cpu()
        if config.mode == QuantMode.BINARY:
            packed, pad = pack_binary(codes)
            encoding = "binary_1bit_lsb"
        else:
            packed, pad = pack_ternary_2bit(codes)
            encoding = "ternary_2bit_slots_lsb"
        tensors[name] = {
            "shape": list(codes.shape),
            "encoding": encoding,
            "padding_symbols": int(pad),
            "packed_codes": packed.cpu(),
            "scales_fp16": result.scales.cpu(),
        }

    payload = {
        "format": "ENGIBONA_G128_RESEARCH_V1",
        "config": asdict(config),
        "tensors": tensors,
    }
    torch.save(payload, Path(path))
    return payload
