from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .embedding_shared import SharedBinaryTernaryEmbeddingState
from .packing import pack_binary, unpack_binary


@torch.no_grad()
def export_shared_embedding_pair(
    state: SharedBinaryTernaryEmbeddingState,
    path: str | Path,
) -> dict[str, Any]:
    """Store one sign codebook, one scale tensor, and one ternary mask.

    Public 1.7B/4B checkpoint forensics supports

        W_binary  = s * b
        W_ternary = s * b * m

    for token embeddings. This research format stores that relation directly
    instead of duplicating binary and ternary embedding codebooks and scales.
    It is a joint archival/interchange format; ordinary model export still
    emits the mode-specific runtime representation.
    """
    hard = state.hard_state()
    binary_codes = hard.binary_codes.detach().cpu().to(torch.int8)
    mask_codes = torch.where(
        hard.mask.detach().cpu(),
        torch.ones_like(binary_codes),
        -torch.ones_like(binary_codes),
    )
    packed_binary, binary_padding = pack_binary(binary_codes)
    packed_mask, mask_padding = pack_binary(mask_codes)
    payload = {
        "format": "ENGIBONA_SHARED_EMBEDDING_PAIR_V1",
        "shape": list(binary_codes.shape),
        "group_size": int(state.group_size),
        "binary_encoding": "binary_1bit_lsb",
        "mask_encoding": "ternary_nonzero_mask_1bit_lsb",
        "binary_padding_symbols": int(binary_padding),
        "mask_padding_symbols": int(mask_padding),
        "packed_binary_codes": packed_binary.cpu(),
        "packed_ternary_mask": packed_mask.cpu(),
        "scales_fp16": hard.scales.detach().cpu().to(torch.float16),
    }
    torch.save(payload, Path(path))
    return payload


def decode_shared_embedding_pair(
    payload_or_path: dict[str, Any] | str | Path,
) -> dict[str, torch.Tensor]:
    """Decode the exact binary and ternary embedding views."""
    payload = (
        torch.load(Path(payload_or_path), map_location="cpu", weights_only=True)
        if isinstance(payload_or_path, (str, Path))
        else payload_or_path
    )
    if payload.get("format") != "ENGIBONA_SHARED_EMBEDDING_PAIR_V1":
        raise ValueError("unsupported shared embedding pair format")
    shape = tuple(int(value) for value in payload["shape"])
    count = 1
    for value in shape:
        count *= value
    group_size = int(payload["group_size"])
    if group_size <= 0 or shape[-1] % group_size:
        raise ValueError("invalid shared embedding group size")

    binary_codes = unpack_binary(
        payload["packed_binary_codes"], count
    ).reshape(shape).to(torch.int8)
    mask_codes = unpack_binary(
        payload["packed_ternary_mask"], count
    ).reshape(shape)
    mask = mask_codes > 0
    ternary_codes = binary_codes * mask.to(torch.int8)

    scales = payload["scales_fp16"].float()
    rows = count // shape[-1]
    groups_per_row = shape[-1] // group_size
    scale_rows = scales.reshape(rows, groups_per_row)
    expanded_scales = scale_rows.repeat_interleave(
        group_size, dim=1
    ).reshape(shape)
    binary_weight = binary_codes.float() * expanded_scales
    ternary_weight = binary_weight * mask.to(binary_weight.dtype)
    return {
        "binary_codes": binary_codes,
        "ternary_codes": ternary_codes,
        "mask": mask,
        "scales": payload["scales_fp16"],
        "binary_weight": binary_weight,
        "ternary_weight": ternary_weight,
    }
