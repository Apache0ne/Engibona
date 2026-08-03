from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .packing import unpack_binary, unpack_ternary_2bit


@dataclass(frozen=True, slots=True)
class DecodedPackedWeight:
    codes: torch.Tensor
    scales: torch.Tensor
    weight: torch.Tensor


def decode_packed_weight(item: dict[str, Any]) -> DecodedPackedWeight:
    """Decode one Engibona research-format tensor exactly.

    This is a correctness reference, not an optimized low-bit kernel. It proves
    that the packed codes and FP16 scales reconstruct the trained hard weight
    without a latent or residual tensor.
    """
    shape = tuple(int(value) for value in item["shape"])
    count = 1
    for value in shape:
        count *= value
    encoding = item["encoding"]
    if encoding == "binary_1bit_lsb":
        codes = unpack_binary(item["packed_codes"], count)
    elif encoding == "ternary_2bit_slots_lsb":
        codes = unpack_ternary_2bit(item["packed_codes"], count)
    else:
        raise ValueError(f"unsupported encoding: {encoding}")
    codes = codes.reshape(shape)
    scales = item["scales_fp16"].float()

    if len(shape) < 2:
        raise ValueError("packed matrix weight must have at least two dimensions")
    last = shape[-1]
    leading = count // last
    code_rows = codes.reshape(leading, last).float()
    scale_rows = scales.reshape(leading, scales.shape[-1])
    group_size = (last + scale_rows.shape[-1] - 1) // scale_rows.shape[-1]
    group_index = torch.arange(last, device=code_rows.device) // group_size
    if int(group_index.max()) >= scale_rows.shape[-1]:
        raise ValueError("scale layout cannot cover packed codes")
    weight = (code_rows * scale_rows[:, group_index]).reshape(shape)
    return DecodedPackedWeight(codes=codes, scales=scales, weight=weight)


def packed_linear(
    x: torch.Tensor,
    item: dict[str, Any],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    decoded = decode_packed_weight(item)
    return F.linear(
        x,
        decoded.weight.to(device=x.device, dtype=x.dtype),
        None if bias is None else bias.to(device=x.device, dtype=x.dtype),
    )


def packed_embedding(
    input_ids: torch.Tensor,
    item: dict[str, Any],
    padding_idx: int | None = None,
) -> torch.Tensor:
    decoded = decode_packed_weight(item)
    return F.embedding(
        input_ids,
        decoded.weight.to(device=input_ids.device),
        padding_idx=padding_idx,
    )
