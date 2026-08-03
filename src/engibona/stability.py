from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CodeStabilityReport:
    zero_ratio: torch.Tensor
    flip_ratio: torch.Tensor | None
    saturated_ratio: torch.Tensor


def code_stability(
    codes: torch.Tensor,
    previous_codes: torch.Tensor | None = None,
) -> CodeStabilityReport:
    """Report ternary sparsity and code oscillation per group."""
    c = codes.float()
    zero_ratio = (c == 0).float().mean(dim=-1)
    saturated_ratio = (c.abs() == 1).float().mean(dim=-1)
    flip_ratio = None
    if previous_codes is not None:
        if previous_codes.shape != codes.shape:
            raise ValueError("previous_codes shape mismatch")
        flip_ratio = (previous_codes != codes).float().mean(dim=-1)
    return CodeStabilityReport(zero_ratio, flip_ratio, saturated_ratio)
