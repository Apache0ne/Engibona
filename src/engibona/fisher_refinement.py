from __future__ import annotations

from dataclasses import dataclass

import torch

from .grouping import group_last_dim


@dataclass(slots=True)
class FisherFlipCandidates:
    predicted_delta: torch.Tensor
    flat_indices: torch.Tensor


def binary_flip_predicted_delta(
    codes: torch.Tensor,
    scales: torch.Tensor,
    gradient: torch.Tensor,
    fisher_diag: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Diagonal empirical-Fisher Taylor score for each exact sign flip.

    For `w = s*c` and `c' = -c`, `delta_w = -2*s*c`. The local loss
    approximation is

        delta L ~= g*delta_w + 0.5*F_diag*delta_w^2.

    Negative scores predict improving flips. A real recovery loop must still
    validate candidate prefixes against the teacher/task calibration loss.
    """
    code_groups, _ = group_last_dim(codes.float(), group_size)
    gradient_groups, _ = group_last_dim(gradient.float(), group_size)
    fisher_groups, _ = group_last_dim(fisher_diag.float(), group_size)
    if scales.shape != code_groups.shape[:-1]:
        raise ValueError("scale shape mismatch")
    delta_weight = -2.0 * scales.float()[..., None] * code_groups
    predicted = (
        gradient_groups * delta_weight
        + 0.5 * fisher_groups.clamp_min(0.0) * delta_weight.square()
    )
    return predicted.reshape(codes.shape)


def rank_binary_flips(
    codes: torch.Tensor,
    scales: torch.Tensor,
    gradient: torch.Tensor,
    fisher_diag: torch.Tensor,
    group_size: int = 128,
    topk: int = 128,
) -> FisherFlipCandidates:
    predicted = binary_flip_predicted_delta(
        codes, scales, gradient, fisher_diag, group_size
    )
    flat = predicted.reshape(-1)
    count = min(int(topk), flat.numel())
    values, indices = torch.topk(-flat, k=count)
    return FisherFlipCandidates(
        predicted_delta=-values,
        flat_indices=indices,
    )


def apply_binary_flips(
    codes: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    result = codes.clone()
    flat = result.reshape(-1)
    flat[flat_indices] *= -1
    return result
