from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .grouping import group_last_dim


@dataclass(slots=True)
class FisherFlipCandidates:
    predicted_delta: torch.Tensor
    flat_indices: torch.Tensor


@dataclass(slots=True)
class ValidatedFlipResult:
    codes: torch.Tensor
    accepted_count: int
    baseline_loss: float
    final_loss: float


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

    Negative scores predict improving flips. Candidate prefixes must still be
    validated against the real teacher/task calibration loss.
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
    result.reshape(-1)[flat_indices] *= -1
    return result


def validated_prefix_search(
    codes: torch.Tensor,
    ranked_indices: torch.Tensor,
    evaluate_loss: Callable[[torch.Tensor], float],
    prefix_sizes: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
) -> ValidatedFlipResult:
    """Accept the candidate prefix that improves real calibration loss most.

    The predictor ranks discrete moves; this function prevents Taylor-model
    error or flip interactions from silently degrading the true objective.
    """
    baseline = float(evaluate_loss(codes))
    best_loss = baseline
    best_codes = codes.clone()
    best_count = 0
    maximum = int(ranked_indices.numel())
    for requested in prefix_sizes:
        count = min(int(requested), maximum)
        if count <= 0:
            continue
        trial = apply_binary_flips(codes, ranked_indices[:count])
        value = float(evaluate_loss(trial))
        if value < best_loss:
            best_loss = value
            best_codes = trial
            best_count = count
    return ValidatedFlipResult(
        codes=best_codes,
        accepted_count=best_count,
        baseline_loss=baseline,
        final_loss=best_loss,
    )


def selected_hessian_diagonal(
    loss: torch.Tensor,
    parameter: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    """Exact selected Hessian diagonal entries for tiny-model validation.

    Complexity is one reverse-mode pass per coordinate. This is a research
    oracle for checking approximations, not a production 27B algorithm.
    """
    gradient = torch.autograd.grad(
        loss, parameter, create_graph=True, retain_graph=True
    )[0].reshape(-1)
    values = []
    for index in flat_indices.tolist():
        second = torch.autograd.grad(
            gradient[index], parameter, retain_graph=True
        )[0].reshape(-1)[index]
        values.append(second)
    return torch.stack(values)
