from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .config import QuantMode
from .grouping import group_last_dim, ungroup_last_dim

MetricKind = Literal["identity", "diag", "full"]


@dataclass(slots=True)
class ProjectionResult:
    dequantized: torch.Tensor
    codes: torch.Tensor
    scales: torch.Tensor
    initial_error: torch.Tensor
    final_error: torch.Tensor
    iterations: int


def _metric_kind(metric: torch.Tensor | None, group_size: int) -> MetricKind:
    if metric is None:
        return "identity"
    if metric.shape[-1] != group_size:
        raise ValueError("metric group dimension does not match group_size")
    if metric.ndim >= 2 and metric.shape[-2:] == (group_size, group_size):
        return "full"
    return "diag"


def _broadcast_metric(
    metric: torch.Tensor | None,
    target_prefix: torch.Size,
    group_size: int,
) -> torch.Tensor | None:
    if metric is None:
        return None
    kind = _metric_kind(metric, group_size)
    suffix = (group_size, group_size) if kind == "full" else (group_size,)
    desired = (*target_prefix, *suffix)
    try:
        return torch.broadcast_to(metric, desired)
    except RuntimeError as exc:
        raise ValueError(f"metric shape {tuple(metric.shape)} cannot broadcast to {desired}") from exc


def _apply_metric(metric: torch.Tensor | None, vector: torch.Tensor) -> torch.Tensor:
    if metric is None:
        return vector
    if metric.ndim == vector.ndim:
        return metric * vector
    return torch.einsum("...ij,...j->...i", metric, vector)


def _quadratic(metric: torch.Tensor | None, vector: torch.Tensor) -> torch.Tensor:
    return (vector * _apply_metric(metric, vector)).sum(dim=-1)


def optimal_scale(
    weights: torch.Tensor,
    codes: torch.Tensor,
    metric: torch.Tensor | None = None,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Closed-form positive scale minimizing (w - s c)^T M (w - s c)."""
    mw = _apply_metric(metric, weights)
    mc = _apply_metric(metric, codes)
    numerator = (codes * mw).sum(dim=-1)
    denominator = (codes * mc).sum(dim=-1).clamp_min(epsilon)
    return (numerator / denominator).clamp_min(epsilon)


def weighted_error(
    weights: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    metric: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = weights - scales[..., None] * codes
    return _quadratic(metric, residual)


def _binary_init(weights: torch.Tensor) -> torch.Tensor:
    return torch.where(weights >= 0, torch.ones_like(weights), -torch.ones_like(weights))


def _ternary_init(
    weights: torch.Tensor,
    metric: torch.Tensor | None,
    iterations: int = 8,
) -> torch.Tensor:
    scale = weights.abs().mean(dim=-1).clamp_min(1.0e-12)
    codes = torch.zeros_like(weights)
    for _ in range(iterations):
        codes = torch.where(
            weights > 0.5 * scale[..., None],
            torch.ones_like(weights),
            torch.where(
                weights < -0.5 * scale[..., None],
                -torch.ones_like(weights),
                torch.zeros_like(weights),
            ),
        )
        dead = codes.square().sum(dim=-1) == 0
        if dead.any():
            index = weights.abs().argmax(dim=-1, keepdim=True)
            fallback = torch.zeros_like(codes).scatter_(
                -1, index, torch.gather(weights.sign(), -1, index)
            )
            codes = torch.where(dead[..., None], fallback, codes)
        scale = optimal_scale(weights, codes, metric)
    return codes


def _coordinate_refine(
    weights: torch.Tensor,
    codes: torch.Tensor,
    metric: torch.Tensor | None,
    mode: QuantMode,
    max_steps: int,
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    """Exact one-coordinate local search under a quadratic group metric.

    For fixed code c, the optimal scale gives objective
      w^T M w - (c^T M w)^2 / (c^T M c).
    Each candidate code change is scored with this closed form, so the search
    jointly re-optimizes the scale after every code move.
    """
    states = (
        torch.tensor([-1.0, 1.0], device=weights.device, dtype=weights.dtype)
        if mode == QuantMode.BINARY
        else torch.tensor([-1.0, 0.0, 1.0], device=weights.device, dtype=weights.dtype)
    )

    mw = _apply_metric(metric, weights)
    mc = _apply_metric(metric, codes)
    numerator = (codes * mw).sum(dim=-1)
    denominator = (codes * mc).sum(dim=-1).clamp_min(1.0e-12)
    current_score = numerator.square() / denominator

    if metric is None:
        diagonal = torch.ones_like(weights)
    elif metric.ndim == weights.ndim:
        diagonal = metric
    else:
        diagonal = metric.diagonal(dim1=-2, dim2=-1)

    steps_taken = 0
    for _ in range(max_steps):
        delta = states.view(*([1] * codes.ndim), -1) - codes.unsqueeze(-1)
        num_new = numerator[..., None, None] + delta * mw.unsqueeze(-1)
        den_new = (
            denominator[..., None, None]
            + 2.0 * delta * mc.unsqueeze(-1)
            + delta.square() * diagonal.unsqueeze(-1)
        ).clamp_min(1.0e-12)
        candidate_score = num_new.square() / den_new
        candidate_score = torch.where(
            num_new > 0.0, candidate_score, torch.full_like(candidate_score, -torch.inf)
        )
        candidate_score = torch.where(
            delta != 0.0, candidate_score, torch.full_like(candidate_score, -torch.inf)
        )

        flat = candidate_score.reshape(*candidate_score.shape[:-2], -1)
        best_score, best_flat = flat.max(dim=-1)
        improve = best_score > current_score + tolerance
        if not bool(improve.any()):
            break

        state_count = states.numel()
        position = best_flat // state_count
        state_index = best_flat % state_count
        old = torch.gather(codes, -1, position[..., None]).squeeze(-1)
        new = states[state_index]
        change = torch.where(improve, new - old, torch.zeros_like(old))

        codes = codes.scatter(-1, position[..., None], (old + change)[..., None])

        if metric is None or metric.ndim == weights.ndim:
            basis_effect = torch.gather(diagonal, -1, position[..., None]).squeeze(-1)
            mc = mc.scatter(
                -1,
                position[..., None],
                (
                    torch.gather(mc, -1, position[..., None]).squeeze(-1)
                    + change * basis_effect
                )[..., None],
            )
        else:
            column = torch.gather(
                metric,
                -1,
                position[..., None, None].expand(*position.shape, metric.shape[-2], 1),
            ).squeeze(-1)
            mc = mc + change[..., None] * column

        numerator = (codes * mw).sum(dim=-1)
        denominator = (codes * mc).sum(dim=-1).clamp_min(1.0e-12)
        current_score = numerator.square() / denominator
        steps_taken += 1

    return codes, steps_taken


def metric_project(
    weight: torch.Tensor,
    mode: QuantMode | str,
    group_size: int = 128,
    metric: torch.Tensor | None = None,
    refine_steps: int = 8,
    tolerance: float = 1.0e-10,
) -> ProjectionResult:
    """Project a weight tensor into exact grouped binary or ternary form.

    `metric` may be:
      - None: Euclidean projection,
      - [..., num_groups, group_size]: diagonal sensitivity,
      - [..., num_groups, group_size, group_size]: full group covariance.
    """
    mode = QuantMode(mode)
    groups, layout = group_last_dim(weight.float(), group_size)
    metric = _broadcast_metric(metric, groups.shape[:-1], group_size)

    if mode == QuantMode.BINARY:
        codes = _binary_init(groups)
    else:
        codes = _ternary_init(groups, metric)

    initial_scales = optimal_scale(groups, codes, metric)
    initial_error = weighted_error(groups, codes, initial_scales, metric)

    codes, iterations = _coordinate_refine(
        groups,
        codes,
        metric,
        mode,
        max_steps=refine_steps,
        tolerance=tolerance,
    )
    scales = optimal_scale(groups, codes, metric)
    final_error = weighted_error(groups, codes, scales, metric)
    dequantized = ungroup_last_dim(scales[..., None] * codes, layout).to(weight.dtype)
    code_tensor = ungroup_last_dim(codes, layout).to(torch.int8)

    return ProjectionResult(
        dequantized=dequantized,
        codes=code_tensor,
        scales=scales.to(torch.float16),
        initial_error=initial_error,
        final_error=final_error,
        iterations=iterations,
    )
