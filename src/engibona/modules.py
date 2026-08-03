from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EngibonaConfig, QuantMode
from .grouping import group_last_dim, ungroup_last_dim
from .relaxation import (
    categorical_relaxation,
    catq_ternary_relaxation,
    compression_pressure,
    effective_weight,
    hard_codes,
    hessian_guided_temperature,
)


class CATQGroupModulation(nn.Module):
    """Per-group CAT-Q-style redistribution parameters for ternary recovery.

    The proxy transform is

        w_hat = (w - mu) / alpha
        mu = mu0 + tanh(delta_mu) * alpha0
        alpha = softplus(raw_alpha) * alpha0
        threshold = softplus(raw_threshold) * 0.5

    The deployed approximation is alpha * ternary_code, without a stored offset.
    This preserves a scale-and-code-only inference representation.
    """

    def __init__(self, prefix_shape: torch.Size, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.delta_mu_raw = nn.Parameter(torch.zeros(prefix_shape, dtype=dtype))
        init_one = torch.log(torch.expm1(torch.tensor(1.0, dtype=dtype)))
        self.delta_alpha_raw = nn.Parameter(torch.full(prefix_shape, init_one, dtype=dtype))
        self.delta_threshold_raw = nn.Parameter(torch.full(prefix_shape, init_one, dtype=dtype))

    def transform(self, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu0 = groups.mean(dim=-1, keepdim=True)
        alpha0 = (groups - mu0).abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
        delta_mu = torch.tanh(self.delta_mu_raw)[..., None]
        delta_alpha = F.softplus(self.delta_alpha_raw)[..., None].clamp_min(1.0e-4)
        delta_threshold = F.softplus(self.delta_threshold_raw)[..., None].clamp_min(1.0e-4)
        mu = mu0 + delta_mu * alpha0
        alpha = delta_alpha * alpha0
        threshold = delta_threshold * 0.5
        normalized = (groups - mu) / alpha
        return normalized, alpha, threshold


class GroupQuantizedLinear(nn.Module):
    """Training-time grouped binary/ternary linear layer.

    The latent tensor and optional CAT-Q modulation variables exist only during
    recovery. Export projects to exact codes and scales; no latent residual is
    required at inference.
    """

    def __init__(self, source: nn.Linear, config: EngibonaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = copy.deepcopy(config)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.latent_weight = nn.Parameter(source.weight.detach().float().clone())
        self.bias = (
            nn.Parameter(source.bias.detach().float().clone())
            if source.bias is not None
            else None
        )
        self.register_buffer("sensitivity", torch.tensor(0.0), persistent=True)
        self.register_buffer("projection_metric", None, persistent=False)
        self.step = 0
        self.total_steps = 1

        groups, _ = group_last_dim(self.latent_weight.detach(), self.config.group_size)
        if self.config.mode == QuantMode.TERNARY:
            self.modulation: CATQGroupModulation | None = CATQGroupModulation(groups.shape[:-1])
        else:
            self.modulation = None

    def set_schedule(self, step: int, total_steps: int) -> None:
        self.step = int(step)
        self.total_steps = max(int(total_steps), 1)

    def set_sensitivity(self, value: torch.Tensor | float) -> None:
        self.sensitivity.copy_(torch.as_tensor(value, device=self.sensitivity.device))

    def set_projection_metric(self, metric: torch.Tensor | None) -> None:
        self.projection_metric = None if metric is None else metric.detach()

    def _surrogate(self) -> torch.Tensor:
        groups, layout = group_last_dim(self.latent_weight, self.config.group_size)
        relaxation = self.config.relaxation
        if relaxation == "auto":
            relaxation = "catq" if self.config.mode == QuantMode.TERNARY else "categorical"

        tau = hessian_guided_temperature(
            self.step,
            self.total_steps,
            self.config.initial_temperature,
            self.sensitivity,
            self.config.sensitivity_alpha,
            self.config.compression_fraction,
        ).to(groups.device, groups.dtype)

        if relaxation == "catq":
            if self.config.mode != QuantMode.TERNARY or self.modulation is None:
                raise ValueError("CAT-Q relaxation is ternary-only")
            normalized, scales, threshold = self.modulation.transform(groups)
            progress = self.step / max(self.total_steps - 1, 1)
            sharpness = max(1.0e-4, 30.0 * progress)
            code = catq_ternary_relaxation(normalized, sharpness, threshold)
            if self.step >= self.total_steps - 1:
                code = torch.where(
                    normalized > threshold,
                    torch.ones_like(normalized),
                    torch.where(normalized < -threshold, -torch.ones_like(normalized), torch.zeros_like(normalized)),
                )
        else:
            scales = groups.abs().mean(dim=-1, keepdim=True).detach().clamp_min(1.0e-8)
            normalized = groups / scales
            code = categorical_relaxation(normalized, self.config.mode, tau)
            if self.step >= self.total_steps - 1:
                code = hard_codes(normalized, self.config.mode)

        return ungroup_last_dim(scales * code, layout)

    @torch.no_grad()
    def hard_surrogate(self) -> torch.Tensor:
        groups, layout = group_last_dim(self.latent_weight, self.config.group_size)
        if self.config.mode == QuantMode.TERNARY and self.modulation is not None:
            normalized, scales, threshold = self.modulation.transform(groups)
            code = torch.where(
                normalized > threshold,
                torch.ones_like(normalized),
                torch.where(normalized < -threshold, -torch.ones_like(normalized), torch.zeros_like(normalized)),
            )
        else:
            scales = groups.abs().mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
            code = hard_codes(groups / scales, self.config.mode)
        return ungroup_last_dim(scales * code, layout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        surrogate = self._surrogate()
        pressure = compression_pressure(
            self.step, self.total_steps, self.config.compression_fraction
        )
        weight = effective_weight(self.latent_weight, surrogate, pressure).to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)


def replace_linear_modules(
    model: nn.Module,
    config: EngibonaConfig,
    include_embeddings: bool = False,
    exclude_names: tuple[str, ...] = ("norm",),
) -> dict[str, nn.Module]:
    """Replace linear layers recursively. Embeddings remain explicit opt-in."""
    replaced: dict[str, nn.Module] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(token in full.lower() for token in exclude_names):
                continue
            if isinstance(child, nn.Linear):
                wrapped = GroupQuantizedLinear(child, config)
                setattr(parent, name, wrapped)
                replaced[full] = wrapped
            elif include_embeddings and isinstance(child, nn.Embedding):
                continue
            else:
                visit(child, full)

    visit(model)
    return replaced
