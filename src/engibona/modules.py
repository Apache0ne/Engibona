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
    hard_ste_weight,
    hessian_guided_temperature,
)


class _GroupedQuantState(nn.Module):
    """Shared trainable state for exact grouped low-bit modules."""

    def __init__(self, weight: torch.Tensor, config: EngibonaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = copy.deepcopy(config)
        self.latent_weight = nn.Parameter(weight.detach().float().clone())
        groups, _ = group_last_dim(
            self.latent_weight.detach(), self.config.group_size
        )
        initial_scale = groups.abs().mean(dim=-1).clamp_min(
            self.config.scale_min
        )
        self.log_scale = nn.Parameter(initial_scale.log())
        self.register_buffer(
            "initial_log_scale", initial_scale.log(), persistent=True
        )
        self.assignment_shift_raw = (
            nn.Parameter(torch.zeros_like(initial_scale))
            if self.config.mode == QuantMode.TERNARY
            else None
        )
        self.threshold_raw = (
            nn.Parameter(torch.zeros_like(initial_scale))
            if self.config.mode == QuantMode.TERNARY
            else None
        )
        self.register_buffer("sensitivity", torch.tensor(0.0), persistent=True)
        self.register_buffer("projection_metric", None, persistent=False)
        self.step = 0
        self.total_steps = 1

    def set_schedule(self, step: int, total_steps: int) -> None:
        self.step = int(step)
        self.total_steps = max(int(total_steps), 1)

    def set_sensitivity(self, value: torch.Tensor | float) -> None:
        self.sensitivity.copy_(
            torch.as_tensor(value, device=self.sensitivity.device)
        )

    def set_projection_metric(self, metric: torch.Tensor | None) -> None:
        self.projection_metric = None if metric is None else metric.detach()

    def _scales(self) -> torch.Tensor:
        lower = self.initial_log_scale - self.config.scale_log_trust_radius
        upper = self.initial_log_scale + self.config.scale_log_trust_radius
        return self.log_scale.clamp(lower, upper).exp().clamp(
            self.config.scale_min, self.config.scale_max
        )

    def _assignment(
        self,
        groups: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.mode == QuantMode.BINARY:
            zero = torch.tensor(
                0.0, device=groups.device, dtype=groups.dtype
            )
            return groups / scales[..., None], zero
        shift = (
            torch.tanh(self.assignment_shift_raw)[..., None]
            * scales[..., None]
        )
        threshold = 0.15 + 0.70 * torch.sigmoid(self.threshold_raw)
        return (groups - shift) / scales[..., None], threshold[..., None]

    def hard_codes_and_scales(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups, layout = group_last_dim(
            self.latent_weight, self.config.group_size
        )
        scales = self._scales()
        normalized, threshold = self._assignment(groups, scales)
        codes = hard_codes(
            normalized,
            self.config.mode,
            threshold=threshold if self.config.mode == QuantMode.TERNARY else 0.5,
        )
        return (
            ungroup_last_dim(codes, layout).to(torch.int8),
            scales.to(torch.float16),
            ungroup_last_dim(scales[..., None] * codes, layout),
        )

    def _surrogate(self) -> torch.Tensor:
        groups, layout = group_last_dim(
            self.latent_weight, self.config.group_size
        )
        scales = self._scales()
        normalized, threshold = self._assignment(groups, scales)
        method = self.config.relaxation
        if method == "auto":
            method = (
                "hard_ste"
                if self.config.mode == QuantMode.BINARY
                else "catq"
            )
        progress = self.step / max(self.total_steps - 1, 1)
        hard_now = (
            method == "hard_ste"
            or progress >= self.config.hard_recovery_start
        )
        if hard_now:
            codes = hard_codes(
                normalized,
                self.config.mode,
                threshold=threshold
                if self.config.mode == QuantMode.TERNARY
                else 0.5,
            )
            hard = ungroup_last_dim(scales[..., None] * codes, layout)
            return hard_ste_weight(self.latent_weight, hard)

        temperature = hessian_guided_temperature(
            self.step,
            self.total_steps,
            self.config.initial_temperature,
            self.sensitivity,
            self.config.sensitivity_alpha,
            self.config.compression_fraction,
            self.config.final_temperature,
        ).to(groups.device, groups.dtype)
        if method == "catq":
            if self.config.mode != QuantMode.TERNARY:
                raise ValueError("CAT-Q is ternary-only")
            sharpness = max(1.0e-3, 1.0 + 29.0 * progress)
            codes = catq_ternary_relaxation(
                normalized, sharpness, threshold
            )
        else:
            codes = categorical_relaxation(
                normalized, self.config.mode, temperature
            )
        soft = ungroup_last_dim(scales[..., None] * codes, layout)
        pressure = compression_pressure(
            self.step,
            self.total_steps,
            self.config.compression_fraction,
        )
        return effective_weight(self.latent_weight, soft, pressure)

    def regularization_loss(self) -> torch.Tensor:
        tether = (
            self.log_scale - self.initial_log_scale
        ).square().mean() * self.config.scale_tether_weight
        if (
            self.config.mode != QuantMode.TERNARY
            or self.config.ternary_zero_weight <= 0
        ):
            return tether
        codes, _, _ = self.hard_codes_and_scales()
        zero_ratio = (codes == 0).float().mean()
        return tether + self.config.ternary_zero_weight * (
            zero_ratio - self.config.ternary_zero_target
        ).square()

    @torch.no_grad()
    def load_hard_codes_and_scales(
        self,
        codes: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        groups, layout = group_last_dim(
            self.latent_weight, self.config.group_size
        )
        code_groups, _ = group_last_dim(
            codes.to(groups.dtype), self.config.group_size
        )
        if scales.shape != self.log_scale.shape:
            raise ValueError("scale shape mismatch")
        self.log_scale.copy_(
            scales.float().clamp_min(self.config.scale_min).log()
        )
        magnitude = groups.abs().clamp_min(1.0e-3)
        self.latent_weight.copy_(
            ungroup_last_dim(magnitude * code_groups, layout)
        )


class GroupQuantizedLinear(_GroupedQuantState):
    def __init__(self, source: nn.Linear, config: EngibonaConfig) -> None:
        super().__init__(source.weight, config)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.bias = (
            nn.Parameter(source.bias.detach().float().clone())
            if source.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self._surrogate().to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

    @torch.no_grad()
    def hard_surrogate(self) -> torch.Tensor:
        return self.hard_codes_and_scales()[2]


class GroupQuantizedEmbedding(_GroupedQuantState):
    """Embedding with release-matched binary/ternary code policies."""

    def __init__(self, source: nn.Embedding, config: EngibonaConfig) -> None:
        super().__init__(source.weight, config)
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        initial_groups, _ = group_last_dim(
            source.weight.detach().float(), self.config.group_size
        )
        self.register_buffer(
            "initial_sign_groups",
            torch.where(initial_groups >= 0, 1.0, -1.0),
            persistent=True,
        )
        if (
            self.config.mode == QuantMode.BINARY
            and self.config.binary_embedding_strategy == "frozen_ptq"
        ) or (
            self.config.mode == QuantMode.TERNARY
            and self.config.ternary_embedding_strategy == "frozen_ptq"
        ):
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def _sign_locked(self) -> bool:
        return (
            self.config.mode == QuantMode.TERNARY
            and self.config.ternary_embedding_strategy == "sign_locked_recovery"
        )

    def hard_codes_and_scales(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        groups, layout = group_last_dim(
            self.latent_weight, self.config.group_size
        )
        scales = self._scales()
        normalized, threshold = self._assignment(groups, scales)
        if self._sign_locked():
            codes = self.initial_sign_groups * (
                normalized.abs() > threshold
            ).to(normalized.dtype)
        else:
            codes = hard_codes(
                normalized,
                self.config.mode,
                threshold=threshold
                if self.config.mode == QuantMode.TERNARY
                else 0.5,
            )
        return (
            ungroup_last_dim(codes, layout).to(torch.int8),
            scales.to(torch.float16),
            ungroup_last_dim(scales[..., None] * codes, layout),
        )

    def _surrogate(self) -> torch.Tensor:
        frozen = (
            self.config.mode == QuantMode.BINARY
            and self.config.binary_embedding_strategy == "frozen_ptq"
        ) or (
            self.config.mode == QuantMode.TERNARY
            and self.config.ternary_embedding_strategy == "frozen_ptq"
        )
        if frozen:
            return self.hard_codes_and_scales()[2]
        if not self._sign_locked():
            return super()._surrogate()

        groups, layout = group_last_dim(
            self.latent_weight, self.config.group_size
        )
        scales = self._scales()
        normalized, threshold = self._assignment(groups, scales)
        progress = self.step / max(self.total_steps - 1, 1)
        method = self.config.relaxation
        if method == "auto":
            method = "catq"
        hard_now = (
            method == "hard_ste"
            or progress >= self.config.hard_recovery_start
        )
        if hard_now:
            codes = self.initial_sign_groups * (
                normalized.abs() > threshold
            ).to(normalized.dtype)
            hard = ungroup_last_dim(scales[..., None] * codes, layout)
            return hard_ste_weight(self.latent_weight, hard)

        temperature = hessian_guided_temperature(
            self.step,
            self.total_steps,
            self.config.initial_temperature,
            self.sensitivity,
            self.config.sensitivity_alpha,
            self.config.compression_fraction,
            self.config.final_temperature,
        ).to(groups.device, groups.dtype)
        if method == "catq":
            sharpness = max(1.0e-3, 1.0 + 29.0 * progress)
            magnitude = catq_ternary_relaxation(
                normalized.abs(), sharpness, threshold
            ).clamp(0.0, 1.0)
        else:
            magnitude = categorical_relaxation(
                normalized.abs(), QuantMode.TERNARY, temperature
            ).abs().clamp(0.0, 1.0)
        codes = self.initial_sign_groups * magnitude
        soft = ungroup_last_dim(scales[..., None] * codes, layout)
        pressure = compression_pressure(
            self.step,
            self.total_steps,
            self.config.compression_fraction,
        )
        return effective_weight(self.latent_weight, soft, pressure)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            input_ids,
            self._surrogate(),
            padding_idx=self.padding_idx,
        )

    @torch.no_grad()
    def hard_surrogate(self) -> torch.Tensor:
        return self.hard_codes_and_scales()[2]


def replace_linear_modules(
    model: nn.Module,
    config: EngibonaConfig,
    include_embeddings: bool = True,
    exclude_names: tuple[str, ...] = ("norm",),
    preserve_tied_weights: bool = True,
) -> dict[str, nn.Module]:
    """Replace matrix-heavy modules and preserve shared embedding/head weights."""
    from .modules_tied import TiedGroupQuantizedLMHead

    replaced: dict[str, nn.Module] = {}
    tied_embeddings: dict[int, GroupQuantizedEmbedding] = {}

    def wrap_embeddings(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(token in full.lower() for token in exclude_names):
                continue
            if include_embeddings and isinstance(child, nn.Embedding):
                source_parameter_id = id(child.weight)
                wrapped = GroupQuantizedEmbedding(child, config)
                setattr(parent, name, wrapped)
                tied_embeddings[source_parameter_id] = wrapped
                replaced[full] = wrapped
            else:
                wrap_embeddings(child, full)

    def wrap_linears(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(token in full.lower() for token in exclude_names):
                continue
            if isinstance(child, nn.Linear):
                shared = tied_embeddings.get(id(child.weight))
                if preserve_tied_weights and shared is not None:
                    wrapped = TiedGroupQuantizedLMHead(shared)
                else:
                    wrapped = GroupQuantizedLinear(child, config)
                setattr(parent, name, wrapped)
                replaced[full] = wrapped
            elif not isinstance(child, GroupQuantizedEmbedding):
                wrap_linears(child, full)

    wrap_embeddings(model)
    wrap_linears(model)
    return replaced
