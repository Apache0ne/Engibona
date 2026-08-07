from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grouping import group_last_dim, ungroup_last_dim


@dataclass(frozen=True, slots=True)
class SharedEmbeddingHardState:
    """Exact released-format views of one shared embedding state."""

    binary_codes: torch.Tensor
    ternary_codes: torch.Tensor
    scales: torch.Tensor
    binary_weight: torch.Tensor
    ternary_weight: torch.Tensor
    mask: torch.Tensor


class SharedBinaryTernaryEmbeddingState(nn.Module):
    """One binary sign codebook and scale with a learned ternary zero mask.

    Direct public-checkpoint forensics supports the released-state relation

        W_binary  = s * b
        W_ternary = s * b * m

    for token embeddings at both 1.7B and 4B.  `b` is a shared binary codebook,
    `s` is one positive FP16-compatible scale per contiguous group, and `m` is a
    binary ternary mask.  The class exposes binary and ternary embedding/LM-head
    views without duplicating trainable state.

    The training surrogate is intentionally explicit rather than claiming the
    private PrismML gradient estimator.  Hard forward values use exact signs and
    masks; identity and sigmoid straight-through paths carry gradients.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        group_size: int = 128,
        scale_min: float = 1.0e-8,
        scale_max: float = 1.0e4,
        scale_log_trust_radius: float = 1.5,
        mask_logit_magnitude: float = 4.0,
        freeze_binary_codebook: bool = False,
        freeze_scales: bool = False,
        active_sign_gradient_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("shared embedding weight must be a matrix")
        if group_size <= 0 or weight.shape[-1] % group_size:
            raise ValueError(
                "embedding width must be divisible by positive group_size"
            )
        if not 0.0 <= active_sign_gradient_scale <= 1.0:
            raise ValueError("active_sign_gradient_scale must be in [0,1]")
        self.num_embeddings = int(weight.shape[0])
        self.embedding_dim = int(weight.shape[1])
        self.group_size = int(group_size)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.scale_log_trust_radius = float(scale_log_trust_radius)
        self.freeze_binary_codebook = bool(freeze_binary_codebook)
        self.freeze_scales = bool(freeze_scales)
        self.active_sign_gradient_scale = float(active_sign_gradient_scale)

        source = weight.detach().float()
        groups, _ = group_last_dim(source, self.group_size)
        initial_scale = groups.abs().mean(dim=-1).clamp_min(self.scale_min)
        initial_sign = torch.where(
            groups < 0,
            -torch.ones_like(groups),
            torch.ones_like(groups),
        )
        initial_mask = self._least_squares_mask(groups)

        self.sign_latent = nn.Parameter(source.clone())
        self.log_scale = nn.Parameter(initial_scale.log())
        mask_logits = torch.where(
            initial_mask,
            torch.full_like(groups, float(mask_logit_magnitude)),
            torch.full_like(groups, -float(mask_logit_magnitude)),
        )
        self.mask_logits = nn.Parameter(mask_logits)
        self.register_buffer(
            "initial_sign_groups", initial_sign, persistent=True
        )
        self.register_buffer(
            "initial_log_scale", initial_scale.log(), persistent=True
        )
        self.register_buffer(
            "initial_mask_groups", initial_mask, persistent=True
        )

        if self.freeze_binary_codebook:
            self.sign_latent.requires_grad_(False)
        if self.freeze_scales:
            self.log_scale.requires_grad_(False)

    @staticmethod
    def _least_squares_mask(
        groups: torch.Tensor,
        iterations: int = 16,
    ) -> torch.Tensor:
        scales = groups.abs().mean(dim=-1).clamp_min(1.0e-12)
        for _ in range(iterations):
            active = groups.abs() > 0.5 * scales[..., None]
            scales = (
                (groups.abs() * active).sum(dim=-1)
                / active.sum(dim=-1).clamp_min(1)
            ).clamp_min(1.0e-12)
        return groups.abs() > 0.5 * scales[..., None]

    def _scales(self) -> torch.Tensor:
        lower = self.initial_log_scale - self.scale_log_trust_radius
        upper = self.initial_log_scale + self.scale_log_trust_radius
        return self.log_scale.clamp(lower, upper).exp().clamp(
            self.scale_min, self.scale_max
        )

    def _hard_sign_groups(self) -> torch.Tensor:
        groups, _ = group_last_dim(self.sign_latent, self.group_size)
        return torch.where(
            groups < 0,
            -torch.ones_like(groups),
            torch.ones_like(groups),
        )

    def _hard_mask_groups(self) -> torch.Tensor:
        return self.mask_logits >= 0

    def _sign_surrogate_groups(self) -> torch.Tensor:
        groups, _ = group_last_dim(self.sign_latent, self.group_size)
        hard = self._hard_sign_groups()
        if self.freeze_binary_codebook:
            return hard
        mask = self._hard_mask_groups().to(groups.dtype)
        gradient_gate = (
            1.0 - mask
            + mask * self.active_sign_gradient_scale
        )
        return hard + gradient_gate * (groups - groups.detach())

    def _mask_surrogate_groups(self) -> torch.Tensor:
        probability = torch.sigmoid(self.mask_logits)
        hard = self._hard_mask_groups().to(probability.dtype)
        return hard + probability - probability.detach()

    def hard_state(self) -> SharedEmbeddingHardState:
        sign_groups = self._hard_sign_groups()
        mask_groups = self._hard_mask_groups()
        scales = self._scales()
        binary_groups = scales[..., None] * sign_groups
        ternary_groups = binary_groups * mask_groups.to(binary_groups.dtype)
        _, layout = group_last_dim(self.sign_latent, self.group_size)
        binary_codes = ungroup_last_dim(sign_groups, layout).to(torch.int8)
        mask = ungroup_last_dim(
            mask_groups.to(sign_groups.dtype), layout
        ).to(torch.bool)
        ternary_codes = binary_codes * mask.to(torch.int8)
        binary_weight = ungroup_last_dim(binary_groups, layout)
        ternary_weight = ungroup_last_dim(ternary_groups, layout)
        return SharedEmbeddingHardState(
            binary_codes=binary_codes,
            ternary_codes=ternary_codes,
            scales=scales.to(torch.float16),
            binary_weight=binary_weight,
            ternary_weight=ternary_weight,
            mask=mask,
        )

    def binary_weight(self, hard: bool = False) -> torch.Tensor:
        if hard:
            return self.hard_state().binary_weight
        sign = self._sign_surrogate_groups()
        groups = self._scales()[..., None] * sign
        _, layout = group_last_dim(self.sign_latent, self.group_size)
        return ungroup_last_dim(groups, layout)

    def ternary_weight(self, hard: bool = False) -> torch.Tensor:
        if hard:
            return self.hard_state().ternary_weight
        sign = self._sign_surrogate_groups()
        mask = self._mask_surrogate_groups()
        groups = self._scales()[..., None] * sign * mask
        _, layout = group_last_dim(self.sign_latent, self.group_size)
        return ungroup_last_dim(groups, layout)

    def sign_change_fraction(self) -> torch.Tensor:
        return (
            self._hard_sign_groups() != self.initial_sign_groups
        ).float().mean()

    def zero_fraction(self) -> torch.Tensor:
        return (~self._hard_mask_groups()).float().mean()

    def regularization_loss(
        self,
        scale_tether_weight: float = 0.0,
        active_sign_tether_weight: float = 0.0,
        zero_target: float | None = None,
        zero_weight: float = 0.0,
    ) -> torch.Tensor:
        loss = (
            self.log_scale - self.initial_log_scale
        ).square().mean() * float(scale_tether_weight)
        if active_sign_tether_weight > 0 and not self.freeze_binary_codebook:
            active = self._hard_mask_groups().to(self.sign_latent.dtype)
            groups, _ = group_last_dim(self.sign_latent, self.group_size)
            signed_margin = groups * self.initial_sign_groups
            loss = loss + float(active_sign_tether_weight) * (
                F.relu(-signed_margin) * active
            ).square().mean()
        if zero_target is not None and zero_weight > 0:
            loss = loss + float(zero_weight) * (
                self.zero_fraction() - float(zero_target)
            ).square()
        return loss


class SharedEmbeddingView(nn.Module):
    """Binary or ternary embedding view over one shared state."""

    def __init__(
        self,
        state: SharedBinaryTernaryEmbeddingState,
        mode: str,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"binary", "ternary"}:
            raise ValueError("mode must be binary or ternary")
        object.__setattr__(self, "_state", state)
        self.mode = mode
        self.padding_idx = padding_idx
        self.num_embeddings = state.num_embeddings
        self.embedding_dim = state.embedding_dim

    @property
    def state(self) -> SharedBinaryTernaryEmbeddingState:
        return object.__getattribute__(self, "_state")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = (
            self.state.binary_weight()
            if self.mode == "binary"
            else self.state.ternary_weight()
        )
        return F.embedding(
            input_ids,
            weight,
            padding_idx=self.padding_idx,
        )

    def hard_codes_and_scales(self):
        hard = self.state.hard_state()
        codes = (
            hard.binary_codes
            if self.mode == "binary"
            else hard.ternary_codes
        )
        weight = (
            hard.binary_weight
            if self.mode == "binary"
            else hard.ternary_weight
        )
        return codes, hard.scales, weight


class SharedEmbeddingLMHeadView(nn.Module):
    """Tied binary or ternary LM-head view over one shared embedding state."""

    def __init__(
        self,
        state: SharedBinaryTernaryEmbeddingState,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"binary", "ternary"}:
            raise ValueError("mode must be binary or ternary")
        object.__setattr__(self, "_state", state)
        self.mode = mode
        self.in_features = state.embedding_dim
        self.out_features = state.num_embeddings
        self.bias = None

    @property
    def state(self) -> SharedBinaryTernaryEmbeddingState:
        return object.__getattribute__(self, "_state")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = (
            self.state.binary_weight()
            if self.mode == "binary"
            else self.state.ternary_weight()
        )
        return F.linear(hidden, weight.to(hidden.dtype))

    def hard_codes_and_scales(self):
        hard = self.state.hard_state()
        codes = (
            hard.binary_codes
            if self.mode == "binary"
            else hard.ternary_codes
        )
        weight = (
            hard.binary_weight
            if self.mode == "binary"
            else hard.ternary_weight
        )
        return codes, hard.scales, weight
