from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMode(str, Enum):
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(slots=True)
class EngibonaConfig:
    """Evidence-selected grouped binary/ternary recovery settings.

    Plain construction is the release-matched profile: it prioritizes agreement
    with direct public Bonsai weight geometry. Use `behavior_maximizing()` to
    make every embedding code trainable when the target is miniature-task
    recovery rather than released-checkpoint lineage.
    """

    mode: QuantMode = QuantMode.TERNARY
    group_size: int = 128

    metric: str = "activation_covariance"
    ridge: float = 1.0e-4
    code_refine_steps: int = 8
    code_refine_tolerance: float = 1.0e-10

    # auto => exact-hard binary; CAT-Q then exact-hard ternary.
    relaxation: str = "auto"
    initial_temperature: float = 1.0
    final_temperature: float = 0.08
    sensitivity_alpha: float = 1.5
    compression_fraction: float = 0.30
    hard_recovery_start: float = 0.50

    scale_min: float = 1.0e-6
    scale_max: float = 16.0
    scale_log_trust_radius: float = 2.0
    scale_tether_weight: float = 1.0e-5
    ternary_zero_target: float = 0.35
    ternary_zero_weight: float = 0.0

    # Direct public-weight forensics found an almost exact binary sign/absmean
    # embedding and a ternary embedding with trainable zeros/scales but preserved
    # nonzero signs. These are the release-matched defaults.
    binary_embedding_strategy: str = "frozen_ptq"
    ternary_embedding_strategy: str = "sign_locked_recovery"

    # Pure teacher KL is the selected behavior-preservation default. CE and
    # hidden/block terms remain explicit Pareto tradeoffs.
    ce_weight: float = 0.00
    kd_weight: float = 1.00
    window_reconstruction_weight: float = 0.00
    hidden_mse_weight: float = 0.00
    cka_weight: float = 0.00
    kd_temperature: float = 1.0

    export_strategy: str = "trained"

    enable_dynamic_curriculum: bool = False
    enable_cka: bool = False
    enable_state_loss: bool = False
    enable_fisher_code_refinement: bool = False

    @classmethod
    def release_matched(
        cls,
        mode: QuantMode | str = QuantMode.TERNARY,
        **overrides,
    ) -> "EngibonaConfig":
        """Prioritize matching released Bonsai code/scale fingerprints."""
        values = {
            "mode": QuantMode(mode),
            "binary_embedding_strategy": "frozen_ptq",
            "ternary_embedding_strategy": "sign_locked_recovery",
            "ce_weight": 0.0,
            "kd_weight": 1.0,
            "hidden_mse_weight": 0.0,
            "window_reconstruction_weight": 0.0,
            "export_strategy": "trained",
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def behavior_maximizing(
        cls,
        mode: QuantMode | str = QuantMode.TERNARY,
        **overrides,
    ) -> "EngibonaConfig":
        """Allow embedding sign/code movement for task-specific recovery.

        This profile can improve small-model teacher KL, especially at greater
        depth, but it intentionally does not match the released binary embedding
        fingerprint.
        """
        values = {
            "mode": QuantMode(mode),
            "binary_embedding_strategy": "train",
            "ternary_embedding_strategy": "train",
            "ce_weight": 0.0,
            "kd_weight": 1.0,
            "hidden_mse_weight": 0.0,
            "window_reconstruction_weight": 0.0,
            "export_strategy": "trained",
        }
        values.update(overrides)
        return cls(**values)

    def validate(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.metric not in {"identity", "activation_diag", "activation_covariance"}:
            raise ValueError(f"unsupported metric: {self.metric}")
        if self.relaxation not in {"auto", "hard_ste", "categorical", "catq"}:
            raise ValueError(f"unsupported relaxation: {self.relaxation}")
        if self.export_strategy not in {"trained", "metric_reproject"}:
            raise ValueError(f"unsupported export_strategy: {self.export_strategy}")
        if self.binary_embedding_strategy not in {"frozen_ptq", "train"}:
            raise ValueError(
                f"unsupported binary_embedding_strategy: {self.binary_embedding_strategy}"
            )
        if self.ternary_embedding_strategy not in {
            "sign_locked_recovery", "train", "frozen_ptq"
        }:
            raise ValueError(
                f"unsupported ternary_embedding_strategy: {self.ternary_embedding_strategy}"
            )
        if not 0.0 <= self.compression_fraction <= 1.0:
            raise ValueError("compression_fraction must be in [0, 1]")
        if not 0.0 <= self.hard_recovery_start <= 1.0:
            raise ValueError("hard_recovery_start must be in [0, 1]")
        if self.scale_min <= 0 or self.scale_max <= self.scale_min:
            raise ValueError("invalid scale bounds")
        if self.scale_log_trust_radius <= 0:
            raise ValueError("scale_log_trust_radius must be positive")
        if not 0.0 <= self.ternary_zero_target <= 1.0:
            raise ValueError("ternary_zero_target must be in [0, 1]")
        for name, value in (
            ("ce_weight", self.ce_weight),
            ("kd_weight", self.kd_weight),
            ("window_reconstruction_weight", self.window_reconstruction_weight),
            ("hidden_mse_weight", self.hidden_mse_weight),
            ("cka_weight", self.cka_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
