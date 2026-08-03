from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMode(str, Enum):
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(slots=True)
class EngibonaConfig:
    """Evidence-driven grouped binary/ternary recovery settings.

    Defaults follow the strongest result from the architecture-faithful CPU
    ablation: exact hard binary forwards with trainable positive group scales,
    teacher-behavior recovery, and trained-state export. Local metric projection
    remains available for initialization and diagnostics, but is not the default
    finalizer because it can improve its local quadratic objective while harming
    end-to-end sequence behavior.
    """

    mode: QuantMode = QuantMode.TERNARY
    group_size: int = 128

    metric: str = "activation_covariance"
    ridge: float = 1.0e-4
    code_refine_steps: int = 8
    code_refine_tolerance: float = 1.0e-10

    # auto => hard_ste for binary; catq for ternary.
    relaxation: str = "auto"
    initial_temperature: float = 1.0
    final_temperature: float = 0.08
    sensitivity_alpha: float = 1.5
    compression_fraction: float = 0.30
    hard_recovery_start: float = 0.55

    # Positive learned group scales. The trust region protects against the
    # ternary zero-ratio/scale feedback failure observed in low-bit recovery.
    scale_min: float = 1.0e-6
    scale_max: float = 16.0
    scale_log_trust_radius: float = 2.0
    scale_tether_weight: float = 1.0e-5
    ternary_zero_target: float = 0.35
    ternary_zero_weight: float = 0.0

    ce_weight: float = 0.20
    kd_weight: float = 1.00
    window_reconstruction_weight: float = 1.00
    hidden_mse_weight: float = 0.10
    cka_weight: float = 0.00
    kd_temperature: float = 1.0

    # trained: preserve globally recovered exact codes/scales.
    # metric_reproject: diagnostic/PTQ path only.
    export_strategy: str = "trained"

    enable_dynamic_curriculum: bool = False
    enable_cka: bool = False
    enable_state_loss: bool = False
    enable_fisher_code_refinement: bool = False

    def validate(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.metric not in {"identity", "activation_diag", "activation_covariance"}:
            raise ValueError(f"unsupported metric: {self.metric}")
        if self.relaxation not in {"auto", "hard_ste", "categorical", "catq"}:
            raise ValueError(f"unsupported relaxation: {self.relaxation}")
        if self.export_strategy not in {"trained", "metric_reproject"}:
            raise ValueError(f"unsupported export_strategy: {self.export_strategy}")
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
