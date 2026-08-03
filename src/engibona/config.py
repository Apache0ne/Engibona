from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMode(str, Enum):
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(slots=True)
class EngibonaConfig:
    """Evidence-driven grouped binary/ternary recovery settings."""

    mode: QuantMode = QuantMode.TERNARY
    group_size: int = 128

    metric: str = "activation_covariance"
    ridge: float = 1.0e-4
    code_refine_steps: int = 8
    code_refine_tolerance: float = 1.0e-10

    # auto => hard_ste for binary; CAT-Q then exact-hard for ternary.
    relaxation: str = "auto"
    initial_temperature: float = 1.0
    final_temperature: float = 0.08
    sensitivity_alpha: float = 1.5
    compression_fraction: float = 0.30
    # A clean official Qwen3-VL 2/4-layer matrix selected a midpoint
    # soft-to-hard transition as the strongest teacher-behavior compromise.
    hard_recovery_start: float = 0.50

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
