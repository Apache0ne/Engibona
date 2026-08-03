from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantMode(str, Enum):
    BINARY = "binary"
    TERNARY = "ternary"


@dataclass(slots=True)
class EngibonaConfig:
    """Highest-confidence default transformation settings.

    Defaults deliberately avoid lower-confidence mechanisms such as learned
    rotations, mixed precision escape paths, or an inference-time residual.
    """

    mode: QuantMode = QuantMode.TERNARY
    group_size: int = 128

    # High-confidence path: PTQ-quality initialization -> smooth hardening ->
    # metric-aware exact projection -> output reconstruction -> global recovery.
    metric: str = "activation_covariance"  # identity | activation_diag | activation_covariance
    ridge: float = 1.0e-4
    code_refine_steps: int = 8
    code_refine_tolerance: float = 1.0e-10

    # Smooth-to-hard continuation.
    relaxation: str = "auto"  # auto | categorical | catq
    initial_temperature: float = 1.0
    sensitivity_alpha: float = 1.5
    compression_fraction: float = 0.30

    # Loss mixture. Teacher and window-output reconstruction are enabled by
    # default; CKA and adaptive curriculum are available but not forced.
    ce_weight: float = 0.20
    kd_weight: float = 1.00
    window_reconstruction_weight: float = 1.00
    hidden_mse_weight: float = 0.10
    cka_weight: float = 0.00
    kd_temperature: float = 1.0

    # Lower-confidence extensions are opt-in.
    enable_dynamic_curriculum: bool = False
    enable_cka: bool = False
    enable_state_loss: bool = False

    def validate(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.metric not in {"identity", "activation_diag", "activation_covariance"}:
            raise ValueError(f"unsupported metric: {self.metric}")
        if self.relaxation not in {"auto", "categorical", "catq"}:
            raise ValueError(f"unsupported relaxation: {self.relaxation}")
        if not 0.0 <= self.compression_fraction <= 1.0:
            raise ValueError("compression_fraction must be in [0, 1]")
