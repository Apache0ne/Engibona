"""Engibona: evidence-driven grouped binary/ternary transformation research."""

from .config import EngibonaConfig, QuantMode
from .projection import ProjectionResult, metric_project
from .relaxation import (
    categorical_relaxation,
    catq_ternary_relaxation,
    compression_pressure,
    hessian_guided_temperature,
)
from .packing import (
    pack_binary,
    unpack_binary,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)

__all__ = [
    "EngibonaConfig",
    "QuantMode",
    "ProjectionResult",
    "metric_project",
    "categorical_relaxation",
    "catq_ternary_relaxation",
    "compression_pressure",
    "hessian_guided_temperature",
    "pack_binary",
    "unpack_binary",
    "pack_ternary_2bit",
    "unpack_ternary_2bit",
]
