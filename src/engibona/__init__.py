"""Engibona: evidence-driven grouped binary/ternary transformation research."""

from .config import EngibonaConfig, QuantMode
from .projection import ProjectionResult, metric_project
from .modules import (
    GroupQuantizedLinear,
    GroupQuantizedEmbedding,
    replace_linear_modules,
)
from .relaxation import (
    categorical_relaxation,
    catq_ternary_relaxation,
    compression_pressure,
    hessian_guided_temperature,
    hard_ste_weight,
)
from .packing import (
    pack_binary,
    unpack_binary,
    pack_ternary_2bit,
    unpack_ternary_2bit,
)
from .fisher_refinement import (
    FisherFlipCandidates,
    ValidatedFlipResult,
    binary_flip_predicted_delta,
    rank_binary_flips,
    apply_binary_flips,
    validated_prefix_search,
    selected_hessian_diagonal,
)

__all__ = [
    "EngibonaConfig",
    "QuantMode",
    "ProjectionResult",
    "metric_project",
    "GroupQuantizedLinear",
    "GroupQuantizedEmbedding",
    "replace_linear_modules",
    "categorical_relaxation",
    "catq_ternary_relaxation",
    "compression_pressure",
    "hessian_guided_temperature",
    "hard_ste_weight",
    "pack_binary",
    "unpack_binary",
    "pack_ternary_2bit",
    "unpack_ternary_2bit",
    "FisherFlipCandidates",
    "ValidatedFlipResult",
    "binary_flip_predicted_delta",
    "rank_binary_flips",
    "apply_binary_flips",
    "validated_prefix_search",
    "selected_hessian_diagonal",
]
