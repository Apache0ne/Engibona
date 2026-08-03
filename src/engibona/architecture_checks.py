from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .modules import GroupQuantizedEmbedding, GroupQuantizedLinear
from .modules_tied import TiedGroupQuantizedLMHead


@dataclass(frozen=True, slots=True)
class CoverageReport:
    embeddings: int
    tied_heads: int
    linears: int
    unsupported_dense_matrices: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unsupported_dense_matrices


def low_bit_coverage_report(model: nn.Module) -> CoverageReport:
    """Report matrix modules that remain outside Engibona's low-bit wrappers."""
    embeddings = 0
    tied_heads = 0
    linears = 0
    unsupported: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, GroupQuantizedEmbedding):
            embeddings += 1
        elif isinstance(module, TiedGroupQuantizedLMHead):
            tied_heads += 1
        elif isinstance(module, GroupQuantizedLinear):
            linears += 1
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            unsupported.append(name)
    return CoverageReport(
        embeddings=embeddings,
        tied_heads=tied_heads,
        linears=linears,
        unsupported_dense_matrices=tuple(sorted(unsupported)),
    )
