from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class GroupLayout:
    original_shape: torch.Size
    padded_last_dim: int
    group_size: int

    @property
    def original_last_dim(self) -> int:
        return int(self.original_shape[-1])


def group_last_dim(tensor: torch.Tensor, group_size: int) -> tuple[torch.Tensor, GroupLayout]:
    """Group contiguous weights along the input/last dimension.

    A matrix [out_features, in_features] becomes
    [out_features, ceil(in_features/group_size), group_size].
    """
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least one dimension")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    last = tensor.shape[-1]
    pad = (-last) % group_size
    padded = F.pad(tensor, (0, pad)) if pad else tensor
    groups = padded.reshape(*padded.shape[:-1], padded.shape[-1] // group_size, group_size)
    return groups, GroupLayout(tensor.shape, pad, group_size)


def ungroup_last_dim(groups: torch.Tensor, layout: GroupLayout) -> torch.Tensor:
    expected_group = layout.group_size
    if groups.shape[-1] != expected_group:
        raise ValueError(
            f"last dimension {groups.shape[-1]} does not match group_size {expected_group}"
        )
    flat = groups.reshape(*groups.shape[:-2], groups.shape[-2] * groups.shape[-1])
    if layout.padded_last_dim:
        flat = flat[..., : layout.original_last_dim]
    return flat.reshape(layout.original_shape)
