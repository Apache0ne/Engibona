import torch

from engibona.grouping import group_last_dim, ungroup_last_dim


def test_non_multiple_last_dimension_roundtrip() -> None:
    value = torch.arange(3 * 130, dtype=torch.float32).reshape(3, 130)
    groups, layout = group_last_dim(value, 128)
    assert groups.shape == (3, 2, 128)
    restored = ungroup_last_dim(groups, layout)
    assert torch.equal(restored, value)


def test_multiple_last_dimension_has_no_padding() -> None:
    value = torch.randn(4, 256)
    groups, layout = group_last_dim(value, 128)
    assert layout.padded_last_dim == 0
    assert torch.equal(ungroup_last_dim(groups, layout), value)
