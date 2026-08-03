import torch

from engibona.packing import (
    pack_binary,
    pack_ternary_2bit,
    unpack_binary,
    unpack_ternary_2bit,
)


def test_binary_roundtrip() -> None:
    codes = torch.tensor([-1, 1, -1, -1, 1, 1, -1, 1, 1], dtype=torch.int8)
    packed, _ = pack_binary(codes)
    assert torch.equal(unpack_binary(packed, codes.numel()), codes)


def test_ternary_roundtrip() -> None:
    codes = torch.tensor([-1, 0, 1, 1, 0, -1, 1], dtype=torch.int8)
    packed, _ = pack_ternary_2bit(codes)
    assert torch.equal(unpack_ternary_2bit(packed, codes.numel()), codes)
