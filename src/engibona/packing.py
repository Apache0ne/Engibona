from __future__ import annotations

import torch


def pack_binary(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack {-1,+1} codes into uint8, least-significant bit first."""
    flat = codes.reshape(-1)
    if not bool(((flat == -1) | (flat == 1)).all()):
        raise ValueError("binary codes must be exactly -1 or +1")
    bits = (flat > 0).to(torch.uint8)
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.view(-1, 8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8)
    return (bits << shifts).sum(dim=-1).to(torch.uint8), pad


def unpack_binary(packed: torch.Tensor, count: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed.reshape(-1, 1) >> shifts) & 1).reshape(-1)[:count]
    return torch.where(bits.bool(), torch.ones_like(bits, dtype=torch.int8), -torch.ones_like(bits, dtype=torch.int8))


def pack_ternary_2bit(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pack {-1,0,+1} into hardware-friendly 2-bit slots."""
    flat = codes.reshape(-1).to(torch.int8)
    if not bool(((flat == -1) | (flat == 0) | (flat == 1)).all()):
        raise ValueError("ternary codes must be exactly -1, 0, or +1")
    symbols = (flat + 1).to(torch.uint8)
    pad = (-symbols.numel()) % 4
    if pad:
        symbols = torch.nn.functional.pad(symbols, (0, pad), value=1)
    symbols = symbols.view(-1, 4)
    shifts = torch.tensor([0, 2, 4, 6], device=symbols.device, dtype=torch.uint8)
    return (symbols << shifts).sum(dim=-1).to(torch.uint8), pad


def unpack_ternary_2bit(packed: torch.Tensor, count: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    symbols = ((packed.reshape(-1, 1) >> shifts) & 0b11).reshape(-1)[:count]
    if bool((symbols == 3).any()):
        raise ValueError("reserved ternary symbol encountered")
    return symbols.to(torch.int8) - 1
