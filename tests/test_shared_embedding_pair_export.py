import tempfile

import torch

from engibona.embedding_shared import SharedBinaryTernaryEmbeddingState
from engibona.embedding_shared_export import (
    decode_shared_embedding_pair,
    export_shared_embedding_pair,
)


def test_shared_pair_export_round_trip_is_exact() -> None:
    generator = torch.Generator().manual_seed(551)
    weight = torch.randn(37, 256, generator=generator) * 0.035
    state = SharedBinaryTernaryEmbeddingState(weight, group_size=128)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        payload = export_shared_embedding_pair(state, handle.name)
        decoded = decode_shared_embedding_pair(handle.name)

    hard = state.hard_state()
    assert payload["format"] == "ENGIBONA_SHARED_EMBEDDING_PAIR_V1"
    assert torch.equal(decoded["binary_codes"], hard.binary_codes.cpu())
    assert torch.equal(decoded["ternary_codes"], hard.ternary_codes.cpu())
    assert torch.equal(decoded["mask"], hard.mask.cpu())
    assert torch.equal(decoded["scales"], hard.scales.cpu())
    assert torch.equal(
        decoded["binary_weight"],
        hard.binary_codes.cpu().float()
        * hard.scales.cpu().float().repeat_interleave(128, dim=1),
    )
    assert torch.equal(
        decoded["ternary_weight"],
        decoded["binary_weight"] * decoded["mask"].float(),
    )


def test_shared_pair_storage_does_not_duplicate_codebook_or_scales() -> None:
    generator = torch.Generator().manual_seed(887)
    state = SharedBinaryTernaryEmbeddingState(
        torch.randn(16, 128, generator=generator),
        group_size=128,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        payload = export_shared_embedding_pair(state, handle.name)
    assert "packed_binary_codes" in payload
    assert "packed_ternary_mask" in payload
    assert "scales_fp16" in payload
    assert "packed_ternary_codes" not in payload
    assert "ternary_scales_fp16" not in payload
