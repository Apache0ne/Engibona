import tempfile

import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.embedding_shared import (
    SharedBinaryTernaryEmbeddingState,
    SharedEmbeddingLMHeadView,
    SharedEmbeddingView,
)
from engibona.export import export_packed
from engibona.packing import unpack_binary, unpack_ternary_2bit


class JointEmbeddingContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(772)
        weight = torch.randn(48, 256, generator=generator) * 0.03
        self.shared_state = SharedBinaryTernaryEmbeddingState(
            weight,
            group_size=128,
        )
        self.binary_embedding = SharedEmbeddingView(
            self.shared_state, "binary"
        )
        self.binary_head = SharedEmbeddingLMHeadView(
            self.shared_state, "binary"
        )
        self.ternary_embedding = SharedEmbeddingView(
            self.shared_state, "ternary"
        )
        self.ternary_head = SharedEmbeddingLMHeadView(
            self.shared_state, "ternary"
        )


def decoded_codes(item: dict, mode: QuantMode) -> torch.Tensor:
    count = 1
    for value in item["shape"]:
        count *= int(value)
    if mode == QuantMode.BINARY:
        codes = unpack_binary(item["packed_codes"], count)
    else:
        codes = unpack_ternary_2bit(item["packed_codes"], count)
    return codes.reshape(item["shape"])


def test_mode_specific_export_uses_one_shared_codebook_and_scale() -> None:
    model = JointEmbeddingContainer()
    binary_config = EngibonaConfig(
        mode=QuantMode.BINARY,
        export_strategy="trained",
    )
    ternary_config = EngibonaConfig(
        mode=QuantMode.TERNARY,
        export_strategy="trained",
    )
    with tempfile.NamedTemporaryFile(suffix=".pt") as binary_file, tempfile.NamedTemporaryFile(
        suffix=".pt"
    ) as ternary_file:
        binary_payload = export_packed(
            model, binary_file.name, binary_config
        )
        ternary_payload = export_packed(
            model, ternary_file.name, ternary_config
        )

    assert set(binary_payload["tensors"]) == {
        "binary_embedding",
        "binary_head",
    }
    assert set(ternary_payload["tensors"]) == {
        "ternary_embedding",
        "ternary_head",
    }

    binary_embedding = binary_payload["tensors"]["binary_embedding"]
    binary_head = binary_payload["tensors"]["binary_head"]
    ternary_embedding = ternary_payload["tensors"]["ternary_embedding"]
    ternary_head = ternary_payload["tensors"]["ternary_head"]

    assert torch.equal(
        binary_embedding["packed_codes"], binary_head["packed_codes"]
    )
    assert torch.equal(
        ternary_embedding["packed_codes"], ternary_head["packed_codes"]
    )
    assert torch.equal(
        binary_embedding["scales_fp16"],
        ternary_embedding["scales_fp16"],
    )

    binary_codes = decoded_codes(binary_embedding, QuantMode.BINARY)
    ternary_codes = decoded_codes(ternary_embedding, QuantMode.TERNARY)
    nonzero = ternary_codes != 0
    assert torch.equal(binary_codes[nonzero], ternary_codes[nonzero])
    assert torch.equal(
        ternary_codes,
        binary_codes * nonzero.to(torch.int8),
    )
