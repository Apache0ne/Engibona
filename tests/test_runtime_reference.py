import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.export import export_packed
from engibona.modules import GroupQuantizedEmbedding, GroupQuantizedLinear
from engibona.runtime_reference import (
    decode_packed_weight,
    packed_embedding,
    packed_linear,
)


def _export_item(module, config):
    model = nn.Sequential(module)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        return export_packed(model, handle.name, config)["tensors"]["0"]


def test_binary_packed_linear_matches_hard_reference_exactly() -> None:
    torch.manual_seed(0)
    config = EngibonaConfig(mode=QuantMode.BINARY)
    source = nn.Linear(130, 7, bias=True)
    module = GroupQuantizedLinear(source, config)
    item = _export_item(module, config)
    x = torch.randn(5, 130)
    expected = F.linear(x, module.hard_surrogate(), module.bias)
    actual = packed_linear(x, item, module.bias)
    decoded = decode_packed_weight(item)
    assert torch.equal(decoded.codes, module.hard_codes_and_scales()[0].cpu())
    assert torch.allclose(decoded.weight, module.hard_surrogate().cpu(), atol=2e-3)
    assert torch.allclose(actual, expected, atol=3e-3)


def test_ternary_packed_linear_matches_hard_reference_exactly() -> None:
    torch.manual_seed(1)
    config = EngibonaConfig(mode=QuantMode.TERNARY)
    source = nn.Linear(257, 5, bias=False)
    module = GroupQuantizedLinear(source, config)
    item = _export_item(module, config)
    x = torch.randn(4, 257)
    expected = F.linear(x, module.hard_surrogate())
    actual = packed_linear(x, item)
    decoded = decode_packed_weight(item)
    assert set(decoded.codes.unique().tolist()) <= {-1, 0, 1}
    assert torch.allclose(decoded.weight, module.hard_surrogate().cpu(), atol=2e-3)
    assert torch.allclose(actual, expected, atol=3e-3)


def test_packed_embedding_matches_hard_reference() -> None:
    torch.manual_seed(2)
    config = EngibonaConfig(mode=QuantMode.BINARY)
    module = GroupQuantizedEmbedding(nn.Embedding(17, 130), config)
    item = _export_item(module, config)
    ids = torch.tensor([[1, 3, 5], [2, 4, 6]])
    expected = F.embedding(ids, module.hard_surrogate())
    actual = packed_embedding(ids, item)
    assert torch.allclose(actual, expected, atol=2e-3)
