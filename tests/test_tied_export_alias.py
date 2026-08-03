import tempfile

import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.export import export_packed
from engibona.modules import replace_linear_modules


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 128)
        self.lm_head = nn.Linear(128, 32, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_tied_export_does_not_create_different_codebooks() -> None:
    model = TiedModel()
    config = EngibonaConfig(mode=QuantMode.BINARY)
    replace_linear_modules(model, config, preserve_tied_weights=True)
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        payload = export_packed(model, handle.name, config)
    embedding = payload["tensors"]["embed_tokens"]
    head = payload["tensors"]["lm_head"]
    assert torch.equal(embedding["packed_codes"], head["packed_codes"])
    assert torch.equal(embedding["scales_fp16"], head["scales_fp16"])
