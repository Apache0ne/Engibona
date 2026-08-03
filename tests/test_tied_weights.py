import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedEmbedding, replace_linear_modules
from engibona.modules_tied import TiedGroupQuantizedLMHead


class TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 128)
        self.lm_head = nn.Linear(128, 32, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


def test_replacement_preserves_embedding_lm_head_tie() -> None:
    model = TinyTiedModel()
    replaced = replace_linear_modules(
        model,
        EngibonaConfig(mode=QuantMode.BINARY),
        preserve_tied_weights=True,
    )
    assert isinstance(model.embed_tokens, GroupQuantizedEmbedding)
    assert isinstance(model.lm_head, TiedGroupQuantizedLMHead)
    assert model.lm_head.embedding is model.embed_tokens
    assert "embed_tokens" in replaced
    assert "lm_head" in replaced

    ids = torch.tensor([[1, 2, 3]])
    logits = model(ids)
    expected = torch.nn.functional.linear(
        model.embed_tokens(ids), model.embed_tokens._surrogate()
    )
    assert torch.allclose(logits, expected)


def test_tied_state_is_not_duplicated_in_parameters() -> None:
    model = TinyTiedModel()
    replace_linear_modules(
        model,
        EngibonaConfig(mode=QuantMode.BINARY),
        preserve_tied_weights=True,
    )
    latent_ids = [
        id(parameter)
        for name, parameter in model.named_parameters()
        if "latent_weight" in name
    ]
    assert len(latent_ids) == 1
