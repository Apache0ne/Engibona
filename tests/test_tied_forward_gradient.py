import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 128)
        self.head = nn.Linear(128, 32, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(ids))


def test_tied_embedding_head_share_one_gradient_state() -> None:
    model = TiedModel()
    replace_linear_modules(
        model,
        EngibonaConfig(mode=QuantMode.BINARY),
        preserve_tied_weights=True,
    )
    model(torch.tensor([[1, 2, 3]])).square().mean().backward()
    assert model.embed.latent_weight.grad is not None
    assert float(model.embed.latent_weight.grad.abs().sum()) > 0
    latent = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("latent_weight")
    ]
    assert len(latent) == 1
