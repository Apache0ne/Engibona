import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedLinear


def test_ternary_module_reaches_hard_alphabet() -> None:
    torch.manual_seed(4)
    module = GroupQuantizedLinear(
        nn.Linear(256, 16, bias=False),
        EngibonaConfig(mode=QuantMode.TERNARY),
    )
    module.set_schedule(9, 10)
    hard = module.hard_surrogate()
    assert hard.shape == module.latent_weight.shape
    assert torch.isfinite(hard).all()


def test_binary_forward() -> None:
    module = GroupQuantizedLinear(
        nn.Linear(128, 8),
        EngibonaConfig(mode=QuantMode.BINARY),
    )
    module.set_schedule(2, 10)
    out = module(torch.randn(3, 128))
    assert out.shape == (3, 8)
