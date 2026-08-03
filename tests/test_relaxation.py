import torch

from engibona.config import QuantMode
from engibona.relaxation import categorical_relaxation, catq_ternary_relaxation


def test_categorical_low_temperature_is_discrete() -> None:
    x = torch.tensor([-2.0, -0.7, 0.2, 0.8, 2.0])
    y = categorical_relaxation(x, QuantMode.TERNARY, 1e-4)
    assert torch.allclose(y, torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0]), atol=1e-3)


def test_catq_large_sharpness_is_ternary_like() -> None:
    x = torch.tensor([-1.0, -0.2, 0.0, 0.2, 1.0])
    y = catq_ternary_relaxation(x, sharpness=40.0, threshold=0.5)
    assert torch.allclose(y, torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-3)
