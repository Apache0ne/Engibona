import torch

from engibona.config import QuantMode
from engibona.metrics import activation_covariance, activation_diagonal
from engibona.projection import metric_project, optimal_scale


def test_binary_exact_and_nonincreasing_error() -> None:
    torch.manual_seed(0)
    weight = torch.randn(8, 256)
    x = torch.randn(128, 256)
    metric = activation_covariance(x, 128)
    result = metric_project(weight, QuantMode.BINARY, 128, metric, refine_steps=8)
    assert set(result.codes.unique().tolist()) <= {-1, 1}
    assert torch.all(result.final_error <= result.initial_error + 1e-5)


def test_ternary_exact_and_nonincreasing_error() -> None:
    torch.manual_seed(1)
    weight = torch.randn(8, 256)
    x = torch.randn(128, 256)
    metric = activation_diagonal(x, 128)
    result = metric_project(weight, QuantMode.TERNARY, 128, metric, refine_steps=8)
    assert set(result.codes.unique().tolist()) <= {-1, 0, 1}
    assert torch.all(result.final_error <= result.initial_error + 1e-5)


def test_closed_form_scale_full_metric() -> None:
    torch.manual_seed(2)
    w = torch.randn(4, 16)
    c = torch.sign(w)
    a = torch.randn(4, 16, 16)
    h = a.transpose(-1, -2) @ a + 1e-3 * torch.eye(16)
    scale = optimal_scale(w, c, h)
    eps = 1e-3

    def error(s):
        r = w - s[:, None] * c
        return torch.einsum("bi,bij,bj->b", r, h, r)

    assert torch.all(error(scale) <= error(scale + eps) + 1e-5)
    assert torch.all(error(scale) <= error(scale - eps) + 1e-5)
