import importlib.util
from pathlib import Path

import pytest
import torch


pytest.importorskip("transformers")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "public_bonsai_forensics"
    / "run_full_model_functional_forensics.py"
)
SPEC = importlib.util.spec_from_file_location("functional_forensics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
functional = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(functional)


def test_grouped_binary_is_exact_absmean_projection() -> None:
    torch.manual_seed(0)
    weight = torch.randn(5, 256)
    quantized = functional.grouped_binary(weight, 128)
    groups = quantized.reshape(5, 2, 128)
    expected_scale = weight.reshape(5, 2, 128).abs().mean(dim=-1)
    assert torch.allclose(groups.abs().mean(dim=-1), expected_scale, atol=1e-6)
    assert set(torch.sign(groups).unique().tolist()) <= {-1.0, 1.0}


def test_grouped_ternary_uses_exact_three_state_alphabet() -> None:
    torch.manual_seed(1)
    weight = torch.randn(3, 256)
    quantized = functional.grouped_ternary(weight, 128)
    groups = quantized.reshape(3, 2, 128)
    scales = groups.abs().amax(dim=-1).clamp_min(1e-12)
    normalized = groups / scales[..., None]
    distance = torch.minimum(normalized.abs(), (normalized.abs() - 1.0).abs())
    assert float(distance.max()) < 1e-6


def test_linear_cka_identical_representation_is_one() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(2, 9, 16)
    assert abs(functional.linear_cka(hidden, hidden) - 1.0) < 1e-5
