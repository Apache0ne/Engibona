import importlib.util
from pathlib import Path

import pytest
import torch


pytest.importorskip("pandas")
pytest.importorskip("requests")
pytest.importorskip("huggingface_hub")
pytest.importorskip("safetensors")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "public_bonsai_forensics"
    / "run_code_structure_forensics.py"
)
SPEC = importlib.util.spec_from_file_location("code_structure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
structure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(structure)


def test_binary_affine_threshold_is_identified_exactly() -> None:
    torch.manual_seed(0)
    base = torch.randn(32, 128)
    threshold = torch.randn(32, 1) * 0.2
    codes = torch.where(base >= threshold, 1.0, -1.0)
    positive, either, transitions = structure.best_binary_threshold(base, codes)
    assert torch.allclose(positive, torch.ones_like(positive))
    assert torch.allclose(either, torch.ones_like(either))
    assert float(transitions.max()) <= 1.0 / 127.0 + 1e-7


def test_binary_nonmonotone_codes_cannot_be_explained_by_one_threshold() -> None:
    base = torch.arange(128).float()[None, :].repeat(8, 1)
    codes = torch.where((base.long() % 2) == 0, 1.0, -1.0)
    positive, _, transitions = structure.best_binary_threshold(base, codes)
    assert float(positive.mean()) < 0.52
    assert float(transitions.mean()) > 0.95


def test_sign_locked_ternary_magnitude_threshold_is_exact() -> None:
    torch.manual_seed(1)
    base = torch.randn(24, 128)
    threshold = base.abs().quantile(0.4, dim=1, keepdim=True)
    codes = torch.sign(base) * (base.abs() > threshold).float()
    full, mask, _ = structure.best_ternary_sign_locked_threshold(base, codes)
    assert torch.allclose(full, torch.ones_like(full))
    assert torch.allclose(mask, torch.ones_like(mask))


def test_sign_flips_reduce_sign_locked_ternary_explanation() -> None:
    torch.manual_seed(2)
    base = torch.randn(16, 128)
    codes = torch.sign(base) * (base.abs() > 0.4).float()
    active = codes != 0
    active_indices = active.nonzero()[:128]
    for row, column in active_indices:
        codes[row, column] *= -1
    full, _, _ = structure.best_ternary_sign_locked_threshold(base, codes)
    assert float(full.mean()) < 0.96
