import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "public_bonsai_forensics"
    / "run_streaming_weight_forensics.py"
)
SPEC = importlib.util.spec_from_file_location("forensics_math", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
forensics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forensics)


def test_naive_binary_release_has_exact_expected_fingerprint() -> None:
    torch.manual_seed(0)
    base = torch.randn(64, 128)
    binary_codes, binary_scales, binary = forensics.naive_binary(base)
    ternary_codes, ternary_scales, ternary = forensics.naive_ternary(base)
    item = {
        "shape": (64, 128),
        "base": base.half(),
        "binary": binary.half(),
        "ternary": ternary.half(),
    }
    metrics = forensics.analyze_tensor(
        "model.layers.0.self_attn.q_proj.weight", item
    )
    assert metrics["binary_sign_agreement_base"] == 1.0
    assert metrics["ternary_code_agreement_naive"] == 1.0
    assert metrics["binary_actual_over_naive_nmse"] < 1.01
    assert metrics["ternary_actual_over_naive_nmse"] < 1.01
    assert set(binary_codes.unique().tolist()) <= {-1.0, 1.0}
    assert set(ternary_codes.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.isfinite(binary_scales).all()
    assert torch.isfinite(ternary_scales).all()


def test_code_reassignment_is_detected() -> None:
    torch.manual_seed(1)
    base = torch.randn(32, 128)
    binary_codes, binary_scales, binary = forensics.naive_binary(base)
    binary = binary.clone()
    binary[:, :16] *= -1
    _, _, ternary = forensics.naive_ternary(base)
    metrics = forensics.analyze_tensor(
        "model.layers.1.mlp.down_proj.weight",
        {
            "shape": (32, 128),
            "base": base.half(),
            "binary": binary.half(),
            "ternary": ternary.half(),
        },
    )
    assert 0.12 < metrics["binary_sign_flip_rate"] < 0.13
    assert metrics["binary_sign_agreement_base"] < 0.88
