import tempfile

import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.export import export_packed
from engibona.fisher_refinement import (
    apply_binary_flips,
    binary_flip_predicted_delta,
    rank_binary_flips,
    selected_hessian_diagonal,
    validated_prefix_search,
)
from engibona.modules import GroupQuantizedEmbedding, GroupQuantizedLinear
from engibona.packing import unpack_binary


def test_binary_forward_is_exact_hard_and_scales_train() -> None:
    torch.manual_seed(0)
    module = GroupQuantizedLinear(
        nn.Linear(128, 8, bias=False),
        EngibonaConfig(mode=QuantMode.BINARY),
    )
    module.set_schedule(0, 10)
    module(torch.randn(4, 128)).square().mean().backward()
    assert module.log_scale.grad is not None
    assert float(module.log_scale.grad.abs().sum()) > 0

    codes, scales, weight = module.hard_codes_and_scales()
    assert set(codes.unique().tolist()) <= {-1, 1}
    groups = weight.reshape(8, 1, 128)
    assert torch.allclose(
        groups.abs().mean(dim=-1), scales.float(), atol=2.0e-3
    )


def test_trained_export_preserves_learned_scales() -> None:
    torch.manual_seed(1)
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        export_strategy="trained",
    )
    model = nn.Sequential(
        GroupQuantizedLinear(nn.Linear(128, 4, bias=False), config)
    )
    with torch.no_grad():
        model[0].log_scale.add_(0.23)
    expected_codes, expected_scales, _ = model[0].hard_codes_and_scales()

    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        payload = export_packed(model, handle.name, config)
    item = payload["tensors"]["0"]
    assert torch.equal(item["scales_fp16"], expected_scales.cpu())
    assert item["shape"] == list(expected_codes.shape)


def test_embedding_is_in_exact_low_bit_coverage() -> None:
    module = GroupQuantizedEmbedding(
        nn.Embedding(16, 128),
        EngibonaConfig(mode=QuantMode.BINARY),
    )
    output = module(torch.tensor([[1, 2, 3]]))
    assert output.shape == (1, 3, 128)
    assert set(module.hard_codes_and_scales()[0].unique().tolist()) <= {-1, 1}


def test_fisher_flip_formula_and_application() -> None:
    codes = torch.tensor([[1] * 128], dtype=torch.int8)
    scales = torch.tensor([[0.5]])
    gradient = torch.zeros(1, 128)
    fisher = torch.ones(1, 128)
    gradient[0, 7] = 2.0

    predicted = binary_flip_predicted_delta(
        codes, scales, gradient, fisher
    )
    assert torch.isclose(predicted[0, 7], torch.tensor(-1.5))

    ranked = rank_binary_flips(
        codes, scales, gradient, fisher, topk=1
    )
    assert int(ranked.flat_indices[0]) == 7
    changed = apply_binary_flips(codes, ranked.flat_indices)
    assert int(changed[0, 7]) == -1


def test_validated_prefix_search_rejects_bad_prefixes() -> None:
    codes = torch.ones(8, dtype=torch.int8)
    ranked = torch.tensor([0, 1, 2, 3])
    target = codes.clone()
    target[:2] = -1
    result = validated_prefix_search(
        codes,
        ranked,
        lambda value: float(
            (value.float() - target.float()).square().sum()
        ),
        prefix_sizes=(1, 2, 4),
    )
    assert result.accepted_count == 2
    assert result.final_loss == 0.0


def test_selected_hessian_diagonal_matches_quadratic() -> None:
    value = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    curvature = torch.tensor([2.0, 4.0, 6.0])
    loss = 0.5 * (curvature * value.square()).sum()
    diagonal = selected_hessian_diagonal(
        loss, value, torch.tensor([0, 2])
    )
    assert torch.allclose(diagonal, torch.tensor([2.0, 6.0]))


def test_export_roundtrip_reconstructs_exact_hard_weight() -> None:
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        export_strategy="trained",
    )
    model = nn.Sequential(
        GroupQuantizedLinear(nn.Linear(128, 3, bias=False), config)
    )
    codes, scales, expected = model[0].hard_codes_and_scales()
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        item = export_packed(model, handle.name, config)["tensors"]["0"]

    decoded = unpack_binary(
        item["packed_codes"], codes.numel()
    ).reshape(codes.shape)
    reconstructed = (
        decoded.float().reshape(3, 1, 128)
        * item["scales_fp16"].float()[..., None]
    ).reshape_as(expected)

    assert torch.equal(decoded, codes.cpu())
    assert torch.allclose(reconstructed, expected.cpu(), atol=2.0e-3)
    assert "latent_weight" not in item
