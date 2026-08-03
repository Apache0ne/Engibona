import torch
import torch.nn as nn

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedEmbedding


def test_binary_embedding_default_is_exact_frozen_absmean_projection() -> None:
    torch.manual_seed(20)
    source = nn.Embedding(32, 128)
    module = GroupQuantizedEmbedding(
        source,
        EngibonaConfig(mode=QuantMode.BINARY),
    )
    codes, scales, weight = module.hard_codes_and_scales()
    expected_codes = torch.where(source.weight >= 0, 1, -1).to(torch.int8)
    expected_scales = source.weight.abs().mean(dim=-1, keepdim=True).half()
    assert torch.equal(codes, expected_codes)
    assert torch.allclose(scales, expected_scales, atol=2e-3)
    assert torch.allclose(
        weight,
        expected_codes.float() * expected_scales.float(),
        atol=2e-3,
    )
    assert not module.latent_weight.requires_grad
    assert not module.log_scale.requires_grad


def test_ternary_embedding_sign_locked_recovery_never_flips_nonzero_signs() -> None:
    torch.manual_seed(21)
    source = nn.Embedding(32, 128)
    module = GroupQuantizedEmbedding(
        source,
        EngibonaConfig(mode=QuantMode.TERNARY),
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=5e-3)
    initial_sign = torch.where(source.weight >= 0, 1, -1)
    ids = torch.randint(0, 32, (8, 12))
    for step in range(30):
        module.set_schedule(step, 30)
        output = module(ids)
        # An arbitrary asymmetric target pushes assignments and scales, but the
        # policy must preserve the original sign of every nonzero state.
        target = torch.sin(torch.arange(128).float())[None, None, :]
        loss = (output - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    codes = module.hard_codes_and_scales()[0]
    nonzero = codes != 0
    assert torch.equal(codes[nonzero].sign(), initial_sign[nonzero].to(torch.int8))
    assert module.latent_weight.requires_grad
    assert module.log_scale.requires_grad


def test_ternary_embedding_full_training_remains_opt_in() -> None:
    module = GroupQuantizedEmbedding(
        nn.Embedding(16, 128),
        EngibonaConfig(
            mode=QuantMode.TERNARY,
            ternary_embedding_strategy="train",
        ),
    )
    assert module.latent_weight.requires_grad
    assert module.log_scale.requires_grad
