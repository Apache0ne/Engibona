import torch

from engibona.embedding_shared import (
    SharedBinaryTernaryEmbeddingState,
    SharedEmbeddingLMHeadView,
    SharedEmbeddingView,
)


def make_weight() -> torch.Tensor:
    generator = torch.Generator().manual_seed(441)
    return torch.randn(32, 256, generator=generator) * 0.04


def test_hard_ternary_is_binary_codebook_times_mask() -> None:
    state = SharedBinaryTernaryEmbeddingState(make_weight(), group_size=128)
    hard = state.hard_state()
    assert torch.equal(
        hard.ternary_codes,
        hard.binary_codes * hard.mask.to(torch.int8),
    )
    assert torch.equal(
        hard.ternary_weight,
        hard.binary_weight * hard.mask.to(hard.binary_weight.dtype),
    )
    assert set(hard.binary_codes.unique().tolist()) <= {-1, 1}
    assert set(hard.ternary_codes.unique().tolist()) <= {-1, 0, 1}


def test_binary_and_ternary_views_share_scales_and_nonzero_signs() -> None:
    state = SharedBinaryTernaryEmbeddingState(make_weight(), group_size=128)
    binary = SharedEmbeddingView(state, "binary")
    ternary = SharedEmbeddingView(state, "ternary")
    binary_codes, binary_scales, binary_weight = binary.hard_codes_and_scales()
    ternary_codes, ternary_scales, ternary_weight = ternary.hard_codes_and_scales()
    nonzero = ternary_codes != 0
    assert torch.equal(binary_scales, ternary_scales)
    assert torch.equal(binary_codes[nonzero], ternary_codes[nonzero])
    assert torch.equal(binary_weight[nonzero], ternary_weight[nonzero])
    assert torch.equal(ternary_weight[~nonzero], torch.zeros_like(ternary_weight[~nonzero]))


def test_tied_lm_head_views_use_the_same_hard_weights() -> None:
    state = SharedBinaryTernaryEmbeddingState(make_weight(), group_size=128)
    binary_embedding = SharedEmbeddingView(state, "binary")
    ternary_embedding = SharedEmbeddingView(state, "ternary")
    binary_head = SharedEmbeddingLMHeadView(state, "binary")
    ternary_head = SharedEmbeddingLMHeadView(state, "ternary")
    ids = torch.tensor([[1, 7, 12]])
    hidden = torch.randn(1, 3, state.embedding_dim)

    assert torch.allclose(
        binary_embedding(ids),
        torch.nn.functional.embedding(ids, state.binary_weight()),
    )
    assert torch.allclose(
        ternary_embedding(ids),
        torch.nn.functional.embedding(ids, state.ternary_weight()),
    )
    assert torch.allclose(
        binary_head(hidden),
        torch.nn.functional.linear(hidden, state.binary_weight()),
    )
    assert torch.allclose(
        ternary_head(hidden),
        torch.nn.functional.linear(hidden, state.ternary_weight()),
    )


def test_joint_binary_and_ternary_losses_reach_one_shared_state() -> None:
    state = SharedBinaryTernaryEmbeddingState(make_weight(), group_size=128)
    ids = torch.tensor([[0, 2, 4, 8]])
    binary = SharedEmbeddingView(state, "binary")
    ternary = SharedEmbeddingView(state, "ternary")
    loss = binary(ids).square().mean() + 0.7 * ternary(ids).abs().mean()
    loss.backward()
    assert state.sign_latent.grad is not None
    assert state.log_scale.grad is not None
    assert state.mask_logits.grad is not None
    assert torch.isfinite(state.sign_latent.grad).all()
    assert torch.isfinite(state.log_scale.grad).all()
    assert torch.isfinite(state.mask_logits.grad).all()


def test_active_sign_gradient_can_be_suppressed_without_breaking_zero_positions() -> None:
    state = SharedBinaryTernaryEmbeddingState(
        make_weight(),
        group_size=128,
        active_sign_gradient_scale=0.0,
    )
    state.mask_logits.data.fill_(4.0)
    active_loss = state.binary_weight().sum()
    active_gradient = torch.autograd.grad(
        active_loss,
        state.sign_latent,
        retain_graph=False,
    )[0]
    assert torch.count_nonzero(active_gradient) == 0

    state.mask_logits.data.fill_(-4.0)
    zero_loss = state.binary_weight().sum()
    zero_gradient = torch.autograd.grad(
        zero_loss,
        state.sign_latent,
        retain_graph=False,
    )[0]
    assert torch.count_nonzero(zero_gradient) == state.sign_latent.numel()


def test_frozen_binary_codebook_still_allows_mask_and_scale_recovery() -> None:
    state = SharedBinaryTernaryEmbeddingState(
        make_weight(),
        group_size=128,
        freeze_binary_codebook=True,
    )
    assert not state.sign_latent.requires_grad
    loss = state.ternary_weight().square().mean()
    loss.backward()
    assert state.log_scale.grad is not None
    assert state.mask_logits.grad is not None


def test_initial_mask_is_nontrivial_and_scale_is_positive() -> None:
    state = SharedBinaryTernaryEmbeddingState(make_weight(), group_size=128)
    hard = state.hard_state()
    assert 0 < int(hard.mask.sum()) < hard.mask.numel()
    assert torch.all(hard.scales > 0)
    assert 0.1 < float(state.zero_fraction()) < 0.6
