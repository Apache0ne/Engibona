import torch

from engibona.curriculum import DamageTracker, select_diverse_coreset


def test_damage_variation() -> None:
    tracker = DamageTracker()
    tracker.update(torch.tensor([1.0, 2.0]))
    tracker.update(torch.tensor([2.0, 2.5]))
    assert torch.allclose(tracker.variation(), torch.tensor([1.0, 0.5]))


def test_coreset_unique() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(20, 8)
    importance = torch.rand(20)
    selected = select_diverse_coreset(embeddings, importance, budget=5)
    assert selected.unique().numel() == 5
