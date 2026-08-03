from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DamageTracker:
    """Optional GRACE/PADP-inspired recovery-data scorer."""

    history: list[torch.Tensor] = field(default_factory=list)

    def update(self, damage: torch.Tensor) -> None:
        self.history.append(damage.detach().float().cpu())

    def variation(self) -> torch.Tensor:
        if not self.history:
            raise RuntimeError("no damage history")
        if len(self.history) == 1:
            return torch.zeros_like(self.history[0])
        deltas = [
            (self.history[i] - self.history[i - 1]).abs()
            for i in range(1, len(self.history))
        ]
        return torch.stack(deltas).mean(dim=0)


def teacher_student_damage(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    teacher_prob = F.softmax(teacher_logits.detach().float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (teacher_prob * (teacher_prob.clamp_min(1e-12).log() - student_log_prob)).sum(dim=-1)
    return token_kl.mean(dim=-1)


def select_diverse_coreset(
    embeddings: torch.Tensor,
    importance: torch.Tensor,
    budget: int,
    diversity_weight: float = 0.5,
) -> torch.Tensor:
    """Greedy facility-location selection with model-aware importance."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [samples, features]")
    if importance.shape != (embeddings.shape[0],):
        raise ValueError("importance must be [samples]")
    if not 0 < budget <= embeddings.shape[0]:
        raise ValueError("invalid budget")

    z = F.normalize(embeddings.float(), dim=-1)
    similarity = z @ z.T
    importance = (importance.float() - importance.float().min())
    importance = importance / importance.max().clamp_min(1.0e-12)

    selected = []
    covered = torch.full((embeddings.shape[0],), -torch.inf, device=embeddings.device)
    available = torch.ones(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)

    for _ in range(budget):
        new_coverage = torch.maximum(covered[:, None], similarity)
        coverage_gain = new_coverage.sum(dim=0) - covered.masked_fill(~torch.isfinite(covered), 0).sum()
        score = diversity_weight * coverage_gain + (1.0 - diversity_weight) * importance
        score = score.masked_fill(~available, -torch.inf)
        index = int(score.argmax())
        selected.append(index)
        covered = torch.maximum(covered, similarity[:, index])
        available[index] = False

    return torch.tensor(selected, device=embeddings.device, dtype=torch.long)
