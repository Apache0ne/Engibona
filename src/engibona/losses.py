from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class RecoveryLossWeights:
    ce: float = 0.20
    kd: float = 1.00
    window: float = 1.00
    hidden_mse: float = 0.10
    cka: float = 0.00
    state: float = 0.00


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    t = float(temperature)
    teacher_prob = F.softmax(teacher_logits.detach().float() / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits.float() / t, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (t * t)


def next_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


def hidden_mse(
    student: Sequence[torch.Tensor],
    teacher: Sequence[torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    count = min(len(student), len(teacher))
    if count == 0:
        raise ValueError("hidden-state sequences must be non-empty")
    terms = []
    for index in range(count):
        s = student[index].float()
        t = teacher[index].detach().float()
        if attention_mask is None:
            terms.append(F.mse_loss(s, t))
        else:
            mask = attention_mask[..., None].to(s.dtype)
            terms.append(((s - t).square() * mask).sum() / (mask.sum().clamp_min(1) * s.shape[-1]))
    return torch.stack(terms).mean()


def linear_cka(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1.0e-12) -> torch.Tensor:
    x = x.float().reshape(-1, x.shape[-1])
    y = y.float().reshape(-1, y.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross = x.T @ y
    numerator = cross.square().sum()
    denominator = (x.T @ x).square().sum().sqrt() * (y.T @ y).square().sum().sqrt()
    return numerator / denominator.clamp_min(epsilon)


def cka_loss(student: Sequence[torch.Tensor], teacher: Sequence[torch.Tensor]) -> torch.Tensor:
    count = min(len(student), len(teacher))
    if count == 0:
        raise ValueError("hidden-state sequences must be non-empty")
    return torch.stack(
        [1.0 - linear_cka(student[i], teacher[i].detach()) for i in range(count)]
    ).mean()


def window_output_loss(student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_output.float(), teacher_output.detach().float())


def state_rollout_loss(student_states: Sequence[torch.Tensor], teacher_states: Sequence[torch.Tensor]) -> torch.Tensor:
    count = min(len(student_states), len(teacher_states))
    if count == 0:
        raise ValueError("state sequences must be non-empty")
    return torch.stack(
        [F.mse_loss(student_states[i].float(), teacher_states[i].detach().float()) for i in range(count)]
    ).mean()
