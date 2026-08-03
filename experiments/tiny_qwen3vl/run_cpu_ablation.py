#!/usr/bin/env python3
"""CPU-only architecture-faithful binary recovery ablation.

This is intentionally tiny, not a quality model. It retains the Qwen3-VL text
operator topology needed to test transformation mathematics:
- pre-norm RMSNorm decoder blocks;
- 4 query heads and 2 KV heads (GQA);
- q/k head RMSNorm;
- RoPE with theta 5,000,000;
- bias-free q/k/v/o and SwiGLU gate/up/down projections;
- embedding and LM head coverage;
- exact contiguous g128 binary weights.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules
from engibona.projection import metric_project


@dataclass
class TinyConfig:
    vocab_size: int = 128
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 32
    max_position_embeddings: int = 64
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1.0e-6
    group_size: int = 128


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return value.to(x.dtype) * self.weight


class Rotary(nn.Module):
    def __init__(self, width: int, theta: float, length: int) -> None:
        super().__init__()
        half = width // 2
        inverse = 1.0 / (
            theta ** (torch.arange(half).float() / half)
        )
        angles = torch.arange(length).float()[:, None] * inverse
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = self.cosine[: q.shape[-2]][None, None]
        sine = self.sine[: q.shape[-2]][None, None]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            first, second = x[..., :half], x[..., half:]
            return torch.cat(
                (
                    first * cosine - second * sine,
                    first * sine + second * cosine,
                ),
                dim=-1,
            )

        return rotate(q), rotate(k)


class Attention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = Rotary(
            self.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q = self.q_norm(
            self.q_proj(x).view(
                batch, length, self.heads, self.head_dim
            )
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(x).view(
                batch, length, self.kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, length, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        q, k = self.rope(q, k)
        repeat = self.heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.full((length, length), float("-inf")), diagonal=1
        )
        output = scores.add(mask).softmax(dim=-1) @ v
        return self.o_proj(output.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class Block(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyQwen3VLText(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ):
        hidden = self.embed_tokens(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(self.norm(hidden))
        return (logits, states) if return_hidden else logits


def make_pool(
    seed: int, count: int, length: int, vocabulary: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(
        0, vocabulary, (count, length + 1), generator=generator
    )
    for position in range(2, length + 1):
        values[:, position] = (
            values[:, position - 1] + values[:, position - 2]
        ) % vocabulary
    return values[:, :-1], values[:, 1:]


def teacher_kl(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(teacher.float(), dim=-1)
    return F.kl_div(
        F.log_softmax(student.float(), dim=-1),
        probability,
        reduction="batchmean",
    ) / teacher.shape[1]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    teacher: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    input_ids, labels = data
    logits, hidden = model(input_ids, return_hidden=True)
    teacher_logits, teacher_hidden = teacher(
        input_ids, return_hidden=True
    )
    cross_entropy = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten()
    )
    hidden_cosine = torch.stack(
        [
            F.cosine_similarity(
                left.flatten(0, 1), right.flatten(0, 1), dim=-1
            ).mean()
            for left, right in zip(hidden, teacher_hidden)
        ]
    ).mean()
    return {
        "ce": float(cross_entropy),
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "teacher_kl": float(teacher_kl(teacher_logits, logits)),
        "hidden_cosine": float(hidden_cosine),
    }


def source_modules(model: nn.Module) -> dict[str, nn.Module]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    }


@torch.no_grad()
def naive_binary(model: nn.Module) -> nn.Module:
    quantized = copy.deepcopy(model)
    for module in source_modules(quantized).values():
        result = metric_project(
            module.weight,
            QuantMode.BINARY,
            group_size=128,
            metric=None,
            refine_steps=0,
        )
        module.weight.copy_(result.dequantized)
    return quantized


def hard_qat(
    base: nn.Module,
    teacher: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    batch_size: int,
) -> nn.Module:
    student = copy.deepcopy(base)
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        relaxation="hard_ste",
        export_strategy="trained",
    )
    modules = replace_linear_modules(
        student, config, include_embeddings=True
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=7.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, labels = data
    for step in range(steps):
        for module in modules.values():
            module.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch_size,))
        batch_ids, batch_labels = input_ids[indices], labels[indices]
        with torch.no_grad():
            teacher_logits = teacher(batch_ids)
        logits = student(batch_ids)
        loss = teacher_kl(teacher_logits, logits)
        loss = loss + 0.2 * F.cross_entropy(
            logits.flatten(0, 1), batch_labels.flatten()
        )
        loss = loss + sum(
            module.regularization_loss() for module in modules.values()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    return student


def run_seed(
    seed: int,
    fp_steps: int,
    recovery_steps: int,
    batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    config = TinyConfig()
    training = make_pool(seed + 1, 384, 20, config.vocab_size)
    validation = make_pool(seed + 2, 128, 32, config.vocab_size)
    model = TinyQwen3VLText(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    for _ in range(fp_steps):
        indices = torch.randint(0, len(training[0]), (batch_size,))
        logits = model(training[0][indices])
        loss = F.cross_entropy(
            logits.flatten(0, 1), training[1][indices].flatten()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    teacher = copy.deepcopy(model).eval()
    naive = naive_binary(model)
    recovered = hard_qat(
        model, teacher, training, recovery_steps, batch_size
    )
    return {
        "seed": seed,
        "config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fp32": evaluate(model, teacher, validation),
        "naive_absmean": evaluate(naive, teacher, validation),
        "hard_ste_kd_trained_state": evaluate(
            recovered, teacher, validation
        ),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in ("fp32", "naive_absmean", "hard_ste_kd_trained_state"):
        result[method] = {}
        for key in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
            values = torch.tensor([row[method][key] for row in rows])
            result[method][f"{key}_mean"] = float(values.mean())
            result[method][f"{key}_std"] = float(
                values.std(unbiased=False)
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--fp-steps", type=int, default=300)
    parser.add_argument("--recovery-steps", type=int, default=220)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="results_package_v2.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.time()
    rows = [
        run_seed(
            7700 + index,
            args.fp_steps,
            args.recovery_steps,
            args.batch,
        )
        for index in range(args.seeds)
    ]
    payload = {
        "arguments": vars(args),
        "runs": rows,
        "aggregate": aggregate(rows),
        "seconds": time.time() - start,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
