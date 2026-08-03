#!/usr/bin/env python3
"""CPU smoke using Hugging Face's official Qwen3-VL text implementation.

No checkpoint download is required. The model is initialized from a tiny
Qwen3VLTextConfig, trained briefly on a synthetic autoregressive task, then
recovered into exact g128 binary weights through Engibona.

This validates integration with the official architecture code. It does not
prove PrismML's private quantization method.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules

try:
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Install transformers>=4.57 to run the official architecture smoke"
    ) from exc


class TinyOfficialQwen3VL(nn.Module):
    def __init__(self, layers: int = 2, tied: bool = True) -> None:
        super().__init__()
        rope = {
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        }
        common = dict(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            rms_norm_eps=1.0e-6,
            attention_bias=False,
            attention_dropout=0.0,
            hidden_act="silu",
            tie_word_embeddings=tied,
            use_cache=False,
        )
        try:
            config = Qwen3VLTextConfig(
                **common,
                rope_scaling=rope,
                rope_theta=5_000_000.0,
            )
        except TypeError:
            config = Qwen3VLTextConfig(**common, rope_parameters=rope)
        config._attn_implementation = "sdpa"
        self.text = Qwen3VLTextModel(config)
        self.lm_head = nn.Linear(128, 128, bias=False)
        if tied:
            self.lm_head.weight = self.text.embed_tokens.weight
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        output = self.text(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(output.last_hidden_state)
        if output_hidden_states:
            hidden = output.hidden_states or (output.last_hidden_state,)
            return logits, hidden
        return logits


def make_data(seed: int, count: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 128, (count, length + 1), generator=generator)
    for position in range(2, length + 1):
        values[:, position] = (
            values[:, position - 1] + values[:, position - 2]
        ) % 128
    return values[:, :-1], values[:, 1:]


def kd_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    probability = teacher.float().softmax(dim=-1)
    return F.kl_div(
        student.float().log_softmax(dim=-1),
        probability,
        reduction="batchmean",
    ) / teacher.shape[1]


@torch.no_grad()
def evaluate(model, teacher, data) -> dict[str, float]:
    input_ids, labels = data
    logits, hidden = model(input_ids, output_hidden_states=True)
    teacher_logits, teacher_hidden = teacher(
        input_ids, output_hidden_states=True
    )
    hidden_count = min(len(hidden), len(teacher_hidden))
    cosine = torch.stack(
        [
            F.cosine_similarity(
                hidden[index].reshape(-1, hidden[index].shape[-1]),
                teacher_hidden[index].reshape(
                    -1, teacher_hidden[index].shape[-1]
                ),
                dim=-1,
            ).mean()
            for index in range(hidden_count)
        ]
    ).mean()
    return {
        "ce": float(
            F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        ),
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "teacher_kl": float(kd_loss(teacher_logits, logits)),
        "hidden_cosine": float(cosine),
    }


def train_teacher(model, data, steps: int, batch: int) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    input_ids, labels = data
    model.train()
    for _ in range(steps):
        index = torch.randint(0, len(input_ids), (batch,))
        logits = model(input_ids[index])
        loss = F.cross_entropy(logits.flatten(0, 1), labels[index].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def recover_binary(model, teacher, data, steps: int, batch: int):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        relaxation="hard_ste",
        export_strategy="trained",
    )
    replaced = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    unique_modules = list(dict.fromkeys(id(module) for module in replaced.values()))
    assert unique_modules
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=7.0e-4, betas=(0.9, 0.95), weight_decay=0.001
    )
    input_ids, labels = data
    student.train()
    for step in range(steps):
        for module in replaced.values():
            if hasattr(module, "set_schedule"):
                module.set_schedule(step, steps)
        index = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[index])
        logits = student(input_ids[index])
        loss = kd_loss(teacher_logits, logits)
        loss = loss + 0.2 * F.cross_entropy(
            logits.flatten(0, 1), labels[index].flatten()
        )
        regularizers = [
            module.regularization_loss()
            for module in replaced.values()
            if hasattr(module, "regularization_loss")
        ]
        if regularizers:
            loss = loss + torch.stack(regularizers).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    return student, replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--teacher-steps", type=int, default=30)
    parser.add_argument("--recovery-steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen3vl_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(991)
    started = time.time()

    training = make_data(992, 128, 16)
    validation = make_data(993, 64, 24)
    model = TinyOfficialQwen3VL(layers=args.layers, tied=True)
    train_teacher(model, training, args.teacher_steps, args.batch)
    teacher = copy.deepcopy(model).eval()
    recovered, replaced = recover_binary(
        model, teacher, training, args.recovery_steps, args.batch
    )
    recovered.eval()

    exact_alphabet = True
    for module in replaced.values():
        if hasattr(module, "hard_codes_and_scales"):
            codes = module.hard_codes_and_scales()[0]
            exact_alphabet = exact_alphabet and set(codes.unique().tolist()) <= {-1, 1}

    result = {
        "architecture": {
            "implementation": "transformers.Qwen3VLTextModel",
            "layers": args.layers,
            "hidden_size": 128,
            "attention_heads": 4,
            "key_value_heads": 2,
            "head_dim": 32,
            "intermediate_size": 256,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
            "rope_theta": 5_000_000.0,
            "tied_embedding_lm_head": True,
        },
        "parameter_count_fp": sum(parameter.numel() for parameter in model.parameters()),
        "replaced_module_count": len(replaced),
        "exact_binary_alphabet": exact_alphabet,
        "teacher": evaluate(teacher, teacher, validation),
        "recovered": evaluate(recovered, teacher, validation),
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not exact_alphabet:
        raise SystemExit("exact binary alphabet validation failed")


if __name__ == "__main__":
    main()
