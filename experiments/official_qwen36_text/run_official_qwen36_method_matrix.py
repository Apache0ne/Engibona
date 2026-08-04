#!/usr/bin/env python3
"""Method matrix on the official Qwen3.6/Qwen3.5-text hybrid architecture.

PrismML's public 27B checkpoint names `Qwen/Qwen3.6-27B` as its source. That
model uses the Hugging Face `qwen3_5_text` architecture: three linear-attention
layers followed by one full-attention layer, repeated through depth. Earlier
Engibona matrices used Qwen3-VL decoder blocks as an architecture-faithful proxy.
This experiment replaces that proxy with the official hybrid text model class.

It compares naive g128 projection, exact-hard recovery, categorical binary
recovery, and CAT-Q-to-hard ternary recovery. All students preserve the official
hybrid block sequence, linear attention, full GQA attention, Q/K normalization,
RoPE configuration, tied embedding/head state, and exact final alphabets.

This is public method-selection evidence, not a claim about PrismML's private
optimizer, data mixture, token count, or schedule.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules


class TinyOfficialQwen36(nn.Module):
    def __init__(
        self,
        layers: int,
        vocab_size: int = 256,
        hidden_size: int = 128,
        intermediate_size: int = 384,
        tied: bool = True,
    ) -> None:
        super().__init__()
        layer_types = [
            "linear_attention" if index % 4 != 3 else "full_attention"
            for index in range(layers)
        ]
        config = AutoConfig.for_model(
            "qwen3_5_text",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            layer_types=layer_types,
            hidden_act="silu",
            max_position_embeddings=1024,
            initializer_range=0.02,
            rms_norm_eps=1.0e-6,
            use_cache=False,
            attention_bias=False,
            rope_theta=10_000_000.0,
            partial_rotary_factor=0.25,
            tie_word_embeddings=tied,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        self.config = config
        self.text = AutoModel.from_config(config)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tied:
            self.lm_head.weight = self.text.embed_tokens.weight

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
            return logits, output.last_hidden_state, output.hidden_states
        return logits


def make_data(seed: int, examples: int, length: int, vocab_size: int = 256):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        3,
        vocab_size,
        (examples, length),
        generator=generator,
    )
    # Deterministic structured next-token relation creates learnable local and
    # long-range dependencies without relying on external corpora.
    rolled = torch.roll(input_ids, shifts=1, dims=1)
    positions = torch.arange(length)[None, :]
    targets = (
        input_ids
        + 3 * rolled
        + 5 * positions
        + (input_ids[:, :1] % 17)
    ) % (vocab_size - 3) + 3
    return input_ids, targets


def next_token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )


def kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    teacher_probability = teacher_logits.detach().float().softmax(dim=-1)
    return F.kl_div(
        student_logits.float().log_softmax(dim=-1),
        teacher_probability,
        reduction="batchmean",
    ) / student_logits.shape[1]


def train_teacher(
    model: nn.Module,
    training,
    steps: int,
    batch: int,
    learning_rate: float = 2.0e-3,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, targets = training
    model.train()
    for _ in range(steps):
        indices = torch.randint(0, len(input_ids), (batch,))
        logits = model(input_ids[indices])
        loss = next_token_ce(logits, targets[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()


@torch.no_grad()
def evaluate(student, teacher, validation) -> dict[str, float]:
    input_ids, targets = validation
    teacher_logits, teacher_hidden, _ = teacher(
        input_ids,
        output_hidden_states=True,
    )
    student_logits, student_hidden, _ = student(
        input_ids,
        output_hidden_states=True,
    )
    return {
        "ce": float(next_token_ce(student_logits, targets)),
        "accuracy": float(
            (student_logits.argmax(dim=-1) == targets).float().mean()
        ),
        "teacher_kl": float(kd_loss(teacher_logits, student_logits)),
        "hidden_cosine": float(
            F.cosine_similarity(
                teacher_hidden.reshape(-1, teacher_hidden.shape[-1]),
                student_hidden.reshape(-1, student_hidden.shape[-1]),
                dim=-1,
            ).mean()
        ),
    }


def exact_alphabet(modules: dict[str, nn.Module], mode: QuantMode) -> bool:
    allowed = {-1, 1} if mode == QuantMode.BINARY else {-1, 0, 1}
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            values = set(module.hard_codes_and_scales()[0].unique().tolist())
            if not values <= allowed:
                return False
    return True


def code_change_fraction(
    initial: dict[str, torch.Tensor],
    modules: dict[str, nn.Module],
) -> float:
    changed = 0
    total = 0
    for name, before in initial.items():
        after = modules[name].hard_codes_and_scales()[0].detach().cpu()
        changed += int((before != after).sum())
        total += before.numel()
    return changed / max(total, 1)


def snapshot_codes(modules: dict[str, nn.Module]):
    return {
        name: module.hard_codes_and_scales()[0].detach().cpu()
        for name, module in modules.items()
        if hasattr(module, "hard_codes_and_scales")
    }


def module_family(name: str) -> str:
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if ".mlp." in name:
        return "mlp"
    if "embed_tokens" in name or name == "lm_head":
        return "embedding_head"
    return "other"


def family_code_changes(
    initial: dict[str, torch.Tensor],
    modules: dict[str, nn.Module],
) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for name, before in initial.items():
        after = modules[name].hard_codes_and_scales()[0].detach().cpu()
        family = module_family(name)
        if family not in counts:
            counts[family] = [0, 0]
        counts[family][0] += int((before != after).sum())
        counts[family][1] += before.numel()
    return {
        family: changed / max(total, 1)
        for family, (changed, total) in sorted(counts.items())
    }


def naive_student(
    fp: nn.Module,
    mode: QuantMode,
) -> tuple[nn.Module, dict[str, nn.Module], dict[str, Any]]:
    student = copy.deepcopy(fp)
    config = EngibonaConfig.release_matched(
        mode=mode,
        relaxation="hard_ste",
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
        ternary_zero_weight=0.0,
    )
    modules = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    for module in modules.values():
        if hasattr(module, "set_schedule"):
            module.set_schedule(1, 1)
    return student.eval(), modules, {"config": config}


def recover_student(
    fp: nn.Module,
    teacher: nn.Module,
    training,
    mode: QuantMode,
    relaxation: str,
    steps: int,
    batch: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, nn.Module], dict[str, Any]]:
    student = copy.deepcopy(fp)
    config = EngibonaConfig.release_matched(
        mode=mode,
        relaxation=relaxation,
        hard_recovery_start=0.50,
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
        ternary_zero_weight=0.0,
    )
    modules = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    initial = snapshot_codes(modules)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, _ = training
    student.train()
    for step in range(steps):
        for module in modules.values():
            if hasattr(module, "set_schedule"):
                module.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[indices])
        student_logits = student(input_ids[indices])
        loss = kd_loss(teacher_logits, student_logits)
        regularizers = [
            module.regularization_loss()
            for module in modules.values()
            if hasattr(module, "regularization_loss")
        ]
        if regularizers:
            loss = loss + torch.stack(regularizers).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    for module in modules.values():
        if hasattr(module, "set_schedule"):
            module.set_schedule(steps, steps)
    student.eval()
    return student, modules, {
        "config": config,
        "code_change_fraction": code_change_fraction(initial, modules),
        "family_code_change_fraction": family_code_changes(initial, modules),
    }


def run_seed(
    seed: int,
    layers: int,
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    training = make_data(seed + 1, 256, 24)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen36(layers=layers, tied=True)
    train_teacher(fp, training, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    specifications = [
        ("binary_naive", QuantMode.BINARY, "naive"),
        ("binary_hard", QuantMode.BINARY, "hard_ste"),
        ("binary_categorical", QuantMode.BINARY, "categorical"),
        ("ternary_naive", QuantMode.TERNARY, "naive"),
        ("ternary_hard", QuantMode.TERNARY, "hard_ste"),
        ("ternary_catq_hard", QuantMode.TERNARY, "auto"),
    ]
    output = {
        "seed": seed,
        "layers": layers,
        "layer_types": fp.config.layer_types,
        "methods": {},
    }
    for name, mode, relaxation in specifications:
        torch.manual_seed(seed + (0 if mode == QuantMode.BINARY else 500000))
        if relaxation == "naive":
            student, modules, metadata = naive_student(fp, mode)
            metadata["code_change_fraction"] = 0.0
            metadata["family_code_change_fraction"] = {
                family: 0.0
                for family in {
                    module_family(module_name) for module_name in modules
                }
            }
        else:
            student, modules, metadata = recover_student(
                fp,
                teacher,
                training,
                mode,
                relaxation,
                recovery_steps,
                batch,
                learning_rate,
            )
        metrics = evaluate(student, teacher, validation)
        metrics.update(
            {
                "mode": mode.value,
                "relaxation": relaxation,
                "exact_alphabet": exact_alphabet(modules, mode),
                "code_change_fraction": metadata["code_change_fraction"],
                "family_code_change_fraction": metadata[
                    "family_code_change_fraction"
                ],
            }
        )
        output["methods"][name] = metrics
    return output


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    methods = runs[0]["methods"].keys()
    output = {}
    for method in methods:
        output[method] = {}
        for metric in (
            "ce",
            "accuracy",
            "teacher_kl",
            "hidden_cosine",
            "code_change_fraction",
        ):
            values = [float(run["methods"][method][metric]) for run in runs]
            output[method][metric + "_mean"] = statistics.mean(values)
            output[method][metric + "_pstdev"] = statistics.pstdev(values)
        families = sorted(
            {
                family
                for run in runs
                for family in run["methods"][method][
                    "family_code_change_fraction"
                ]
            }
        )
        output[method]["family_code_change_fraction_mean"] = {
            family: statistics.mean(
                float(
                    run["methods"][method][
                        "family_code_change_fraction"
                    ].get(family, 0.0)
                )
                for run in runs
            )
            for family in families
        }
        output[method]["all_exact_alphabet"] = all(
            run["methods"][method]["exact_alphabet"] for run in runs
        )
    return output


def select(by_depth: dict[str, Any]) -> dict[str, Any]:
    methods = next(iter(by_depth.values()))["aggregate"].keys()
    mean_kl = {
        method: statistics.mean(
            depth["aggregate"][method]["teacher_kl_mean"]
            for depth in by_depth.values()
        )
        for method in methods
    }
    return {
        "binary_behavior_winner": min(
            (name for name in methods if name.startswith("binary_")),
            key=lambda name: mean_kl[name],
        ),
        "ternary_behavior_winner": min(
            (name for name in methods if name.startswith("ternary_")),
            key=lambda name: mean_kl[name],
        ),
        "mean_teacher_kl": mean_kl,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen36_method_matrix.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_seed(
                36000 + layers * 100 + index,
                layers,
                args.teacher_steps,
                args.recovery_steps,
                args.batch,
                args.learning_rate,
            )
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {
            "runs": runs,
            "aggregate": aggregate(runs),
        }
    payload = {
        "implementation": "transformers qwen3_5_text official hybrid architecture",
        "arguments": vars(args),
        "by_depth": by_depth,
        "selection": select(by_depth),
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        "selection": payload["selection"],
        "aggregate": {
            depth: values["aggregate"]
            for depth, values in by_depth.items()
        },
    }, indent=2, allow_nan=True))
    for depth in by_depth.values():
        for method, metrics in depth["aggregate"].items():
            if not metrics["all_exact_alphabet"]:
                raise SystemExit(f"invalid exact alphabet for {method}")


if __name__ == "__main__":
    main()
