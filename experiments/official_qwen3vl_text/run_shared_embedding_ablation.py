#!/usr/bin/env python3
"""Ablate the public shared binary-codebook/ternary-mask embedding state.

Two low-bit students are recovered jointly from one teacher:

* a binary transformer student;
* a ternary transformer student.

Transformer matrices remain separately quantized. The embedding/LM-head policy
is varied between independent per-mode states and one shared state satisfying
exactly:

    W_binary = s * b
    W_ternary = s * b * m

Shared variants test a frozen sign codebook, sign gradients concentrated on
ternary-zero coordinates, and fully trainable signs. The experiment measures
behavior, exact released-format invariants, parameter count, sign changes, zero
rate, and whether independent training spontaneously recovers the public shared
relation.

This validates the public representation choice; it does not identify the exact
private joint-training schedule.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.embedding_shared import (
    SharedBinaryTernaryEmbeddingState,
    SharedEmbeddingLMHeadView,
    SharedEmbeddingView,
)
from engibona.modules import replace_linear_modules


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "official_smoke", HERE / "run_official_cpu_smoke.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load official Qwen3-VL miniature")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class StudentPair(nn.Module):
    def __init__(
        self,
        binary: nn.Module,
        ternary: nn.Module,
        shared_state: SharedBinaryTernaryEmbeddingState | None = None,
    ) -> None:
        super().__init__()
        self.binary = binary
        self.ternary = ternary
        if shared_state is not None:
            self.shared_embedding_state = shared_state
        else:
            self.shared_embedding_state = None


def exact_alphabet(modules: dict[str, nn.Module], mode: QuantMode) -> bool:
    allowed = {-1, 1} if mode == QuantMode.BINARY else {-1, 0, 1}
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            values = set(module.hard_codes_and_scales()[0].unique().tolist())
            if not values <= allowed:
                return False
    return True


def embedding_module(modules: dict[str, nn.Module]) -> nn.Module:
    candidates = [
        module
        for name, module in modules.items()
        if name.endswith("embed_tokens") and hasattr(module, "hard_codes_and_scales")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one embedding module, found {len(candidates)}")
    return candidates[0]


def tensor_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-20))


def independent_relation(
    binary_modules: dict[str, nn.Module],
    ternary_modules: dict[str, nn.Module],
) -> dict[str, float]:
    binary_codes, binary_scales, binary_weight = embedding_module(
        binary_modules
    ).hard_codes_and_scales()
    ternary_codes, ternary_scales, ternary_weight = embedding_module(
        ternary_modules
    ).hard_codes_and_scales()
    nonzero = ternary_codes != 0
    mask_reconstruction = binary_weight * nonzero.to(binary_weight.dtype)
    return {
        "binary_ternary_nonzero_sign_agreement": float(
            (binary_codes[nonzero] == ternary_codes[nonzero]).float().mean()
        ),
        "binary_ternary_scale_correlation": tensor_correlation(
            binary_scales, ternary_scales
        ),
        "binary_codebook_mask_exact_fraction": float(
            (mask_reconstruction == ternary_weight).float().mean()
        ),
        "binary_codebook_mask_mean_abs_error": float(
            (mask_reconstruction - ternary_weight).abs().mean()
        ),
        "ternary_zero_fraction": float((~nonzero).float().mean()),
    }


def make_independent_pair(fp: nn.Module):
    binary = copy.deepcopy(fp)
    ternary = copy.deepcopy(fp)
    binary_config = EngibonaConfig.release_matched(
        mode=QuantMode.BINARY,
        relaxation="hard_ste",
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
    )
    ternary_config = EngibonaConfig.release_matched(
        mode=QuantMode.TERNARY,
        relaxation="auto",
        hard_recovery_start=0.50,
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
        ternary_zero_weight=0.0,
    )
    binary_modules = replace_linear_modules(
        binary,
        binary_config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    ternary_modules = replace_linear_modules(
        ternary,
        ternary_config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    return (
        StudentPair(binary, ternary),
        binary_modules,
        ternary_modules,
        None,
    )


def make_shared_pair(
    fp: nn.Module,
    policy: str,
):
    binary = copy.deepcopy(fp)
    ternary = copy.deepcopy(fp)
    source_weight = fp.text.embed_tokens.weight.detach().clone()
    if policy == "shared_frozen_sign":
        freeze_sign = True
        active_gradient = 0.0
    elif policy == "shared_mask_focused_sign":
        freeze_sign = False
        active_gradient = 0.0
    elif policy == "shared_full_sign":
        freeze_sign = False
        active_gradient = 1.0
    else:
        raise ValueError(f"unsupported shared policy: {policy}")
    state = SharedBinaryTernaryEmbeddingState(
        source_weight,
        group_size=128,
        freeze_binary_codebook=freeze_sign,
        active_sign_gradient_scale=active_gradient,
    )

    binary_config = EngibonaConfig.release_matched(
        mode=QuantMode.BINARY,
        relaxation="hard_ste",
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
    )
    ternary_config = EngibonaConfig.release_matched(
        mode=QuantMode.TERNARY,
        relaxation="auto",
        hard_recovery_start=0.50,
        ce_weight=0.0,
        kd_weight=1.0,
        scale_tether_weight=1.0e-5,
        ternary_zero_weight=0.0,
    )
    binary_modules = replace_linear_modules(
        binary,
        binary_config,
        include_embeddings=False,
        preserve_tied_weights=False,
    )
    ternary_modules = replace_linear_modules(
        ternary,
        ternary_config,
        include_embeddings=False,
        preserve_tied_weights=False,
    )
    # Remove orphaned temporary LM-head quantizers from schedules and metrics.
    binary_modules = {
        name: module
        for name, module in binary_modules.items()
        if name != "lm_head"
    }
    ternary_modules = {
        name: module
        for name, module in ternary_modules.items()
        if name != "lm_head"
    }

    binary.text.embed_tokens = SharedEmbeddingView(
        state, "binary", padding_idx=binary.config.pad_token_id
    )
    binary.lm_head = SharedEmbeddingLMHeadView(state, "binary")
    ternary.text.embed_tokens = SharedEmbeddingView(
        state, "ternary", padding_idx=ternary.config.pad_token_id
    )
    ternary.lm_head = SharedEmbeddingLMHeadView(state, "ternary")
    return (
        StudentPair(binary, ternary, state),
        binary_modules,
        ternary_modules,
        state,
    )


def evaluate_pair(pair: StudentPair, teacher, validation):
    return {
        "binary": smoke.evaluate(pair.binary, teacher, validation),
        "ternary": smoke.evaluate(pair.ternary, teacher, validation),
    }


def train_pair(
    pair: StudentPair,
    binary_modules: dict[str, nn.Module],
    ternary_modules: dict[str, nn.Module],
    shared_state: SharedBinaryTernaryEmbeddingState | None,
    teacher,
    training,
    steps: int,
    batch: int,
    learning_rate: float,
    zero_target: float,
) -> None:
    optimizer = torch.optim.AdamW(
        pair.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, _ = training
    pair.train()
    for step in range(steps):
        for module in binary_modules.values():
            if hasattr(module, "set_schedule"):
                module.set_schedule(step, steps)
        for module in ternary_modules.values():
            if hasattr(module, "set_schedule"):
                module.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[indices])
        binary_logits = pair.binary(input_ids[indices])
        ternary_logits = pair.ternary(input_ids[indices])
        loss = smoke.kd_loss(teacher_logits, binary_logits)
        loss = loss + smoke.kd_loss(teacher_logits, ternary_logits)

        regularizers = []
        for module in list(binary_modules.values()) + list(ternary_modules.values()):
            if hasattr(module, "regularization_loss"):
                regularizers.append(module.regularization_loss())
        if regularizers:
            loss = loss + torch.stack(regularizers).sum()
        if shared_state is not None:
            loss = loss + shared_state.regularization_loss(
                scale_tether_weight=1.0e-5,
                active_sign_tether_weight=1.0e-4,
                zero_target=zero_target,
                zero_weight=0.05,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pair.parameters(), 1.0)
        optimizer.step()

    for module in binary_modules.values():
        if hasattr(module, "set_schedule"):
            module.set_schedule(steps, steps)
    for module in ternary_modules.values():
        if hasattr(module, "set_schedule"):
            module.set_schedule(steps, steps)
    pair.eval()


def run_policy(
    policy: str,
    fp,
    teacher,
    training,
    validation,
    steps,
    batch,
    learning_rate,
    zero_target,
):
    if policy == "independent":
        pair, binary_modules, ternary_modules, state = make_independent_pair(fp)
    else:
        pair, binary_modules, ternary_modules, state = make_shared_pair(fp, policy)
    initial_parameters = sum(parameter.numel() for parameter in pair.parameters())
    train_pair(
        pair,
        binary_modules,
        ternary_modules,
        state,
        teacher,
        training,
        steps,
        batch,
        learning_rate,
        zero_target,
    )
    metrics = evaluate_pair(pair, teacher, validation)
    relation = (
        independent_relation(binary_modules, ternary_modules)
        if state is None
        else {
            "binary_ternary_nonzero_sign_agreement": 1.0,
            "binary_ternary_scale_correlation": 1.0,
            "binary_codebook_mask_exact_fraction": 1.0,
            "binary_codebook_mask_mean_abs_error": 0.0,
            "ternary_zero_fraction": float(state.zero_fraction()),
            "binary_sign_change_fraction": float(state.sign_change_fraction()),
        }
    )
    return {
        "policy": policy,
        "trainable_parameter_count": initial_parameters,
        "behavior": metrics,
        "combined_teacher_kl": (
            metrics["binary"]["teacher_kl"]
            + metrics["ternary"]["teacher_kl"]
        ),
        "embedding_relation": relation,
        "binary_exact_alphabet": exact_alphabet(
            binary_modules, QuantMode.BINARY
        ),
        "ternary_exact_alphabet": exact_alphabet(
            ternary_modules, QuantMode.TERNARY
        ),
    }


def run_seed(
    seed: int,
    layers: int,
    policies: list[str],
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
    zero_target: float,
):
    torch.manual_seed(seed)
    training = smoke.make_data(seed + 1, 256, 20)
    validation = smoke.make_data(seed + 2, 96, 32)
    fp = smoke.TinyOfficialQwen3VL(layers=layers, tied=True)
    smoke.train_teacher(fp, training, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    output = {"seed": seed, "layers": layers, "policies": {}}
    for policy in policies:
        torch.manual_seed(seed + 50000)
        output["policies"][policy] = run_policy(
            policy,
            fp,
            teacher,
            training,
            validation,
            recovery_steps,
            batch,
            learning_rate,
            zero_target,
        )
    return output


def aggregate(runs: list[dict[str, Any]], policies: list[str]):
    output = {}
    for policy in policies:
        rows = [run["policies"][policy] for run in runs]
        output[policy] = {}
        getters = {
            "combined_teacher_kl": lambda row: row["combined_teacher_kl"],
            "binary_teacher_kl": lambda row: row["behavior"]["binary"]["teacher_kl"],
            "ternary_teacher_kl": lambda row: row["behavior"]["ternary"]["teacher_kl"],
            "binary_hidden_cosine": lambda row: row["behavior"]["binary"]["hidden_cosine"],
            "ternary_hidden_cosine": lambda row: row["behavior"]["ternary"]["hidden_cosine"],
            "relation_nonzero_sign_agreement": lambda row: row["embedding_relation"]["binary_ternary_nonzero_sign_agreement"],
            "relation_scale_correlation": lambda row: row["embedding_relation"]["binary_ternary_scale_correlation"],
            "relation_exact_mask_fraction": lambda row: row["embedding_relation"]["binary_codebook_mask_exact_fraction"],
            "ternary_zero_fraction": lambda row: row["embedding_relation"]["ternary_zero_fraction"],
            "parameter_count": lambda row: row["trainable_parameter_count"],
        }
        if policy != "independent":
            getters["binary_sign_change_fraction"] = lambda row: row["embedding_relation"]["binary_sign_change_fraction"]
        for name, getter in getters.items():
            values = [float(getter(row)) for row in rows]
            output[policy][name + "_mean"] = statistics.mean(values)
            output[policy][name + "_pstdev"] = statistics.pstdev(values)
        output[policy]["all_exact_alphabet"] = all(
            row["binary_exact_alphabet"] and row["ternary_exact_alphabet"]
            for row in rows
        )
    baseline = output["independent"]["combined_teacher_kl_mean"]
    for policy in policies:
        output[policy]["combined_kl_over_independent"] = (
            output[policy]["combined_teacher_kl_mean"] / baseline
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=150)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--zero-target", type=float, default=0.312)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="shared_embedding_ablation.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    policies = [
        "independent",
        "shared_frozen_sign",
        "shared_mask_focused_sign",
        "shared_full_sign",
    ]
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_seed(
                18600 + layers * 100 + index,
                layers,
                policies,
                args.teacher_steps,
                args.recovery_steps,
                args.batch,
                args.learning_rate,
                args.zero_target,
            )
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {
            "runs": runs,
            "aggregate": aggregate(runs, policies),
        }
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "policies": policies,
        "by_depth": by_depth,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        depth: values["aggregate"] for depth, values in by_depth.items()
    }, indent=2, allow_nan=True))
    for depth in by_depth.values():
        for values in depth["aggregate"].values():
            if not values["all_exact_alphabet"]:
                raise SystemExit("invalid final alphabet")


if __name__ == "__main__":
    main()
