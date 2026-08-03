#!/usr/bin/env python3
"""Official Qwen3-VL embedding-policy ablation under exact low-bit recovery."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedEmbedding, replace_linear_modules
from engibona.modules_tied import TiedGroupQuantizedLMHead
from run_official_cpu_smoke import TinyOfficialQwen3VL, evaluate, kd_loss, make_data, train_teacher


def unique_states(replaced):
    states = {}
    embedding = None
    for module in replaced.values():
        state = module.embedding if isinstance(module, TiedGroupQuantizedLMHead) else module
        if hasattr(state, "hard_codes_and_scales"):
            states[id(state)] = state
        if isinstance(state, GroupQuantizedEmbedding):
            embedding = state
    if embedding is None:
        raise RuntimeError("quantized embedding not found")
    return list(states.values()), embedding


def snapshot(state):
    codes, scales, _ = state.hard_codes_and_scales()
    source_sign = state.initial_sign_groups.reshape_as(codes).detach().cpu().to(torch.int8)
    return (
        codes.detach().cpu().clone(),
        scales.detach().float().cpu().clone(),
        source_sign.clone(),
    )


def state_delta(initial, state):
    initial_codes, initial_scales, source_sign = initial
    codes, scales, _ = state.hard_codes_and_scales()
    codes = codes.detach().cpu()
    scales = scales.detach().float().cpu()
    nonzero = codes != 0
    true_source_sign_flip = (
        (codes.sign() != source_sign) & nonzero
    ).float().sum() / nonzero.float().sum().clamp_min(1.0)
    return {
        "code_change_rate": float((codes != initial_codes).float().mean()),
        "nonzero_sign_flip_rate_vs_source": float(true_source_sign_flip),
        "zero_ratio": float((codes == 0).float().mean()),
        "mean_abs_log_scale_change": float(
            (
                scales.clamp_min(1e-12).log()
                - initial_scales.clamp_min(1e-12).log()
            ).abs().mean()
        ),
    }


def recover(model, teacher, data, mode, embedding_strategy, steps, batch):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=mode,
        relaxation="hard_ste" if mode == QuantMode.BINARY else "catq",
        hard_recovery_start=0.50,
        binary_embedding_strategy=(
            embedding_strategy if mode == QuantMode.BINARY else "frozen_ptq"
        ),
        ternary_embedding_strategy=(
            embedding_strategy if mode == QuantMode.TERNARY else "sign_locked_recovery"
        ),
        ce_weight=0.0,
        kd_weight=1.0,
        hidden_mse_weight=0.0,
        export_strategy="trained",
    )
    replaced = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    states, embedding = unique_states(replaced)
    initial_embedding = snapshot(embedding)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=7e-4,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, _ = data
    student.train()
    for step in range(steps):
        for state in states:
            state.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[indices])
        logits = student(input_ids[indices])
        loss = kd_loss(teacher_logits, logits)
        regularizers = []
        for state in states:
            value = state.regularization_loss()
            if value.requires_grad:
                regularizers.append(value)
        if regularizers:
            loss = loss + torch.stack(regularizers).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    for state in states:
        state.set_schedule(steps, steps)
    return student.eval(), state_delta(initial_embedding, embedding)


def run_one(seed, layers, teacher_steps, recovery_steps, batch):
    torch.manual_seed(seed)
    training = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, training, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    methods = {"teacher": evaluate(teacher, teacher, validation)}
    specifications = [
        ("binary_frozen_ptq_embedding", QuantMode.BINARY, "frozen_ptq"),
        ("binary_train_embedding", QuantMode.BINARY, "train"),
        ("ternary_sign_locked_embedding", QuantMode.TERNARY, "sign_locked_recovery"),
        ("ternary_train_embedding", QuantMode.TERNARY, "train"),
        ("ternary_frozen_ptq_embedding", QuantMode.TERNARY, "frozen_ptq"),
    ]
    for name, mode, strategy in specifications:
        student, embedding_delta = recover(
            fp,
            teacher,
            training,
            mode,
            strategy,
            recovery_steps,
            batch,
        )
        methods[name] = evaluate(student, teacher, validation) | {
            "embedding": embedding_delta
        }
    return {"seed": seed, "layers": layers, "methods": methods}


def aggregate(runs):
    result = {}
    for method in runs[0]["methods"]:
        result[method] = {}
        for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
            values = [run["methods"][method][metric] for run in runs]
            result[method][metric + "_mean"] = statistics.mean(values)
            result[method][metric + "_pstdev"] = statistics.pstdev(values)
        if method != "teacher":
            for metric in (
                "code_change_rate",
                "nonzero_sign_flip_rate_vs_source",
                "zero_ratio",
                "mean_abs_log_scale_change",
            ):
                values = [run["methods"][method]["embedding"][metric] for run in runs]
                result[method]["embedding_" + metric + "_mean"] = statistics.mean(values)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen3vl_embedding_policy_matrix.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_one(
                8100 + layers * 100 + index,
                layers,
                args.teacher_steps,
                args.recovery_steps,
                args.batch,
            )
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {"runs": runs, "aggregate": aggregate(runs)}
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "metric_correction": (
            "nonzero_sign_flip_rate_vs_source compares final nonzero signs "
            "against original FP embedding signs, including positions that were initially zero"
        ),
        "by_depth": by_depth,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({depth: value["aggregate"] for depth, value in by_depth.items()}, indent=2))


if __name__ == "__main__":
    main()
