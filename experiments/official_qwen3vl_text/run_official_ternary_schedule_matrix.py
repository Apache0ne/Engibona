#!/usr/bin/env python3
"""Resolve ternary soft-to-hard scheduling on official Qwen3-VL text code."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules
from engibona.modules_tied import TiedGroupQuantizedLMHead
from run_official_cpu_smoke import TinyOfficialQwen3VL, evaluate, kd_loss, make_data, train_teacher


def unique_quant_states(replaced):
    states = {}
    for module in replaced.values():
        state = module.embedding if isinstance(module, TiedGroupQuantizedLMHead) else module
        if hasattr(state, "hard_codes_and_scales"):
            states[id(state)] = state
    return list(states.values())


def snapshot(states):
    return [
        (
            state.hard_codes_and_scales()[0].detach().cpu().clone(),
            state.hard_codes_and_scales()[1].detach().float().cpu().clone(),
        )
        for state in states
    ]


def state_statistics(initial, states):
    changed = 0
    total = 0
    zeros = 0
    scale_log_change = []
    exact = True
    for (initial_codes, initial_scales), state in zip(initial, states):
        codes, scales, _ = state.hard_codes_and_scales()
        codes = codes.detach().cpu()
        scales = scales.detach().float().cpu()
        exact = exact and set(codes.unique().tolist()) <= {-1, 0, 1}
        changed += int((codes != initial_codes).sum())
        total += codes.numel()
        zeros += int((codes == 0).sum())
        scale_log_change.append(
            (scales.clamp_min(1e-12).log() - initial_scales.clamp_min(1e-12).log()).abs().mean()
        )
    return {
        "exact_alphabet": exact,
        "code_change_rate": changed / max(total, 1),
        "zero_ratio": zeros / max(total, 1),
        "mean_abs_log_scale_change": float(torch.stack(scale_log_change).mean()),
    }


def recover(model, teacher, data, relaxation, hard_start, steps, batch):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=QuantMode.TERNARY,
        relaxation=relaxation,
        hard_recovery_start=hard_start,
        export_strategy="trained",
        ternary_zero_weight=1.0e-4,
        scale_tether_weight=1.0e-5,
    )
    replaced = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    states = unique_quant_states(replaced)
    initial = snapshot(states)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=7.0e-4, betas=(0.9, 0.95), weight_decay=0.001
    )
    input_ids, labels = data
    student.train()
    for step in range(steps):
        for state in states:
            state.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[indices])
        logits = student(input_ids[indices])
        loss = kd_loss(teacher_logits, logits)
        loss = loss + 0.2 * F.cross_entropy(
            logits.flatten(0, 1), labels[indices].flatten()
        )
        loss = loss + torch.stack([state.regularization_loss() for state in states]).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
    for state in states:
        state.set_schedule(steps, steps)
    student.eval()
    return student, state_statistics(initial, states)


def run_one(seed, layers, teacher_steps, recovery_steps, batch):
    torch.manual_seed(seed)
    train = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    methods = {"teacher": evaluate(teacher, teacher, validation)}
    specifications = [
        ("hard", "hard_ste", 0.0),
        ("catq_hard25", "catq", 0.25),
        ("catq_hard50", "catq", 0.50),
        ("catq_hard75", "catq", 0.75),
        ("categorical_hard50", "categorical", 0.50),
    ]
    for name, relaxation, hard_start in specifications:
        student, state = recover(
            fp, teacher, train, relaxation, hard_start, recovery_steps, batch
        )
        methods[name] = evaluate(student, teacher, validation) | state
    return {"seed": seed, "layers": layers, "methods": methods}


def aggregate(runs):
    output = {}
    for method in runs[0]["methods"]:
        output[method] = {}
        for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
            values = [run["methods"][method][metric] for run in runs]
            output[method][metric + "_mean"] = statistics.mean(values)
            output[method][metric + "_pstdev"] = statistics.pstdev(values)
        if method != "teacher":
            for metric in ("code_change_rate", "zero_ratio", "mean_abs_log_scale_change"):
                values = [run["methods"][method][metric] for run in runs]
                output[method][metric + "_mean"] = statistics.mean(values)
                output[method][metric + "_pstdev"] = statistics.pstdev(values)
            output[method]["all_exact_alphabet"] = all(
                run["methods"][method]["exact_alphabet"] for run in runs
            )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen3vl_ternary_schedule_matrix.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_one(6100 + layers * 100 + index, layers, args.teacher_steps, args.recovery_steps, args.batch)
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {"runs": runs, "aggregate": aggregate(runs)}
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "by_depth": by_depth,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({depth: value["aggregate"] for depth, value in by_depth.items()}, indent=2))
    for value in by_depth.values():
        for method, metrics in value["aggregate"].items():
            if method != "teacher" and not metrics["all_exact_alphabet"]:
                raise SystemExit(f"invalid alphabet: {method}")


if __name__ == "__main__":
    main()
