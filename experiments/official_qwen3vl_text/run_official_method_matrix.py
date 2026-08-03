#!/usr/bin/env python3
"""Multi-seed official Qwen3-VL text architecture quantization matrix."""
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
from run_official_cpu_smoke import (
    TinyOfficialQwen3VL,
    evaluate,
    kd_loss,
    make_data,
    train_teacher,
)


def quantize_without_recovery(model, mode: QuantMode):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=mode,
        relaxation="hard_ste",
        export_strategy="trained",
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
    return student.eval(), modules


def recover(
    model,
    teacher,
    data,
    mode: QuantMode,
    relaxation: str,
    steps: int,
    batch: int,
):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=mode,
        relaxation=relaxation,
        hard_recovery_start=0.65,
        export_strategy="trained",
        ternary_zero_weight=1.0e-4 if mode == QuantMode.TERNARY else 0.0,
    )
    modules = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=7.0e-4,
        betas=(0.9, 0.95),
        weight_decay=0.001,
    )
    input_ids, labels = data
    student.train()
    for step in range(steps):
        for module in modules.values():
            if hasattr(module, "set_schedule"):
                module.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        with torch.no_grad():
            teacher_logits = teacher(input_ids[indices])
        logits = student(input_ids[indices])
        loss = kd_loss(teacher_logits, logits)
        loss = loss + 0.2 * F.cross_entropy(
            logits.flatten(0, 1), labels[indices].flatten()
        )
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
    return student.eval(), modules


def exact_alphabet(modules, mode: QuantMode) -> bool:
    allowed = {-1, 1} if mode == QuantMode.BINARY else {-1, 0, 1}
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            if not set(module.hard_codes_and_scales()[0].unique().tolist()) <= allowed:
                return False
    return True


def zero_ratio(modules) -> float | None:
    values = []
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            codes = module.hard_codes_and_scales()[0]
            values.append(float((codes == 0).float().mean()))
    return statistics.mean(values) if values else None


def run_one(seed: int, layers: int, teacher_steps: int, recovery_steps: int, batch: int):
    torch.manual_seed(seed)
    train = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()

    methods = {}
    methods["teacher"] = evaluate(teacher, teacher, validation)

    specifications = [
        ("binary_naive", QuantMode.BINARY, None),
        ("binary_hard", QuantMode.BINARY, "hard_ste"),
        ("binary_categorical", QuantMode.BINARY, "categorical"),
        ("ternary_naive", QuantMode.TERNARY, None),
        ("ternary_hard", QuantMode.TERNARY, "hard_ste"),
        ("ternary_catq", QuantMode.TERNARY, "catq"),
        ("ternary_categorical", QuantMode.TERNARY, "categorical"),
    ]
    for name, mode, relaxation in specifications:
        if relaxation is None:
            student, modules = quantize_without_recovery(fp, mode)
        else:
            student, modules = recover(
                fp,
                teacher,
                train,
                mode,
                relaxation,
                recovery_steps,
                batch,
            )
        metrics = evaluate(student, teacher, validation)
        metrics["exact_alphabet"] = exact_alphabet(modules, mode)
        metrics["zero_ratio"] = zero_ratio(modules) if mode == QuantMode.TERNARY else None
        methods[name] = metrics
    return {"seed": seed, "layers": layers, "methods": methods}


def aggregate(runs):
    output = {}
    method_names = runs[0]["methods"].keys()
    for method in method_names:
        output[method] = {}
        for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
            values = [run["methods"][method][metric] for run in runs]
            output[method][metric + "_mean"] = statistics.mean(values)
            output[method][metric + "_pstdev"] = statistics.pstdev(values)
        output[method]["all_exact_alphabet"] = all(
            run["methods"][method].get("exact_alphabet", True) for run in runs
        )
        zero_values = [
            run["methods"][method].get("zero_ratio")
            for run in runs
            if run["methods"][method].get("zero_ratio") is not None
        ]
        if zero_values:
            output[method]["zero_ratio_mean"] = statistics.mean(zero_values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen3vl_method_matrix.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()

    by_depth = {}
    all_runs = []
    for layers in args.layers:
        runs = [
            run_one(4100 + layers * 100 + index, layers, args.teacher_steps, args.recovery_steps, args.batch)
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {"runs": runs, "aggregate": aggregate(runs)}
        all_runs.extend(runs)

    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "by_depth": by_depth,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({depth: data["aggregate"] for depth, data in by_depth.items()}, indent=2))

    for depth_data in by_depth.values():
        for method, values in depth_data["aggregate"].items():
            if method != "teacher" and not values["all_exact_alphabet"]:
                raise SystemExit(f"invalid final alphabet for {method}")


if __name__ == "__main__":
    main()
