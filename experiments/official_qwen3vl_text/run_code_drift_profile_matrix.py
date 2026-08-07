#!/usr/bin/env python3
"""Select recovery paths using behavior and public released-code geometry.

A behavior-only miniature can choose methods that recover teacher logits while
producing a very different discrete trajectory from the public Bonsai models.
This experiment adds a second axis: final code-change profiles by decoder depth
and module type, measured relative to the same sign/threshold initialization.

It compares uniform optimization with a mild public-profile learning-rate prior.
The prior scales quantized linear-module learning rates by the released 1.7B
layer and module code-change fractions. Random minibatch sequences are paired
across methods. Released geometry is a method-selection target, not proof of the
private optimizer.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import GroupQuantizedLinear, replace_linear_modules
from run_official_cpu_smoke import (
    TinyOfficialQwen3VL,
    evaluate,
    kd_loss,
    make_data,
    train_teacher,
)


HERE = Path(__file__).resolve().parent
TARGET_PATH = (
    HERE.parent
    / "public_bonsai_forensics"
    / "public_1p7b_code_drift_targets.json"
)


def layer_index(name: str) -> int:
    match = re.search(r"(?:layers|h|blocks)\.(\d+)", name)
    return int(match.group(1)) if match else -1


def module_type(name: str) -> str:
    for token in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ):
        if token in name:
            return token
    return "other"


def load_targets() -> dict[str, Any]:
    return json.loads(TARGET_PATH.read_text(encoding="utf-8"))


def nearest_target_layer(tiny_layer: int, tiny_layers: int, target_layers: int) -> int:
    if tiny_layers <= 1:
        return target_layers - 1
    return int(round(tiny_layer / (tiny_layers - 1) * (target_layers - 1)))


def target_multiplier(
    name: str,
    mode: QuantMode,
    tiny_layers: int,
    targets: dict[str, Any],
) -> float:
    layer = layer_index(name)
    module = module_type(name)
    layer_values = targets["layer_code_change_fraction"][mode.value]
    module_values = targets["module_code_change_fraction"][mode.value]
    layer_mean = statistics.mean(layer_values)
    module_mean = statistics.mean(module_values.values())
    layer_factor = 1.0
    if layer >= 0:
        mapped = nearest_target_layer(layer, tiny_layers, len(layer_values))
        layer_factor = layer_values[mapped] / layer_mean
    module_factor = module_values.get(module, module_mean) / module_mean
    return float(min(1.5, max(0.5, layer_factor * module_factor)))


def make_optimizer(
    student: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    mode: QuantMode,
    tiny_layers: int,
    targets: dict[str, Any],
    base_lr: float,
    profile: str,
) -> tuple[torch.optim.Optimizer, dict[str, float]]:
    used: set[int] = set()
    groups = []
    multipliers = {}
    for name, module in modules.items():
        trainable = [
            parameter
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad and id(parameter) not in used
        ]
        if not trainable:
            continue
        for parameter in trainable:
            used.add(id(parameter))
        multiplier = (
            target_multiplier(name, mode, tiny_layers, targets)
            if profile == "public"
            else 1.0
        )
        multipliers[name] = multiplier
        groups.append(
            {
                "params": trainable,
                "lr": base_lr * multiplier,
                "weight_decay": 0.001,
            }
        )

    remaining = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad and id(parameter) not in used
    ]
    if remaining:
        groups.append(
            {
                "params": remaining,
                "lr": base_lr,
                "weight_decay": 0.001,
            }
        )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    return optimizer, multipliers


def snapshot_codes(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    output = {}
    for name, module in modules.items():
        if isinstance(module, GroupQuantizedLinear):
            output[name] = module.hard_codes_and_scales()[0].detach().cpu()
    return output


def exact_alphabet(modules: dict[str, torch.nn.Module], mode: QuantMode) -> bool:
    allowed = {-1, 1} if mode == QuantMode.BINARY else {-1, 0, 1}
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            if not set(module.hard_codes_and_scales()[0].unique().tolist()) <= allowed:
                return False
    return True


def code_profile(
    initial: dict[str, torch.Tensor],
    modules: dict[str, torch.nn.Module],
    mode: QuantMode,
    tiny_layers: int,
    targets: dict[str, Any],
) -> dict[str, Any]:
    by_layer_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    by_module_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_changed = 0
    total_values = 0
    tensor_rows = {}
    for name, before in initial.items():
        after = modules[name].hard_codes_and_scales()[0].detach().cpu()
        changed = int((before != after).sum())
        total = before.numel()
        layer = layer_index(name)
        module = module_type(name)
        if layer >= 0:
            by_layer_counts[layer][0] += changed
            by_layer_counts[layer][1] += total
        if module != "other":
            by_module_counts[module][0] += changed
            by_module_counts[module][1] += total
        total_changed += changed
        total_values += total
        tensor_rows[name] = {
            "layer": layer,
            "module": module,
            "change_fraction": changed / max(total, 1),
            "values": total,
        }

    layer_profile = [
        by_layer_counts[index][0] / max(by_layer_counts[index][1], 1)
        for index in range(tiny_layers)
    ]
    module_profile = {
        module: counts[0] / max(counts[1], 1)
        for module, counts in sorted(by_module_counts.items())
    }
    target_layers = targets["layer_code_change_fraction"][mode.value]
    mapped_layer_targets = [
        target_layers[
            nearest_target_layer(index, tiny_layers, len(target_layers))
        ]
        for index in range(tiny_layers)
    ]
    target_modules = targets["module_code_change_fraction"][mode.value]
    common_modules = sorted(set(module_profile) & set(target_modules))
    layer_error = np.asarray(layer_profile) - np.asarray(mapped_layer_targets)
    module_error = np.asarray(
        [module_profile[name] - target_modules[name] for name in common_modules]
    )
    if tiny_layers >= 2 and np.std(layer_profile) > 0 and np.std(mapped_layer_targets) > 0:
        correlation = float(np.corrcoef(layer_profile, mapped_layer_targets)[0, 1])
    else:
        correlation = float("nan")
    return {
        "overall_change_fraction": total_changed / max(total_values, 1),
        "layer_profile": layer_profile,
        "mapped_public_layer_target": mapped_layer_targets,
        "layer_profile_rmse": float(np.sqrt(np.mean(layer_error**2))),
        "layer_profile_correlation": correlation,
        "module_profile": module_profile,
        "public_module_target": {
            name: target_modules[name] for name in common_modules
        },
        "module_profile_rmse": float(np.sqrt(np.mean(module_error**2))),
        "tensor_profile": tensor_rows,
    }


def recover(
    fp,
    teacher,
    train,
    validation,
    mode: QuantMode,
    relaxation: str,
    optimizer_profile: str,
    tiny_layers: int,
    targets: dict[str, Any],
    steps: int,
    batch: int,
    learning_rate: float,
) -> dict[str, Any]:
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
    initial_codes = snapshot_codes(modules)
    optimizer, multipliers = make_optimizer(
        student,
        modules,
        mode,
        tiny_layers,
        targets,
        learning_rate,
        optimizer_profile,
    )
    input_ids, _ = train
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
    metrics = evaluate(student, teacher, validation)
    metrics.update(
        {
            "mode": mode.value,
            "relaxation": relaxation,
            "optimizer_profile": optimizer_profile,
            "exact_alphabet": exact_alphabet(modules, mode),
            "learning_rate_multipliers": multipliers,
            "code_profile": code_profile(
                initial_codes,
                modules,
                mode,
                tiny_layers,
                targets,
            ),
        }
    )
    return metrics


def run_seed(
    seed: int,
    layers: int,
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
    targets: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    teacher_baseline = evaluate(teacher, teacher, validation)
    methods = [
        (QuantMode.BINARY, "hard_ste", "uniform"),
        (QuantMode.BINARY, "hard_ste", "public"),
        (QuantMode.BINARY, "categorical", "uniform"),
        (QuantMode.BINARY, "categorical", "public"),
        (QuantMode.TERNARY, "hard_ste", "uniform"),
        (QuantMode.TERNARY, "hard_ste", "public"),
        (QuantMode.TERNARY, "auto", "uniform"),
        (QuantMode.TERNARY, "auto", "public"),
    ]
    output = {
        "seed": seed,
        "layers": layers,
        "teacher_baseline": teacher_baseline,
        "methods": {},
    }
    for mode, relaxation, profile in methods:
        name = f"{mode.value}_{relaxation}_{profile}"
        # Paired random minibatches within each mode.
        torch.manual_seed(seed + (0 if mode == QuantMode.BINARY else 700000))
        output["methods"][name] = recover(
            fp,
            teacher,
            train,
            validation,
            mode,
            relaxation,
            profile,
            layers,
            targets,
            recovery_steps,
            batch,
            learning_rate,
        )
    return output


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    names = runs[0]["methods"].keys()
    output = {}
    for name in names:
        output[name] = {}
        scalar_paths = {
            "ce": lambda row: row["ce"],
            "accuracy": lambda row: row["accuracy"],
            "teacher_kl": lambda row: row["teacher_kl"],
            "hidden_cosine": lambda row: row["hidden_cosine"],
            "overall_code_change": lambda row: row["code_profile"]["overall_change_fraction"],
            "layer_profile_rmse": lambda row: row["code_profile"]["layer_profile_rmse"],
            "layer_profile_correlation": lambda row: row["code_profile"]["layer_profile_correlation"],
            "module_profile_rmse": lambda row: row["code_profile"]["module_profile_rmse"],
        }
        for metric, getter in scalar_paths.items():
            values = [float(getter(run["methods"][name])) for run in runs]
            finite = [value for value in values if math.isfinite(value)]
            output[name][metric + "_mean"] = statistics.mean(finite) if finite else float("nan")
            output[name][metric + "_pstdev"] = statistics.pstdev(finite) if len(finite) > 1 else 0.0
        output[name]["all_exact_alphabet"] = all(
            run["methods"][name]["exact_alphabet"] for run in runs
        )
    return output


def aggregate_teacher_baseline(runs: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
        values = [float(run["teacher_baseline"][metric]) for run in runs]
        output[metric + "_mean"] = statistics.mean(values)
        output[metric + "_pstdev"] = statistics.pstdev(values)
    return output


def pareto(aggregate_by_depth: dict[str, Any]) -> dict[str, Any]:
    names = next(iter(aggregate_by_depth.values()))["aggregate"].keys()
    summary = {}
    for name in names:
        summary[name] = {
            "mean_teacher_kl": statistics.mean(
                depth["aggregate"][name]["teacher_kl_mean"]
                for depth in aggregate_by_depth.values()
            ),
            "mean_layer_profile_rmse": statistics.mean(
                depth["aggregate"][name]["layer_profile_rmse_mean"]
                for depth in aggregate_by_depth.values()
            ),
            "mean_module_profile_rmse": statistics.mean(
                depth["aggregate"][name]["module_profile_rmse_mean"]
                for depth in aggregate_by_depth.values()
            ),
        }
    frontier = []
    for name, values in summary.items():
        dominated = False
        for other, candidate in summary.items():
            if other == name:
                continue
            no_worse = all(
                candidate[key] <= values[key]
                for key in (
                    "mean_teacher_kl",
                    "mean_layer_profile_rmse",
                    "mean_module_profile_rmse",
                )
            )
            strictly_better = any(
                candidate[key] < values[key]
                for key in (
                    "mean_teacher_kl",
                    "mean_layer_profile_rmse",
                    "mean_module_profile_rmse",
                )
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(name)
    return {"methods": summary, "pareto_frontier": sorted(frontier)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="code_drift_profile_matrix.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    targets = load_targets()
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_seed(
                9100 + layers * 100 + index,
                layers,
                args.teacher_steps,
                args.recovery_steps,
                args.batch,
                args.learning_rate,
                targets,
            )
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {
            "runs": runs,
            "aggregate": aggregate(runs),
            "teacher_baseline": aggregate_teacher_baseline(runs),
        }
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "target_path": str(TARGET_PATH),
        "arguments": vars(args),
        "by_depth": by_depth,
        "selection": pareto(by_depth),
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        "selection": payload["selection"],
        "aggregate": {
            depth: data["aggregate"] for depth, data in by_depth.items()
        },
    }, indent=2, allow_nan=True))
    for depth in by_depth.values():
        for name, metrics in depth["aggregate"].items():
            if not metrics["all_exact_alphabet"]:
                raise SystemExit(f"invalid final alphabet for {name}")


if __name__ == "__main__":
    main()
