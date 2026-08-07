#!/usr/bin/env python3
"""Test the scale structure inferred from public Bonsai checkpoints.

Direct public-weight forensics found that released/naive log-scale ratios are
partly separable into output-row and input-group-column effects. This experiment
tests whether a soft residual penalty for that structure improves behavior
recovery on the official Hugging Face Qwen3-VL text architecture miniature.

The penalty is applied only to quantized linear matrices, not embeddings:

    delta[i, g] = log_scale[i, g] - initial_log_scale[i, g]
    residual = delta - row_mean(delta) - column_mean(delta) + global_mean(delta)

A coefficient of zero is the current free-per-group baseline. This is a public
method-selection experiment; it does not claim PrismML used this regularizer.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

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


def scale_structure_terms(modules: dict[str, torch.nn.Module]):
    residual_losses = []
    total_energies = []
    residual_energies = []
    row_energies = []
    column_energies = []
    for module in modules.values():
        if not isinstance(module, GroupQuantizedLinear):
            continue
        delta = module.log_scale - module.initial_log_scale
        if delta.ndim != 2 or min(delta.shape) < 2:
            continue
        mean = delta.mean()
        row = delta.mean(dim=1, keepdim=True) - mean
        column = delta.mean(dim=0, keepdim=True) - mean
        additive = mean + row + column
        residual = delta - additive
        residual_losses.append(residual.square().mean())
        total_energies.append((delta - mean).square().sum())
        residual_energies.append(residual.square().sum())
        row_energies.append(row.expand_as(delta).square().sum())
        column_energies.append(column.expand_as(delta).square().sum())
    if not residual_losses:
        zero = torch.tensor(0.0)
        return zero, {
            "additive_r2": float("nan"),
            "row_energy_fraction": float("nan"),
            "column_energy_fraction": float("nan"),
        }
    penalty = torch.stack(residual_losses).sum()
    total = torch.stack(total_energies).sum().clamp_min(1e-20)
    residual = torch.stack(residual_energies).sum()
    return penalty, {
        "additive_r2": float(1.0 - residual / total),
        "row_energy_fraction": float(torch.stack(row_energies).sum() / total),
        "column_energy_fraction": float(torch.stack(column_energies).sum() / total),
    }


def code_snapshot(modules: dict[str, torch.nn.Module]) -> dict[str, torch.Tensor]:
    output = {}
    for name, module in modules.items():
        if isinstance(module, GroupQuantizedLinear):
            output[name] = module.hard_codes_and_scales()[0].detach().cpu()
    return output


def code_change_fraction(
    initial: dict[str, torch.Tensor],
    modules: dict[str, torch.nn.Module],
) -> float:
    changed = 0
    total = 0
    for name, before in initial.items():
        after = modules[name].hard_codes_and_scales()[0].detach().cpu()
        changed += int((before != after).sum())
        total += before.numel()
    return changed / max(total, 1)


def exact_alphabet(
    modules: dict[str, torch.nn.Module],
    mode: QuantMode,
) -> bool:
    allowed = {-1, 1} if mode == QuantMode.BINARY else {-1, 0, 1}
    for module in modules.values():
        if hasattr(module, "hard_codes_and_scales"):
            values = set(module.hard_codes_and_scales()[0].unique().tolist())
            if not values <= allowed:
                return False
    return True


def recover(
    fp,
    teacher,
    train,
    validation,
    mode: QuantMode,
    coefficient: float,
    steps: int,
    batch: int,
    learning_rate: float,
) -> dict[str, Any]:
    student = copy.deepcopy(fp)
    config = EngibonaConfig.release_matched(
        mode=mode,
        relaxation="hard_ste" if mode == QuantMode.BINARY else "auto",
        hard_recovery_start=0.50,
        scale_tether_weight=1.0e-5,
        ternary_zero_weight=0.0,
    )
    modules = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    initial_codes = code_snapshot(modules)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.001,
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
        module_regularizers = [
            module.regularization_loss()
            for module in modules.values()
            if hasattr(module, "regularization_loss")
        ]
        if module_regularizers:
            loss = loss + torch.stack(module_regularizers).sum()
        structure_penalty, _ = scale_structure_terms(modules)
        loss = loss + coefficient * structure_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

    for module in modules.values():
        if hasattr(module, "set_schedule"):
            module.set_schedule(steps, steps)
    student.eval()
    metrics = evaluate(student, teacher, validation)
    final_penalty, structure = scale_structure_terms(modules)
    metrics.update(
        {
            "coefficient": coefficient,
            "mode": mode.value,
            "scale_structure_penalty": float(final_penalty),
            "scale_additive_r2": structure["additive_r2"],
            "scale_row_energy_fraction": structure["row_energy_fraction"],
            "scale_column_energy_fraction": structure["column_energy_fraction"],
            "linear_code_change_fraction": code_change_fraction(initial_codes, modules),
            "exact_alphabet": exact_alphabet(modules, mode),
        }
    )
    return metrics


def run_seed(
    seed: int,
    layers: int,
    coefficients: list[float],
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    output = {"seed": seed, "layers": layers, "runs": {}}
    for mode in (QuantMode.BINARY, QuantMode.TERNARY):
        for coefficient in coefficients:
            key = f"{mode.value}_structure_{coefficient:.0e}"
            torch.manual_seed(seed + int(coefficient * 1.0e9) + (0 if mode == QuantMode.BINARY else 500000))
            output["runs"][key] = recover(
                fp,
                teacher,
                train,
                validation,
                mode,
                coefficient,
                recovery_steps,
                batch,
                learning_rate,
            )
    return output


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    names = runs[0]["runs"].keys()
    metrics = (
        "ce",
        "accuracy",
        "teacher_kl",
        "hidden_cosine",
        "scale_structure_penalty",
        "scale_additive_r2",
        "scale_row_energy_fraction",
        "scale_column_energy_fraction",
        "linear_code_change_fraction",
    )
    output = {}
    for name in names:
        output[name] = {}
        for metric in metrics:
            values = [float(run["runs"][name][metric]) for run in runs]
            output[name][metric + "_mean"] = statistics.mean(values)
            output[name][metric + "_pstdev"] = statistics.pstdev(values)
        output[name]["all_exact_alphabet"] = all(
            run["runs"][name]["exact_alphabet"] for run in runs
        )
    return output


def select(aggregate_by_depth: dict[str, Any], coefficients: list[float]) -> dict[str, Any]:
    output = {}
    for mode in ("binary", "ternary"):
        scores = {}
        for coefficient in coefficients:
            key = f"{mode}_structure_{coefficient:.0e}"
            scores[key] = statistics.mean(
                depth["aggregate"][key]["teacher_kl_mean"]
                for depth in aggregate_by_depth.values()
            )
        winner = min(scores, key=scores.get)
        baseline = f"{mode}_structure_{0.0:.0e}"
        output[mode] = {
            "winner": winner,
            "mean_teacher_kl_by_method": scores,
            "winner_over_free_baseline": scores[winner] / max(scores[baseline], 1e-20),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        default=[0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="scale_structure_matrix.json")
    args = parser.parse_args()

    if 0.0 not in args.coefficients:
        raise ValueError("coefficients must include the zero baseline")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_seed(
                7800 + layers * 100 + index,
                layers,
                args.coefficients,
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
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "by_depth": by_depth,
        "selection": select(by_depth, args.coefficients),
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
