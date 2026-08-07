#!/usr/bin/env python3
"""Sweep recovery duration and learning-rate budget against public code drift.

The first code-profile matrix reproduced the released layer/module ordering but
changed only one quarter to one third as many codes. This experiment tests
whether longer or stronger continuation closes that magnitude gap without
sacrificing teacher behavior.

All runs use the public layer/module learning-rate profile. Within each mode and
seed, minibatch streams are paired: resetting the same RNG means a shorter run
is an exact prefix of a longer run at the same learning rate.
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


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "code_drift_original", HERE / "run_code_drift_profile_matrix.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load code-drift profile experiment")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def budget_name(steps: int, learning_rate: float) -> str:
    return f"s{steps}_lr{learning_rate:.1e}"


def run_seed(
    seed: int,
    layers: int,
    budgets: list[tuple[int, float]],
    teacher_steps: int,
    batch: int,
    targets: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train = experiment.make_data(seed + 1, 256, 20)
    validation = experiment.make_data(seed + 2, 96, 32)
    fp = experiment.TinyOfficialQwen3VL(layers=layers, tied=True)
    experiment.train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    teacher_baseline = experiment.evaluate(teacher, teacher, validation)
    methods = [
        (experiment.QuantMode.BINARY, "hard_ste"),
        (experiment.QuantMode.BINARY, "categorical"),
        (experiment.QuantMode.TERNARY, "hard_ste"),
        (experiment.QuantMode.TERNARY, "auto"),
    ]
    output = {
        "seed": seed,
        "layers": layers,
        "teacher_baseline": {
            "ce": teacher_baseline["ce"],
            "accuracy": teacher_baseline["accuracy"],
        },
        "runs": {},
    }
    for mode, relaxation in methods:
        paired_seed = seed + (
            0 if mode == experiment.QuantMode.BINARY else 700000
        )
        for steps, learning_rate in budgets:
            name = (
                f"{mode.value}_{relaxation}_public_"
                f"{budget_name(steps, learning_rate)}"
            )
            torch.manual_seed(paired_seed)
            output["runs"][name] = experiment.recover(
                fp,
                teacher,
                train,
                validation,
                mode,
                relaxation,
                "public",
                layers,
                targets,
                steps,
                batch,
                learning_rate,
            )
            output["runs"][name]["steps"] = steps
            output["runs"][name]["learning_rate"] = learning_rate
            output["runs"][name]["nominal_budget"] = steps * learning_rate
    return output


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    names = runs[0]["runs"].keys()
    output = {}
    for name in names:
        sample = runs[0]["runs"][name]
        output[name] = {
            "mode": sample["mode"],
            "relaxation": sample["relaxation"],
            "steps": sample["steps"],
            "learning_rate": sample["learning_rate"],
            "nominal_budget": sample["nominal_budget"],
        }
        paths = {
            "teacher_kl": lambda row: row["teacher_kl"],
            "ce": lambda row: row["ce"],
            "accuracy": lambda row: row["accuracy"],
            "hidden_cosine": lambda row: row["hidden_cosine"],
            "overall_code_change": lambda row: row["code_profile"]["overall_change_fraction"],
            "layer_profile_rmse": lambda row: row["code_profile"]["layer_profile_rmse"],
            "layer_profile_correlation": lambda row: row["code_profile"]["layer_profile_correlation"],
            "module_profile_rmse": lambda row: row["code_profile"]["module_profile_rmse"],
        }
        for metric, getter in paths.items():
            values = [float(getter(run["runs"][name])) for run in runs]
            finite = [value for value in values if math.isfinite(value)]
            output[name][metric + "_mean"] = (
                statistics.mean(finite) if finite else float("nan")
            )
            output[name][metric + "_pstdev"] = (
                statistics.pstdev(finite) if finite else float("nan")
            )
        output_fidelity = [
            math.exp(-float(run["runs"][name]["teacher_kl"]))
            for run in runs
        ]
        task_likelihood_retention = [
            math.exp(
                float(run["teacher_baseline"]["ce"])
                - float(run["runs"][name]["ce"])
            )
            for run in runs
        ]
        output[name]["output_fidelity_proxy_mean"] = statistics.mean(
            output_fidelity
        )
        output[name]["output_fidelity_proxy_pstdev"] = statistics.pstdev(
            output_fidelity
        )
        output[name]["task_likelihood_retention_mean"] = statistics.mean(
            task_likelihood_retention
        )
        output[name]["task_likelihood_retention_pstdev"] = statistics.pstdev(
            task_likelihood_retention
        )
        output[name]["all_exact_alphabet"] = all(
            run["runs"][name]["exact_alphabet"] for run in runs
        )
    return output


def aggregate_teacher_baseline(runs: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for metric in ("ce", "accuracy"):
        values = [float(run["teacher_baseline"][metric]) for run in runs]
        output[metric + "_mean"] = statistics.mean(values)
        output[metric + "_pstdev"] = statistics.pstdev(values)
    return output


def target_overall(mode: str, targets: dict[str, Any]) -> float:
    values = targets["layer_code_change_fraction"][mode]
    return statistics.mean(values)


def selection(aggregate: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for mode in ("binary", "ternary"):
        target = target_overall(mode, targets)
        selected = {
            name: values
            for name, values in aggregate.items()
            if values["mode"] == mode
        }
        for values in selected.values():
            values["overall_change_target"] = target
            values["overall_change_absolute_error"] = abs(
                values["overall_code_change_mean"] - target
            )
        minimum_kl = min(
            values["teacher_kl_mean"] for values in selected.values()
        )
        behavior_tolerance = minimum_kl * 1.10
        acceptable = {
            name: values
            for name, values in selected.items()
            if values["teacher_kl_mean"] <= behavior_tolerance
        }
        closest = min(
            acceptable,
            key=lambda name: (
                acceptable[name]["overall_change_absolute_error"],
                acceptable[name]["layer_profile_rmse_mean"],
                acceptable[name]["teacher_kl_mean"],
            ),
        )
        frontier = []
        objectives = (
            "teacher_kl_mean",
            "overall_change_absolute_error",
            "layer_profile_rmse_mean",
            "module_profile_rmse_mean",
        )
        for name, values in selected.items():
            dominated = False
            for other_name, other in selected.items():
                if other_name == name:
                    continue
                if all(other[key] <= values[key] for key in objectives) and any(
                    other[key] < values[key] for key in objectives
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(name)
        output[mode] = {
            "target_overall_change": target,
            "minimum_teacher_kl": minimum_kl,
            "behavior_tolerance_10_percent": behavior_tolerance,
            "closest_geometry_within_behavior_tolerance": closest,
            "pareto_frontier": sorted(frontier),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=[
            "120:7e-4",
            "300:7e-4",
            "600:7e-4",
            "1200:7e-4",
            "300:1.4e-3",
            "600:1.4e-3",
        ],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=16700)
    parser.add_argument("--output", default="recovery_budget_matrix.json")
    args = parser.parse_args()
    budgets = []
    for item in args.budgets:
        steps_text, lr_text = item.split(":", 1)
        budgets.append((int(steps_text), float(lr_text)))
    if len(set(budgets)) != len(budgets):
        raise ValueError("budgets contain duplicates")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    targets = experiment.load_targets()
    started = time.time()
    runs = [
        run_seed(
            args.seed + index,
            args.layers,
            budgets,
            args.teacher_steps,
            args.batch,
            targets,
        )
        for index in range(args.seeds)
    ]
    summary = aggregate(runs)
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "parsed_budgets": [
            {
                "steps": steps,
                "learning_rate": learning_rate,
                "nominal_budget": steps * learning_rate,
            }
            for steps, learning_rate in budgets
        ],
        "aggregate": summary,
        "teacher_baseline": aggregate_teacher_baseline(runs),
        "selection": selection(summary, targets),
        "runs": runs,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps({
        "teacher_baseline": payload["teacher_baseline"],
        "selection": payload["selection"],
        "aggregate": summary,
    }, indent=2, allow_nan=True))
    for metrics in summary.values():
        if not metrics["all_exact_alphabet"]:
            raise SystemExit("invalid final alphabet")


if __name__ == "__main__":
    main()
