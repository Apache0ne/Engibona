#!/usr/bin/env python3
"""Larger paired confirmation of scale-structure regularization.

The effective-range pilot found a mild binary signal near coefficient 0.1 and no
stable ternary winner. This follow-up uses more seeds, an additional depth, and
mode-specific local grids. It reports paired bootstrap intervals and exact sign
randomization probabilities versus the free-scale baseline.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "scale_structure_original", HERE / "run_scale_structure_matrix.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load scale-structure experiment")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def run_seed(
    seed: int,
    layers: int,
    binary_coefficients: list[float],
    ternary_coefficients: list[float],
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train = experiment.make_data(seed + 1, 256, 20)
    validation = experiment.make_data(seed + 2, 96, 32)
    fp = experiment.TinyOfficialQwen3VL(layers=layers, tied=True)
    experiment.train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    output = {
        "seed": seed,
        "layers": layers,
        "teacher_baseline": experiment.evaluate(teacher, teacher, validation),
        "runs": {},
    }
    grids = {
        experiment.QuantMode.BINARY: binary_coefficients,
        experiment.QuantMode.TERNARY: ternary_coefficients,
    }
    for mode, coefficients in grids.items():
        paired_seed = seed + (
            0 if mode == experiment.QuantMode.BINARY else 500000
        )
        for coefficient in coefficients:
            key = f"{mode.value}_structure_{coefficient:.6g}"
            torch.manual_seed(paired_seed)
            output["runs"][key] = experiment.recover(
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


def bootstrap_interval(
    values: list[float],
    seed: int,
    iterations: int = 20000,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(array), size=(iterations, len(array))
    )
    samples = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "low_95": float(np.quantile(samples, 0.025)),
        "high_95": float(np.quantile(samples, 0.975)),
        "standard_error": float(array.std(ddof=1) / math.sqrt(len(array))),
    }


def exact_sign_randomization(values: list[float]) -> float:
    observed = abs(sum(values))
    if len(values) <= 20:
        totals = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            totals.append(abs(sum(sign * value for sign, value in zip(signs, values))))
        return sum(total >= observed - 1e-15 for total in totals) / len(totals)
    generator = np.random.default_rng(5519)
    count = 200000
    signs = generator.choice((-1.0, 1.0), size=(count, len(values)))
    totals = np.abs(signs @ np.asarray(values))
    return float(np.mean(totals >= observed - 1e-15))


def aggregate(
    by_depth: dict[str, list[dict[str, Any]]],
    binary_coefficients: list[float],
    ternary_coefficients: list[float],
    seed: int,
) -> dict[str, Any]:
    output = {}
    for mode, coefficients in (
        ("binary", binary_coefficients),
        ("ternary", ternary_coefficients),
    ):
        baseline_key = f"{mode}_structure_{0.0:.6g}"
        output[mode] = {"methods": {}}
        all_records = [
            (depth, run)
            for depth, runs in by_depth.items()
            for run in runs
        ]
        for coefficient in coefficients:
            key = f"{mode}_structure_{coefficient:.6g}"
            kl_values = [
                float(run["runs"][key]["teacher_kl"])
                for _, run in all_records
            ]
            differences = [
                float(
                    run["runs"][key]["teacher_kl"]
                    - run["runs"][baseline_key]["teacher_kl"]
                )
                for _, run in all_records
            ]
            depth_differences = {
                depth: [
                    float(
                        run["runs"][key]["teacher_kl"]
                        - run["runs"][baseline_key]["teacher_kl"]
                    )
                    for run in runs
                ]
                for depth, runs in by_depth.items()
            }
            output[mode]["methods"][key] = {
                "coefficient": coefficient,
                "teacher_kl_mean": statistics.mean(kl_values),
                "paired_kl_difference": bootstrap_interval(
                    differences,
                    seed + int(abs(coefficient) * 100000) + (0 if mode == "binary" else 900000),
                ),
                "improved_pairs": sum(value < 0 for value in differences),
                "total_pairs": len(differences),
                "exact_sign_randomization_p_two_sided": exact_sign_randomization(differences),
                "depth_mean_differences": {
                    depth: statistics.mean(values)
                    for depth, values in depth_differences.items()
                },
                "scale_additive_r2_mean": statistics.mean(
                    float(run["runs"][key]["scale_additive_r2"])
                    for _, run in all_records
                ),
                "code_change_fraction_mean": statistics.mean(
                    float(run["runs"][key]["linear_code_change_fraction"])
                    for _, run in all_records
                ),
                "all_exact_alphabet": all(
                    run["runs"][key]["exact_alphabet"]
                    for _, run in all_records
                ),
            }
        winner = min(
            output[mode]["methods"],
            key=lambda name: output[mode]["methods"][name]["teacher_kl_mean"],
        )
        output[mode]["winner"] = winner
    return output


def aggregate_teacher_baseline(runs: list[dict[str, Any]]) -> dict[str, float]:
    output = {}
    for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
        values = [float(run["teacher_baseline"][metric]) for run in runs]
        output[metric + "_mean"] = statistics.mean(values)
        output[metric + "_pstdev"] = statistics.pstdev(values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary-coefficients",
        type=float,
        nargs="+",
        default=[0.0, 0.03, 0.1, 0.3, 1.0],
    )
    parser.add_argument(
        "--ternary-coefficients",
        type=float,
        nargs="+",
        default=[0.0, 0.3, 1.0, 3.0, 10.0],
    )
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7.0e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=14822)
    parser.add_argument("--output", default="scale_structure_confirmatory.json")
    args = parser.parse_args()

    if 0.0 not in args.binary_coefficients or 0.0 not in args.ternary_coefficients:
        raise ValueError("both coefficient grids must include zero")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    by_depth: dict[str, list[dict[str, Any]]] = {}
    for layers in args.layers:
        by_depth[str(layers)] = [
            run_seed(
                args.seed + layers * 1000 + index,
                layers,
                args.binary_coefficients,
                args.ternary_coefficients,
                args.teacher_steps,
                args.recovery_steps,
                args.batch,
                args.learning_rate,
            )
            for index in range(args.seeds)
        ]
    summary = aggregate(
        by_depth,
        args.binary_coefficients,
        args.ternary_coefficients,
        args.seed,
    )
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "summary": summary,
        "by_depth": by_depth,
        "teacher_baseline_by_depth": {
            depth: aggregate_teacher_baseline(runs)
            for depth, runs in by_depth.items()
        },
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    for mode in summary.values():
        for metrics in mode["methods"].values():
            if not metrics["all_exact_alphabet"]:
                raise SystemExit("invalid final alphabet")


if __name__ == "__main__":
    main()
