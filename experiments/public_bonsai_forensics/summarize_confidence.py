#!/usr/bin/env python3
"""Tensor-cluster bootstrap uncertainty for public Bonsai forensics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "binary_sign_agreement_base",
    "binary_scale_corr_mean_abs_base",
    "binary_actual_over_naive_nmse",
    "ternary_zero_rate",
    "ternary_code_agreement_naive",
    "ternary_zero_base_magnitude_percentile",
    "ternary_scale_corr_naive",
    "ternary_actual_over_naive_nmse",
    "binary_ternary_sign_agreement_nonzero",
    "binary_ternary_scale_corr",
]


def arrays(frame: pd.DataFrame, column: str):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    weights = pd.to_numeric(frame["groups_sampled"], errors="coerce").to_numpy(np.float64)
    modules = frame["module"].astype(str).to_numpy()
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[mask], weights[mask], modules[mask]


def interval(samples: np.ndarray, estimate: float, **metadata) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "low_95": float(np.quantile(samples, 0.025)),
        "high_95": float(np.quantile(samples, 0.975)),
        "bootstrap_standard_error": float(samples.std(ddof=1)),
        **metadata,
    }


def bootstrap(frame: pd.DataFrame, column: str, iterations: int, seed: int):
    values, weights, _ = arrays(frame, column)
    if values.size == 0:
        return {"estimate": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
    generator = np.random.default_rng(seed)
    count = values.size
    indices = generator.integers(0, count, size=(iterations, count))
    selected_weights = weights[indices]
    samples = (values[indices] * selected_weights).sum(axis=1) / selected_weights.sum(axis=1)
    estimate = np.average(values, weights=weights)
    return interval(samples, estimate, tensor_clusters=int(count))


def stratified_module_bootstrap(
    frame: pd.DataFrame,
    column: str,
    iterations: int,
    seed: int,
):
    values, weights, modules = arrays(frame, column)
    if values.size == 0:
        return {"estimate": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
    generator = np.random.default_rng(seed)
    numerator = np.zeros(iterations, dtype=np.float64)
    denominator = np.zeros(iterations, dtype=np.float64)
    unique_modules = np.unique(modules)
    for module in unique_modules:
        mask = modules == module
        module_values = values[mask]
        module_weights = weights[mask]
        count = module_values.size
        indices = generator.integers(0, count, size=(iterations, count))
        selected_weights = module_weights[indices]
        numerator += (module_values[indices] * selected_weights).sum(axis=1)
        denominator += selected_weights.sum(axis=1)
    samples = numerator / denominator
    estimate = np.average(values, weights=weights)
    return interval(
        samples,
        estimate,
        module_strata=int(unique_modules.size),
        tensor_clusters=int(values.size),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="public_bonsai_forensics/tensor_metrics.csv")
    parser.add_argument("--output", default="public_bonsai_forensics/confidence_intervals.json")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    metrics = [metric for metric in DEFAULT_METRICS if metric in frame.columns]
    result = {
        "method": "tensor-cluster bootstrap with module-stratified sensitivity check",
        "iterations": args.iterations,
        "seed": args.seed,
        "metrics": {},
    }
    for offset, metric in enumerate(metrics):
        result["metrics"][metric] = {
            "tensor_cluster": bootstrap(frame, metric, args.iterations, args.seed + offset),
            "module_stratified": stratified_module_bootstrap(
                frame, metric, args.iterations, args.seed + 1000 + offset
            ),
        }
    Path(args.output).write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
