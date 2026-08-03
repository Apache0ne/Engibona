#!/usr/bin/env python3
"""Cluster-bootstrap uncertainty for public Bonsai weight forensics.

Tensors, rather than individual weights, are resampled. This avoids reporting
artificially tiny intervals from millions of highly correlated weights inside
the same matrix.
"""
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


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    weights = pd.to_numeric(frame["groups_sampled"], errors="coerce").to_numpy(float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def bootstrap(
    frame: pd.DataFrame,
    column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    clean = frame.loc[
        pd.to_numeric(frame[column], errors="coerce").notna()
        & pd.to_numeric(frame["groups_sampled"], errors="coerce").notna()
    ].reset_index(drop=True)
    if clean.empty:
        return {"estimate": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
    generator = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    count = len(clean)
    for index in range(iterations):
        sampled = clean.iloc[generator.integers(0, count, size=count)]
        values[index] = weighted_mean(sampled, column)
    return {
        "estimate": weighted_mean(clean, column),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "tensor_clusters": int(count),
    }


def stratified_module_bootstrap(
    frame: pd.DataFrame,
    column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    clean = frame.loc[
        pd.to_numeric(frame[column], errors="coerce").notna()
        & pd.to_numeric(frame["groups_sampled"], errors="coerce").notna()
    ].reset_index(drop=True)
    if clean.empty:
        return {"estimate": float("nan"), "low_95": float("nan"), "high_95": float("nan")}
    groups = [group.reset_index(drop=True) for _, group in clean.groupby("module")]
    generator = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        pieces = []
        for group in groups:
            count = len(group)
            pieces.append(group.iloc[generator.integers(0, count, size=count)])
        values[iteration] = weighted_mean(pd.concat(pieces, ignore_index=True), column)
    return {
        "estimate": weighted_mean(clean, column),
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "module_strata": int(len(groups)),
        "tensor_clusters": int(len(clean)),
    }


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
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
