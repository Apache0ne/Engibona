#!/usr/bin/env python3
"""Test whether Bonsai code reassignment is a simple hidden-channel gauge.

This experiment uses structured Cartesian samples of rows and contiguous g128
input groups from the public Qwen3-1.7B, binary Bonsai, and ternary Bonsai
checkpoints.  It measures how much of the released sign disagreement can be
removed by:

* independent output-row sign flips;
* independent input-column sign flips;
* an alternating row+column sign gauge;
* a g128-block column gauge;
* separable row/group rescaling of the released group scales.

A high row+column-adjusted agreement would support a change of internal basis
or sign gauge.  A modest gain with substantial residual disagreement supports
true discrete code re-optimization.  Final weights still cannot identify the
private optimizer, data, learning rate, or training schedule.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def structured_indices(
    rows: int,
    groups_per_row: int,
    max_rows: int,
    max_group_columns: int,
    seed: int,
) -> torch.Tensor:
    selected_rows = core.fixed_indices(rows, min(rows, max_rows), seed)
    selected_columns = core.fixed_indices(
        groups_per_row,
        min(groups_per_row, max_group_columns),
        seed ^ 0x5EED5EED,
    )
    grid = selected_rows[:, None] * groups_per_row + selected_columns[None, :]
    return grid.reshape(-1).sort().values


def collect_structured_base(
    repo_id: str,
    work: Path,
    group_size: int,
    max_rows: int,
    max_group_columns: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    files, weight_map = core.repo_layout(repo_id, work)
    selected_by_file: dict[str, list[str]] = defaultdict(list)
    if weight_map is not None:
        for name, filename in weight_map.items():
            if core.TARGET.search(name):
                selected_by_file[filename].append(name)
    else:
        selected_by_file[files[0]] = []

    samples: dict[str, dict[str, Any]] = {}
    for filename in files:
        if weight_map is not None and filename not in selected_by_file:
            continue
        local = core.download_file(repo_id, filename, work / Path(filename).name)
        with safe_open(str(local), framework="pt", device="cpu") as handle:
            keys = selected_by_file.get(filename) or [
                key for key in handle.keys() if core.TARGET.search(key)
            ]
            for name in sorted(keys):
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2 or tensor.shape[-1] % group_size:
                    continue
                rows = int(tensor.shape[0])
                groups_per_row = int(tensor.shape[1] // group_size)
                indices = structured_indices(
                    rows,
                    groups_per_row,
                    max_rows,
                    max_group_columns,
                    core.stable_seed(name, seed),
                )
                groups = core.sample_groups(tensor, indices, group_size)
                samples[name] = {
                    "shape": tuple(int(value) for value in tensor.shape),
                    "indices": indices,
                    "base": groups.to(torch.float16),
                }
                del tensor, groups
        local.unlink(missing_ok=True)
        gc.collect()
    if not samples:
        raise RuntimeError("no target tensors collected")
    return samples


def sign_of(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))


def compact(values: torch.Tensor) -> tuple[torch.Tensor, int]:
    _, inverse = torch.unique(values.to(torch.long), sorted=True, return_inverse=True)
    return inverse, int(inverse.max().item()) + 1 if inverse.numel() else 0


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = torch.zeros(size, dtype=torch.float64)
    result.scatter_add_(0, index, values.to(torch.float64))
    return result


def majority_factor(scores: torch.Tensor) -> torch.Tensor:
    return torch.where(scores < 0, -torch.ones_like(scores), torch.ones_like(scores))


def agreement(products: torch.Tensor, factor: torch.Tensor | None = None) -> float:
    adjusted = products if factor is None else products * factor
    return float((adjusted > 0).to(torch.float64).mean())


def optimize_sign_gauge(
    products: torch.Tensor,
    row_ids: torch.Tensor,
    column_ids: torch.Tensor,
    seed: int,
    iterations: int = 40,
    restarts: int = 8,
) -> dict[str, float]:
    if products.numel() == 0:
        return {
            "raw": float("nan"),
            "row_only": float("nan"),
            "column_only": float("nan"),
            "row_column": float("nan"),
            "row_flip_fraction": float("nan"),
            "column_flip_fraction": float("nan"),
        }
    products = sign_of(products.detach().cpu().reshape(-1)).to(torch.float64)
    rows, row_count = compact(row_ids.detach().cpu().reshape(-1))
    columns, column_count = compact(column_ids.detach().cpu().reshape(-1))

    row_scores = scatter_sum(products, rows, row_count)
    row_factor = majority_factor(row_scores)
    row_only = agreement(products, row_factor[rows])

    column_scores = scatter_sum(products, columns, column_count)
    column_factor = majority_factor(column_scores)
    column_only = agreement(products, column_factor[columns])

    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts: list[torch.Tensor] = [
        torch.ones(column_count, dtype=torch.float64),
        column_factor.clone(),
    ]
    while len(starts) < max(2, restarts):
        random_bits = torch.randint(
            0, 2, (column_count,), generator=generator, dtype=torch.int64
        )
        starts.append((random_bits * 2 - 1).to(torch.float64))

    best_score = -1.0
    best_rows = torch.ones(row_count, dtype=torch.float64)
    best_columns = torch.ones(column_count, dtype=torch.float64)
    for initial_columns in starts:
        current_columns = initial_columns
        current_rows = torch.ones(row_count, dtype=torch.float64)
        previous = -1.0
        for _ in range(iterations):
            current_rows = majority_factor(
                scatter_sum(products * current_columns[columns], rows, row_count)
            )
            current_columns = majority_factor(
                scatter_sum(products * current_rows[rows], columns, column_count)
            )
            score = agreement(
                products,
                current_rows[rows] * current_columns[columns],
            )
            if abs(score - previous) < 1e-12:
                break
            previous = score
        if score > best_score:
            best_score = score
            best_rows = current_rows.clone()
            best_columns = current_columns.clone()

    return {
        "raw": agreement(products),
        "row_only": row_only,
        "column_only": column_only,
        "row_column": best_score,
        "row_flip_fraction": float((best_rows < 0).to(torch.float64).mean()),
        "column_flip_fraction": float((best_columns < 0).to(torch.float64).mean()),
    }


def mean_by(values: torch.Tensor, index: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    sums = scatter_sum(values, index, size)
    counts = scatter_sum(torch.ones_like(values, dtype=torch.float64), index, size)
    means = sums / counts.clamp_min(1.0)
    return means, counts


def r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.to(torch.float64)
    prediction = prediction.to(torch.float64)
    denominator = (target - target.mean()).square().sum()
    if float(denominator) <= 1e-20:
        return float("nan")
    return float(1.0 - (target - prediction).square().sum() / denominator)


def fit_scale_separability(
    actual: torch.Tensor,
    baseline: torch.Tensor,
    row_ids: torch.Tensor,
    group_column_ids: torch.Tensor,
    seed: int,
    iterations: int = 30,
) -> dict[str, float]:
    ratio = (actual.to(torch.float64) / baseline.to(torch.float64).clamp_min(1e-20)).clamp_min(1e-20)
    y = ratio.log().reshape(-1)
    rows, row_count = compact(row_ids.reshape(-1))
    columns, column_count = compact(group_column_ids.reshape(-1))
    count = y.numel()
    if count < 16:
        return {
            "row_r2": float("nan"),
            "group_column_r2": float("nan"),
            "additive_r2": float("nan"),
            "additive_cv_r2": float("nan"),
            "cv_coverage": 0.0,
        }

    row_mean, _ = mean_by(y, rows, row_count)
    column_mean, _ = mean_by(y, columns, column_count)
    row_prediction = row_mean[rows]
    column_prediction = column_mean[columns]

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    train_size = max(8, int(round(count * 0.8)))
    train_mask = torch.zeros(count, dtype=torch.bool)
    train_mask[permutation[:train_size]] = True

    def fit(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        train_y = y[mask]
        train_rows = rows[mask]
        train_columns = columns[mask]
        row_effect = torch.zeros(row_count, dtype=torch.float64)
        column_effect = torch.zeros(column_count, dtype=torch.float64)
        intercept = float(train_y.mean())
        row_counts = torch.zeros(row_count, dtype=torch.float64)
        column_counts = torch.zeros(column_count, dtype=torch.float64)
        for _ in range(iterations):
            row_effect, row_counts = mean_by(
                train_y - intercept - column_effect[train_columns],
                train_rows,
                row_count,
            )
            column_effect, column_counts = mean_by(
                train_y - intercept - row_effect[train_rows],
                train_columns,
                column_count,
            )
            intercept = float(
                (train_y - row_effect[train_rows] - column_effect[train_columns]).mean()
            )
        return row_effect, column_effect, row_counts, column_counts, intercept

    full_mask = torch.ones(count, dtype=torch.bool)
    full_row, full_column, _, _, full_intercept = fit(full_mask)
    additive_prediction = full_intercept + full_row[rows] + full_column[columns]

    train_row, train_column, train_row_counts, train_column_counts, train_intercept = fit(train_mask)
    test_mask = ~train_mask
    covered = test_mask & (train_row_counts[rows] > 0) & (train_column_counts[columns] > 0)
    if int(covered.sum()) >= 4:
        cv_prediction = train_intercept + train_row[rows[covered]] + train_column[columns[covered]]
        cv_r2 = r2_score(y[covered], cv_prediction)
    else:
        cv_r2 = float("nan")

    return {
        "row_r2": r2_score(y, row_prediction),
        "group_column_r2": r2_score(y, column_prediction),
        "additive_r2": r2_score(y, additive_prediction),
        "additive_cv_r2": cv_r2,
        "cv_coverage": float(covered.to(torch.float64).mean()),
    }


def tensor_coordinates(item: dict[str, Any], group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, input_width = (int(value) for value in item["shape"])
    del rows
    groups_per_row = input_width // group_size
    group_indices = item["indices"].to(torch.long)
    row_group = group_indices // groups_per_row
    group_column = group_indices % groups_per_row
    offsets = torch.arange(group_size, dtype=torch.long)
    row_weight = row_group[:, None].expand(-1, group_size).reshape(-1)
    column_weight = (group_column[:, None] * group_size + offsets[None, :]).reshape(-1)
    block_weight = group_column[:, None].expand(-1, group_size).reshape(-1)
    return row_group, group_column, row_weight, column_weight, block_weight


def ternary_codes_and_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    absolute = weight.abs()
    nonzero = absolute > 0
    scales = (
        (absolute * nonzero).sum(dim=1)
        / nonzero.sum(dim=1).clamp_min(1)
    ).clamp_min(1e-12)
    normalized = weight / scales[:, None]
    codes = torch.where(
        normalized > 0.5,
        torch.ones_like(normalized),
        torch.where(normalized < -0.5, -torch.ones_like(normalized), torch.zeros_like(normalized)),
    )
    return codes, scales


def analyze_tensor(name: str, item: dict[str, Any], group_size: int, seed: int) -> dict[str, Any]:
    base = item["base"].float()
    binary = item["binary"].float()
    ternary = item["ternary"].float()
    row_group, group_column, row_weight, column_weight, block_weight = tensor_coordinates(item, group_size)

    base_sign = sign_of(base).reshape(-1)
    binary_sign = sign_of(binary).reshape(-1)
    binary_product = base_sign * binary_sign

    binary_column = optimize_sign_gauge(
        binary_product,
        row_weight,
        column_weight,
        seed=core.stable_seed(name + ":binary-column", seed),
    )
    binary_block = optimize_sign_gauge(
        binary_product,
        row_weight,
        block_weight,
        seed=core.stable_seed(name + ":binary-block", seed),
    )

    ternary_codes, ternary_scales = ternary_codes_and_scales(ternary)
    ternary_flat = ternary_codes.reshape(-1)
    ternary_nonzero = ternary_flat != 0
    ternary_product = base_sign[ternary_nonzero] * ternary_flat[ternary_nonzero]
    ternary_gauge = optimize_sign_gauge(
        ternary_product,
        row_weight[ternary_nonzero],
        column_weight[ternary_nonzero],
        seed=core.stable_seed(name + ":ternary-column", seed),
    )

    binary_ternary_product = binary_sign[ternary_nonzero] * ternary_flat[ternary_nonzero]
    lineage_gauge = optimize_sign_gauge(
        binary_ternary_product,
        row_weight[ternary_nonzero],
        column_weight[ternary_nonzero],
        seed=core.stable_seed(name + ":lineage-column", seed),
    )

    _, naive_binary_scales, _ = core.naive_binary(base)
    binary_scales = binary.abs().median(dim=1).values.clamp_min(1e-12)
    _, naive_ternary_scales, _ = core.naive_ternary(base)

    binary_scale = fit_scale_separability(
        binary_scales,
        naive_binary_scales,
        row_group,
        group_column,
        seed=core.stable_seed(name + ":binary-scale", seed),
    )
    ternary_scale = fit_scale_separability(
        ternary_scales,
        naive_ternary_scales,
        row_group,
        group_column,
        seed=core.stable_seed(name + ":ternary-scale", seed),
    )

    return {
        "tensor": name,
        "layer": core.layer_index(name),
        "module": core.module_type(name),
        "shape": "x".join(str(value) for value in item["shape"]),
        "groups_sampled": int(base.shape[0]),
        "weights_sampled": int(base.numel()),
        "observed_rows": int(torch.unique(row_group).numel()),
        "observed_group_columns": int(torch.unique(group_column).numel()),
        "binary_raw_sign_agreement": binary_column["raw"],
        "binary_row_only_agreement": binary_column["row_only"],
        "binary_column_only_agreement": binary_column["column_only"],
        "binary_row_column_agreement": binary_column["row_column"],
        "binary_row_column_gain": binary_column["row_column"] - binary_column["raw"],
        "binary_row_flip_fraction": binary_column["row_flip_fraction"],
        "binary_column_flip_fraction": binary_column["column_flip_fraction"],
        "binary_row_block_agreement": binary_block["row_column"],
        "binary_row_block_gain": binary_block["row_column"] - binary_block["raw"],
        "ternary_nonzero_raw_sign_agreement": ternary_gauge["raw"],
        "ternary_nonzero_row_column_agreement": ternary_gauge["row_column"],
        "ternary_nonzero_row_column_gain": ternary_gauge["row_column"] - ternary_gauge["raw"],
        "binary_ternary_raw_sign_agreement": lineage_gauge["raw"],
        "binary_ternary_row_column_agreement": lineage_gauge["row_column"],
        "binary_ternary_row_column_gain": lineage_gauge["row_column"] - lineage_gauge["raw"],
        "binary_scale_row_r2": binary_scale["row_r2"],
        "binary_scale_group_column_r2": binary_scale["group_column_r2"],
        "binary_scale_additive_r2": binary_scale["additive_r2"],
        "binary_scale_additive_cv_r2": binary_scale["additive_cv_r2"],
        "binary_scale_cv_coverage": binary_scale["cv_coverage"],
        "ternary_scale_row_r2": ternary_scale["row_r2"],
        "ternary_scale_group_column_r2": ternary_scale["group_column_r2"],
        "ternary_scale_additive_r2": ternary_scale["additive_r2"],
        "ternary_scale_additive_cv_r2": ternary_scale["additive_cv_r2"],
        "ternary_scale_cv_coverage": ternary_scale["cv_coverage"],
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["groups_sampled"], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    identifiers = {
        "tensor", "layer", "module", "shape", "groups_sampled",
        "weights_sampled", "observed_rows", "observed_group_columns",
    }
    metrics = {
        column: weighted_mean(frame, column)
        for column in frame.columns
        if column not in identifiers
    }
    binary_adjusted = metrics["binary_row_column_agreement"]
    binary_gain = metrics["binary_row_column_gain"]
    scale_cv = metrics["binary_scale_additive_cv_r2"]
    conclusions = [
        "These results test simple sign-gauge and separable-scale explanations; they do not identify the private optimizer."
    ]
    if binary_adjusted >= 0.95:
        conclusions.append(
            "Most binary sign reassignment is explainable by an output-row/input-column sign gauge."
        )
    elif binary_gain >= 0.10:
        conclusions.append(
            "A row/column sign gauge explains a material minority of binary changes, but substantial discrete reassignment remains."
        )
    else:
        conclusions.append(
            "Row/column sign gauges explain little of the binary reassignment; broad code re-optimization is required."
        )
    if math.isfinite(scale_cv) and scale_cv >= 0.80:
        conclusions.append(
            "Released binary scale changes are largely separable into row and input-group factors."
        )
    elif math.isfinite(scale_cv) and scale_cv >= 0.30:
        conclusions.append(
            "Separable row/group scaling is present but does not fully explain released binary scales."
        )
    else:
        conclusions.append(
            "Released binary scale changes are not well explained by a separable row/group rescaling."
        )
    if metrics["binary_ternary_row_column_agreement"] >= 0.98:
        conclusions.append(
            "Binary and ternary signs become nearly identical under a shared-form row/column gauge."
        )
    else:
        conclusions.append(
            "Binary and ternary checkpoints retain genuine sign differences after row/column gauge optimization."
        )
    return {
        "tensor_count": int(len(frame)),
        "sampled_groups": int(frame["groups_sampled"].sum()),
        "sampled_weights": int(frame["weights_sampled"].sum()),
        "metrics": metrics,
        "conclusions": conclusions,
    }


def module_summary(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "tensor", "layer", "module", "shape", "groups_sampled",
        "weights_sampled", "observed_rows", "observed_group_columns",
    }
    rows = []
    for module, group in frame.groupby("module", dropna=False):
        row: dict[str, Any] = {
            "module": module,
            "tensor_count": int(len(group)),
            "groups_sampled": int(group["groups_sampled"].sum()),
        }
        for column in frame.columns:
            if column not in identifiers:
                row[column] = weighted_mean(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--rows-per-tensor", type=int, default=128)
    parser.add_argument("--group-columns-per-tensor", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="gauge_forensics")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="bonsai-gauge-") as temporary:
        work = Path(temporary)
        samples = collect_structured_base(
            args.base,
            work / "base",
            args.group_size,
            args.rows_per_tensor,
            args.group_columns_per_tensor,
            args.seed,
        )
        dropped = sorted(name for name in samples if name.endswith("lm_head.weight"))
        for name in dropped:
            samples.pop(name)
        core.collect_matching(
            args.binary, work / "binary", samples, "binary", args.group_size
        )
        core.collect_matching(
            args.ternary, work / "ternary", samples, "ternary", args.group_size
        )

        rows = [
            analyze_tensor(name, item, args.group_size, args.seed)
            for name, item in sorted(samples.items())
        ]
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tensor_metrics.csv", index=False)
        module_summary(frame).to_csv(output / "module_metrics.csv", index=False)
        summary = summarize(frame)
        summary.update(
            {
                "repositories": {
                    "base": args.base,
                    "binary": args.binary,
                    "ternary": args.ternary,
                },
                "group_size": args.group_size,
                "rows_per_tensor_limit": args.rows_per_tensor,
                "group_columns_per_tensor_limit": args.group_columns_per_tensor,
                "seed": args.seed,
                "dropped_tied_duplicate_tensors": dropped,
                "seconds": time.time() - started,
            }
        )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        (output / "interpretation.txt").write_text(
            "\n".join(summary["conclusions"]) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
