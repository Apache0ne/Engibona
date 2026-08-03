#!/usr/bin/env python3
"""Test whether released codes are explainable by simple per-group thresholds.

A learned affine shift before quantization would make binary codes nearly a
monotone threshold of the original group values. A sign-preserving ternary
threshold would make zero/nonzero assignments nearly a magnitude threshold.
This experiment measures the best possible agreement of those simple families.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def best_binary_threshold(base: torch.Tensor, codes: torch.Tensor):
    order = base.argsort(dim=1)
    labels = (codes.gather(1, order) > 0).to(torch.int32)
    prefix = torch.cat(
        [torch.zeros(labels.shape[0], 1, dtype=torch.int32), labels.cumsum(dim=1)],
        dim=1,
    )
    n = labels.shape[1]
    k = torch.arange(n + 1, dtype=torch.int32)[None, :]
    positive_total = prefix[:, -1:]
    # Positive orientation: low values -> -1, high values -> +1.
    positive_correct = k + positive_total - 2 * prefix
    # Reverse orientation is recorded to detect a global sign reversal.
    reverse_correct = n - k - positive_total + 2 * prefix
    positive_best = positive_correct.max(dim=1).values.float() / n
    either_best = torch.maximum(
        positive_correct.max(dim=1).values,
        reverse_correct.max(dim=1).values,
    ).float() / n
    sorted_codes = codes.gather(1, order)
    transitions = (sorted_codes[:, 1:] != sorted_codes[:, :-1]).float().mean(dim=1)
    return positive_best, either_best, transitions


def best_ternary_sign_locked_threshold(base: torch.Tensor, codes: torch.Tensor):
    magnitude = base.abs()
    order = magnitude.argsort(dim=1)
    sorted_codes = codes.gather(1, order)
    sorted_sign = torch.where(base.gather(1, order) >= 0, 1.0, -1.0)
    n = codes.shape[1]

    zero_correct = (sorted_codes == 0).to(torch.int32)
    active_sign_correct = (sorted_codes == sorted_sign).to(torch.int32)
    zero_prefix = torch.cat(
        [torch.zeros(codes.shape[0], 1, dtype=torch.int32), zero_correct.cumsum(dim=1)],
        dim=1,
    )
    active_prefix = torch.cat(
        [torch.zeros(codes.shape[0], 1, dtype=torch.int32), active_sign_correct.cumsum(dim=1)],
        dim=1,
    )
    active_total = active_prefix[:, -1:]
    full_correct = zero_prefix + active_total - active_prefix
    full_best = full_correct.max(dim=1).values.float() / n

    active = (sorted_codes != 0).to(torch.int32)
    active_prefix_mask = torch.cat(
        [torch.zeros(codes.shape[0], 1, dtype=torch.int32), active.cumsum(dim=1)],
        dim=1,
    )
    k = torch.arange(n + 1, dtype=torch.int32)[None, :]
    active_total_mask = active_prefix_mask[:, -1:]
    mask_correct = k + active_total_mask - 2 * active_prefix_mask
    mask_best = mask_correct.max(dim=1).values.float() / n

    value_order = base.argsort(dim=1)
    value_codes = codes.gather(1, value_order)
    transitions = (value_codes[:, 1:] != value_codes[:, :-1]).float().mean(dim=1)
    return full_best, mask_best, transitions


def entropy_from_codes(codes: torch.Tensor, states: tuple[int, ...]) -> torch.Tensor:
    probabilities = []
    for state in states:
        probabilities.append((codes == state).float().mean(dim=1))
    probability = torch.stack(probabilities, dim=1).clamp_min(1e-12)
    return -(probability * probability.log2()).sum(dim=1)


def analyze(name: str, item: dict):
    base = item["base"].float()
    binary = item["binary"].float()
    ternary = item["ternary"].float()

    binary_scale = binary.abs().median(dim=1).values.clamp_min(1e-12)
    binary_codes = torch.where(binary >= 0, 1.0, -1.0)
    ternary_abs = ternary.abs()
    nonzero = ternary_abs > 0
    ternary_scale = (
        (ternary_abs * nonzero).sum(dim=1)
        / nonzero.sum(dim=1).clamp_min(1)
    ).clamp_min(1e-12)
    normalized = ternary / ternary_scale[:, None]
    ternary_codes = torch.where(
        normalized > 0.5,
        1.0,
        torch.where(normalized < -0.5, -1.0, 0.0),
    )

    binary_positive, binary_either, binary_transitions = best_binary_threshold(
        base, binary_codes
    )
    ternary_full, ternary_mask, ternary_transitions = (
        best_ternary_sign_locked_threshold(base, ternary_codes)
    )
    return {
        "tensor": name,
        "layer": core.layer_index(name),
        "module": core.module_type(name),
        "groups_sampled": int(base.shape[0]),
        "binary_direct_sign_agreement": float(
            (binary_codes == torch.where(base >= 0, 1.0, -1.0)).float().mean()
        ),
        "binary_best_positive_threshold_agreement": float(binary_positive.mean()),
        "binary_best_either_orientation_threshold_agreement": float(binary_either.mean()),
        "binary_sorted_transition_fraction": float(binary_transitions.mean()),
        "binary_code_entropy_bits": float(
            entropy_from_codes(binary_codes, (-1, 1)).mean()
        ),
        "ternary_best_sign_locked_threshold_agreement": float(ternary_full.mean()),
        "ternary_best_zero_mask_threshold_agreement": float(ternary_mask.mean()),
        "ternary_sorted_transition_fraction": float(ternary_transitions.mean()),
        "ternary_code_entropy_bits": float(
            entropy_from_codes(ternary_codes, (-1, 0, 1)).mean()
        ),
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["groups_sampled"], errors="coerce")
    mask = values.notna() & weights.notna()
    return float(np.average(values[mask], weights=weights[mask]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7441)
    parser.add_argument("--output-dir", default="code_structure_forensics")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="code-structure-") as temporary:
        work = Path(temporary)
        samples = core.collect_base(
            args.base, work / "base", args.group_size, args.groups_per_tensor, args.seed
        )
        for name in list(samples):
            if name.endswith("lm_head.weight"):
                samples.pop(name)
        core.collect_matching(args.binary, work / "binary", samples, "binary", args.group_size)
        core.collect_matching(args.ternary, work / "ternary", samples, "ternary", args.group_size)
        rows = [analyze(name, item) for name, item in sorted(samples.items())]
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tensor_code_structure.csv", index=False)
        identifiers = {"tensor", "layer", "module", "groups_sampled"}
        summary = {
            "tensor_count": int(len(frame)),
            "sampled_groups": int(frame["groups_sampled"].sum()),
            "metrics": {
                column: weighted_mean(frame, column)
                for column in frame.columns
                if column not in identifiers
            },
            "seconds": time.time() - started,
        }
        module_rows = []
        for module, group in frame.groupby("module"):
            row = {"module": module, "groups_sampled": int(group["groups_sampled"].sum())}
            for column in frame.columns:
                if column not in identifiers:
                    row[column] = weighted_mean(group, column)
            module_rows.append(row)
        pd.DataFrame(module_rows).to_csv(output / "module_code_structure.csv", index=False)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
