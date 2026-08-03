#!/usr/bin/env python3
"""Tied-head-aware streaming public Bonsai forensics.

The base checkpoint serializes `lm_head.weight`, while the public Bonsai
checkpoints tie it to token embeddings. The duplicate base tensor is removed.
This entry point also adds conditional binary/ternary code-lineage statistics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def safe_fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = int(denominator.sum())
    if value == 0:
        return float("nan")
    return float(numerator.sum().float() / value)


def analyze_tensor_extended(name: str, item: dict):
    metrics = core.analyze_tensor(name, item)
    base = item["base"].float()
    binary = item["binary"].float()
    ternary = item["ternary"].float()

    base_sign = torch.where(base >= 0, 1.0, -1.0)
    binary_codes = torch.where(binary >= 0, 1.0, -1.0)
    binary_flip = binary_codes != base_sign

    ternary_abs = ternary.abs()
    raw_nonzero = ternary_abs > 0
    ternary_scale = (
        (ternary_abs * raw_nonzero).sum(dim=1)
        / raw_nonzero.sum(dim=1).clamp_min(1)
    ).clamp_min(1e-12)
    normalized = ternary / ternary_scale[:, None]
    ternary_codes = torch.where(
        normalized > 0.5,
        1.0,
        torch.where(normalized < -0.5, -1.0, 0.0),
    )
    ternary_zero = ternary_codes == 0
    ternary_nonzero = ~ternary_zero
    intersection = binary_flip & ternary_zero
    union = binary_flip | ternary_zero

    metrics.update(
        {
            "binary_flip_given_ternary_zero": safe_fraction(
                binary_flip & ternary_zero, ternary_zero
            ),
            "binary_flip_given_ternary_nonzero": safe_fraction(
                binary_flip & ternary_nonzero, ternary_nonzero
            ),
            "ternary_zero_given_binary_flip": safe_fraction(
                ternary_zero & binary_flip, binary_flip
            ),
            "binary_base_agreement_on_ternary_zero": safe_fraction(
                (binary_codes == base_sign) & ternary_zero, ternary_zero
            ),
            "binary_base_agreement_on_ternary_nonzero": safe_fraction(
                (binary_codes == base_sign) & ternary_nonzero, ternary_nonzero
            ),
            "binary_flip_ternary_zero_jaccard": safe_fraction(
                intersection, union
            ),
            "ternary_nonzero_sign_disagreement_binary": safe_fraction(
                (binary_codes != ternary_codes) & ternary_nonzero,
                ternary_nonzero,
            ),
            "ternary_sign_flip_given_nonzero": safe_fraction(
                (ternary_codes != base_sign) & ternary_nonzero,
                ternary_nonzero,
            ),
        }
    )
    if intersection.any():
        metrics["joint_flip_zero_base_magnitude_percentile"] = core.percentile_rank(
            base.abs(), intersection
        )
    else:
        metrics["joint_flip_zero_base_magnitude_percentile"] = float("nan")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="public_bonsai_forensics")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="bonsai-forensics-") as temporary:
        work = Path(temporary)
        samples = core.collect_base(
            args.base,
            work / "base",
            args.group_size,
            args.groups_per_tensor,
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
            analyze_tensor_extended(name, item)
            for name, item in sorted(samples.items())
        ]
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tensor_metrics.csv", index=False)
        module = (
            frame.groupby("module", dropna=False)
            .apply(
                lambda group: pd.Series({
                    column: core.weighted_mean(group, column)
                    for column in frame.columns
                    if column not in {
                        "tensor", "layer", "module", "shape",
                        "groups_sampled", "weights_sampled",
                    }
                }),
                include_groups=False,
            )
            .reset_index()
        )
        module.to_csv(output / "module_metrics.csv", index=False)
        summary = core.summarize(frame)
        summary["repositories"] = {
            "base": args.base,
            "binary": args.binary,
            "ternary": args.ternary,
        }
        summary["group_size"] = args.group_size
        summary["groups_per_tensor_limit"] = args.groups_per_tensor
        summary["seed"] = args.seed
        summary["dropped_tied_duplicate_tensors"] = dropped
        summary["seconds"] = time.time() - started
        summary["interpretation"] = core.interpretation(summary)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        (output / "interpretation.txt").write_text(
            "\n".join(summary["interpretation"]) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
