#!/usr/bin/env python3
"""Run shared embedding-codebook forensics at 1.7B, 4B, 8B, and 27B."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "embedding_lineage_original",
    HERE / "run_embedding_lineage_forensics.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load embedding-lineage forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def conclusions(results):
    lines = []
    for size, result in results.items():
        exact = result["exact_binary_codebook_plus_mask_fraction"]
        scale_correlation = result["binary_ternary_scale_corr"]
        nonzero_sign = result["binary_ternary_nonzero_sign_agreement"]
        lines.append(
            f"{size}: shared-codebook mask exact fraction={exact:.6f}, "
            f"binary/ternary scale correlation={scale_correlation:.8f}, "
            f"nonzero sign agreement={nonzero_sign:.8f}."
        )
        lines.append(
            f"{size}: binary sign agreement with source Qwen={result['binary_sign_agreement_base']:.6f}; "
            f"{100*result['ternary_zero_given_binary_flip']:.2f}% of source-relative binary flips occur at ternary-zero positions."
        )
    binary_agreements = {
        size: result["binary_sign_agreement_base"]
        for size, result in results.items()
    }
    lines.append(
        "The released binary and ternary embeddings share one sign/scale state plus a ternary zero mask across every tested scale when exact fraction and scale/sign metrics remain near one."
    )
    lines.append(
        "Source-Qwen sign retention is a scale/checkpoint property rather than the shared representation itself: "
        + ", ".join(
            f"{size}={100*value:.2f}%" for size, value in binary_agreements.items()
        )
        + "."
    )
    lines.append(
        "Final weights cannot determine whether the common binary codebook was optimized first, inferred from ternary nonzeros, or jointly recovered with the mask."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-count", type=int, default=24)
    parser.add_argument("--tail-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-dir", default="embedding_lineage_all_scales")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    specifications = {
        "1.7B": {
            "base": "Qwen/Qwen3-1.7B",
            "binary": "prism-ml/Bonsai-1.7B-unpacked",
            "ternary": "prism-ml/Ternary-Bonsai-1.7B-unpacked",
        },
        "4B": {
            "base": "Qwen/Qwen3-4B",
            "binary": "prism-ml/Bonsai-4B-unpacked",
            "ternary": "prism-ml/Ternary-Bonsai-4B-unpacked",
        },
        "8B": {
            "base": "Qwen/Qwen3-8B",
            "binary": "prism-ml/Bonsai-8B-unpacked",
            "ternary": "prism-ml/Ternary-Bonsai-8B-unpacked",
        },
        "27B": {
            "base": "Qwen/Qwen3.6-27B",
            "binary": "prism-ml/Bonsai-27B-unpacked",
            "ternary": "prism-ml/Ternary-Bonsai-27B-unpacked",
        },
    }

    results = {}
    bin_frames = []
    for offset, (size, repositories) in enumerate(specifications.items()):
        result, bins = core.run_size(
            size,
            repositories,
            args.group_size,
            args.block_size,
            args.block_count,
            args.tail_rows,
            args.seed + offset * 1000,
        )
        results[size] = result
        bin_frames.append(bins)

    summary = {
        "method": "remote safetensors byte-range cross-scale embedding forensics",
        "arguments": vars(args),
        "results": results,
        "conclusions": conclusions(results),
        "seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.concat(bin_frames, ignore_index=True).to_csv(
        output / "row_bin_metrics.csv", index=False
    )
    (output / "interpretation.txt").write_text(
        "\n".join(summary["conclusions"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
