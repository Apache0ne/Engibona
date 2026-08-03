#!/usr/bin/env python3
"""Tied-head-aware entry point for the public Bonsai 1.7B forensics.

The stock Qwen checkpoint serializes `lm_head.weight`, while the public Bonsai
unpacked checkpoint omits it because the output head is tied to token embeddings.
The duplicate base tensor is removed before cross-checkpoint sampling; the shared
embedding codebook remains in the analysis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


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
            core.analyze_tensor(name, item)
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
