#!/usr/bin/env python3
"""Replay FP32 teachers omitted by three historical matrix schemas."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = (
    ROOT
    / "experiments"
    / "official_qwen3vl_text"
    / "run_official_cpu_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("official_smoke", SMOKE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load official Qwen3-VL miniature")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


SEED_GROUPS = {
    "profiles": [9500, 9501, 9502],
    "scale_structure": [18822, 18823, 18824],
    "shared_embedding": [19000, 19001, 19002],
}


def run_teacher(seed: int, layers: int, steps: int, batch: int) -> dict:
    torch.manual_seed(seed)
    training = smoke.make_data(seed + 1, 256, 20)
    validation = smoke.make_data(seed + 2, 96, 32)
    model = smoke.TinyOfficialQwen3VL(layers=layers, tied=True)
    smoke.train_teacher(model, training, steps, batch)
    teacher = copy.deepcopy(model).eval()
    return {"seed": seed, **smoke.evaluate(teacher, teacher, validation)}


def aggregate(runs: list[dict]) -> dict:
    result = {}
    for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
        values = [float(run[metric]) for run in runs]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_pstdev"] = statistics.pstdev(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(SEED_GROUPS), required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    runs = [
        run_teacher(seed, args.layers, args.teacher_steps, args.batch)
        for seed in SEED_GROUPS[args.group]
    ]
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "purpose": "FP32 baseline replay for a historical matrix schema",
        "group": args.group,
        "arguments": vars(args),
        "runs": runs,
        "aggregate": aggregate(runs),
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
