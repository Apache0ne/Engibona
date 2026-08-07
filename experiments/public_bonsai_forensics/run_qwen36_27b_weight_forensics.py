#!/usr/bin/env python3
"""Direct public-weight forensics for Qwen3.6-27B Bonsai checkpoints.

Qwen3.6-27B is the actual source checkpoint named in PrismML's public notice.
Unlike Qwen3-1.7B/4B/8B, it is a hybrid architecture containing full-attention
and linear-attention language blocks plus a separate vision tower. This script
selects every two-dimensional language-model weight whose input width supports
contiguous g128 grouping, while excluding the vision tower and high-precision
one-dimensional state.

It streams one shard at a time and samples deterministic groups, then applies
the same binary/ternary geometry tests used at smaller scales. The experiment
can identify released-state structure but not the private optimizer, dataset,
schedule, or intermediate trajectory.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
COMMON_SPEC = importlib.util.spec_from_file_location(
    "streaming_common_rows", HERE / "streaming_common_rows.py"
)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise ImportError("cannot load common-row streaming core")
core = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(core)


def is_language_matrix_name(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    if not name.startswith("model.language_model."):
        return False
    if not name.endswith(".weight"):
        return False
    if "norm" in name.lower():
        return False
    return True


def module_type(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if name == "lm_head.weight":
        return "lm_head"
    for family in ("linear_attn", "self_attn", "mlp"):
        marker = f".{family}."
        if marker in name:
            suffix = name.split(marker, 1)[1]
            component = suffix.rsplit(".weight", 1)[0]
            return f"{family}.{component}"
    return name.rsplit(".weight", 1)[0].split(".")[-1]


def layer_family(name: str) -> str:
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if ".mlp." in name:
        return "mlp"
    if "embed_tokens" in name or name == "lm_head.weight":
        return "embedding_head"
    return "other"


def collect_base(
    repo_id: str,
    work: Path,
    group_size: int,
    groups_per_tensor: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    files, weight_map = core.repo_layout(repo_id, work)
    selected_by_file: dict[str, list[str]] = defaultdict(list)
    if weight_map is not None:
        for name, filename in weight_map.items():
            if is_language_matrix_name(name):
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
                key for key in handle.keys() if is_language_matrix_name(key)
            ]
            for name in sorted(keys):
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2 or tensor.shape[-1] % group_size:
                    continue
                total_groups = tensor.numel() // group_size
                indices = core.fixed_indices(
                    total_groups,
                    groups_per_tensor,
                    core.stable_seed(name, seed),
                )
                groups = core.sample_groups(tensor, indices, group_size)
                samples[name] = {
                    "shape": tuple(int(value) for value in tensor.shape),
                    "indices": indices,
                    "base": groups.to(core.torch.float16),
                    "family": layer_family(name),
                }
                del tensor, groups
        local.unlink(missing_ok=True)
        gc.collect()
    if not samples:
        raise RuntimeError("no Qwen3.6 language matrices collected")
    return samples


def family_summary(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "tensor", "layer", "module", "family", "shape",
        "groups_sampled", "weights_sampled",
    }
    rows = []
    for family, group in frame.groupby("family", dropna=False):
        row: dict[str, Any] = {
            "family": family,
            "tensor_count": int(len(group)),
            "groups_sampled": int(group["groups_sampled"].sum()),
            "weights_sampled": int(group["weights_sampled"].sum()),
        }
        for column in frame.columns:
            if column not in identifiers:
                row[column] = core.weighted_mean(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def layer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["layer"] >= 0]
    rows = []
    metrics = [
        "binary_sign_agreement_base",
        "binary_actual_over_naive_nmse",
        "ternary_zero_rate",
        "ternary_code_agreement_naive",
        "ternary_actual_over_naive_nmse",
        "binary_ternary_sign_agreement_nonzero",
        "binary_ternary_scale_corr",
    ]
    for layer, group in selected.groupby("layer"):
        row = {
            "layer": int(layer),
            "tensor_count": int(len(group)),
            "groups_sampled": int(group["groups_sampled"].sum()),
            "has_linear_attention": bool(
                (group["family"] == "linear_attention").any()
            ),
            "has_full_attention": bool(
                (group["family"] == "full_attention").any()
            ),
        }
        for metric in metrics:
            row[metric] = core.weighted_mean(group, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation(summary: dict[str, Any], families: pd.DataFrame) -> list[str]:
    metrics = summary["metrics"]
    lines = [
        "These conclusions describe released Qwen3.6-27B weight geometry and do not identify PrismML's private optimizer or data."
    ]
    if metrics["binary_sign_agreement_base"] < 0.9:
        lines.append(
            "The actual 27B binary checkpoint contains broad sign reassignment beyond sign(W)."
        )
    if metrics["ternary_code_agreement_naive"] < 0.85:
        lines.append(
            "The actual 27B ternary checkpoint is not explained by ordinary magnitude thresholding."
        )
    if (
        metrics["binary_actual_over_naive_nmse"] > 1.5
        and metrics["ternary_actual_over_naive_nmse"] > 1.5
    ):
        lines.append(
            "Both released alphabets sacrifice raw weight MSE relative to naive projection, supporting a functional recovery objective."
        )
    for family in ("linear_attention", "full_attention", "mlp"):
        selected = families[families["family"] == family]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"{family}: binary sign agreement={row['binary_sign_agreement_base']:.4f}, ternary naive-code agreement={row['ternary_code_agreement_naive']:.4f}."
            )
    lines.append(
        "The family split determines whether linear-attention matrices follow the same recovery fingerprint as full attention and MLP matrices."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-27B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-27B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="public_bonsai_forensics_27b")
    args = parser.parse_args()

    # analyze_tensor resolves module names dynamically through this global.
    core.module_type = module_type
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="bonsai-27b-") as temporary:
        work = Path(temporary)
        samples = collect_base(
            args.base,
            work / "base",
            args.group_size,
            args.groups_per_tensor,
            args.seed,
        )
        core.collect_matching(
            args.binary,
            work / "binary",
            samples,
            "binary",
            args.group_size,
        )
        core.collect_matching(
            args.ternary,
            work / "ternary",
            samples,
            "ternary",
            args.group_size,
        )

        rows = []
        for name, item in sorted(samples.items()):
            row = core.analyze_tensor(name, item)
            row["family"] = item["family"]
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tensor_metrics.csv", index=False)
        module = (
            frame.groupby("module", dropna=False)
            .apply(
                lambda group: pd.Series({
                    column: core.weighted_mean(group, column)
                    for column in frame.columns
                    if column not in {
                        "tensor", "layer", "module", "family", "shape",
                        "groups_sampled", "weights_sampled",
                    }
                }),
                include_groups=False,
            )
            .reset_index()
        )
        module.to_csv(output / "module_metrics.csv", index=False)
        families = family_summary(frame)
        families.to_csv(output / "family_metrics.csv", index=False)
        layers = layer_summary(frame)
        layers.to_csv(output / "layer_metrics.csv", index=False)

        summary = core.summarize(frame)
        summary.update(
            {
                "repositories": {
                    "base": args.base,
                    "binary": args.binary,
                    "ternary": args.ternary,
                },
                "architecture": "Qwen3.6 hybrid language model; vision tower excluded",
                "group_size": args.group_size,
                "groups_per_tensor_limit": args.groups_per_tensor,
                "seed": args.seed,
                "language_matrix_tensor_count": int(len(frame)),
                "layer_count_observed": int((layers["layer"].max() + 1) if len(layers) else 0),
                "family_tensor_counts": frame.groupby("family").size().to_dict(),
                "seconds": time.time() - started,
            }
        )
        summary["interpretation"] = interpretation(summary, families)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        (output / "interpretation.txt").write_text(
            "\n".join(summary["interpretation"]) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
