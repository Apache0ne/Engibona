#!/usr/bin/env python3
"""Cross-scale lineage decomposition for released binary and ternary codes.

Public embeddings satisfy an almost exact shared binary-codebook-plus-mask
relation. Transformer matrices are less exact but still show roughly 89-90%
nonzero sign agreement. This experiment decomposes that relation directly at
1.7B, 4B, 8B, and the actual Qwen3.6-27B source.

For every sampled position with a nonzero ternary code, exactly one of four
released-state categories applies:

* stable_common: binary == ternary == source sign;
* shared_moved: binary == ternary != source sign;
* binary_only: binary != source sign, ternary == source sign;
* ternary_only: ternary != source sign, binary == source sign.

Ternary-zero positions are split by whether the released binary sign stayed at
or moved away from the source sign. This identifies how much of the final state
can be represented by a shared recovered sign backbone plus a ternary mask and
how much requires branch-specific sign corrections.

Weights are sampled through deterministic HTTP byte ranges. The experiment
identifies final-state algebra and cannot determine whether the private trainer
used a shared latent codebook, independent students, or another equivalent
parameterization.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "remote_forensics",
    HERE / "run_qwen36_27b_remote_forensics.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load remote safetensors forensic helpers")
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


def is_common_language_matrix(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    if not name.endswith(".weight"):
        return False
    lowered = name.lower()
    if "visual" in lowered or "vision" in lowered:
        return False
    if "norm" in lowered:
        return False
    if "embed_tokens" in name:
        return True
    return ".layers." in name and (
        ".self_attn." in name
        or ".linear_attn." in name
        or ".mlp." in name
    )


def layer_index(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def module_type(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if name == "lm_head.weight":
        return "lm_head"
    for marker, family in (
        (".linear_attn.", "linear_attn"),
        (".self_attn.", "self_attn"),
        (".mlp.", "mlp"),
    ):
        if marker in name:
            component = name.split(marker, 1)[1].rsplit(".weight", 1)[0]
            return f"{family}.{component}"
    return "other"


def family(name: str) -> str:
    if "embed_tokens" in name or name == "lm_head.weight":
        return "embedding_head"
    if ".linear_attn." in name:
        return "linear_attention"
    if ".self_attn." in name:
        return "full_attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def infer_codes_scales(weight: torch.Tensor, mode: str):
    if mode == "binary":
        scale = weight.abs().median(dim=1).values.clamp_min(1e-12)
        code = torch.where(
            weight < 0,
            -torch.ones_like(weight),
            torch.ones_like(weight),
        )
    else:
        nonzero = weight != 0
        scale = (
            (weight.abs() * nonzero).sum(dim=1)
            / nonzero.sum(dim=1).clamp_min(1)
        ).clamp_min(1e-12)
        normalized = weight / scale[:, None]
        code = torch.where(
            normalized > 0.5,
            torch.ones_like(normalized),
            torch.where(
                normalized < -0.5,
                -torch.ones_like(normalized),
                torch.zeros_like(normalized),
            ),
        )
    return code, scale


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-20))


def fraction(mask: torch.Tensor, condition: torch.Tensor | None = None) -> float:
    if condition is None:
        return float(mask.float().mean())
    if not bool(condition.any()):
        return float("nan")
    return float(mask[condition].float().mean())


def analyze_sample(size: str, item: dict[str, Any]) -> dict[str, Any]:
    base = item["base"].float()
    binary = item["binary"].float()
    ternary = item["ternary"].float()
    base_sign = torch.where(
        base < 0,
        -torch.ones_like(base),
        torch.ones_like(base),
    )
    binary_code, binary_scale = infer_codes_scales(binary, "binary")
    ternary_code, ternary_scale = infer_codes_scales(ternary, "ternary")
    ternary_zero = ternary_code == 0
    ternary_nonzero = ~ternary_zero
    agree = ternary_nonzero & (binary_code == ternary_code)
    mismatch = ternary_nonzero & (binary_code != ternary_code)

    stable_common = agree & (binary_code == base_sign)
    shared_moved = agree & (binary_code != base_sign)
    binary_only = mismatch & (binary_code != base_sign)
    ternary_only = mismatch & (ternary_code != base_sign)
    zero_binary_stable = ternary_zero & (binary_code == base_sign)
    zero_binary_moved = ternary_zero & (binary_code != base_sign)

    decomposition = (
        stable_common.to(torch.int8)
        + shared_moved.to(torch.int8)
        + binary_only.to(torch.int8)
        + ternary_only.to(torch.int8)
        + zero_binary_stable.to(torch.int8)
        + zero_binary_moved.to(torch.int8)
    )
    if not bool((decomposition == 1).all()):
        raise RuntimeError(
            f"lineage categories do not form a partition for {item['name']}"
        )

    shared_backbone_mask_code = torch.where(
        ternary_zero,
        torch.zeros_like(binary_code),
        binary_code,
    )
    scale_ratio = ternary_scale / binary_scale.clamp_min(1e-12)
    log_scale_ratio = scale_ratio.log()
    binary_ternary_weight_mask = binary * ternary_nonzero.float()

    return {
        "size": size,
        "tensor": item["name"],
        "layer": layer_index(item["name"]),
        "module": module_type(item["name"]),
        "family": family(item["name"]),
        "shape": "x".join(str(value) for value in item["shape"]),
        "groups_sampled": int(base.shape[0]),
        "weights_sampled": int(base.numel()),
        "sample_start_group": int(item["sample_start_group"]),
        "ternary_zero_fraction": fraction(ternary_zero),
        "stable_common_fraction_all": fraction(stable_common),
        "shared_moved_fraction_all": fraction(shared_moved),
        "binary_only_fraction_all": fraction(binary_only),
        "ternary_only_fraction_all": fraction(ternary_only),
        "zero_binary_stable_fraction_all": fraction(zero_binary_stable),
        "zero_binary_moved_fraction_all": fraction(zero_binary_moved),
        "stable_common_fraction_nonzero": fraction(
            stable_common, ternary_nonzero
        ),
        "shared_moved_fraction_nonzero": fraction(
            shared_moved, ternary_nonzero
        ),
        "binary_only_fraction_nonzero": fraction(
            binary_only, ternary_nonzero
        ),
        "ternary_only_fraction_nonzero": fraction(
            ternary_only, ternary_nonzero
        ),
        "shared_backbone_mask_code_agreement": fraction(
            shared_backbone_mask_code == ternary_code
        ),
        "branch_specific_sign_residual_fraction_all": fraction(mismatch),
        "branch_specific_sign_residual_fraction_nonzero": fraction(
            mismatch, ternary_nonzero
        ),
        "shared_sign_move_given_nonzero_agreement": fraction(
            shared_moved, agree
        ),
        "binary_branch_given_nonzero_mismatch": fraction(
            binary_only, mismatch
        ),
        "ternary_branch_given_nonzero_mismatch": fraction(
            ternary_only, mismatch
        ),
        "binary_source_sign_agreement": fraction(binary_code == base_sign),
        "ternary_nonzero_source_sign_agreement": fraction(
            ternary_code == base_sign, ternary_nonzero
        ),
        "binary_flip_given_ternary_zero": fraction(
            binary_code != base_sign, ternary_zero
        ),
        "binary_ternary_scale_correlation": correlation(
            binary_scale, ternary_scale
        ),
        "binary_ternary_scale_ratio_median": float(scale_ratio.median()),
        "binary_ternary_log_scale_ratio_std": float(
            log_scale_ratio.std(unbiased=False)
        ),
        "binary_mask_ternary_weight_relative_rmse": float(
            (binary_ternary_weight_mask - ternary).square().mean().sqrt()
            / ternary.square().mean().sqrt().clamp_min(1e-20)
        ),
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["weights_sampled"], errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    identifiers = {
        "size", "tensor", "layer", "module", "family", "shape",
        "groups_sampled", "weights_sampled", "sample_start_group",
    }
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {name: value for name, value in zip(keys, group_key)}
        row.update({
            "tensor_count": int(len(group)),
            "groups_sampled": int(group["groups_sampled"].sum()),
            "weights_sampled": int(group["weights_sampled"].sum()),
        })
        for column in frame.columns:
            if column in identifiers:
                continue
            numeric = pd.to_numeric(group[column], errors="coerce")
            if numeric.notna().any():
                row[column] = weighted_mean(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def run_size(
    size: str,
    repository_ids: dict[str, str],
    group_size: int,
    groups_per_tensor: int,
    workers: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    repositories = {
        label: remote.RemoteSafeTensorsRepository(repo_id)
        for label, repo_id in repository_ids.items()
    }
    for repository in repositories.values():
        repository.preload_headers()
    common_keys = set.intersection(
        *(repository.keys for repository in repositories.values())
    )
    candidates = []
    for name in sorted(common_keys):
        if not is_common_language_matrix(name):
            continue
        item = remote.common_descriptor(name, repositories, group_size)
        if item is not None:
            candidates.append(item)
    if not candidates:
        raise RuntimeError(f"{size}: no common language matrices")

    samples = []
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        future_to_name = {
            executor.submit(
                remote.read_tensor_sample,
                item,
                repositories,
                group_size,
                groups_per_tensor,
                seed,
            ): item["name"]
            for item in candidates
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_name), start=1
        ):
            name = future_to_name[future]
            try:
                samples.append(future.result())
            except Exception as error:
                failures.append({"tensor": name, "error": repr(error)})
            if completed % 100 == 0 or completed == len(candidates):
                print(
                    f"{size}: {completed}/{len(candidates)} complete; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"{size}: {len(failures)} tensor reads failed; first={failures[:3]}"
        )
    rows = [analyze_sample(size, item) for item in samples]
    frame = pd.DataFrame(rows).sort_values("tensor").reset_index(drop=True)
    metadata = {
        "size": size,
        "repositories": repository_ids,
        "candidate_tensor_count": len(candidates),
        "sampled_tensor_count": len(frame),
        "sampled_groups": int(frame["groups_sampled"].sum()),
        "sampled_weights": int(frame["weights_sampled"].sum()),
    }
    return frame, metadata


def conclusions(global_frame: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in global_frame.iterrows():
        size = row["size"]
        lines.append(
            f"{size}: shared binary-backbone plus ternary-mask code agreement is "
            f"{100*row['shared_backbone_mask_code_agreement']:.2f}%; "
            f"branch-specific nonzero sign residual is "
            f"{100*row['branch_specific_sign_residual_fraction_nonzero']:.2f}%."
        )
        lines.append(
            f"{size}: among nonzero positions where both branches agree, "
            f"{100*row['shared_sign_move_given_nonzero_agreement']:.2f}% moved together away from the source sign."
        )
        lines.append(
            f"{size}: among nonzero sign mismatches, "
            f"{100*row['binary_branch_given_nonzero_mismatch']:.2f}% are binary-only and "
            f"{100*row['ternary_branch_given_nonzero_mismatch']:.2f}% are ternary-only relative to the source sign."
        )
    lines.append(
        "High shared-backbone agreement supports a compact final-state representation with a common sign backbone, ternary mask, and sparse branch-specific sign residuals; it does not prove the private trainer explicitly shared those variables."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--output-dir", default="transformer_lineage_all_scales")
    args = parser.parse_args()
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
    frames = []
    metadata = {}
    for offset, (size, repository_ids) in enumerate(specifications.items()):
        frame, info = run_size(
            size,
            repository_ids,
            args.group_size,
            args.groups_per_tensor,
            args.workers,
            args.seed + offset * 1000,
        )
        frames.append(frame)
        metadata[size] = info
    tensor_frame = pd.concat(frames, ignore_index=True)
    global_frame = aggregate_frame(tensor_frame, ["size"])
    family_frame = aggregate_frame(tensor_frame, ["size", "family"])
    module_frame = aggregate_frame(tensor_frame, ["size", "module"])
    layer_frame = aggregate_frame(
        tensor_frame[tensor_frame["layer"] >= 0],
        ["size", "layer"],
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tensor_frame.to_csv(output / "tensor_metrics.csv", index=False)
    global_frame.to_csv(output / "global_metrics.csv", index=False)
    family_frame.to_csv(output / "family_metrics.csv", index=False)
    module_frame.to_csv(output / "module_metrics.csv", index=False)
    layer_frame.to_csv(output / "layer_metrics.csv", index=False)
    summary = {
        "method": "remote byte-range binary/ternary lineage decomposition",
        "arguments": vars(args),
        "metadata": metadata,
        "global": global_frame.to_dict(orient="records"),
        "conclusions": conclusions(global_frame),
        "seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output / "interpretation.txt").write_text(
        "\n".join(summary["conclusions"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
