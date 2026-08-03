#!/usr/bin/env python3
"""Streaming public-weight forensics for Qwen3-1.7B and Bonsai 1.7B.

The three public checkpoints are downloaded one shard at a time. Only fixed,
deterministic g128 samples are retained, so a standard GitHub-hosted runner can
compare the released binary and ternary code geometry without keeping all three
models on disk simultaneously.

This is the most direct public test of competing transformation hypotheses. It
cannot reveal the exact private optimizer, data, learning rate, or step count.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi, hf_hub_url
from safetensors import safe_open


TARGET = re.compile(
    r"(embed_tokens|lm_head|q_proj|k_proj|v_proj|o_proj|"
    r"gate_proj|up_proj|down_proj)\.weight$"
)


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) & 0x7FFFFFFF


def layer_index(name: str) -> int:
    match = re.search(r"(?:layers|h|blocks)\.(\d+)", name)
    return int(match.group(1)) if match else -1


def module_type(name: str) -> str:
    for token in (
        "embed_tokens", "lm_head", "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ):
        if token in name:
            return token
    return "other"


def download_file(repo_id: str, filename: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision="main")
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, 5):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def repo_layout(repo_id: str, work: Path) -> tuple[list[str], dict[str, str] | None]:
    files = HfApi().list_repo_files(repo_id=repo_id, revision="main")
    tensor_files = sorted(name for name in files if name.endswith(".safetensors"))
    indexes = sorted(name for name in files if name.endswith(".safetensors.index.json"))
    if not tensor_files:
        raise RuntimeError(f"{repo_id}: no safetensors files")
    if not indexes:
        return tensor_files, None
    index_path = download_file(repo_id, indexes[0], work / indexes[0])
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"{repo_id}: invalid weight map")
    return tensor_files, {str(key): str(value) for key, value in weight_map.items()}


def fixed_indices(total: int, maximum: int, seed: int) -> torch.Tensor:
    if total <= maximum:
        return torch.arange(total, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(total, generator=generator)[:maximum].sort().values


def sample_groups(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected matrix, got {tuple(tensor.shape)}")
    if tensor.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {tensor.shape[-1]} is not divisible by {group_size}"
        )
    groups = tensor.detach().float().contiguous().reshape(-1, group_size)
    if indices.numel() and int(indices.max()) >= groups.shape[0]:
        raise ValueError("sample index exceeds group count")
    return groups.index_select(0, indices)


def collect_base(
    repo_id: str,
    work: Path,
    group_size: int,
    groups_per_tensor: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    files, weight_map = repo_layout(repo_id, work)
    selected_by_file: dict[str, list[str]] = defaultdict(list)
    if weight_map is not None:
        for name, filename in weight_map.items():
            if TARGET.search(name):
                selected_by_file[filename].append(name)
    else:
        selected_by_file[files[0]] = []

    samples: dict[str, dict[str, Any]] = {}
    for filename in files:
        if weight_map is not None and filename not in selected_by_file:
            continue
        local = download_file(repo_id, filename, work / Path(filename).name)
        with safe_open(str(local), framework="pt", device="cpu") as handle:
            keys = selected_by_file.get(filename) or [
                key for key in handle.keys() if TARGET.search(key)
            ]
            for name in sorted(keys):
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2 or tensor.shape[-1] % group_size:
                    continue
                total_groups = tensor.numel() // group_size
                indices = fixed_indices(
                    total_groups,
                    groups_per_tensor,
                    stable_seed(name, seed),
                )
                groups = sample_groups(tensor, indices, group_size)
                samples[name] = {
                    "shape": tuple(int(value) for value in tensor.shape),
                    "indices": indices,
                    "base": groups.to(torch.float16),
                }
                del tensor, groups
        local.unlink(missing_ok=True)
        gc.collect()
    if not samples:
        raise RuntimeError("no target base tensors collected")
    return samples


def collect_matching(
    repo_id: str,
    work: Path,
    base_samples: dict[str, dict[str, Any]],
    field: str,
    group_size: int,
) -> None:
    files, weight_map = repo_layout(repo_id, work)
    selected_by_file: dict[str, list[str]] = defaultdict(list)
    if weight_map is not None:
        for name in base_samples:
            filename = weight_map.get(name)
            if filename is not None:
                selected_by_file[filename].append(name)
    else:
        selected_by_file[files[0]] = list(base_samples)

    seen: set[str] = set()
    for filename in files:
        names = selected_by_file.get(filename, [])
        if not names:
            continue
        local = download_file(repo_id, filename, work / Path(filename).name)
        with safe_open(str(local), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(names):
                if name not in available:
                    continue
                tensor = handle.get_tensor(name)
                expected_shape = tuple(base_samples[name]["shape"])
                if tuple(tensor.shape) != expected_shape:
                    if tensor.ndim == 2 and tensor.shape[1:] == expected_shape[1:]:
                        rows = min(tensor.shape[0], expected_shape[0])
                        tensor = tensor[:rows]
                    else:
                        continue
                base_samples[name][field] = sample_groups(
                    tensor,
                    base_samples[name]["indices"],
                    group_size,
                ).to(torch.float16)
                seen.add(name)
                del tensor
        local.unlink(missing_ok=True)
        gc.collect()
    missing = sorted(set(base_samples) - seen)
    if missing:
        raise RuntimeError(
            f"{repo_id}: {len(missing)} selected tensors missing; first={missing[:3]}"
        )


def corr(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    mask = torch.isfinite(left) & torch.isfinite(right)
    left, right = left[mask], right[mask]
    if left.numel() < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator) <= 0:
        return float("nan")
    return float((left * right).sum() / denominator)


def nmse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = reference.float().square().mean().clamp_min(1e-20)
    return float((candidate.float() - reference.float()).square().mean() / denominator)


def naive_binary(base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = torch.where(base >= 0, 1.0, -1.0)
    scales = base.abs().mean(dim=1).clamp_min(1e-12)
    return codes, scales, codes * scales[:, None]


def naive_ternary(
    base: torch.Tensor,
    iterations: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = base.abs().mean(dim=1).clamp_min(1e-12)
    for _ in range(iterations):
        active = base.abs() > 0.5 * scales[:, None]
        scales = (
            (base.abs() * active).sum(dim=1)
            / active.sum(dim=1).clamp_min(1)
        ).clamp_min(1e-12)
    active = base.abs() > 0.5 * scales[:, None]
    codes = torch.sign(base) * active.float()
    return codes, scales, codes * scales[:, None]


def percentile_rank(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    rank = values.argsort(dim=1).argsort(dim=1).float()
    rank = rank / max(values.shape[1] - 1, 1)
    return float(rank[mask].mean())


def analyze_tensor(name: str, item: dict[str, Any]) -> dict[str, Any]:
    base = item["base"].float()
    binary = item["binary"].float()
    ternary = item["ternary"].float()

    naive_b_codes, naive_b_scales, naive_b_weights = naive_binary(base)
    binary_scales = binary.abs().median(dim=1).values.clamp_min(1e-12)
    binary_codes = torch.where(binary >= 0, 1.0, -1.0)
    binary_weights = binary_codes * binary_scales[:, None]
    binary_normalized = binary.abs() / binary_scales[:, None]
    binary_flip = binary_codes != naive_b_codes

    naive_t_codes, naive_t_scales, naive_t_weights = naive_ternary(base)
    ternary_abs = ternary.abs()
    ternary_nonzero_raw = ternary_abs > 0
    ternary_scales = (
        (ternary_abs * ternary_nonzero_raw).sum(dim=1)
        / ternary_nonzero_raw.sum(dim=1).clamp_min(1)
    ).clamp_min(1e-12)
    ternary_normalized = ternary / ternary_scales[:, None]
    ternary_codes = torch.where(
        ternary_normalized > 0.5,
        1.0,
        torch.where(ternary_normalized < -0.5, -1.0, 0.0),
    )
    ternary_weights = ternary_codes * ternary_scales[:, None]
    ternary_zero = ternary_codes == 0
    ternary_nonzero = ~ternary_zero

    binary_actual_nmse = nmse(binary_weights, base)
    binary_naive_nmse = nmse(naive_b_weights, base)
    ternary_actual_nmse = nmse(ternary_weights, base)
    ternary_naive_nmse = nmse(naive_t_weights, base)

    ternary_alphabet_distance = torch.minimum(
        ternary_normalized.abs(),
        (ternary_normalized.abs() - 1.0).abs(),
    )

    return {
        "tensor": name,
        "layer": layer_index(name),
        "module": module_type(name),
        "shape": "x".join(str(value) for value in item["shape"]),
        "groups_sampled": int(base.shape[0]),
        "weights_sampled": int(base.numel()),
        "binary_alphabet_max_error": float((binary_normalized - 1.0).abs().max()),
        "binary_sign_agreement_base": float((binary_codes == naive_b_codes).float().mean()),
        "binary_sign_flip_rate": float(binary_flip.float().mean()),
        "binary_flip_base_magnitude_percentile": percentile_rank(base.abs(), binary_flip),
        "binary_scale_corr_mean_abs_base": corr(binary_scales, naive_b_scales),
        "binary_scale_corr_rms_base": corr(
            binary_scales, base.square().mean(dim=1).sqrt()
        ),
        "binary_scale_median_ratio_mean_abs": float(
            (binary_scales / naive_b_scales).median()
        ),
        "binary_actual_nmse": binary_actual_nmse,
        "binary_naive_nmse": binary_naive_nmse,
        "binary_actual_over_naive_nmse": binary_actual_nmse / max(binary_naive_nmse, 1e-20),
        "ternary_alphabet_max_error": float(ternary_alphabet_distance.max()),
        "ternary_zero_rate": float(ternary_zero.float().mean()),
        "ternary_code_agreement_naive": float((ternary_codes == naive_t_codes).float().mean()),
        "ternary_zero_base_magnitude_percentile": percentile_rank(base.abs(), ternary_zero),
        "ternary_nonzero_sign_agreement_base": float(
            (ternary_codes[ternary_nonzero] == naive_b_codes[ternary_nonzero]).float().mean()
        ) if ternary_nonzero.any() else float("nan"),
        "ternary_scale_corr_naive": corr(ternary_scales, naive_t_scales),
        "ternary_scale_median_ratio_naive": float(
            (ternary_scales / naive_t_scales).median()
        ),
        "ternary_actual_nmse": ternary_actual_nmse,
        "ternary_naive_nmse": ternary_naive_nmse,
        "ternary_actual_over_naive_nmse": ternary_actual_nmse / max(ternary_naive_nmse, 1e-20),
        "binary_ternary_sign_agreement_nonzero": float(
            (binary_codes[ternary_nonzero] == ternary_codes[ternary_nonzero]).float().mean()
        ) if ternary_nonzero.any() else float("nan"),
        "binary_ternary_scale_corr": corr(binary_scales, ternary_scales),
    }


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame["groups_sampled"], errors="coerce")
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    identifiers = {
        "tensor", "layer", "module", "shape", "groups_sampled", "weights_sampled"
    }
    metrics = {
        column: weighted_mean(frame, column)
        for column in frame.columns
        if column not in identifiers
    }
    return {
        "tensor_count": int(len(frame)),
        "sampled_groups": int(frame["groups_sampled"].sum()),
        "sampled_weights": int(frame["weights_sampled"].sum()),
        "metrics": metrics,
    }


def interpretation(summary: dict[str, Any]) -> list[str]:
    m = summary["metrics"]
    lines = [
        "These conclusions concern released-weight geometry, not private optimizer identity."
    ]
    binary_agreement = m["binary_sign_agreement_base"]
    ternary_agreement = m["ternary_code_agreement_naive"]
    lineage = m["binary_ternary_sign_agreement_nonzero"]
    if binary_agreement >= 0.995:
        lines.append(
            "Binary signs are nearly direct sign(W); scale/global recovery may dominate code reassignment."
        )
    elif binary_agreement >= 0.95:
        lines.append(
            "Binary sign(W) is a strong initializer, but released codes contain material reassignment."
        )
    else:
        lines.append(
            "Released binary codes require substantial discrete re-optimization beyond sign(W)."
        )
    if ternary_agreement >= 0.97:
        lines.append(
            "Ternary assignments are close to ordinary least-squares magnitude thresholding."
        )
    elif ternary_agreement >= 0.85:
        lines.append(
            "Magnitude thresholding explains most ternary codes, followed by learned reassignment."
        )
    else:
        lines.append(
            "The ternary zero mask is not explained by ordinary magnitude thresholding."
        )
    if lineage >= 0.99:
        lines.append(
            "Binary and ternary signs strongly support a shared recovered lineage."
        )
    elif lineage >= 0.95:
        lines.append(
            "Binary and ternary models share strong initialization structure but diverged during recovery."
        )
    else:
        lines.append(
            "Binary and ternary code geometry is more consistent with independently recovered models."
        )
    if (
        m["binary_actual_over_naive_nmse"] > 1.05
        or m["ternary_actual_over_naive_nmse"] > 1.05
    ):
        lines.append(
            "Released weights are farther from Qwen in raw MSE than naive projection, supporting a functional rather than weight-MSE objective."
        )
    return lines


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
        samples = collect_base(
            args.base, work / "base", args.group_size, args.groups_per_tensor, args.seed
        )
        collect_matching(args.binary, work / "binary", samples, "binary", args.group_size)
        collect_matching(args.ternary, work / "ternary", samples, "ternary", args.group_size)

        rows = [analyze_tensor(name, item) for name, item in sorted(samples.items())]
        frame = pd.DataFrame(rows)
        frame.to_csv(output / "tensor_metrics.csv", index=False)
        module = (
            frame.groupby("module", dropna=False)
            .apply(
                lambda group: pd.Series({
                    column: weighted_mean(group, column)
                    for column in frame.columns
                    if column not in {"tensor", "layer", "module", "shape", "groups_sampled", "weights_sampled"}
                }),
                include_groups=False,
            )
            .reset_index()
        )
        module.to_csv(output / "module_metrics.csv", index=False)
        summary = summarize(frame)
        summary["repositories"] = {
            "base": args.base,
            "binary": args.binary,
            "ternary": args.ternary,
        }
        summary["group_size"] = args.group_size
        summary["groups_per_tensor_limit"] = args.groups_per_tensor
        summary["seed"] = args.seed
        summary["seconds"] = time.time() - started
        summary["interpretation"] = interpretation(summary)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        (output / "interpretation.txt").write_text(
            "\n".join(summary["interpretation"]) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
