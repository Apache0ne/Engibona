#!/usr/bin/env python3
"""Cross-scale embedding lineage for public Bonsai checkpoints.

The 1.7B checkpoint suggested an almost direct binary embedding projection, while
4B sampling showed material binary sign movement concentrated at ternary-zero
positions. This experiment tests a stronger shared-codebook hypothesis:

    binary embedding:  W_b = s * b
    ternary embedding: W_t = s * b * m,  m in {0,1}

It reads deterministic contiguous row blocks directly from remote safetensors
using HTTP byte ranges, avoiding full multi-gigabyte model downloads. It measures
exact shared scales, exact nonzero equality, conditional sign changes, row-index
localization, and the residual error of the binary-codebook-plus-mask model at
1.7B and 4B.

Final checkpoints cannot identify whether binary or ternary was optimized first;
this experiment tests only the released-state relation.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi, hf_hub_url


DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}


def request_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=(30, 300),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(
                    f"range request returned {response.status_code}, expected 206"
                )
            expected = end - start + 1
            if len(response.content) != expected:
                raise RuntimeError(
                    f"range length {len(response.content)} != {expected}"
                )
            return response.content
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def remote_safetensors_header(repo_id: str, filename: str) -> tuple[str, int, dict[str, Any]]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision="main")
    first = request_range(url, 0, 7)
    header_length = struct.unpack("<Q", first)[0]
    if header_length <= 0 or header_length > 100_000_000:
        raise RuntimeError(f"invalid safetensors header length: {header_length}")
    raw = request_range(url, 8, 8 + header_length - 1)
    header = json.loads(raw.decode("utf-8").rstrip(" \t\r\n\x00"))
    return url, 8 + header_length, header


def repository_embedding(repo_id: str) -> tuple[str, str, tuple[int, int], str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, revision="main")
    indexes = sorted(
        name for name in files if name.endswith(".safetensors.index.json")
    )
    tensor_files = sorted(name for name in files if name.endswith(".safetensors"))
    if not tensor_files:
        raise RuntimeError(f"{repo_id}: no safetensors files")

    weight_map = None
    if indexes:
        response = requests.get(
            hf_hub_url(repo_id=repo_id, filename=indexes[0], revision="main"),
            timeout=(30, 120),
        )
        response.raise_for_status()
        weight_map = response.json()["weight_map"]
        candidates = sorted(
            key for key in weight_map if key.endswith("embed_tokens.weight")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"{repo_id}: embedding candidates {candidates}")
        key = candidates[0]
        filename = weight_map[key]
    else:
        filename = tensor_files[0]
        _, _, header = remote_safetensors_header(repo_id, filename)
        candidates = sorted(
            key
            for key in header
            if key != "__metadata__" and key.endswith("embed_tokens.weight")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"{repo_id}: embedding candidates {candidates}")
        key = candidates[0]

    _, _, header = remote_safetensors_header(repo_id, filename)
    item = header[key]
    shape = tuple(int(value) for value in item["shape"])
    if len(shape) != 2:
        raise RuntimeError(f"{repo_id}: embedding is not a matrix: {shape}")
    dtype = str(item["dtype"])
    if dtype not in DTYPES:
        raise RuntimeError(f"{repo_id}: unsupported embedding dtype {dtype}")
    return filename, key, shape, dtype


def deterministic_blocks(
    rows: int,
    block_size: int,
    block_count: int,
    tail_rows: int,
    seed: int,
) -> list[tuple[int, int]]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    block_size = min(block_size, rows)
    generator = np.random.default_rng(seed)
    starts: list[int] = []
    edges = np.linspace(0, max(rows - block_size, 0), block_count + 1)
    for index in range(block_count):
        low = int(math.floor(edges[index]))
        high = int(math.floor(edges[index + 1]))
        if high < low:
            high = low
        starts.append(int(generator.integers(low, high + 1)))
    if tail_rows > 0:
        starts.append(max(0, rows - min(tail_rows, rows)))

    intervals = sorted((start, min(rows, start + block_size)) for start in starts)
    if tail_rows > block_size:
        intervals[-1] = (max(0, rows - tail_rows), rows)

    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def read_embedding_rows(
    repo_id: str,
    filename: str,
    key: str,
    expected_shape: tuple[int, int],
    intervals: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    url, data_start, header = remote_safetensors_header(repo_id, filename)
    item = header[key]
    shape = tuple(int(value) for value in item["shape"])
    if shape != expected_shape:
        raise RuntimeError(f"{repo_id}: shape changed {shape} != {expected_shape}")
    dtype_name = str(item["dtype"])
    dtype, item_size = DTYPES[dtype_name]
    offsets = [int(value) for value in item["data_offsets"]]
    rows, width = shape
    row_bytes = width * item_size
    expected_bytes = rows * row_bytes
    if offsets[1] - offsets[0] != expected_bytes:
        raise RuntimeError("tensor byte length does not match shape")

    row_ids = []
    parts = []
    for start_row, end_row in intervals:
        if start_row < 0 or end_row > rows or end_row <= start_row:
            raise ValueError(f"invalid row interval {(start_row, end_row)}")
        byte_start = data_start + offsets[0] + start_row * row_bytes
        byte_end = data_start + offsets[0] + end_row * row_bytes - 1
        raw = bytearray(request_range(url, byte_start, byte_end))
        tensor = torch.frombuffer(raw, dtype=dtype).clone().reshape(
            end_row - start_row, width
        )
        parts.append(tensor.to(torch.float16))
        row_ids.append(torch.arange(start_row, end_row, dtype=torch.long))
    return (
        torch.cat(parts, dim=0),
        torch.cat(row_ids, dim=0),
        {
            "repo": repo_id,
            "filename": filename,
            "key": key,
            "shape": list(shape),
            "dtype": dtype_name,
            "intervals": [list(interval) for interval in intervals],
        },
    )


def naive_ternary(base_groups: torch.Tensor, iterations: int = 16):
    scales = base_groups.abs().mean(dim=-1).clamp_min(1e-12)
    for _ in range(iterations):
        active = base_groups.abs() > 0.5 * scales[..., None]
        scales = (
            (base_groups.abs() * active).sum(dim=-1)
            / active.sum(dim=-1).clamp_min(1)
        ).clamp_min(1e-12)
    active = base_groups.abs() > 0.5 * scales[..., None]
    return torch.sign(base_groups) * active.float(), scales


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-20))


def safe_fraction(mask: torch.Tensor, condition: torch.Tensor | None = None) -> float:
    if condition is None:
        return float(mask.float().mean())
    if not bool(condition.any()):
        return float("nan")
    return float(mask[condition].float().mean())


def row_quantiles(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    result = torch.quantile(values.float(), quantiles)
    names = ["min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"]
    return {name: float(value) for name, value in zip(names, result)}


def analyze(
    size: str,
    base: torch.Tensor,
    binary: torch.Tensor,
    ternary: torch.Tensor,
    row_ids: torch.Tensor,
    common_rows: int,
    group_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if base.shape != binary.shape or base.shape != ternary.shape:
        raise ValueError("sampled embedding shapes do not match")
    if base.shape[1] % group_size:
        raise ValueError("embedding width is not divisible by group size")
    rows, width = base.shape
    groups_per_row = width // group_size
    base_groups = base.float().reshape(rows, groups_per_row, group_size)
    binary_groups = binary.float().reshape(rows, groups_per_row, group_size)
    ternary_groups = ternary.float().reshape(rows, groups_per_row, group_size)

    base_sign = torch.where(base_groups < 0, -torch.ones_like(base_groups), torch.ones_like(base_groups))
    binary_sign = torch.where(binary_groups < 0, -torch.ones_like(binary_groups), torch.ones_like(binary_groups))
    binary_scale = binary_groups.abs().median(dim=-1).values.clamp_min(1e-12)
    binary_code = binary_sign

    ternary_nonzero_raw = ternary_groups != 0
    ternary_scale = (
        (ternary_groups.abs() * ternary_nonzero_raw).sum(dim=-1)
        / ternary_nonzero_raw.sum(dim=-1).clamp_min(1)
    ).clamp_min(1e-12)
    ternary_normalized = ternary_groups / ternary_scale[..., None]
    ternary_code = torch.where(
        ternary_normalized > 0.5,
        torch.ones_like(ternary_normalized),
        torch.where(
            ternary_normalized < -0.5,
            -torch.ones_like(ternary_normalized),
            torch.zeros_like(ternary_normalized),
        ),
    )
    ternary_zero = ternary_code == 0
    ternary_nonzero = ~ternary_zero
    naive_t_code, naive_t_scale = naive_ternary(base_groups)
    naive_b_scale = base_groups.abs().mean(dim=-1).clamp_min(1e-12)

    binary_flip = binary_code != base_sign
    ternary_sign_flip = ternary_nonzero & (ternary_code != base_sign)
    binary_equals_ternary_nonzero = (
        binary_groups == ternary_groups
    ) & ternary_nonzero
    exact_mask_model = (ternary_groups == 0) | (ternary_groups == binary_groups)
    inferred_mask_reconstruction = binary_groups * ternary_nonzero.float()

    row_binary_agreement = (binary_code == base_sign).float().mean(dim=(1, 2))
    row_ternary_naive_agreement = (ternary_code == naive_t_code).float().mean(dim=(1, 2))
    row_mask_exact = exact_mask_model.float().mean(dim=(1, 2))
    row_zero_rate = ternary_zero.float().mean(dim=(1, 2))
    normalized_row_index = row_ids.float() / max(common_rows - 1, 1)

    bin_index = torch.clamp(
        (normalized_row_index * 10).floor().to(torch.long), max=9
    )
    records = []
    for bin_value in range(10):
        selected = bin_index == bin_value
        if not bool(selected.any()):
            continue
        records.append(
            {
                "size": size,
                "row_bin": bin_value,
                "row_start_fraction": bin_value / 10,
                "row_end_fraction": (bin_value + 1) / 10,
                "sampled_rows": int(selected.sum()),
                "binary_sign_agreement": float(row_binary_agreement[selected].mean()),
                "ternary_naive_code_agreement": float(row_ternary_naive_agreement[selected].mean()),
                "ternary_zero_rate": float(row_zero_rate[selected].mean()),
                "exact_binary_mask_model": float(row_mask_exact[selected].mean()),
            }
        )
    tail = row_ids >= max(0, common_rows - 1024)
    if bool(tail.any()):
        records.append(
            {
                "size": size,
                "row_bin": 10,
                "row_start_fraction": max(0, common_rows - 1024) / common_rows,
                "row_end_fraction": 1.0,
                "sampled_rows": int(tail.sum()),
                "binary_sign_agreement": float(row_binary_agreement[tail].mean()),
                "ternary_naive_code_agreement": float(row_ternary_naive_agreement[tail].mean()),
                "ternary_zero_rate": float(row_zero_rate[tail].mean()),
                "exact_binary_mask_model": float(row_mask_exact[tail].mean()),
            }
        )

    scale_difference = (binary_scale - ternary_scale).abs()
    relative_scale_difference = scale_difference / binary_scale.clamp_min(1e-12)
    result = {
        "size": size,
        "common_rows": common_rows,
        "sampled_rows": int(rows),
        "sampled_weights": int(base.numel()),
        "embedding_width": int(width),
        "groups_per_row": int(groups_per_row),
        "binary_sign_agreement_base": safe_fraction(binary_code == base_sign),
        "binary_sign_flip_rate": safe_fraction(binary_flip),
        "binary_scale_corr_naive": correlation(binary_scale, naive_b_scale),
        "binary_scale_median_ratio_naive": float((binary_scale / naive_b_scale).median()),
        "ternary_zero_rate": safe_fraction(ternary_zero),
        "ternary_code_agreement_naive": safe_fraction(ternary_code == naive_t_code),
        "ternary_nonzero_sign_agreement_base": safe_fraction(
            ternary_code == base_sign, ternary_nonzero
        ),
        "ternary_scale_corr_naive": correlation(ternary_scale, naive_t_scale),
        "binary_ternary_nonzero_sign_agreement": safe_fraction(
            binary_code == ternary_code, ternary_nonzero
        ),
        "binary_flip_given_ternary_zero": safe_fraction(binary_flip, ternary_zero),
        "binary_flip_given_ternary_nonzero": safe_fraction(binary_flip, ternary_nonzero),
        "ternary_zero_given_binary_flip": safe_fraction(ternary_zero, binary_flip),
        "ternary_sign_flip_given_nonzero": safe_fraction(
            ternary_sign_flip, ternary_nonzero
        ),
        "binary_ternary_scale_corr": correlation(binary_scale, ternary_scale),
        "binary_ternary_scale_exact_fraction": safe_fraction(scale_difference == 0),
        "binary_ternary_scale_mean_relative_error": float(relative_scale_difference.mean()),
        "binary_ternary_scale_max_relative_error": float(relative_scale_difference.max()),
        "binary_equals_ternary_exact_on_nonzero": safe_fraction(
            binary_equals_ternary_nonzero, ternary_nonzero
        ),
        "exact_binary_codebook_plus_mask_fraction": safe_fraction(exact_mask_model),
        "binary_codebook_plus_mask_max_abs_error": float(
            (inferred_mask_reconstruction - ternary_groups).abs().max()
        ),
        "binary_codebook_plus_mask_mean_abs_error": float(
            (inferred_mask_reconstruction - ternary_groups).abs().mean()
        ),
        "row_binary_sign_agreement_quantiles": row_quantiles(row_binary_agreement),
        "row_ternary_zero_rate_quantiles": row_quantiles(row_zero_rate),
        "row_exact_mask_model_quantiles": row_quantiles(row_mask_exact),
        "row_index_binary_agreement_correlation": correlation(
            normalized_row_index, row_binary_agreement
        ),
        "row_index_ternary_zero_rate_correlation": correlation(
            normalized_row_index, row_zero_rate
        ),
    }
    return result, pd.DataFrame(records)


def run_size(
    size: str,
    repositories: dict[str, str],
    group_size: int,
    block_size: int,
    block_count: int,
    tail_rows: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    layouts = {
        name: repository_embedding(repo_id)
        for name, repo_id in repositories.items()
    }
    widths = {layout[2][1] for layout in layouts.values()}
    if len(widths) != 1:
        raise RuntimeError(f"{size}: embedding widths differ: {widths}")
    common_rows = min(layout[2][0] for layout in layouts.values())
    intervals = deterministic_blocks(
        common_rows,
        block_size,
        block_count,
        tail_rows,
        seed,
    )

    sampled = {}
    metadata = {}
    expected_ids = None
    for name, repo_id in repositories.items():
        filename, key, shape, _ = layouts[name]
        values, row_ids, info = read_embedding_rows(
            repo_id,
            filename,
            key,
            shape,
            intervals,
        )
        if expected_ids is None:
            expected_ids = row_ids
        elif not torch.equal(expected_ids, row_ids):
            raise RuntimeError("sampled row IDs differ")
        sampled[name] = values
        metadata[name] = info

    result, bins = analyze(
        size,
        sampled["base"],
        sampled["binary"],
        sampled["ternary"],
        expected_ids,
        common_rows,
        group_size,
    )
    result["repositories"] = repositories
    result["remote_tensor_metadata"] = metadata
    result["sampled_intervals"] = [list(interval) for interval in intervals]
    return result, bins


def conclusions(results: dict[str, Any]) -> list[str]:
    lines = []
    for size, result in results.items():
        exact = result["exact_binary_codebook_plus_mask_fraction"]
        scale_exact = result["binary_ternary_scale_exact_fraction"]
        if exact >= 0.99999 and scale_exact >= 0.99999:
            lines.append(
                f"{size}: released ternary embedding is exactly the released binary embedding multiplied by a zero mask on the sampled rows."
            )
        elif exact >= 0.999:
            lines.append(
                f"{size}: a shared binary codebook plus ternary mask explains more than 99.9% of sampled embedding values."
            )
        else:
            lines.append(
                f"{size}: binary-codebook-plus-mask is incomplete; exact fraction={exact:.6f}."
            )
        lines.append(
            f"{size}: {100*result['ternary_zero_given_binary_flip']:.2f}% of binary embedding sign flips occur at ternary-zero positions."
        )
    if (
        results["1.7B"]["binary_sign_agreement_base"] > 0.995
        and results["4B"]["binary_sign_agreement_base"] < 0.98
    ):
        lines.append(
            "The shared embedding representation is cross-scale, but whether its binary sign codebook is frozen to Qwen is scale-dependent."
        )
    lines.append(
        "Released states identify a shared codebook and mask relation, but not whether binary recovery preceded ternary masking or both were optimized jointly."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--block-count", type=int, default=32)
    parser.add_argument("--tail-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-dir", default="embedding_lineage_forensics")
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
    }

    results = {}
    bin_frames = []
    for offset, (size, repositories) in enumerate(specifications.items()):
        result, bins = run_size(
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
