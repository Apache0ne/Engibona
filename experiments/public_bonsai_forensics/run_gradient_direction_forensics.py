#!/usr/bin/env python3
"""Test whether public Bonsai code moves follow behavior-loss gradients.

For Qwen3-1.7B this experiment:

1. caches teacher logits on four calibration prompts;
2. constructs naive exact-g128 binary or ternary students;
3. computes gradients of teacher KL and ordinary next-token CE at the naive state;
4. compares first-order and empirical-Fisher code-move costs with the actual
   released Bonsai code changes at layers 0, 13, and 27.

The central binary score is the predicted loss decrease from flipping the naive
sign at fixed group scale. For ternary, the empirical-Fisher quadratic chooses
among {-s,0,+s}. Released-direction costs also include the released scale change.

Strong alignment with KD but not CE would directly support behavior matching as
an important recovery objective. Final-state gradients cannot identify the exact
private optimizer, dataset, schedule, or later nonlinear optimization path.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_url
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version


PROMPTS = [
    "Explain why the sky appears blue and sunsets appear red, distinguishing wavelength dependence and path length.",
    "A warehouse has 37 aisles with 48 shelves each. Twelve shelves per aisle are reserved. Compute usable shelves and capacity at 16 boxes per shelf.",
    "Write a robust Python function for longest increasing subsequence length using the O(n log n) tails method and explain its invariant.",
    "Describe how a transformer decoder turns token embeddings into a next-token distribution through attention, residuals, MLPs, normalization, and the LM head.",
]
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
DTYPES = {
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}


def stable_seed(text: str, seed: int) -> int:
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) & 0x7FFFFFFF


def layer_index(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def module_type(name: str) -> str:
    for token in TARGET_MODULES:
        if token in name:
            return token
    return "other"


def request_range(url: str, start: int, end: int) -> bytes:
    for attempt in range(1, 6):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 300),
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(f"range status {response.status_code}")
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


class RemoteSafeTensors:
    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        files = HfApi().list_repo_files(repo_id=repo_id, revision="main")
        indexes = sorted(
            name for name in files if name.endswith(".safetensors.index.json")
        )
        self.tensor_files = sorted(
            name for name in files if name.endswith(".safetensors")
        )
        if not self.tensor_files:
            raise RuntimeError(f"{repo_id}: no safetensors files")
        self.weight_map = None
        if indexes:
            response = requests.get(
                hf_hub_url(repo_id=repo_id, filename=indexes[0], revision="main"),
                timeout=(30, 120),
            )
            response.raise_for_status()
            self.weight_map = response.json()["weight_map"]
        self.cache: dict[str, tuple[str, int, dict[str, Any]]] = {}

    def header(self, filename: str) -> tuple[str, int, dict[str, Any]]:
        if filename not in self.cache:
            url = hf_hub_url(
                repo_id=self.repo_id, filename=filename, revision="main"
            )
            header_length = struct.unpack("<Q", request_range(url, 0, 7))[0]
            raw = request_range(url, 8, 8 + header_length - 1)
            header = json.loads(raw.decode("utf-8").rstrip(" \t\r\n\x00"))
            self.cache[filename] = (url, 8 + header_length, header)
        return self.cache[filename]

    def resolve(self, key: str) -> tuple[str, dict[str, Any], str, int]:
        if self.weight_map is not None:
            if key not in self.weight_map:
                raise KeyError(f"{self.repo_id}: missing {key}")
            filename = self.weight_map[key]
            url, data_start, header = self.header(filename)
            return filename, header[key], url, data_start
        for filename in self.tensor_files:
            url, data_start, header = self.header(filename)
            if key in header:
                return filename, header[key], url, data_start
        raise KeyError(f"{self.repo_id}: missing {key}")

    def rows(
        self,
        key: str,
        row_intervals: list[tuple[int, int]],
    ) -> torch.Tensor:
        _, item, url, data_start = self.resolve(key)
        shape = tuple(int(value) for value in item["shape"])
        if len(shape) != 2:
            raise ValueError(f"{key}: expected matrix, got {shape}")
        dtype_name = str(item["dtype"])
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name}")
        dtype, item_size = DTYPES[dtype_name]
        offsets = [int(value) for value in item["data_offsets"]]
        rows, width = shape
        row_bytes = width * item_size
        parts = []
        for start, end in row_intervals:
            if start < 0 or end > rows or end <= start:
                raise ValueError(f"bad row interval {(start, end)} for {shape}")
            byte_start = data_start + offsets[0] + start * row_bytes
            byte_end = data_start + offsets[0] + end * row_bytes - 1
            raw = bytearray(request_range(url, byte_start, byte_end))
            parts.append(
                torch.frombuffer(raw, dtype=dtype)
                .clone()
                .reshape(end - start, width)
                .to(torch.float16)
            )
        return torch.cat(parts, dim=0)


def row_intervals(rows: int, block_rows: int, blocks: int, seed: int):
    generator = np.random.default_rng(seed)
    edges = np.linspace(0, max(rows - block_rows, 0), blocks + 1)
    intervals = []
    for index in range(blocks):
        low = int(edges[index])
        high = int(edges[index + 1])
        start = int(generator.integers(low, high + 1))
        intervals.append((start, min(rows, start + block_rows)))
    return intervals


def flatten_row_ids(intervals: list[tuple[int, int]]) -> torch.Tensor:
    return torch.cat(
        [torch.arange(start, end, dtype=torch.long) for start, end in intervals]
    )


def grouped_binary(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    groups = weight.float().reshape(-1, group_size)
    codes = torch.where(groups < 0, -torch.ones_like(groups), torch.ones_like(groups))
    scales = groups.abs().mean(dim=1).clamp_min(1e-12)
    return (codes * scales[:, None]).reshape_as(weight)


def grouped_ternary(weight: torch.Tensor, group_size: int, iterations: int = 16):
    groups = weight.float().reshape(-1, group_size)
    scales = groups.abs().mean(dim=1).clamp_min(1e-12)
    for _ in range(iterations):
        active = groups.abs() > 0.5 * scales[:, None]
        scales = (
            (groups.abs() * active).sum(dim=1)
            / active.sum(dim=1).clamp_min(1)
        ).clamp_min(1e-12)
    active = groups.abs() > 0.5 * scales[:, None]
    codes = torch.sign(groups) * active.float()
    return (codes * scales[:, None]).reshape_as(weight)


@torch.no_grad()
def quantize_model(model: nn.Module, mode: str, group_size: int, row_chunk: int = 64):
    seen: set[int] = set()
    count = 0
    values = 0
    for module in model.modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        parameter = module.weight
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim != 2 or parameter.shape[-1] % group_size:
            continue
        for start in range(0, parameter.shape[0], row_chunk):
            end = min(parameter.shape[0], start + row_chunk)
            source = parameter[start:end].float()
            quantized = (
                grouped_binary(source, group_size)
                if mode == "binary"
                else grouped_ternary(source, group_size)
            )
            parameter[start:end].copy_(quantized.to(parameter.dtype))
        count += 1
        values += parameter.numel()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return {"tensors": count, "weights": values}


def load_model(repo_id: str, cache_dir: Path):
    common = dict(
        pretrained_model_name_or_path=repo_id,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            **common, dtype=torch.bfloat16
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            **common, torch_dtype=torch.bfloat16
        )
    model.eval()
    return model


def tokenize(tokenizer, prompts: list[str], max_length: int):
    output = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        output.append(
            {
                "prompt": prompt,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded.get(
                    "attention_mask", torch.ones_like(encoded["input_ids"])
                ),
            }
        )
    return output


@torch.inference_mode()
def teacher_cache(model, batches):
    output = []
    for batch in batches:
        result = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        output.append(result.logits.detach().cpu().to(torch.float16))
    return output


def target_parameters(model, layers: list[int]):
    targets = {}
    suffixes = tuple(f".{token}.weight" for token in TARGET_MODULES)
    for name, parameter in model.named_parameters():
        if layer_index(name) in layers and name.endswith(suffixes):
            targets[name] = parameter
    expected = len(layers) * len(TARGET_MODULES)
    if len(targets) != expected:
        raise RuntimeError(
            f"expected {expected} target matrices, found {len(targets)}: {sorted(targets)}"
        )
    return targets


def prepare_sampling(targets, block_rows: int, blocks: int, seed: int):
    metadata = {}
    for name, parameter in targets.items():
        intervals = row_intervals(
            parameter.shape[0],
            min(block_rows, parameter.shape[0]),
            blocks,
            stable_seed(name, seed),
        )
        metadata[name] = {
            "intervals": intervals,
            "row_ids": flatten_row_ids(intervals),
            "shape": tuple(parameter.shape),
            "layer": layer_index(name),
            "module": module_type(name),
        }
    return metadata


def sample_gradient(gradient: torch.Tensor, intervals: list[tuple[int, int]]):
    return torch.cat([gradient[start:end].detach().cpu().float() for start, end in intervals])


def kd_loss(teacher_logits: torch.Tensor, student_logits: torch.Tensor):
    teacher_probability = teacher_logits.float().softmax(dim=-1)
    return F.kl_div(
        student_logits.float().log_softmax(dim=-1),
        teacher_probability,
        reduction="batchmean",
    ) / student_logits.shape[1]


def next_token_ce(logits: torch.Tensor, input_ids: torch.Tensor):
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


def collect_gradients(
    model,
    targets,
    sampling,
    batches,
    teachers,
):
    accumulators = {
        loss_name: {
            name: {
                "sum": torch.zeros(
                    (sampling[name]["row_ids"].numel(), parameter.shape[1]),
                    dtype=torch.float32,
                ),
                "square_sum": torch.zeros(
                    (sampling[name]["row_ids"].numel(), parameter.shape[1]),
                    dtype=torch.float32,
                ),
            }
            for name, parameter in targets.items()
        }
        for loss_name in ("kd", "ce")
    }

    for batch, teacher_logits in zip(batches, teachers):
        for loss_name in ("kd", "ce"):
            result = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            loss = (
                kd_loss(teacher_logits, result.logits)
                if loss_name == "kd"
                else next_token_ce(result.logits, batch["input_ids"])
            )
            gradients = torch.autograd.grad(
                loss,
                list(targets.values()),
                retain_graph=False,
                create_graph=False,
            )
            for (name, _), gradient in zip(targets.items(), gradients):
                selected = sample_gradient(
                    gradient, sampling[name]["intervals"]
                )
                accumulators[loss_name][name]["sum"] += selected
                accumulators[loss_name][name]["square_sum"] += selected.square()
            del result, loss, gradients
            gc.collect()

    count = len(batches)
    for losses in accumulators.values():
        for values in losses.values():
            values["mean"] = values.pop("sum") / count
            values["second_moment"] = values.pop("square_sum") / count
    return accumulators


def infer_codes_scales(weight: torch.Tensor, mode: str, group_size: int):
    rows, width = weight.shape
    groups = weight.float().reshape(rows, width // group_size, group_size)
    if mode == "binary":
        scales = groups.abs().median(dim=-1).values.clamp_min(1e-12)
        codes = torch.where(groups < 0, -torch.ones_like(groups), torch.ones_like(groups))
    else:
        nonzero = groups != 0
        scales = (
            (groups.abs() * nonzero).sum(dim=-1)
            / nonzero.sum(dim=-1).clamp_min(1)
        ).clamp_min(1e-12)
        normalized = groups / scales[..., None]
        codes = torch.where(
            normalized > 0.5,
            torch.ones_like(normalized),
            torch.where(
                normalized < -0.5,
                -torch.ones_like(normalized),
                torch.zeros_like(normalized),
            ),
        )
    return codes, scales, groups


def auc_score(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.reshape(-1).bool()
    scores = scores.reshape(-1).float()
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def top_fraction_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.reshape(-1).bool()
    scores = scores.reshape(-1)
    k = int(labels.sum())
    if k <= 0:
        return float("nan")
    selected = torch.topk(scores, min(k, scores.numel())).indices
    return float(labels[selected].float().mean())


def binary_metrics(
    naive_weight,
    released_weight,
    gradient,
    second_moment,
    group_size,
):
    naive_code, naive_scale, naive_groups = infer_codes_scales(
        naive_weight, "binary", group_size
    )
    released_code, _, released_groups = infer_codes_scales(
        released_weight, "binary", group_size
    )
    g = gradient.reshape_as(naive_groups)
    h = second_moment.reshape_as(naive_groups)
    label = released_code != naive_code
    fixed_flip_delta = -2.0 * naive_groups
    first_cost = g * fixed_flip_delta
    fisher_cost = first_cost + 0.5 * h * fixed_flip_delta.square()
    first_benefit = -first_cost
    fisher_benefit = -fisher_cost
    released_delta = released_groups - naive_groups
    released_first_cost = g * released_delta
    released_fisher_cost = (
        released_first_cost + 0.5 * h * released_delta.square()
    )
    return {
        "weights": int(label.numel()),
        "actual_flip_rate": float(label.float().mean()),
        "first_order_flip_auc": auc_score(label, first_benefit),
        "fisher_flip_auc": auc_score(label, fisher_benefit),
        "first_order_top_rate_precision": top_fraction_precision(label, first_benefit),
        "fisher_top_rate_precision": top_fraction_precision(label, fisher_benefit),
        "actual_flips_first_order_beneficial_fraction": float(
            (first_cost[label] < 0).float().mean()
        ),
        "actual_flips_fisher_beneficial_fraction": float(
            (fisher_cost[label] < 0).float().mean()
        ),
        "released_direction_first_order_beneficial_fraction": float(
            (released_first_cost[label] < 0).float().mean()
        ),
        "released_direction_fisher_beneficial_fraction": float(
            (released_fisher_cost[label] < 0).float().mean()
        ),
        "mean_first_benefit_actual_flip": float(first_benefit[label].mean()),
        "mean_first_benefit_unchanged": float(first_benefit[~label].mean()),
        "mean_fisher_benefit_actual_flip": float(fisher_benefit[label].mean()),
        "mean_fisher_benefit_unchanged": float(fisher_benefit[~label].mean()),
    }


def ternary_metrics(
    naive_weight,
    released_weight,
    gradient,
    second_moment,
    group_size,
):
    naive_code, naive_scale, naive_groups = infer_codes_scales(
        naive_weight, "ternary", group_size
    )
    released_code, _, released_groups = infer_codes_scales(
        released_weight, "ternary", group_size
    )
    g = gradient.reshape_as(naive_groups)
    h = second_moment.reshape_as(naive_groups)
    label = released_code != naive_code
    candidates = torch.stack(
        [
            -naive_scale[..., None].expand_as(naive_groups),
            torch.zeros_like(naive_groups),
            naive_scale[..., None].expand_as(naive_groups),
        ],
        dim=-1,
    )
    delta = candidates - naive_groups[..., None]
    fisher_costs = g[..., None] * delta + 0.5 * h[..., None] * delta.square()
    predicted_indices = fisher_costs.argmin(dim=-1)
    predicted_code = predicted_indices.float() - 1.0
    released_delta = released_groups - naive_groups
    first_cost = g * released_delta
    fisher_cost = first_cost + 0.5 * h * released_delta.square()
    return {
        "weights": int(label.numel()),
        "actual_change_rate": float(label.float().mean()),
        "fisher_predicted_code_agreement_released": float(
            (predicted_code == released_code).float().mean()
        ),
        "fisher_predicted_change_agreement": float(
            ((predicted_code != naive_code) == label).float().mean()
        ),
        "released_changes_first_order_beneficial_fraction": float(
            (first_cost[label] < 0).float().mean()
        ),
        "released_changes_fisher_beneficial_fraction": float(
            (fisher_cost[label] < 0).float().mean()
        ),
        "released_zero_fraction": float((released_code == 0).float().mean()),
        "predicted_zero_fraction": float((predicted_code == 0).float().mean()),
        "mean_first_cost_actual_change": float(first_cost[label].mean()),
        "mean_fisher_cost_actual_change": float(fisher_cost[label].mean()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for loss_name in sorted({row["loss"] for row in rows}):
        selected = [row for row in rows if row["loss"] == loss_name]
        total = sum(row["weights"] for row in selected)
        metrics = {}
        numeric = sorted(
            key for key, value in selected[0].items()
            if isinstance(value, (int, float))
            and key not in {"layer", "weights"}
        )
        for key in numeric:
            metrics[key] = float(
                sum(row[key] * row["weights"] for row in selected) / total
            )
        metrics["weights"] = total
        output[loss_name] = metrics
    return output


def run_mode(
    mode: str,
    base_repo: str,
    released_repo: str,
    cache_dir: Path,
    batches,
    teachers,
    layers,
    group_size,
    block_rows,
    blocks,
    seed,
):
    model = load_model(base_repo, cache_dir)
    quantization = quantize_model(model, mode, group_size)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = target_parameters(model, layers)
    for parameter in targets.values():
        parameter.requires_grad_(True)
    sampling = prepare_sampling(targets, block_rows, blocks, seed)
    gradients = collect_gradients(
        model, targets, sampling, batches, teachers
    )

    released = RemoteSafeTensors(released_repo)
    rows = []
    for name, parameter in targets.items():
        info = sampling[name]
        naive_weight = torch.cat(
            [
                parameter[start:end].detach().cpu().float()
                for start, end in info["intervals"]
            ]
        )
        released_weight = released.rows(name, info["intervals"]).float()
        for loss_name in ("kd", "ce"):
            values = gradients[loss_name][name]
            metrics = (
                binary_metrics(
                    naive_weight,
                    released_weight,
                    values["mean"],
                    values["second_moment"],
                    group_size,
                )
                if mode == "binary"
                else ternary_metrics(
                    naive_weight,
                    released_weight,
                    values["mean"],
                    values["second_moment"],
                    group_size,
                )
            )
            rows.append(
                {
                    "mode": mode,
                    "loss": loss_name,
                    "tensor": name,
                    "layer": info["layer"],
                    "module": info["module"],
                    **metrics,
                }
            )
    del model, targets, gradients
    gc.collect()
    return {
        "quantization": quantization,
        "rows": rows,
        "aggregate": aggregate(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--output-dir", default="gradient_direction_forensics")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("gradient_direction_cache")
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        cache_dir=str(cache / "tokenizer"),
        trust_remote_code=True,
        use_fast=True,
    )
    batches = tokenize(tokenizer, PROMPTS, args.max_length)
    teacher_model = load_model(args.base, cache / "base")
    teachers = teacher_cache(teacher_model, batches)
    teacher_parameter_count = sum(
        parameter.numel() for parameter in teacher_model.parameters()
    )
    del teacher_model
    gc.collect()

    binary = run_mode(
        "binary",
        args.base,
        args.binary,
        cache / "base",
        batches,
        teachers,
        args.layers,
        args.group_size,
        args.block_rows,
        args.blocks,
        args.seed,
    )
    ternary = run_mode(
        "ternary",
        args.base,
        args.ternary,
        cache / "base",
        batches,
        teachers,
        args.layers,
        args.group_size,
        args.block_rows,
        args.blocks,
        args.seed,
    )

    all_rows = binary["rows"] + ternary["rows"]
    payload = {
        "repositories": {
            "base": args.base,
            "binary": args.binary,
            "ternary": args.ternary,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers_version,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "teacher_parameter_count": teacher_parameter_count,
        "arguments": vars(args),
        "prompt_token_counts": [
            int(batch["attention_mask"].sum()) for batch in batches
        ],
        "binary": {
            "quantization": binary["quantization"],
            "aggregate": binary["aggregate"],
        },
        "ternary": {
            "quantization": ternary["quantization"],
            "aggregate": ternary["aggregate"],
        },
        "seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(output / "tensor_metrics.csv", index=False)
    print(json.dumps(payload, indent=2, allow_nan=True))
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
