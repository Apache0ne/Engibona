#!/usr/bin/env python3
"""Pooled functional alignment for public Qwen and Bonsai 1.7B checkpoints.

The earlier full-model run established that released Bonsai models are vastly
closer to Qwen behavior than naive g128 projections.  This experiment asks a
more specific question: are the remaining hidden-state differences explainable
by a simple channel gauge (sign/scale/offset), or do they require a broader
change of internal basis?

For selected hidden-state depths it reports, on held-out tokens:

* raw cosine and pooled linear CKA;
* one global affine map;
* per-channel diagonal affine alignment;
* signed per-channel standardization;
* a low-dimensional orthogonal Procrustes alignment.

Logits are compared on the shared vocabulary with raw and centered cosine,
teacher KL, affine R2, and top-k overlap.  Models are loaded sequentially in
BF16 and naive baselines are quantized in row chunks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, __version__ as transformers_version


HERE = Path(__file__).resolve().parent
BF16_SPEC = importlib.util.spec_from_file_location(
    "full_functional_bf16",
    HERE / "run_full_model_functional_forensics_bf16.py",
)
if BF16_SPEC is None or BF16_SPEC.loader is None:
    raise ImportError("cannot load BF16 functional helper")
bf16 = importlib.util.module_from_spec(BF16_SPEC)
BF16_SPEC.loader.exec_module(bf16)
core = bf16.core


PROMPTS = [
    (
        "Explain, step by step, why Rayleigh scattering makes a clear daytime sky "
        "look blue while sunsets often appear red. Distinguish wavelength dependence, "
        "path length, and the role of aerosols. End with a two-sentence summary."
    ),
    (
        "A warehouse has 37 aisles, each with 48 shelves. Twelve shelves per aisle are "
        "reserved, and the rest hold 16 boxes each. Compute the usable shelf count and "
        "the total box capacity, showing each arithmetic step and checking the result."
    ),
    (
        "Write a robust Python implementation of longest increasing subsequence length "
        "using the O(n log n) tails method. Explain the invariant, handle an empty input, "
        "include type hints, and give two compact examples without importing NumPy."
    ),
    (
        "Return a JSON object describing a weather-tool request for Tokyo followed by a "
        "validation checklist. The JSON must contain action, location, units, and days. "
        "After the JSON, explain how malformed dates and unsupported units are handled."
    ),
    (
        "Compare binary search, a balanced binary search tree, and a hash table for "
        "lookup, insertion, ordered traversal, memory locality, and worst-case behavior. "
        "Use one concrete workload to explain when each structure is the best choice."
    ),
    (
        "A spacecraft performs three burns: 125.5 m/s prograde, 32.0 m/s radial inward, "
        "and 18.75 m/s retrograde. Compute the net prograde component, the vector magnitude, "
        "and explain why scalar addition of all three burn magnitudes would be incorrect."
    ),
    (
        "Describe how a transformer decoder produces the next-token distribution from "
        "token embeddings through attention, residual streams, normalization, MLP blocks, "
        "and the language-model head. Separate training-time teacher forcing from inference."
    ),
    (
        "Design a deterministic test plan for a quantized neural network converter. Cover "
        "format invariants, packed round trips, tied weights, numerical smoke tests, behavior "
        "comparisons, reproducibility, failure injection, and artifact provenance."
    ),
]


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) & 0x7FFFFFFF


def token_split_mask(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Deterministic 80/20 split shared by every model.
    test = (keys.remainder(5) == 0)
    if int(test.sum()) < 2 or int((~test).sum()) < 4:
        raise RuntimeError("insufficient train/test token split")
    return ~test, test


@torch.inference_mode()
def capture_signature(
    model,
    tokenizer,
    prompts: list[str],
    selected_layers: list[int],
    max_length: int,
    logit_positions: int,
) -> dict[str, Any]:
    hidden: dict[int, list[torch.Tensor]] = {layer: [] for layer in selected_layers}
    hidden_keys: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    logit_keys: list[torch.Tensor] = []
    prompt_lengths: list[int] = []

    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if output.hidden_states is None:
            raise RuntimeError("model did not return hidden states")
        length = int(attention_mask.sum())
        prompt_lengths.append(length)
        positions = torch.arange(length, dtype=torch.long)
        keys = prompt_index * 100000 + positions
        hidden_keys.append(keys)
        for layer in selected_layers:
            if layer >= len(output.hidden_states):
                raise ValueError(
                    f"hidden-state index {layer} exceeds {len(output.hidden_states) - 1}"
                )
            hidden[layer].append(
                output.hidden_states[layer][0, :length].detach().cpu().to(torch.float16)
            )

        count = min(logit_positions, length)
        selected = torch.arange(length - count, length, dtype=torch.long)
        logits.append(
            output.logits[0, selected].detach().cpu().to(torch.float16)
        )
        logit_keys.append(prompt_index * 100000 + selected)
        del output
        gc.collect()

    return {
        "hidden": {
            str(layer): torch.cat(parts, dim=0)
            for layer, parts in hidden.items()
        },
        "hidden_keys": torch.cat(hidden_keys),
        "logits": torch.cat(logits, dim=0),
        "logit_keys": torch.cat(logit_keys),
        "prompt_lengths": prompt_lengths,
    }


def r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.float()
    prediction = prediction.float()
    denominator = (target - target.mean()).square().sum().clamp_min(1e-20)
    return float(1.0 - (target - prediction).square().sum() / denominator)


def normalized_rmse(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.float()
    prediction = prediction.float()
    return float(
        (target - prediction).square().mean().sqrt()
        / target.square().mean().sqrt().clamp_min(1e-20)
    )


def pooled_linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    # Token-space form avoids a hidden_size x hidden_size allocation.
    left = left.float()
    right = right.float()
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    gram_left = left @ left.T
    gram_right = right @ right.T
    numerator = (gram_left * gram_right).sum()
    denominator = (
        gram_left.square().sum().sqrt()
        * gram_right.square().sum().sqrt()
    ).clamp_min(1e-20)
    return float(numerator / denominator)


def fit_global_affine(
    target_train: torch.Tensor,
    source_train: torch.Tensor,
    source_test: torch.Tensor,
) -> torch.Tensor:
    x = source_train.float().reshape(-1)
    y = target_train.float().reshape(-1)
    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).sum() / (
        (x - x_mean).square().sum().clamp_min(1e-20)
    )
    intercept = y_mean - slope * x_mean
    return source_test.float() * slope + intercept


def fit_diagonal_affine(
    target_train: torch.Tensor,
    source_train: torch.Tensor,
    source_test: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_train = target_train.float()
    source_train = source_train.float()
    source_test = source_test.float()
    target_mean = target_train.mean(dim=0)
    source_mean = source_train.mean(dim=0)
    source_centered = source_train - source_mean
    target_centered = target_train - target_mean
    covariance = (source_centered * target_centered).sum(dim=0)
    source_energy = source_centered.square().sum(dim=0).clamp_min(1e-20)
    target_energy = target_centered.square().sum(dim=0).clamp_min(1e-20)
    slope = covariance / source_energy
    mapped = (source_test - source_mean) * slope + target_mean
    correlation = covariance / (source_energy * target_energy).sqrt()
    metadata = {
        "negative_channel_fraction": float((correlation < 0).float().mean()),
        "median_abs_channel_correlation": float(correlation.abs().median()),
        "mean_abs_channel_correlation": float(correlation.abs().mean()),
        "median_abs_slope": float(slope.abs().median()),
    }
    return mapped, metadata


def fit_signed_standardization(
    target_train: torch.Tensor,
    source_train: torch.Tensor,
    source_test: torch.Tensor,
) -> torch.Tensor:
    target_train = target_train.float()
    source_train = source_train.float()
    source_test = source_test.float()
    target_mean = target_train.mean(dim=0)
    source_mean = source_train.mean(dim=0)
    target_std = target_train.std(dim=0, unbiased=False).clamp_min(1e-8)
    source_std = source_train.std(dim=0, unbiased=False).clamp_min(1e-8)
    covariance = (
        (source_train - source_mean) * (target_train - target_mean)
    ).mean(dim=0)
    sign = torch.where(covariance < 0, -torch.ones_like(covariance), torch.ones_like(covariance))
    return (
        (source_test - source_mean)
        / source_std
        * target_std
        * sign
        + target_mean
    )


def deterministic_projection(hidden_size: int, dimension: int, seed: int) -> torch.Tensor:
    dimension = min(dimension, hidden_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randint(
        0,
        2,
        (hidden_size, dimension),
        generator=generator,
        dtype=torch.int8,
    ).float()
    matrix = matrix.mul_(2).sub_(1).div_(math.sqrt(dimension))
    return matrix


def projected_orthogonal_alignment(
    target_train: torch.Tensor,
    source_train: torch.Tensor,
    target_test: torch.Tensor,
    source_test: torch.Tensor,
    dimension: int,
    seed: int,
) -> dict[str, float]:
    projection = deterministic_projection(target_train.shape[1], dimension, seed)
    target_train_p = target_train.float() @ projection
    source_train_p = source_train.float() @ projection
    target_test_p = target_test.float() @ projection
    source_test_p = source_test.float() @ projection
    target_mean = target_train_p.mean(dim=0)
    source_mean = source_train_p.mean(dim=0)
    target_centered = target_train_p - target_mean
    source_centered = source_train_p - source_mean
    cross = source_centered.T @ target_centered
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    identity = source_test_p
    mapped = (source_test_p - source_mean) @ rotation + target_mean
    return {
        "projected_dimension": int(projection.shape[1]),
        "identity_cosine": float(
            F.cosine_similarity(target_test_p, identity, dim=1).mean()
        ),
        "orthogonal_cosine": float(
            F.cosine_similarity(target_test_p, mapped, dim=1).mean()
        ),
        "orthogonal_r2": r2_score(target_test_p, mapped),
        "orthogonal_normalized_rmse": normalized_rmse(target_test_p, mapped),
    }


def hidden_alignment(
    teacher: torch.Tensor,
    candidate: torch.Tensor,
    keys: torch.Tensor,
    projection_dimension: int,
    seed: int,
) -> dict[str, Any]:
    if teacher.shape != candidate.shape:
        raise ValueError(f"hidden shape mismatch {teacher.shape} vs {candidate.shape}")
    train_mask, test_mask = token_split_mask(keys)
    target_train = teacher[train_mask].float()
    source_train = candidate[train_mask].float()
    target_test = teacher[test_mask].float()
    source_test = candidate[test_mask].float()

    global_mapped = fit_global_affine(target_train, source_train, source_test)
    diagonal_mapped, diagonal_metadata = fit_diagonal_affine(
        target_train, source_train, source_test
    )
    signed_mapped = fit_signed_standardization(
        target_train, source_train, source_test
    )
    target_test_centered = target_test - target_train.mean(dim=0)
    source_test_centered = source_test - source_train.mean(dim=0)

    result: dict[str, Any] = {
        "tokens_total": int(keys.numel()),
        "tokens_train": int(train_mask.sum()),
        "tokens_test": int(test_mask.sum()),
        "hidden_size": int(teacher.shape[1]),
        "raw_cosine": float(
            F.cosine_similarity(target_test, source_test, dim=1).mean()
        ),
        "centered_cosine": float(
            F.cosine_similarity(
                target_test_centered,
                source_test_centered,
                dim=1,
            ).mean()
        ),
        "raw_normalized_rmse": normalized_rmse(target_test, source_test),
        "pooled_linear_cka": pooled_linear_cka(teacher, candidate),
        "global_affine_r2": r2_score(target_test, global_mapped),
        "global_affine_cosine": float(
            F.cosine_similarity(target_test, global_mapped, dim=1).mean()
        ),
        "diagonal_affine_r2": r2_score(target_test, diagonal_mapped),
        "diagonal_affine_cosine": float(
            F.cosine_similarity(target_test, diagonal_mapped, dim=1).mean()
        ),
        "diagonal_affine_normalized_rmse": normalized_rmse(
            target_test, diagonal_mapped
        ),
        "signed_standardized_r2": r2_score(target_test, signed_mapped),
        "signed_standardized_cosine": float(
            F.cosine_similarity(target_test, signed_mapped, dim=1).mean()
        ),
        **diagonal_metadata,
    }
    result["projected_orthogonal"] = projected_orthogonal_alignment(
        target_train,
        source_train,
        target_test,
        source_test,
        projection_dimension,
        seed,
    )
    return result


def topk_overlap(left: torch.Tensor, right: torch.Tensor, k: int) -> float:
    k = min(k, left.shape[-1], right.shape[-1])
    left_indices = left.topk(k, dim=-1).indices
    right_indices = right.topk(k, dim=-1).indices
    overlap = (
        left_indices[:, :, None] == right_indices[:, None, :]
    ).any(dim=2).float().mean(dim=1)
    return float(overlap.mean())


def logit_alignment(
    teacher_full: torch.Tensor,
    candidate_full: torch.Tensor,
) -> dict[str, Any]:
    shared = min(teacher_full.shape[-1], candidate_full.shape[-1])
    teacher = teacher_full[..., :shared].float()
    candidate = candidate_full[..., :shared].float()
    teacher_log_prob = teacher.log_softmax(dim=-1)
    candidate_log_prob = candidate.log_softmax(dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl = (teacher_prob * (teacher_log_prob - candidate_log_prob)).sum(dim=-1)

    raw_cosine = F.cosine_similarity(teacher, candidate, dim=-1)
    teacher_centered = teacher - teacher.mean(dim=-1, keepdim=True)
    candidate_centered = candidate - candidate.mean(dim=-1, keepdim=True)
    centered_cosine = F.cosine_similarity(
        teacher_centered, candidate_centered, dim=-1
    )
    candidate_energy = candidate_centered.square().sum(dim=-1, keepdim=True).clamp_min(1e-20)
    slope = (candidate_centered * teacher_centered).sum(
        dim=-1, keepdim=True
    ) / candidate_energy
    intercept = teacher.mean(dim=-1, keepdim=True) - slope * candidate.mean(
        dim=-1, keepdim=True
    )
    mapped = candidate * slope + intercept

    teacher_full_probability = teacher_full.float().softmax(dim=-1)
    candidate_full_probability = candidate_full.float().softmax(dim=-1)
    return {
        "positions": int(teacher.shape[0]),
        "teacher_vocabulary_size": int(teacher_full.shape[-1]),
        "candidate_vocabulary_size": int(candidate_full.shape[-1]),
        "shared_vocabulary_size": int(shared),
        "teacher_shared_probability_mass": float(
            teacher_full_probability[..., :shared].sum(dim=-1).mean()
        ),
        "candidate_shared_probability_mass": float(
            candidate_full_probability[..., :shared].sum(dim=-1).mean()
        ),
        "teacher_kl": float(kl.mean()),
        "last_position_kl": float(kl[-1]),
        "top1_agreement": float(
            (teacher.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean()
        ),
        "raw_logit_cosine": float(raw_cosine.mean()),
        "centered_logit_cosine": float(centered_cosine.mean()),
        "affine_logit_r2": r2_score(teacher, mapped),
        "affine_logit_normalized_rmse": normalized_rmse(teacher, mapped),
        "top10_overlap": topk_overlap(teacher, candidate, 10),
        "top100_overlap": topk_overlap(teacher, candidate, 100),
    }


def compare_signatures(
    teacher: dict[str, Any],
    candidate: dict[str, Any],
    selected_layers: list[int],
    projection_dimension: int,
    seed: int,
) -> dict[str, Any]:
    if not torch.equal(teacher["hidden_keys"], candidate["hidden_keys"]):
        raise RuntimeError("hidden token keys differ")
    if not torch.equal(teacher["logit_keys"], candidate["logit_keys"]):
        raise RuntimeError("logit token keys differ")
    hidden = {}
    for layer in selected_layers:
        hidden[str(layer)] = hidden_alignment(
            teacher["hidden"][str(layer)],
            candidate["hidden"][str(layer)],
            teacher["hidden_keys"],
            projection_dimension,
            stable_seed(f"layer-{layer}", seed),
        )
    return {
        "hidden": hidden,
        "logits": logit_alignment(teacher["logits"], candidate["logits"]),
    }


def clear_model(model, cache_dir: Path | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if cache_dir is not None:
        shutil.rmtree(cache_dir, ignore_errors=True)
    gc.collect()


def run_candidate(
    name: str,
    repo_id: str,
    cache_dir: Path,
    tokenizer,
    teacher_signature: dict[str, Any],
    selected_layers: list[int],
    max_length: int,
    logit_positions: int,
    projection_dimension: int,
    seed: int,
    naive_mode: str | None,
    group_size: int,
) -> tuple[dict[str, Any], Any]:
    started = time.time()
    model = bf16.load_model_bf16(repo_id, cache_dir)
    quantization = None
    if naive_mode is not None:
        quantization = bf16.quantize_model_chunked(
            model, naive_mode, group_size
        )
    signature = capture_signature(
        model,
        tokenizer,
        PROMPTS,
        selected_layers,
        max_length,
        logit_positions,
    )
    metrics = compare_signatures(
        teacher_signature,
        signature,
        selected_layers,
        projection_dimension,
        seed,
    )
    del signature
    gc.collect()
    return {
        "name": name,
        "repository": repo_id,
        "quantization": quantization,
        "metrics": metrics,
        "seconds": time.time() - started,
    }, model


def compact_comparison(results: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for mode in ("binary", "ternary"):
        naive = results[f"naive_{mode}"]["metrics"]
        released = results[f"released_{mode}"]["metrics"]
        layer_rows = {}
        for layer in naive["hidden"]:
            n = naive["hidden"][layer]
            r = released["hidden"][layer]
            layer_rows[layer] = {
                "released_minus_naive_raw_cosine": r["raw_cosine"] - n["raw_cosine"],
                "released_minus_naive_diagonal_affine_r2": r["diagonal_affine_r2"] - n["diagonal_affine_r2"],
                "released_minus_naive_projected_orthogonal_r2": (
                    r["projected_orthogonal"]["orthogonal_r2"]
                    - n["projected_orthogonal"]["orthogonal_r2"]
                ),
                "released_diagonal_fraction_of_orthogonal_r2": (
                    r["diagonal_affine_r2"]
                    / max(r["projected_orthogonal"]["orthogonal_r2"], 1e-20)
                ),
            }
        output[mode] = {
            "released_over_naive_kl": (
                released["logits"]["teacher_kl"]
                / max(naive["logits"]["teacher_kl"], 1e-20)
            ),
            "released_minus_naive_centered_logit_cosine": (
                released["logits"]["centered_logit_cosine"]
                - naive["logits"]["centered_logit_cosine"]
            ),
            "released_minus_naive_top100_overlap": (
                released["logits"]["top100_overlap"]
                - naive["logits"]["top100_overlap"]
            ),
            "layers": layer_rows,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--selected-layers", type=int, nargs="+", default=[0, 7, 14, 21, 28])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logit-positions", type=int, default=8)
    parser.add_argument("--projection-dimension", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260803)
    parser.add_argument("--output", default="pooled_functional_alignment.json")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()
    root = Path("pooled_alignment_cache")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base,
        cache_dir=str(root / "tokenizer"),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_cache = root / "base"
    teacher_model = bf16.load_model_bf16(args.base, base_cache)
    teacher_parameters = sum(p.numel() for p in teacher_model.parameters())
    teacher_signature = capture_signature(
        teacher_model,
        tokenizer,
        PROMPTS,
        args.selected_layers,
        args.max_length,
        args.logit_positions,
    )
    clear_model(teacher_model)

    results: dict[str, Any] = {}
    specifications = [
        ("naive_binary", args.base, base_cache, "binary"),
        ("naive_ternary", args.base, base_cache, "ternary"),
        ("released_binary", args.binary, root / "binary", None),
        ("released_ternary", args.ternary, root / "ternary", None),
    ]
    for name, repo_id, cache_dir, naive_mode in specifications:
        result, model = run_candidate(
            name,
            repo_id,
            cache_dir,
            tokenizer,
            teacher_signature,
            args.selected_layers,
            args.max_length,
            args.logit_positions,
            args.projection_dimension,
            args.seed,
            naive_mode,
            args.group_size,
        )
        results[name] = result
        delete_cache = (
            cache_dir
            if name in {"naive_ternary", "released_binary", "released_ternary"}
            else None
        )
        clear_model(model, delete_cache)

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
        "teacher_parameter_count": teacher_parameters,
        "group_size": args.group_size,
        "selected_layers": args.selected_layers,
        "max_length": args.max_length,
        "logit_positions_per_prompt": args.logit_positions,
        "projection_dimension": args.projection_dimension,
        "prompt_count": len(PROMPTS),
        "prompt_lengths": teacher_signature["prompt_lengths"],
        "pooled_hidden_tokens": int(teacher_signature["hidden_keys"].numel()),
        "sampled_logit_positions": int(teacher_signature["logit_keys"].numel()),
        "results": results,
        "comparisons": compact_comparison(results),
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, allow_nan=True))
    shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
