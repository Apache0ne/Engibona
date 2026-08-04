#!/usr/bin/env python3
"""Test exact transformer symmetry explanations of released Bonsai weights.

Two architecture-preserving symmetries could make released weights look far from
the original checkpoint without requiring broad functional relearning:

* permutation of attention heads with the matching output projection blocks;
* permutation of SwiGLU intermediate neurons across gate/up rows and down columns.

This script builds sign/code fingerprints from public Qwen and unpacked Bonsai
checkpoints.  Candidate permutations are selected on one disjoint feature split
and evaluated on a held-out split, preventing nearest-neighbour overfitting from
being mistaken for a real symmetry.  It cannot identify private optimization
hyperparameters, but it can strongly accept or reject these exact symmetry
families as the source of the released code divergence.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


SUFFIXES = {
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "o": "self_attn.o_proj.weight",
    "gate": "mlp.gate_proj.weight",
    "up": "mlp.up_proj.weight",
    "down": "mlp.down_proj.weight",
}


def fixed_disjoint(total: int, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = min(count, total // 2)
    if count <= 0:
        raise ValueError(f"cannot make disjoint samples from total={total}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)
    return permutation[:count].sort().values, permutation[count : 2 * count].sort().values


def codes(weight: torch.Tensor, mode: str) -> torch.Tensor:
    weight = weight.float()
    if mode == "ternary":
        return torch.sign(weight)
    return torch.where(weight < 0, -torch.ones_like(weight), torch.ones_like(weight))


def normalized_rows(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), p=2, dim=1, eps=1e-12)


class TensorStore:
    def __init__(self, repo_id: str, work: Path) -> None:
        self.repo_id = repo_id
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)
        self.files, self.weight_map = core.repo_layout(repo_id, self.work)
        self.local: dict[str, Path] = {}
        if self.weight_map is not None:
            self.keys = set(self.weight_map)
        else:
            first = self._local_path(self.files[0])
            with safe_open(str(first), framework="pt", device="cpu") as handle:
                self.keys = set(handle.keys())

    def _local_path(self, filename: str) -> Path:
        if filename not in self.local:
            self.local[filename] = core.download_file(
                self.repo_id,
                filename,
                self.work / Path(filename).name,
            )
        return self.local[filename]

    def tensor(self, key: str) -> torch.Tensor:
        if key not in self.keys:
            raise KeyError(f"{self.repo_id}: missing tensor {key}")
        if self.weight_map is None:
            filename = self.files[0]
        else:
            filename = self.weight_map[key]
        local = self._local_path(filename)
        with safe_open(str(local), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def config(self) -> dict[str, Any]:
        path = core.download_file(
            self.repo_id,
            "config.json",
            self.work / "config.json",
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def close(self) -> None:
        self.local.clear()
        shutil.rmtree(self.work, ignore_errors=True)
        gc.collect()


def find_layer_key(keys: set[str], layer: int, suffix: str) -> str:
    marker = f".layers.{layer}."
    candidates = sorted(key for key in keys if marker in key and key.endswith(suffix))
    if len(candidates) != 1:
        raise RuntimeError(
            f"layer {layer} suffix {suffix}: expected one tensor, found {candidates[:5]}"
        )
    return candidates[0]


def mlp_features(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    mode: str,
    input_columns: torch.Tensor,
    output_rows: torch.Tensor,
) -> torch.Tensor:
    if gate.shape != up.shape:
        raise ValueError("gate/up shape mismatch")
    if down.shape[1] != gate.shape[0] or down.shape[0] != gate.shape[1]:
        raise ValueError(
            f"down shape {tuple(down.shape)} incompatible with gate {tuple(gate.shape)}"
        )
    gate_part = codes(gate[:, input_columns], mode)
    up_part = codes(up[:, input_columns], mode)
    down_part = codes(down[output_rows, :].T, mode)
    return torch.cat([gate_part, up_part, down_part], dim=1).to(torch.float16)


def qo_head_features(
    q: torch.Tensor,
    o: torch.Tensor,
    mode: str,
    num_heads: int,
    head_dim: int,
    input_columns: torch.Tensor,
    output_rows: torch.Tensor,
) -> torch.Tensor:
    if q.shape[0] != num_heads * head_dim:
        raise ValueError(f"q rows {q.shape[0]} != heads*dim {num_heads * head_dim}")
    if o.shape[1] != num_heads * head_dim:
        raise ValueError(f"o columns {o.shape[1]} != heads*dim {num_heads * head_dim}")
    rows = []
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        q_part = codes(q[start:end, input_columns], mode).reshape(-1)
        o_part = codes(o[output_rows, start:end], mode).reshape(-1)
        rows.append(torch.cat([q_part, o_part]))
    return torch.stack(rows).to(torch.float16)


def kv_head_features(
    k: torch.Tensor,
    v: torch.Tensor,
    mode: str,
    num_heads: int,
    head_dim: int,
    input_columns: torch.Tensor,
) -> torch.Tensor:
    if k.shape != v.shape or k.shape[0] != num_heads * head_dim:
        raise ValueError("K/V shape is incompatible with configured KV heads")
    rows = []
    for head in range(num_heads):
        start = head * head_dim
        end = start + head_dim
        k_part = codes(k[start:end, input_columns], mode).reshape(-1)
        v_part = codes(v[start:end, input_columns], mode).reshape(-1)
        rows.append(torch.cat([k_part, v_part]))
    return torch.stack(rows).to(torch.float16)


def extract_repo(
    repo_id: str,
    mode: str,
    layers: list[int],
    work: Path,
    feature_count: int,
    seed: int,
) -> dict[str, Any]:
    store = TensorStore(repo_id, work)
    try:
        config = store.config()
        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        num_kv_heads = int(config.get("num_key_value_heads", num_heads))
        head_dim = int(config.get("head_dim", hidden_size // num_heads))
        output: dict[str, Any] = {
            "config": {
                "hidden_size": hidden_size,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
            },
            "layers": {},
        }
        for layer in layers:
            keys = {
                name: find_layer_key(store.keys, layer, suffix)
                for name, suffix in SUFFIXES.items()
            }
            q = store.tensor(keys["q"])
            k = store.tensor(keys["k"])
            v = store.tensor(keys["v"])
            o = store.tensor(keys["o"])
            gate = store.tensor(keys["gate"])
            up = store.tensor(keys["up"])
            down = store.tensor(keys["down"])

            input_train, input_test = fixed_disjoint(
                hidden_size,
                feature_count,
                core.stable_seed(f"layer-{layer}-input", seed),
            )
            output_train, output_test = fixed_disjoint(
                hidden_size,
                feature_count,
                core.stable_seed(f"layer-{layer}-output", seed),
            )
            output["layers"][str(layer)] = {
                "mlp_train": mlp_features(
                    gate, up, down, mode, input_train, output_train
                ),
                "mlp_test": mlp_features(
                    gate, up, down, mode, input_test, output_test
                ),
                "qo_train": qo_head_features(
                    q,
                    o,
                    mode,
                    num_heads,
                    head_dim,
                    input_train,
                    output_train,
                ),
                "qo_test": qo_head_features(
                    q,
                    o,
                    mode,
                    num_heads,
                    head_dim,
                    input_test,
                    output_test,
                ),
                "kv_train": kv_head_features(
                    k,
                    v,
                    mode,
                    num_kv_heads,
                    head_dim,
                    input_train,
                ),
                "kv_test": kv_head_features(
                    k,
                    v,
                    mode,
                    num_kv_heads,
                    head_dim,
                    input_test,
                ),
                "intermediate_size": int(gate.shape[0]),
            }
            del q, k, v, o, gate, up, down
            gc.collect()
        return output
    finally:
        store.close()


def vector_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float(), right.float(), dim=1, eps=1e-12)


def large_permutation_test(
    base_train: torch.Tensor,
    base_test: torch.Tensor,
    candidate_train: torch.Tensor,
    candidate_test: torch.Tensor,
    sample_indices: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    base_train_sample = normalized_rows(base_train[sample_indices])
    candidate_train_normalized = normalized_rows(candidate_train)
    scores = base_train_sample @ candidate_train_normalized.T
    best_scores, best_indices = scores.max(dim=1)
    identity_train = scores[
        torch.arange(sample_indices.numel()), sample_indices
    ]
    identity_percentile = (
        (scores <= identity_train[:, None]).float().mean(dim=1)
    )

    matched_test = vector_cosine(
        base_test[sample_indices], candidate_test[best_indices]
    )
    identity_test = vector_cosine(
        base_test[sample_indices], candidate_test[sample_indices]
    )
    unique_fraction = float(torch.unique(best_indices).numel() / best_indices.numel())
    displacement = (best_indices - sample_indices).abs().float()

    metrics = {
        "sampled_neurons": int(sample_indices.numel()),
        "candidate_neurons": int(candidate_train.shape[0]),
        "identity_top1_fraction": float((best_indices == sample_indices).float().mean()),
        "identity_train_cosine": float(identity_train.mean()),
        "best_train_cosine": float(best_scores.mean()),
        "best_minus_identity_train": float((best_scores - identity_train).mean()),
        "identity_test_cosine": float(identity_test.mean()),
        "matched_test_cosine": float(matched_test.mean()),
        "matched_minus_identity_test": float((matched_test - identity_test).mean()),
        "matched_test_positive_gain_fraction": float((matched_test > identity_test).float().mean()),
        "identity_rank_percentile": float(identity_percentile.mean()),
        "unique_best_match_fraction": unique_fraction,
        "normalized_index_displacement": float(
            displacement.mean() / max(candidate_train.shape[0] - 1, 1)
        ),
        "best_index_sha256": hashlib.sha256(
            best_indices.numpy().astype(np.int32).tobytes()
        ).hexdigest(),
    }
    return metrics, best_indices


def small_assignment_test(
    base_train: torch.Tensor,
    base_test: torch.Tensor,
    candidate_train: torch.Tensor,
    candidate_test: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    train_scores = normalized_rows(base_train) @ normalized_rows(candidate_train).T
    row_indices, column_indices = linear_sum_assignment(
        -train_scores.detach().cpu().numpy()
    )
    rows = torch.tensor(row_indices, dtype=torch.long)
    assignment = torch.tensor(column_indices, dtype=torch.long)
    identity = torch.arange(train_scores.shape[0], dtype=torch.long)
    assigned_train = train_scores[rows, assignment]
    identity_train = train_scores[identity, identity]
    assigned_test = vector_cosine(base_test[rows], candidate_test[assignment])
    identity_test = vector_cosine(base_test, candidate_test)
    metrics = {
        "unit_count": int(train_scores.shape[0]),
        "identity_assignment_fraction": float((assignment == rows).float().mean()),
        "identity_train_cosine": float(identity_train.mean()),
        "assigned_train_cosine": float(assigned_train.mean()),
        "assigned_minus_identity_train": float(
            assigned_train.mean() - identity_train.mean()
        ),
        "identity_test_cosine": float(identity_test.mean()),
        "assigned_test_cosine": float(assigned_test.mean()),
        "assigned_minus_identity_test": float(
            assigned_test.mean() - identity_test.mean()
        ),
        "assignment": assignment.tolist(),
    }
    return metrics, assignment


def analyze_variant(
    base: dict[str, Any],
    candidate: dict[str, Any],
    variant: str,
    layers: list[int],
    sampled_neurons: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    if base["config"] != candidate["config"]:
        raise RuntimeError(
            f"architecture mismatch for {variant}: {base['config']} vs {candidate['config']}"
        )
    rows: list[dict[str, Any]] = []
    mlp_matches: dict[int, torch.Tensor] = {}
    qo_assignments: dict[int, torch.Tensor] = {}
    kv_assignments: dict[int, torch.Tensor] = {}
    for layer in layers:
        base_layer = base["layers"][str(layer)]
        candidate_layer = candidate["layers"][str(layer)]
        intermediate = int(base_layer["intermediate_size"])
        neuron_indices = core.fixed_indices(
            intermediate,
            min(sampled_neurons, intermediate),
            core.stable_seed(f"layer-{layer}-neurons", seed),
        )
        mlp_metrics, mlp_best = large_permutation_test(
            base_layer["mlp_train"],
            base_layer["mlp_test"],
            candidate_layer["mlp_train"],
            candidate_layer["mlp_test"],
            neuron_indices,
        )
        mlp_matches[layer] = mlp_best
        rows.append({"variant": variant, "layer": layer, "symmetry": "mlp_neuron", **mlp_metrics})

        qo_metrics, qo_assignment = small_assignment_test(
            base_layer["qo_train"],
            base_layer["qo_test"],
            candidate_layer["qo_train"],
            candidate_layer["qo_test"],
        )
        qo_assignments[layer] = qo_assignment
        rows.append({"variant": variant, "layer": layer, "symmetry": "attention_qo_head", **qo_metrics})

        kv_metrics, kv_assignment = small_assignment_test(
            base_layer["kv_train"],
            base_layer["kv_test"],
            candidate_layer["kv_train"],
            candidate_layer["kv_test"],
        )
        kv_assignments[layer] = kv_assignment
        rows.append({"variant": variant, "layer": layer, "symmetry": "attention_kv_head", **kv_metrics})
    return rows, mlp_matches, qo_assignments, kv_assignments


def finite_mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(np.mean(clean)) if clean else float("nan")


def summarize(rows: list[dict[str, Any]], cross_variant: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"variants": {}, "cross_variant": cross_variant}
    for variant in sorted({row["variant"] for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {}
        for symmetry in sorted({row["symmetry"] for row in variant_rows}):
            selected = [row for row in variant_rows if row["symmetry"] == symmetry]
            numeric_keys = sorted(
                key
                for key, value in selected[0].items()
                if isinstance(value, (int, float)) and key not in {"layer"}
            )
            summary["variants"][variant][symmetry] = {
                key: finite_mean([float(row[key]) for row in selected])
                for key in numeric_keys
            }

    conclusions = []
    for variant, data in summary["variants"].items():
        mlp = data["mlp_neuron"]
        qo = data["attention_qo_head"]
        kv = data["attention_kv_head"]
        if (
            mlp["matched_minus_identity_test"] > 0.15
            and mlp["matched_test_cosine"] > 0.65
            and mlp["identity_top1_fraction"] < 0.5
        ):
            conclusions.append(
                f"{variant}: held-out MLP fingerprints support a real intermediate-neuron permutation."
            )
        else:
            conclusions.append(
                f"{variant}: MLP neuron permutation does not explain most released-code divergence."
            )
        if (
            qo["assigned_minus_identity_test"] > 0.08
            and qo["identity_assignment_fraction"] < 0.8
        ):
            conclusions.append(
                f"{variant}: Q/O attention-head fingerprints support head reordering."
            )
        else:
            conclusions.append(
                f"{variant}: Q/O attention heads remain predominantly in their original identity order."
            )
        if (
            kv["assigned_minus_identity_test"] > 0.08
            and kv["identity_assignment_fraction"] < 0.8
        ):
            conclusions.append(
                f"{variant}: K/V attention-head fingerprints support head reordering."
            )
        else:
            conclusions.append(
                f"{variant}: K/V attention heads remain predominantly in their original identity order."
            )
    summary["conclusions"] = conclusions
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-1.7B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--feature-count", type=int, default=64)
    parser.add_argument("--sampled-neurons", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=98317)
    parser.add_argument("--output-dir", default="symmetry_forensics")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="bonsai-symmetry-") as temporary:
        root = Path(temporary)
        base = extract_repo(
            args.base,
            "base",
            args.layers,
            root / "base",
            args.feature_count,
            args.seed,
        )
        binary = extract_repo(
            args.binary,
            "binary",
            args.layers,
            root / "binary",
            args.feature_count,
            args.seed,
        )
        ternary = extract_repo(
            args.ternary,
            "ternary",
            args.layers,
            root / "ternary",
            args.feature_count,
            args.seed,
        )

    binary_rows, binary_mlp, binary_qo, binary_kv = analyze_variant(
        base,
        binary,
        "binary",
        args.layers,
        args.sampled_neurons,
        args.seed,
    )
    ternary_rows, ternary_mlp, ternary_qo, ternary_kv = analyze_variant(
        base,
        ternary,
        "ternary",
        args.layers,
        args.sampled_neurons,
        args.seed,
    )
    rows = binary_rows + ternary_rows

    cross_variant = {
        "mlp_best_match_agreement": {
            str(layer): float((binary_mlp[layer] == ternary_mlp[layer]).float().mean())
            for layer in args.layers
        },
        "qo_assignment_agreement": {
            str(layer): float((binary_qo[layer] == ternary_qo[layer]).float().mean())
            for layer in args.layers
        },
        "kv_assignment_agreement": {
            str(layer): float((binary_kv[layer] == ternary_kv[layer]).float().mean())
            for layer in args.layers
        },
    }
    summary = summarize(rows, cross_variant)
    summary.update(
        {
            "repositories": {
                "base": args.base,
                "binary": args.binary,
                "ternary": args.ternary,
            },
            "layers": args.layers,
            "feature_count_per_split": args.feature_count,
            "sampled_neurons_per_layer": args.sampled_neurons,
            "seed": args.seed,
            "seconds": time.time() - started,
        }
    )

    pd.DataFrame(rows).to_csv(output / "symmetry_metrics.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output / "interpretation.txt").write_text(
        "\n".join(summary["conclusions"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
