#!/usr/bin/env python3
"""Byte-range public-weight forensics for the actual Qwen3.6-27B source.

The unpacked Qwen3.6/Bonsai checkpoints are tens of gigabytes each. Downloading
all three checkpoints merely to retain a small deterministic sample is wasteful
and can exceed hosted-runner time limits. This experiment reads safetensors
headers and then fetches one deterministic contiguous g128 block per language
matrix with HTTP byte ranges.

Coverage includes every two-dimensional, g128-compatible language-model tensor
shared by:

* Qwen/Qwen3.6-27B;
* prism-ml/Bonsai-27B-unpacked;
* prism-ml/Ternary-Bonsai-27B-unpacked.

The vision tower and one-dimensional high-precision state are excluded. Metrics
are identical to the smaller-scale public-weight forensic suite, with additional
hybrid-architecture family and layer summaries. Final checkpoints cannot reveal
the private optimizer, corpus, learning-rate schedule, token count, or exact
training trajectory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import HfApi, hf_hub_url


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_core", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F32": (torch.float32, 4),
}
_THREAD_LOCAL = threading.local()


def session() -> requests.Session:
    value = getattr(_THREAD_LOCAL, "session", None)
    if value is None:
        value = requests.Session()
        value.headers.update({
            "User-Agent": "Engibona-public-weight-forensics/1.0",
        })
        token = os.environ.get("HF_TOKEN")
        if token:
            value.headers.update({"Authorization": f"Bearer {token}"})
        _THREAD_LOCAL.session = value
    return value


def request_bytes(
    url: str,
    start: int | None = None,
    end: int | None = None,
    attempts: int = 6,
) -> bytes:
    headers = {}
    expected = None
    if start is not None or end is not None:
        if start is None or end is None or end < start:
            raise ValueError("range requires valid start and end")
        headers["Range"] = f"bytes={start}-{end}"
        expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session().get(
                url,
                headers=headers,
                timeout=(30, 300),
                allow_redirects=True,
            )
            if expected is None:
                response.raise_for_status()
            elif response.status_code != 206:
                raise RuntimeError(
                    f"range request returned {response.status_code}, expected 206"
                )
            content = response.content
            if expected is not None and len(content) != expected:
                raise RuntimeError(
                    f"range length {len(content)} != expected {expected}"
                )
            return content
        except Exception as error:
            last_error = error
            if attempt == attempts:
                break
            time.sleep(min(30.0, 1.5**attempt + random.random()))
    raise RuntimeError(f"request failed after {attempts} attempts: {last_error}")


def stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") ^ seed) & 0x7FFFFFFF


def is_language_matrix_name(name: str) -> bool:
    if name == "lm_head.weight":
        return True
    if not name.startswith("model.language_model."):
        return False
    if not name.endswith(".weight"):
        return False
    lowered = name.lower()
    if "norm" in lowered:
        return False
    return True


def layer_index(name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def module_type(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if name == "lm_head.weight":
        return "lm_head"
    for family in ("linear_attn", "self_attn", "mlp"):
        marker = f".{family}."
        if marker in name:
            component = name.split(marker, 1)[1].rsplit(".weight", 1)[0]
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


class RemoteSafeTensorsRepository:
    """Resolve tensors and read exact byte ranges without full shard downloads."""

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
        self.weight_map: dict[str, str] | None = None
        if indexes:
            raw = request_bytes(
                hf_hub_url(
                    repo_id=repo_id,
                    filename=indexes[0],
                    revision="main",
                )
            )
            index = json.loads(raw.decode("utf-8"))
            self.weight_map = {
                str(key): str(value)
                for key, value in index["weight_map"].items()
            }
            self.keys = set(self.weight_map)
        else:
            self.keys = set()
        self._headers: dict[str, tuple[str, int, dict[str, Any]]] = {}

    def header(self, filename: str) -> tuple[str, int, dict[str, Any]]:
        cached = self._headers.get(filename)
        if cached is not None:
            return cached
        url = hf_hub_url(
            repo_id=self.repo_id,
            filename=filename,
            revision="main",
        )
        length = struct.unpack("<Q", request_bytes(url, 0, 7))[0]
        if length <= 0 or length > 100_000_000:
            raise RuntimeError(
                f"{self.repo_id}/{filename}: invalid header length {length}"
            )
        raw = request_bytes(url, 8, 8 + length - 1)
        header = json.loads(raw.decode("utf-8").rstrip(" \t\r\n\x00"))
        cached = (url, 8 + length, header)
        self._headers[filename] = cached
        if self.weight_map is None:
            self.keys.update(
                key for key in header if key != "__metadata__"
            )
        return cached

    def preload_headers(self) -> None:
        for filename in self.tensor_files:
            self.header(filename)

    def resolve(
        self,
        key: str,
    ) -> tuple[str, str, int, dict[str, Any]]:
        if self.weight_map is not None:
            filename = self.weight_map.get(key)
            if filename is None:
                raise KeyError(f"{self.repo_id}: missing tensor {key}")
            url, data_start, header = self.header(filename)
            return filename, url, data_start, header[key]
        for filename in self.tensor_files:
            url, data_start, header = self.header(filename)
            if key in header:
                return filename, url, data_start, header[key]
        raise KeyError(f"{self.repo_id}: missing tensor {key}")

    def descriptor(self, key: str) -> dict[str, Any]:
        filename, url, data_start, item = self.resolve(key)
        shape = tuple(int(value) for value in item["shape"])
        dtype_name = str(item["dtype"])
        if dtype_name not in DTYPES:
            raise RuntimeError(
                f"{self.repo_id}/{key}: unsupported dtype {dtype_name}"
            )
        offsets = tuple(int(value) for value in item["data_offsets"])
        return {
            "filename": filename,
            "url": url,
            "data_start": data_start,
            "shape": shape,
            "dtype": dtype_name,
            "offsets": offsets,
        }

    def read_flat_values(
        self,
        descriptor: dict[str, Any],
        start_value: int,
        value_count: int,
    ) -> torch.Tensor:
        dtype, item_size = DTYPES[descriptor["dtype"]]
        shape = descriptor["shape"]
        total_values = math.prod(shape)
        if start_value < 0 or value_count <= 0:
            raise ValueError("invalid flat value range")
        if start_value + value_count > total_values:
            raise ValueError(
                f"flat range {start_value}:{start_value + value_count} "
                f"exceeds {total_values}"
            )
        tensor_start = descriptor["data_start"] + descriptor["offsets"][0]
        byte_start = tensor_start + start_value * item_size
        byte_end = byte_start + value_count * item_size - 1
        raw = bytearray(
            request_bytes(descriptor["url"], byte_start, byte_end)
        )
        return torch.frombuffer(raw, dtype=dtype).clone().float()


def common_descriptor(
    name: str,
    repositories: dict[str, RemoteSafeTensorsRepository],
    group_size: int,
) -> dict[str, Any] | None:
    descriptors = {
        label: repository.descriptor(name)
        for label, repository in repositories.items()
    }
    shapes = {label: value["shape"] for label, value in descriptors.items()}
    if any(len(shape) != 2 for shape in shapes.values()):
        return None
    widths = {shape[1] for shape in shapes.values()}
    if len(widths) != 1:
        return None
    width = next(iter(widths))
    if width % group_size:
        return None
    common_rows = min(shape[0] for shape in shapes.values())
    if common_rows <= 0:
        return None
    return {
        "name": name,
        "shape": (common_rows, width),
        "total_groups": common_rows * (width // group_size),
        "descriptors": descriptors,
        "original_shapes": {
            label: list(shape) for label, shape in shapes.items()
        },
    }


def choose_group_block(
    name: str,
    total_groups: int,
    requested_groups: int,
    seed: int,
) -> tuple[int, int]:
    count = min(total_groups, requested_groups)
    if count <= 0:
        raise ValueError("tensor has no groups")
    maximum_start = total_groups - count
    if maximum_start <= 0:
        return 0, count
    generator = np.random.default_rng(stable_seed(name, seed))
    return int(generator.integers(0, maximum_start + 1)), count


def read_tensor_sample(
    item: dict[str, Any],
    repositories: dict[str, RemoteSafeTensorsRepository],
    group_size: int,
    groups_per_tensor: int,
    seed: int,
) -> dict[str, Any]:
    name = item["name"]
    start_group, group_count = choose_group_block(
        name,
        item["total_groups"],
        groups_per_tensor,
        seed,
    )
    start_value = start_group * group_size
    value_count = group_count * group_size
    values = {
        label: repository.read_flat_values(
            item["descriptors"][label],
            start_value,
            value_count,
        ).reshape(group_count, group_size)
        for label, repository in repositories.items()
    }
    return {
        "name": name,
        "shape": item["shape"],
        "indices": torch.arange(
            start_group,
            start_group + group_count,
            dtype=torch.long,
        ),
        "base": values["base"].to(torch.float16),
        "binary": values["binary"].to(torch.float16),
        "ternary": values["ternary"].to(torch.float16),
        "sample_start_group": start_group,
        "sample_group_count": group_count,
        "original_shapes": item["original_shapes"],
    }


def weighted_family_summary(frame: pd.DataFrame) -> pd.DataFrame:
    identifiers = {
        "tensor", "layer", "module", "family", "shape",
        "groups_sampled", "weights_sampled", "sample_start_group",
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
                values = pd.to_numeric(group[column], errors="coerce")
                if values.notna().any():
                    row[column] = core.weighted_mean(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def layer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["layer"] >= 0]
    metrics = [
        "binary_sign_agreement_base",
        "binary_scale_corr_mean_abs_base",
        "binary_actual_over_naive_nmse",
        "ternary_zero_rate",
        "ternary_code_agreement_naive",
        "ternary_scale_corr_naive",
        "ternary_actual_over_naive_nmse",
        "binary_ternary_sign_agreement_nonzero",
        "binary_ternary_scale_corr",
    ]
    rows = []
    for layer, group in selected.groupby("layer"):
        row: dict[str, Any] = {
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


def correlation(left: pd.Series, right: pd.Series) -> float:
    x = pd.to_numeric(left, errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(right, errors="coerce").to_numpy(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def interpretation(
    summary: dict[str, Any],
    families: pd.DataFrame,
    layers: pd.DataFrame,
) -> list[str]:
    metrics = summary["metrics"]
    lines = [
        "These conclusions describe released Qwen3.6-27B language-weight geometry, not the private optimizer, data, schedule, or training trajectory."
    ]
    if metrics["binary_sign_agreement_base"] < 0.90:
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
            "Both 27B alphabets sacrifice raw weight MSE relative to naive projection, supporting behavior-oriented recovery."
        )
    for family in ("linear_attention", "full_attention", "mlp"):
        selected = families[families["family"] == family]
        if not selected.empty:
            row = selected.iloc[0]
            lines.append(
                f"{family}: binary sign agreement={row['binary_sign_agreement_base']:.4f}, ternary naive-code agreement={row['ternary_code_agreement_naive']:.4f}, binary NMSE ratio={row['binary_actual_over_naive_nmse']:.2f}x."
            )
    if len(layers):
        binary_depth = correlation(
            layers["layer"], layers["binary_actual_over_naive_nmse"]
        )
        ternary_depth = correlation(
            layers["layer"], layers["ternary_actual_over_naive_nmse"]
        )
        lines.append(
            f"Depth correlation of released/naive NMSE is {binary_depth:.4f} binary and {ternary_depth:.4f} ternary."
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--binary", default="prism-ml/Bonsai-27B-unpacked")
    parser.add_argument("--ternary", default="prism-ml/Ternary-Bonsai-27B-unpacked")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="public_bonsai_forensics_27b")
    args = parser.parse_args()

    core.module_type = module_type
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repositories = {
        "base": RemoteSafeTensorsRepository(args.base),
        "binary": RemoteSafeTensorsRepository(args.binary),
        "ternary": RemoteSafeTensorsRepository(args.ternary),
    }
    for repository in repositories.values():
        repository.preload_headers()

    common_keys = set.intersection(
        *(repository.keys for repository in repositories.values())
    )
    candidates = []
    rejected = defaultdict(int)
    for name in sorted(common_keys):
        if not is_language_matrix_name(name):
            continue
        try:
            item = common_descriptor(name, repositories, args.group_size)
        except Exception:
            rejected["descriptor_error"] += 1
            raise
        if item is None:
            rejected["incompatible_shape_or_width"] += 1
            continue
        candidates.append(item)
    if not candidates:
        raise RuntimeError("no common Qwen3.6 language matrices selected")

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        future_to_name = {
            executor.submit(
                read_tensor_sample,
                item,
                repositories,
                args.group_size,
                args.groups_per_tensor,
                args.seed,
            ): item["name"]
            for item in candidates
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                samples.append(future.result())
            except Exception as error:
                failures.append({"tensor": name, "error": repr(error)})
            completed += 1
            if completed % 50 == 0 or completed == len(candidates):
                print(
                    f"completed {completed}/{len(candidates)} tensors; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"failed to read {len(failures)} tensors; see failures.json"
        )

    rows = []
    for item in sorted(samples, key=lambda value: value["name"]):
        row = core.analyze_tensor(item["name"], item)
        row["family"] = layer_family(item["name"])
        row["sample_start_group"] = item["sample_start_group"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tensor_metrics.csv", index=False)

    module_rows = []
    identifiers = {
        "tensor", "layer", "module", "family", "shape",
        "groups_sampled", "weights_sampled", "sample_start_group",
    }
    for module, group in frame.groupby("module", dropna=False):
        row: dict[str, Any] = {
            "module": module,
            "tensor_count": int(len(group)),
            "groups_sampled": int(group["groups_sampled"].sum()),
        }
        for column in frame.columns:
            if column in identifiers:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if values.notna().any():
                row[column] = core.weighted_mean(group, column)
        module_rows.append(row)
    pd.DataFrame(module_rows).to_csv(
        output / "module_metrics.csv", index=False
    )

    families = weighted_family_summary(frame)
    families.to_csv(output / "family_metrics.csv", index=False)
    layers = layer_summary(frame)
    layers.to_csv(output / "layer_metrics.csv", index=False)

    numeric_frame = frame.drop(
        columns=["family", "sample_start_group"], errors="ignore"
    )
    summary = core.summarize(numeric_frame)
    summary.update({
        "repositories": {
            "base": args.base,
            "binary": args.binary,
            "ternary": args.ternary,
        },
        "architecture": (
            "Qwen3.6 hybrid language model; vision tower and 1D "
            "high-precision state excluded"
        ),
        "sampling": "one deterministic contiguous g128 block per tensor via HTTP byte ranges",
        "group_size": args.group_size,
        "groups_per_tensor_limit": args.groups_per_tensor,
        "workers": args.workers,
        "seed": args.seed,
        "language_matrix_tensor_count": int(len(frame)),
        "candidate_tensor_count": int(len(candidates)),
        "rejected": dict(rejected),
        "family_tensor_counts": {
            str(key): int(value)
            for key, value in frame.groupby("family").size().items()
        },
        "layer_count_observed": int(
            layers["layer"].max() + 1 if len(layers) else 0
        ),
        "depth_correlations": {
            "binary_sign_agreement": correlation(
                layers["layer"], layers["binary_sign_agreement_base"]
            ) if len(layers) else float("nan"),
            "binary_nmse_ratio": correlation(
                layers["layer"], layers["binary_actual_over_naive_nmse"]
            ) if len(layers) else float("nan"),
            "ternary_code_agreement": correlation(
                layers["layer"], layers["ternary_code_agreement_naive"]
            ) if len(layers) else float("nan"),
            "ternary_zero_rate": correlation(
                layers["layer"], layers["ternary_zero_rate"]
            ) if len(layers) else float("nan"),
            "ternary_nmse_ratio": correlation(
                layers["layer"], layers["ternary_actual_over_naive_nmse"]
            ) if len(layers) else float("nan"),
        },
        "seconds": time.time() - started,
    })
    summary["interpretation"] = interpretation(summary, families, layers)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    (output / "interpretation.txt").write_text(
        "\n".join(summary["interpretation"]) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
