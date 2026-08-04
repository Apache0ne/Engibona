#!/usr/bin/env python3
"""Common-row-safe wrapper around the streaming Bonsai forensic core."""
from __future__ import annotations

import gc
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bonsai_streaming_original", HERE / "run_streaming_weight_forensics.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load streaming forensic core")
_original = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_original)
for _name in dir(_original):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_original, _name)


def _trim_sampled_rows(item: dict[str, Any], rows: int, input_width: int, group_size: int) -> None:
    valid_group_count = rows * (input_width // group_size)
    indices = item["indices"].to(torch.long)
    keep = indices < valid_group_count
    if not bool(keep.all()):
        old_count = int(indices.numel())
        item["indices"] = indices[keep]
        for key, value in list(item.items()):
            if key == "indices":
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == old_count:
                item[key] = value[keep]
        item["rows_trimmed_for_common_vocabulary"] = int((~keep).sum())
    item["shape"] = (int(rows), int(input_width))


def collect_matching(repo_id: str, work: Path, base_samples: dict[str, dict[str, Any]], field: str, group_size: int) -> None:
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
                    if tensor.ndim == 2 and len(expected_shape) == 2 and int(tensor.shape[1]) == int(expected_shape[1]) and int(tensor.shape[1]) % group_size == 0:
                        common_rows = min(int(tensor.shape[0]), int(expected_shape[0]))
                        _trim_sampled_rows(base_samples[name], common_rows, int(tensor.shape[1]), group_size)
                        tensor = tensor[:common_rows]
                    else:
                        continue
                indices = base_samples[name]["indices"]
                if indices.numel() == 0:
                    continue
                base_samples[name][field] = sample_groups(tensor, indices, group_size).to(torch.float16)
                seen.add(name)
                del tensor
        local.unlink(missing_ok=True)
        gc.collect()
    missing = sorted(set(base_samples) - seen)
    if missing:
        raise RuntimeError(f"{repo_id}: {len(missing)} selected tensors missing; first={missing[:3]}")
