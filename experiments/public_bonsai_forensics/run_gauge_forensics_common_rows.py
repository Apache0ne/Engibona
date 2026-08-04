#!/usr/bin/env python3
"""Common-row-safe entry point for Bonsai gauge forensics."""
from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent

COMMON_SPEC = importlib.util.spec_from_file_location(
    "streaming_common_rows", HERE / "streaming_common_rows.py"
)
if COMMON_SPEC is None or COMMON_SPEC.loader is None:
    raise ImportError("cannot load common-row streaming core")
common = importlib.util.module_from_spec(COMMON_SPEC)
COMMON_SPEC.loader.exec_module(common)

EXPERIMENT_SPEC = importlib.util.spec_from_file_location(
    "gauge_forensics_original", HERE / "run_gauge_forensics.py"
)
if EXPERIMENT_SPEC is None or EXPERIMENT_SPEC.loader is None:
    raise ImportError("cannot load gauge forensic experiment")
experiment = importlib.util.module_from_spec(EXPERIMENT_SPEC)
EXPERIMENT_SPEC.loader.exec_module(experiment)
experiment.core = common


if __name__ == "__main__":
    experiment.main()
