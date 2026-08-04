#!/usr/bin/env python3
"""Common-row-safe entry point for tied public Bonsai weight forensics."""
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

TIED_SPEC = importlib.util.spec_from_file_location(
    "streaming_tied_original", HERE / "run_streaming_weight_forensics_tied.py"
)
if TIED_SPEC is None or TIED_SPEC.loader is None:
    raise ImportError("cannot load tied forensic entry point")
tied = importlib.util.module_from_spec(TIED_SPEC)
TIED_SPEC.loader.exec_module(tied)
tied.core = common


if __name__ == "__main__":
    tied.main()
