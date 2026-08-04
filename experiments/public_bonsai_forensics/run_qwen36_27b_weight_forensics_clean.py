#!/usr/bin/env python3
"""Clean-summary entry point for Qwen3.6-27B weight forensics."""
from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "qwen36_27b_original",
    HERE / "run_qwen36_27b_weight_forensics.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load Qwen3.6-27B forensic experiment")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)

_original_summarize = experiment.core.summarize


def summarize_without_family(frame):
    return _original_summarize(frame.drop(columns=["family"], errors="ignore"))


experiment.core.summarize = summarize_without_family


if __name__ == "__main__":
    experiment.main()
