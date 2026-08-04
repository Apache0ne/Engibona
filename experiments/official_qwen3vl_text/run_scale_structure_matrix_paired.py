#!/usr/bin/env python3
"""Paired-randomness entry point for the scale-structure matrix.

Every coefficient within a mode receives the same initialization and sampled
minibatch sequence. Differences are therefore attributable to the structural
penalty rather than random batch order.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "scale_structure_original", HERE / "run_scale_structure_matrix.py"
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load scale-structure matrix")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def paired_run_seed(
    seed: int,
    layers: int,
    coefficients: list[float],
    teacher_steps: int,
    recovery_steps: int,
    batch: int,
    learning_rate: float,
):
    torch.manual_seed(seed)
    train = experiment.make_data(seed + 1, 256, 20)
    validation = experiment.make_data(seed + 2, 96, 32)
    fp = experiment.TinyOfficialQwen3VL(layers=layers, tied=True)
    experiment.train_teacher(fp, train, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    output = {"seed": seed, "layers": layers, "runs": {}}
    for mode in (experiment.QuantMode.BINARY, experiment.QuantMode.TERNARY):
        paired_seed = seed + (
            0 if mode == experiment.QuantMode.BINARY else 500000
        )
        for coefficient in coefficients:
            key = f"{mode.value}_structure_{coefficient:.0e}"
            torch.manual_seed(paired_seed)
            output["runs"][key] = experiment.recover(
                fp,
                teacher,
                train,
                validation,
                mode,
                coefficient,
                recovery_steps,
                batch,
                learning_rate,
            )
    return output


experiment.run_seed = paired_run_seed


if __name__ == "__main__":
    experiment.main()
