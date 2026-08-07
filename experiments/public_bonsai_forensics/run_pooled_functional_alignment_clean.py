#!/usr/bin/env python3
"""Entry point that omits undefined diagonal/orthogonal R2 ratios."""
from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "pooled_alignment_original",
    HERE / "run_pooled_functional_alignment.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError("cannot load pooled functional alignment")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def safe_compact_comparison(results):
    output = {}
    for mode in ("binary", "ternary"):
        naive = results[f"naive_{mode}"]["metrics"]
        released = results[f"released_{mode}"]["metrics"]
        layer_rows = {}
        for layer in naive["hidden"]:
            n = naive["hidden"][layer]
            r = released["hidden"][layer]
            orthogonal_r2 = r["projected_orthogonal"]["orthogonal_r2"]
            layer_rows[layer] = {
                "released_minus_naive_raw_cosine": r["raw_cosine"] - n["raw_cosine"],
                "released_minus_naive_diagonal_affine_r2": r["diagonal_affine_r2"] - n["diagonal_affine_r2"],
                "released_minus_naive_projected_orthogonal_r2": (
                    orthogonal_r2
                    - n["projected_orthogonal"]["orthogonal_r2"]
                ),
                "released_diagonal_fraction_of_orthogonal_r2": (
                    r["diagonal_affine_r2"] / orthogonal_r2
                    if orthogonal_r2 > 0
                    else None
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


experiment.compact_comparison = safe_compact_comparison


if __name__ == "__main__":
    experiment.main()
