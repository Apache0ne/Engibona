#!/usr/bin/env python3
"""Build the compact JSON and Markdown report for the 600x3 benchmark suite."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW_FILES = {
    "surrogates": "results_surrogates_600x3.json",
    "profiles": "results_profiles_600x3.json",
    "losses": "results_losses_600x3.json",
    "ternary_schedules": "results_ternary_schedules_600x3.json",
    "embedding_policies": "results_embedding_policies_600x3.json",
    "scale_structure": "results_scale_structure_600x3.json",
    "shared_embedding": "results_shared_embedding_600x3.json",
    "qwen36_hybrid": "results_qwen36_hybrid_600x3.json",
}
TEACHER_FILES = {
    "profiles": "results_teacher_profiles_600x3.json",
    "scale_structure": "results_teacher_scale_structure_600x3.json",
    "shared_embedding": "results_teacher_shared_embedding_600x3.json",
}


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha256(name: str) -> str:
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def require_contract(payload: dict[str, Any], name: str) -> None:
    arguments = payload["arguments"]
    if arguments["seeds"] != 3:
        raise ValueError(f"{name}: expected three seeds")
    layers = arguments["layers"]
    if layers != [4]:
        raise ValueError(f"{name}: expected four layers, got {layers}")
    if arguments["teacher_steps"] != 600:
        raise ValueError(f"{name}: expected 600 teacher steps")
    if arguments["recovery_steps"] != 600:
        raise ValueError(f"{name}: expected 600 recovery steps")


def fidelity(teacher_kl: float) -> float:
    return math.exp(-float(teacher_kl))


def fmt(mean: float, spread: float | None = None, digits: int = 4) -> str:
    text = f"{float(mean):.{digits}f}"
    if spread is not None:
        text += f" +/- {float(spread):.{digits}f}"
    return text


def pct(mean: float, spread: float | None = None, digits: int = 2) -> str:
    return fmt(100.0 * float(mean), None if spread is None else 100.0 * float(spread), digits) + "%"


def aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["by_depth"]["4"]["aggregate"]


def teacher_baselines(data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    output = {
        "surrogates": aggregate(data["surrogates"])["teacher"],
        "losses": aggregate(data["losses"])["teacher"],
        "ternary_schedules": aggregate(data["ternary_schedules"])["teacher"],
        "embedding_policies": aggregate(data["embedding_policies"])["teacher"],
        "qwen36_hybrid": data["qwen36_hybrid"]["by_depth"]["4"]["teacher_baseline"],
    }
    for family, filename in TEACHER_FILES.items():
        output[family] = load(filename)["aggregate"]
    return output


def validate_exactness(data: dict[str, dict[str, Any]]) -> None:
    for family in ("surrogates", "profiles", "losses", "ternary_schedules", "embedding_policies", "qwen36_hybrid"):
        for method, metrics in aggregate(data[family]).items():
            if (
                method != "teacher"
                and "all_exact_alphabet" in metrics
                and not metrics["all_exact_alphabet"]
            ):
                raise ValueError(f"{family}/{method}: invalid final alphabet")
    for mode in data["scale_structure"]["summary"].values():
        for method, metrics in mode["methods"].items():
            if not metrics["all_exact_alphabet"]:
                raise ValueError(f"scale_structure/{method}: invalid final alphabet")
    for method, metrics in aggregate(data["shared_embedding"]).items():
        if not metrics["all_exact_alphabet"]:
            raise ValueError(f"shared_embedding/{method}: invalid final alphabet")


def winner(rows: dict[str, dict[str, Any]], prefix: str, metric: str = "teacher_kl_mean") -> str:
    selected = {name: values for name, values in rows.items() if name.startswith(prefix)}
    return min(selected, key=lambda name: selected[name][metric])


def selections(data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    surrogate = aggregate(data["surrogates"])
    profile = aggregate(data["profiles"])
    loss = aggregate(data["losses"])
    schedule = aggregate(data["ternary_schedules"])
    embedding = aggregate(data["embedding_policies"])
    shared = aggregate(data["shared_embedding"])
    qwen36 = aggregate(data["qwen36_hybrid"])
    return {
        "surrogates": {
            "binary": winner(surrogate, "binary_"),
            "ternary": winner(surrogate, "ternary_"),
        },
        "profiles": {
            "binary": winner(profile, "binary_"),
            "ternary": winner(profile, "ternary_"),
        },
        "losses": min(
            (name for name in loss if name != "teacher"),
            key=lambda name: loss[name]["teacher_kl_mean"],
        ),
        "ternary_schedules": min(
            (name for name in schedule if name != "teacher"),
            key=lambda name: schedule[name]["teacher_kl_mean"],
        ),
        "embedding_policies": {
            "binary": winner(embedding, "binary_"),
            "ternary": winner(embedding, "ternary_"),
        },
        "scale_structure": {
            mode: values["winner"]
            for mode, values in data["scale_structure"]["summary"].items()
        },
        "shared_embedding": min(
            shared,
            key=lambda name: shared[name]["combined_teacher_kl_mean"],
        ),
        "qwen36_hybrid": {
            "binary": winner(qwen36, "binary_"),
            "ternary": winner(qwen36, "ternary_"),
        },
    }


def standard_table(title: str, rows: dict[str, dict[str, Any]], names: list[str]) -> list[str]:
    output = [
        f"### {title}",
        "",
        "| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        values = rows[name]
        output.append(
            "| " + " | ".join(
                [
                    name,
                    fmt(values["ce_mean"], values["ce_pstdev"]),
                    pct(values["accuracy_mean"], values["accuracy_pstdev"]),
                    fmt(values["teacher_kl_mean"], values["teacher_kl_pstdev"]),
                    pct(fidelity(values["teacher_kl_mean"]), digits=2),
                    fmt(values["hidden_cosine_mean"], values["hidden_cosine_pstdev"]),
                ]
            ) + " |"
        )
    return output + [""]


def build_markdown(data: dict[str, dict[str, Any]], compact: dict[str, Any]) -> str:
    teachers = compact["teacher_baselines"]
    lines = [
        "# Long all-method 600-step CPU benchmark",
        "",
        "## Contract",
        "",
        "Every recovery row used 600 optimizer steps, three seeds, four decoder layers, batch 12, FP32 CPU execution, and exact contiguous g128 codes. Each teacher used 600 FP32 training steps. Qwen3-VL and Qwen3.6 results use different architectures and seed sets, so rank methods only within a table.",
        "",
        "The Qwen3-VL miniature uses the official Hugging Face decoder implementation. The Qwen3.6 miniature uses the official `qwen3_5_text` hybrid pattern: three linear-attention layers plus one full-attention layer. These are architecture-real miniature experiments, not pretrained 27B intelligence benchmarks.",
        "",
        "`Fidelity proxy = exp(-Teacher KL)`. It is a local distribution-agreement proxy, not an intelligence-retention percentage.",
        "",
        "## FP32 teacher references",
        "",
        "| Family | Architecture | FP32 CE ↓ | FP32 accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family in RAW_FILES:
        values = teachers[family]
        architecture = "Qwen3.6 hybrid" if family == "qwen36_hybrid" else "Qwen3-VL"
        lines.append(
            f"| {family} | {architecture} | {fmt(values['ce_mean'], values['ce_pstdev'])} | {pct(values['accuracy_mean'], values['accuracy_pstdev'])} | ~0 | 1.0000 |"
        )
    lines += [
        "",
        "Teacher validation quality varies by seed family. The Qwen3.6 teacher remained at chance accuracy despite the longer run, so its KL/cosine rows test architectural recovery mechanics only.",
        "",
        "## Recovery surrogate matrix",
        "",
    ]
    surrogate = aggregate(data["surrogates"])
    lines += standard_table(
        "Naive, exact-hard, categorical, and CAT-Q",
        surrogate,
        [name for name in surrogate if name != "teacher"],
    )

    profile = aggregate(data["profiles"])
    lines += [
        "### Uniform versus public layer/module pressure",
        "",
        "| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement ↔ target | Layer RMSE ↓ | Module RMSE ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in profile.items():
        lines.append(
            f"| {name} | {fmt(values['teacher_kl_mean'], values['teacher_kl_pstdev'])} | {pct(fidelity(values['teacher_kl_mean']))} | {fmt(values['hidden_cosine_mean'], values['hidden_cosine_pstdev'])} | {pct(values['overall_code_change_mean'], values['overall_code_change_pstdev'])} | {fmt(values['layer_profile_rmse_mean'], values['layer_profile_rmse_pstdev'])} | {fmt(values['module_profile_rmse_mean'], values['module_profile_rmse_pstdev'])} |"
        )
    lines += [""]

    loss = aggregate(data["losses"])
    lines += [
        "### Binary recovery loss objectives",
        "",
        "| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in loss.items():
        if name == "teacher":
            continue
        lines.append(
            f"| {name} | {fmt(values['teacher_kl_mean'], values['teacher_kl_pstdev'])} | {pct(fidelity(values['teacher_kl_mean']))} | {fmt(values['hidden_cosine_mean'], values['hidden_cosine_pstdev'])} | {pct(values['code_change_rate_mean'])} |"
        )
    lines += [""]

    schedule = aggregate(data["ternary_schedules"])
    lines += [
        "### Ternary soft-to-hard schedules",
        "",
        "| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement ↔ target | Zero ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in schedule.items():
        if name == "teacher":
            continue
        lines.append(
            f"| {name} | {fmt(values['teacher_kl_mean'], values['teacher_kl_pstdev'])} | {pct(fidelity(values['teacher_kl_mean']))} | {fmt(values['hidden_cosine_mean'], values['hidden_cosine_pstdev'])} | {pct(values['code_change_rate_mean'], values['code_change_rate_pstdev'])} | {pct(values['zero_ratio_mean'], values['zero_ratio_pstdev'])} |"
        )
    lines += [""]

    embedding = aggregate(data["embedding_policies"])
    lines += standard_table(
        "Embedding policies",
        embedding,
        [name for name in embedding if name != "teacher"],
    )

    lines += [
        "### Scale-structure regularization",
        "",
        "| Mode | Coefficient | Teacher KL ↓ | Paired KL delta ↓ | Improved seeds | Exact p | Additive R2 ↑ | Code movement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, group in data["scale_structure"]["summary"].items():
        for values in group["methods"].values():
            lines.append(
                f"| {mode} | {values['coefficient']:g} | {values['teacher_kl_mean']:.4f} | {values['paired_kl_difference']['mean']:+.4f} | {values['improved_pairs']}/{values['total_pairs']} | {values['exact_sign_randomization_p_two_sided']:.3f} | {values['scale_additive_r2_mean']:.4f} | {pct(values['code_change_fraction_mean'])} |"
            )
    lines += [""]

    shared = aggregate(data["shared_embedding"])
    lines += [
        "### Shared binary-codebook/ternary-mask policies",
        "",
        "| Policy | Combined KL ↓ | Binary KL ↓ | Ternary KL ↓ | Binary cosine ↑ | Ternary cosine ↑ | Exact mask relation ↑ | KL / independent ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in shared.items():
        lines.append(
            f"| {name} | {fmt(values['combined_teacher_kl_mean'], values['combined_teacher_kl_pstdev'])} | {fmt(values['binary_teacher_kl_mean'], values['binary_teacher_kl_pstdev'])} | {fmt(values['ternary_teacher_kl_mean'], values['ternary_teacher_kl_pstdev'])} | {fmt(values['binary_hidden_cosine_mean'], values['binary_hidden_cosine_pstdev'])} | {fmt(values['ternary_hidden_cosine_mean'], values['ternary_hidden_cosine_pstdev'])} | {pct(values['relation_exact_mask_fraction_mean'])} | {values['combined_kl_over_independent']:.4f} |"
        )
    lines += [""]

    qwen36 = aggregate(data["qwen36_hybrid"])
    lines += standard_table(
        "Official Qwen3.6 hybrid architecture",
        qwen36,
        list(qwen36),
    )

    select = compact["selection"]
    lines += [
        "## Main findings",
        "",
        f"- Qwen3-VL behavior winners: binary `{select['surrogates']['binary']}` and ternary `{select['surrogates']['ternary']}`.",
        f"- Public pressure helped both binary surrogates slightly; `{select['profiles']['binary']}` won the binary profile table. Uniform `{select['profiles']['ternary']}` won ternary behavior, while exact-hard retained more code movement.",
        f"- `{select['losses']}` had the lowest binary loss-ablation KL. CE-only was clearly worse at teacher retention.",
        f"- `{select['ternary_schedules']}` had the best ternary schedule KL/cosine, but moved fewer codes than sustained exact-hard recovery.",
        f"- Long embedding behavior favored `{select['embedding_policies']['binary']}` and `{select['embedding_policies']['ternary']}` within their respective modes.",
        "- Binary scale coefficient `0.1` improved all three paired seeds, but the exact two-sided p-value is `0.25`; retain it as experimental. Ternary coefficient `10` won by only `0.00175` KL and improved one of three seeds, so it is not a reliable default.",
        f"- `{select['shared_embedding']}` produced the best combined shared-pair KL and an exact released-format embedding relation. It improved combined KL by about 0.81% versus independent recovery, but slightly worsened binary KL while improving ternary KL.",
        f"- On Qwen3.6, long recovery selected binary `{select['qwen36_hybrid']['binary']}` and ternary `{select['qwen36_hybrid']['ternary']}`. The chance-level FP32 teacher prevents intelligence-retention claims.",
        "- The stronger Qwen3-VL teachers drive fidelity proxies far below the earlier weak-teacher ~98% values. That confirms `exp(-KL)` must not be reported as percent intelligence kept.",
        "",
        "## Coverage boundary",
        "",
        "This suite covers every runnable official-architecture recovery matrix currently present on `main`, plus the branch-preserved scale and shared-embedding runners restored here: 49 named low-bit configurations across eight families. Repeated baselines remain because each family uses its own deterministic seed set.",
        "",
        "Not assigned artificial CE/KL rows: hidden gauges, head/neuron permutations, static affine alignment, and checkpoint lineage are forensic hypotheses rather than trainable recovery methods. Network-dependent 1.7B/4B/8B/27B checkpoint studies retain their existing reports. Fisher/metric-projection primitives and inactive config flags do not yet have an official-architecture end-to-end runner; they are not mislabeled as completed long benchmarks.",
        "",
        "## Provenance",
        "",
        "| Raw file | Runtime | SHA-256 |",
        "|---|---:|---|",
    ]
    for family, values in compact["provenance"].items():
        lines.append(
            f"| `{values['file']}` | {values['seconds']:.2f}s | `{values['sha256']}` |"
        )
    lines += [
        "",
        f"Total recorded matrix runtime: **{compact['total_recorded_seconds'] / 3600.0:.2f} core-hours-equivalent** across concurrently executed CPU processes. No GitHub Actions were used.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = {family: load(filename) for family, filename in RAW_FILES.items()}
    for family, payload in data.items():
        require_contract(payload, family)
    validate_exactness(data)
    teachers = teacher_baselines(data)
    provenance = {
        family: {
            "file": filename,
            "sha256": sha256(filename),
            "seconds": float(data[family]["seconds"]),
        }
        for family, filename in RAW_FILES.items()
    }
    provenance.update(
        {
            "teacher_replay_" + family: {
                "file": filename,
                "sha256": sha256(filename),
                "seconds": float(load(filename)["seconds"]),
            }
            for family, filename in TEACHER_FILES.items()
        }
    )
    compact = {
        "benchmark_contract": {
            "seeds": 3,
            "layers": 4,
            "teacher_steps": 600,
            "recovery_steps": 600,
            "batch": 12,
            "precision": "FP32 CPU",
            "group_size": 128,
            "named_low_bit_configurations": 49,
        },
        "teacher_baselines": teachers,
        "selection": selections(data),
        "exactness": {
            "all_recorded_checks_passed": True,
            "embedding_policy_runner_recorded_field": False,
            "embedding_policy_exactness_enforced_by_shared_quant_module": True,
        },
        "provenance": provenance,
        "total_recorded_seconds": sum(item["seconds"] for item in provenance.values()),
    }
    summary_path = HERE / "results_long_all_methods_summary.json"
    summary_path.write_text(
        json.dumps(compact, indent=2, allow_nan=True), encoding="utf-8"
    )
    report_path = ROOT / "docs" / "LONG_ALL_METHODS_600_STEP.md"
    report_path.write_text(build_markdown(data, compact), encoding="utf-8")
    print(json.dumps(compact["selection"], indent=2))


if __name__ == "__main__":
    main()
