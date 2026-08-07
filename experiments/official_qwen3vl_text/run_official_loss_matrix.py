#!/usr/bin/env python3
"""Equal-budget recovery-loss matrix on official Qwen3-VL text code."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules
from engibona.modules_tied import TiedGroupQuantizedLMHead
from run_official_cpu_smoke import TinyOfficialQwen3VL, evaluate, kd_loss, make_data, train_teacher


def unique_states(replaced):
    states = {}
    for module in replaced.values():
        state = module.embedding if isinstance(module, TiedGroupQuantizedLMHead) else module
        if hasattr(state, "hard_codes_and_scales"):
            states[id(state)] = state
    return list(states.values())


def normalized_hidden_mse(teacher_hidden, student_hidden):
    losses = []
    for teacher_state, student_state in zip(teacher_hidden, student_hidden):
        target = teacher_state.detach().float()
        prediction = student_state.float()
        denominator = target.square().mean().clamp_min(1e-8)
        losses.append((prediction - target).square().mean() / denominator)
    return torch.stack(losses).mean()


def recover(model, teacher, data, weights, steps, batch):
    student = copy.deepcopy(model)
    config = EngibonaConfig(
        mode=QuantMode.BINARY,
        relaxation="hard_ste",
        export_strategy="trained",
        scale_tether_weight=1e-5,
    )
    replaced = replace_linear_modules(
        student,
        config,
        include_embeddings=True,
        preserve_tied_weights=True,
    )
    states = unique_states(replaced)
    initial_codes = [state.hard_codes_and_scales()[0].detach().cpu().clone() for state in states]
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=7e-4, betas=(0.9, 0.95), weight_decay=0.001
    )
    input_ids, labels = data
    student.train()
    for step in range(steps):
        for state in states:
            state.set_schedule(step, steps)
        indices = torch.randint(0, len(input_ids), (batch,))
        batch_ids = input_ids[indices]
        batch_labels = labels[indices]
        need_hidden = weights["hidden"] > 0
        with torch.no_grad():
            teacher_output = teacher(batch_ids, output_hidden_states=need_hidden)
        student_output = student(batch_ids, output_hidden_states=need_hidden)
        if need_hidden:
            teacher_logits, teacher_hidden = teacher_output
            logits, student_hidden = student_output
        else:
            teacher_logits = teacher_output
            logits = student_output

        terms = []
        if weights["kd"] > 0:
            terms.append(weights["kd"] * kd_loss(teacher_logits, logits))
        if weights["ce"] > 0:
            terms.append(
                weights["ce"]
                * F.cross_entropy(logits.flatten(0, 1), batch_labels.flatten())
            )
        if weights["hidden"] > 0:
            terms.append(
                weights["hidden"]
                * normalized_hidden_mse(teacher_hidden, student_hidden)
            )
        loss = torch.stack(terms).sum()
        loss = loss + torch.stack([state.regularization_loss() for state in states]).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

    changed = 0
    total = 0
    exact = True
    for initial, state in zip(initial_codes, states):
        codes = state.hard_codes_and_scales()[0].detach().cpu()
        changed += int((codes != initial).sum())
        total += codes.numel()
        exact = exact and set(codes.unique().tolist()) <= {-1, 1}
    return student.eval(), {
        "code_change_rate": changed / max(total, 1),
        "exact_alphabet": exact,
    }


def run_one(seed, layers, teacher_steps, recovery_steps, batch):
    torch.manual_seed(seed)
    training = make_data(seed + 1, 256, 20)
    validation = make_data(seed + 2, 96, 32)
    fp = TinyOfficialQwen3VL(layers=layers, tied=True)
    train_teacher(fp, training, teacher_steps, batch)
    teacher = copy.deepcopy(fp).eval()
    methods = {"teacher": evaluate(teacher, teacher, validation)}
    objectives = {
        "ce_only": {"ce": 1.0, "kd": 0.0, "hidden": 0.0},
        "kd_only": {"ce": 0.0, "kd": 1.0, "hidden": 0.0},
        "kd_ce": {"ce": 0.2, "kd": 1.0, "hidden": 0.0},
        "kd_hidden": {"ce": 0.0, "kd": 1.0, "hidden": 0.1},
        "kd_ce_hidden": {"ce": 0.2, "kd": 1.0, "hidden": 0.1},
    }
    for name, weights in objectives.items():
        student, state = recover(
            fp, teacher, training, weights, recovery_steps, batch
        )
        methods[name] = evaluate(student, teacher, validation) | state | {"weights": weights}
    return {"seed": seed, "layers": layers, "methods": methods}


def aggregate(runs):
    result = {}
    for method in runs[0]["methods"]:
        result[method] = {}
        for metric in ("ce", "accuracy", "teacher_kl", "hidden_cosine"):
            values = [run["methods"][method][metric] for run in runs]
            result[method][metric + "_mean"] = statistics.mean(values)
            result[method][metric + "_pstdev"] = statistics.pstdev(values)
        if method != "teacher":
            changes = [run["methods"][method]["code_change_rate"] for run in runs]
            result[method]["code_change_rate_mean"] = statistics.mean(changes)
            result[method]["all_exact_alphabet"] = all(
                run["methods"][method]["exact_alphabet"] for run in runs
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--layers", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--teacher-steps", type=int, default=80)
    parser.add_argument("--recovery-steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default="official_qwen3vl_loss_matrix.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    by_depth = {}
    for layers in args.layers:
        runs = [
            run_one(7100 + layers * 100 + index, layers, args.teacher_steps, args.recovery_steps, args.batch)
            for index in range(args.seeds)
        ]
        by_depth[str(layers)] = {"runs": runs, "aggregate": aggregate(runs)}
    payload = {
        "implementation": "transformers.Qwen3VLTextModel",
        "arguments": vars(args),
        "by_depth": by_depth,
        "seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({depth: value["aggregate"] for depth, value in by_depth.items()}, indent=2))
    for value in by_depth.values():
        for method, metrics in value["aggregate"].items():
            if method != "teacher" and not metrics["all_exact_alphabet"]:
                raise SystemExit(f"invalid binary alphabet: {method}")


if __name__ == "__main__":
    main()
