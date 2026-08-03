# Engibona

Engibona is an evidence-driven research implementation of the highest-confidence public reconstruction of a Bonsai-style transformation from pretrained FP/BF16 language-model weights into exact grouped binary or ternary weights.

It is **not PrismML's proprietary converter** and does not claim 1:1 reproduction. Every default is separated into:

- publicly constrained representation facts;
- choices supported by direct Engibona experiments;
- literature-supported but unresolved options;
- private details that cannot be identified without training artifacts.

## Current tested core path

1. Exact contiguous g128 representation:
   - binary `{-1,+1}` codes;
   - ternary `{-1,0,+1}` codes;
   - one FP16 scale per group.
2. Strong sign/threshold initialization.
3. **Binary default: exact hard forward from the first recovery step.**
4. Learned positive group scales with a bounded log-scale trust region.
5. Teacher KL plus task and optional block/hidden recovery losses.
6. Embeddings, attention projections, MLP projections, and LM head included.
7. Preserve globally learned exact codes and scales at export.
8. Optional empirical-Fisher discrete sign refinement with calibration line search.
9. Exact packing with no inference-time latent, LoRA, or FP16 residual path.

Ternary recovery retains CAT-Q-style smooth assignment as the default until an equally deep ternary ablation is complete, followed by a sustained exact-hard recovery phase.

## Why the binary default changed

The first Engibona version used smooth continuation and re-projected recovered latent weights through a local activation-covariance objective at export. A three-seed CPU ablation on a two-layer, four-head, Qwen3-VL-text-topology miniature falsified that choice.

| Binary method | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Naive sign + absmean | 5.03893 | 0.04741 | 3.62201 | 0.56972 |
| Covariance coordinate PTQ | 5.75938 | 0.06158 | 4.00645 | 0.59454 |
| Smooth KD recovery | 3.46709 | 0.28199 | 1.95101 | 0.65287 |
| Staged KD recovery | 3.24553 | 0.29079 | 1.86080 | 0.65059 |
| Exact-hard STE + KD | 3.12885 | 0.33749 | 1.79099 | **0.65755** |
| Exact-hard STE + KD + Fisher flips | **3.10093** | **0.34239** | **1.76733** | 0.65636 |

Exact-hard recovery beat smooth, staged, naive, and covariance-coordinate methods for all three seeds on CE and teacher KL. Fisher-guided exact sign refinement further improved CE and KL for all three seeds.

The decisive finalization test was:

| Finalization after recovery | CE | Teacher KL |
|---|---:|---:|
| Preserve trained codes/scales | **3.2743** | **1.8904** |
| Replace with covariance-optimal scales | 3.8967 | 2.4271 |
| Covariance scale + coordinate re-projection | 4.9813 | 3.4895 |

Therefore local metric projection remains a useful PTQ initializer and diagnostic, but **trained-state export is now the default**.

Full details:

- [`docs/TINY_QWEN3VL_CPU_ABLATION.md`](docs/TINY_QWEN3VL_CPU_ABLATION.md)
- [`experiments/tiny_qwen3vl/results_three_seed_deep_ablation.json`](experiments/tiny_qwen3vl/results_three_seed_deep_ablation.json)
- [`experiments/tiny_qwen3vl/run_cpu_ablation.py`](experiments/tiny_qwen3vl/run_cpu_ablation.py)

## Mathematical binary recovery path

For each group:

```text
hard code:     c = sign(u)
positive scale s = exp(log_s)
hard weight:   q = s c
```

The forward pass is exactly binary. The training expression is:

```text
q_train = q + u - stop_gradient(u)
```

This gives:

- exact `q` in the forward pass;
- an identity surrogate gradient to the latent code carrier `u`;
- real gradients to the learned scale `s`.

The scale is constrained by:

```text
log_s in [log_s_initial - radius, log_s_initial + radius]
```

which prevents uncontrolled scale/code feedback.

## Fisher-guided discrete refinement

After hard recovery, Engibona can rank exact sign flips using an empirical-Fisher diagonal approximation. For `w=s*c`, a sign flip has:

```text
delta_w = -2 s c
```

and predicted loss change:

```text
delta_L ~= g * delta_w + 0.5 * F_diag * delta_w^2
```

Negative candidates are ranked, but candidate prefixes must be validated against the real teacher/task calibration loss before acceptance.

## Metric projection remains available

For PTQ initialization and diagnostics, Engibona can minimize:

```text
E(c,s) = (w - s c)^T M (w - s c)
```

with exact fixed-code scale:

```text
s* = (c^T M w) / (c^T M c)
```

and coordinate code refinement. `M` may be identity, activation diagonal, or full within-group activation covariance.

The experiment shows an important boundary: decreasing this local objective does not guarantee improved full-model sequence behavior after global recovery.

## Installation and tests

```bash
pip install -e ".[test]"
pytest
```

## Training-time wrapper

```python
from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules

config = EngibonaConfig(
    mode=QuantMode.BINARY,
    relaxation="hard_ste",
    export_strategy="trained",
)
modules = replace_linear_modules(model, config, include_embeddings=True)

for step in range(total_steps):
    for module in modules.values():
        module.set_schedule(step, total_steps)

    # teacher KL + task/block losses
    loss.backward()
    optimizer.step()
```

Each module exposes `regularization_loss()` for scale tethering and optional ternary zero-ratio control.

## Export

```python
from engibona.export import export_packed

export_packed(model, "engibona_g128.pt", config)
```

The default export stores the model's trained exact hard codes and learned FP16 scales. Set `export_strategy="metric_reproject"` only for explicit PTQ/reprojection experiments.

The current container is an auditable research format, not GGUF.

## Evidence and scope

See:

- [`docs/MATH_SELECTION.md`](docs/MATH_SELECTION.md)
- [`docs/CONFIDENCE.md`](docs/CONFIDENCE.md)
- [`docs/TINY_QWEN3VL_CPU_ABLATION.md`](docs/TINY_QWEN3VL_CPU_ABLATION.md)

The project still does not claim knowledge of PrismML's exact optimizer, corpus, learning rates, step count, distributed implementation, or private discrete solver. It does not yet provide a verified PrismML-compatible GGUF serializer or production CUDA/Metal/MLX kernels.
