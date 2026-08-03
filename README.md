# Engibona

Engibona is an evidence-driven research implementation of the **highest-confidence public reconstruction** of a Bonsai-style transformation from pretrained FP/BF16 language-model weights into exact grouped binary or ternary weights.

It is **not PrismML's proprietary converter** and does not claim 1:1 reproduction. It deliberately separates:

- facts disclosed by PrismML;
- mechanisms strongly supported by current low-bit research;
- lower-confidence extensions that remain optional;
- implementation details that are still unknowable without training artifacts.

## Selected core path

The default path implements the components with the strongest combined support:

1. **Exact g128 representation**
   - binary codes `{-1,+1}`;
   - ternary codes `{-1,0,+1}`;
   - one FP16 scale per contiguous group of 128 weights.
2. **Strong PTQ-style initialization** rather than immediate full-model hard QAT.
3. **Smooth-to-hard continuation** instead of using a rigid STE from step zero.
4. **Tensor-sensitivity-aware hardening** using a Hessian-trace-compatible temperature schedule.
5. **Metric-aware code and scale optimization** under activation covariance or a diagonal approximation.
6. **Exact coordinate code refinement** with the group scale analytically re-optimized after every candidate move.
7. **Sliding-window / block-output reconstruction** as the intended recovery objective.
8. **Teacher KL, next-token loss, hidden-state reconstruction**, with relational CKA available as an option.
9. **Exact packing and alphabet verification** with no inference-time FP16 residual path.

Lower-confidence features—dynamic curriculum selection, CKA, recurrent-state loss, and rotations—are not mandatory defaults.

## Mathematical core

For a group `w`, code `c`, positive scale `s`, and positive-semidefinite metric `M`, Engibona minimizes

```text
E(c,s) = (w - s c)^T M (w - s c)
```

with

```text
c in {-1,+1}^128
```

or

```text
c in {-1,0,+1}^128.
```

For fixed codes, the exact optimal scale is

```text
s* = (c^T M w) / (c^T M c).
```

After substituting `s*`, code search maximizes

```text
(c^T M w)^2 / (c^T M c).
```

The coordinate-search implementation evaluates every legal one-code transition with this closed form. It therefore performs joint code/scale refinement without approximating the scale update.

For a linear layer `y = xW`, the default full group metric is the calibration activation covariance

```text
M = X^T X + lambda I,
```

which makes the projection minimize layer-output reconstruction error rather than raw weight MSE.

## Installation

```bash
pip install -e ".[test]"
pytest
```

Optional Hugging Face dependencies:

```bash
pip install -e ".[hf,test]"
```

## Toy projection

```bash
python examples/toy_recovery.py
```

## Minimal API

```python
import torch

from engibona.config import QuantMode
from engibona.metrics import activation_covariance
from engibona.projection import metric_project

weight = torch.randn(64, 4096)
calibration_x = torch.randn(512, 4096)
metric = activation_covariance(calibration_x, group_size=128)

result = metric_project(
    weight,
    mode=QuantMode.TERNARY,
    group_size=128,
    metric=metric,
    refine_steps=8,
)

assert set(result.codes.unique().tolist()) <= {-1, 0, 1}
assert torch.all(result.final_error <= result.initial_error + 1e-5)
```

## Training-time wrapper

```python
from engibona.config import EngibonaConfig, QuantMode
from engibona.modules import replace_linear_modules

config = EngibonaConfig(mode=QuantMode.TERNARY)
replaced = replace_linear_modules(model, config)

for step in range(total_steps):
    for module in replaced.values():
        module.set_schedule(step, total_steps)
    # teacher/student recovery step
```

The wrapper uses temporary latent weights only during recovery. `export_packed` projects to exact codes and scales and stores no latent residual.

## Export

```python
from engibona.export import export_packed

export_packed(model, "engibona_g128.pt", config)
```

The current container is an auditable research format, not GGUF. A runtime-specific serializer should be added only after verifying exact tensor naming, group ordering, scale layout, and target-kernel semantics.

## Why these methods were selected

See:

- [`docs/MATH_SELECTION.md`](docs/MATH_SELECTION.md)
- [`docs/CONFIDENCE.md`](docs/CONFIDENCE.md)

The review explicitly rejects several tempting but format-incompatible choices:

- low-rank binary factorization as the deployed representation;
- mixed-precision salient-weight escape channels;
- inference-time LoRA or FP16 residual paths;
- rotation tensors without direct evidence;
- ordinary sign/threshold rounding as the complete method;
- a traditional hard STE as the principal optimization path.

## Scope boundary

Engibona currently provides the quantization mathematics, recovery modules, losses, data-selection option, packing, and tests. It does not yet provide:

- a distributed 27B trainer;
- Qwen3.6-specific hybrid linear-attention state hooks;
- a verified PrismML-compatible GGUF serializer;
- custom CUDA, Metal, or MLX kernels;
- PrismML's undisclosed corpus, optimizer, learning rates, or step schedule.

Those details must not be guessed and represented as fact.
