# Engibona

Engibona is an evidence-driven research implementation of a public, testable reconstruction of a Bonsai-style transformation from pretrained FP/BF16 language-model weights into exact grouped binary or ternary weights.

It is **not PrismML's proprietary converter** and does not claim bit-for-bit identity. The project separates exact public constraints, experimentally selected methods, unresolved alternatives, and private details that final weights cannot reveal.

## Current core path

1. Exact contiguous g128 representation:
   - binary `{-1,+1}`;
   - ternary `{-1,0,+1}`;
   - one FP16 scale per group.
2. Sign or threshold initialization.
3. Whole-model teacher/task recovery rather than one-pass projection.
4. Learned positive group scales with a bounded log-space trust region.
5. Embeddings, Q/K/V/O, MLP gate/up/down, and LM head included.
6. Tied embedding and LM-head parameters remain one quantized state.
7. Globally trained exact codes and scales are preserved at export.
8. Optional empirical-Fisher code refinement with real-loss prefix validation.
9. Exact packing with no inference-time latent, LoRA, or FP residual path.

Binary defaults to exact-hard forward recovery. Ternary defaults to a CAT-Q-style soft phase followed by sustained exact-hard recovery because the current official-architecture evidence favors different paths at different depths.

## Clean official Qwen3-VL validation

A GitHub-hosted clean runner executed the test suite and a multi-seed method matrix using Hugging Face's public `Qwen3VLTextModel` implementation.

The reduced official configuration retained:

- 4 query heads and 2 KV heads;
- official grouped-query attention;
- Qwen3-VL RMSNorm and Q/K normalization;
- official SwiGLU decoder blocks;
- interleaved MRoPE with sections `[6,5,5]`;
- RoPE theta 5,000,000;
- tied embedding and LM head;
- exact g128 binary or ternary weights.

### Two decoder layers, three seeds

| Method | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Binary naive | 4.76875 | 0.01367 | 0.31115 | 0.59272 |
| Binary hard recovery | **4.49671** | **0.02843** | **0.17731** | 0.68224 |
| Binary categorical recovery | 4.51536 | 0.02778 | 0.17838 | **0.68349** |
| Ternary naive | 4.69071 | 0.01866 | 0.21809 | 0.74351 |
| Ternary hard recovery | 4.51400 | 0.03158 | **0.13002** | 0.78396 |
| Ternary CAT-Q recovery | **4.50444** | **0.03190** | 0.13188 | **0.78771** |

### Four decoder layers, three seeds

| Method | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Binary naive | 4.47710 | 0.02040 | 0.12017 | 0.84206 |
| Binary hard recovery | 4.26599 | 0.02387 | **0.05441** | 0.84827 |
| Binary categorical recovery | **4.26291** | **0.02658** | 0.05466 | **0.85057** |
| Ternary naive | 4.46434 | 0.02148 | 0.10019 | **0.90964** |
| Ternary hard recovery | 4.26274 | **0.02637** | 0.04448 | 0.87994 |
| Ternary CAT-Q recovery | 4.28870 | 0.02561 | **0.04133** | **0.89408** |

Every recovery method beat its naive projection baseline on teacher KL for every tested seed. All final code tensors passed exact-alphabet checks.

Full report and raw data:

- [`docs/OFFICIAL_QWEN3VL_METHOD_MATRIX.md`](docs/OFFICIAL_QWEN3VL_METHOD_MATRIX.md)
- [`experiments/official_qwen3vl_text/results_official_method_matrix.json`](experiments/official_qwen3vl_text/results_official_method_matrix.json)
- [`experiments/official_qwen3vl_text/run_official_method_matrix.py`](experiments/official_qwen3vl_text/run_official_method_matrix.py)

Clean workflow result:

```text
workflow run: 30859347649
artifact: 8873796012
artifact SHA-256: 07bb77f7c03bc8b9740c7f0bb35b56915bf16d11598d6a61e44d9057d460fabe
```

## Local 600-step recovery-budget result

A three-seed, eight-layer run tested 600 recovery steps at learning rate `1.4e-3` entirely on the local CPU runtime. Every method retained an exact legal low-bit alphabet. Ternary hard STE was the strongest joint behavior/geometry result: teacher KL `0.02048`, hidden cosine `0.84974`, and `32.81%` code movement versus the `37.69%` public target. Binary hard STE reached `20.08%` movement versus its `27.89%` target.

The apparent `97.84-98.10%` output-fidelity values are `exp(-teacher KL)` proxies, not intelligence-retention scores. The briefly trained synthetic teacher was weak, so the next retention test requires a converged teacher or pretrained checkpoint and full-precision-normalized benchmarks.

- [`docs/RECOVERY_BUDGET_600_MULTI_SEED.md`](docs/RECOVERY_BUDGET_600_MULTI_SEED.md)
- [`experiments/official_qwen3vl_text/results_recovery_budget_600_multiseed_summary.json`](experiments/official_qwen3vl_text/results_recovery_budget_600_multiseed_summary.json)

## Consolidated PR and method ranking

The thirteen research PRs from #4 through #16 have been consolidated into one evidence ranking. Negative experiments remain preserved on their branches, while `main` records why they were rejected or demoted.

Current selection:

- binary: exact-hard STE with free learned g128 scales and public depth/module pressure;
- ternary: optional CAT-Q soft phase followed by sustained exact-hard recovery;
- embeddings: shared binary-codebook/scale plus ternary-mask representation is supported, but independent per-mode recovery remains the behavior default;
- strong scale constraints, hidden gauges, head/neuron permutations, naive projection, and short recovery are rejected.

See:

- [`docs/PR_METHOD_RANKING.md`](docs/PR_METHOD_RANKING.md)
- [`docs/OFFICIAL_QWEN36_LOCAL_RESULT.md`](docs/OFFICIAL_QWEN36_LOCAL_RESULT.md)

## Binary recovery mathematics

For each group:

```text
c = sign(u)
s = exp(log_s)
q = s c
q_train = q + u - stop_gradient(u)
```

The forward pass is exactly binary while the latent carrier receives an identity surrogate gradient and the scale receives its real gradient. The scale is bounded around initialization:

```text
log_s in [log_s_initial - radius, log_s_initial + radius]
```

On the hand-written architecture-faithful miniature, exact-hard recovery plus Fisher refinement was the strongest tested binary path. On the official Qwen3-VL matrix, exact-hard and categorical binary recovery were effectively tied, while both decisively beat naive projection. Exact-hard remains the default because its training forward matches deployment.

## Ternary recovery status

The official matrix did not identify one universal ternary surrogate:

- exact-hard had the lowest two-layer teacher KL;
- CAT-Q had the lowest four-layer teacher KL and strongest recovered hidden alignment;
- categorical ternary sometimes had the lowest CE but worse teacher behavior.

The current default is therefore a hybrid:

```text
soft CAT-Q assignment
-> sustained exact-hard recovery
-> trained-state export
```

Zero ratios remained stable around 29–30% in the clean matrix.

## Fisher-guided discrete refinement

For a binary sign flip:

```text
delta_w = -2 s c
delta_L ~= g * delta_w + 0.5 * F_diag * delta_w^2
```

Empirical Fisher ranks candidate flips. Candidate prefixes are accepted only after they lower the real calibration objective. An exact selected-Hessian implementation is included as a tiny-model oracle, not a production large-model method.

## Export

```python
from engibona.export import export_packed

export_packed(model, "engibona_g128.pt", config)
```

The default export stores exact trained codes and FP16 scales. Packed round-trip tests verify code identity, reconstructed hard weights, tied-state equality, and the absence of latent residual fields.

## Installation

```bash
pip install -e ".[test]"
pytest
```

For official Qwen3-VL integration tests:

```bash
pip install -e ".[hf,test]"
```

## Evidence boundary

Engibona can reach 100% on measurable invariants such as alphabet validity, g128 layout, tied-weight preservation, packing identity, architecture coverage, and clean test execution.

It cannot honestly claim 100% identity with PrismML's private converter without direct source, training configuration, optimizer state, logs, data manifest, or intermediate checkpoints.

See:

- [`docs/CONFIDENCE.md`](docs/CONFIDENCE.md)
- [`docs/MATH_SELECTION.md`](docs/MATH_SELECTION.md)
- [`docs/ONE_HUNDRED_PERCENT_BOUNDARY.md`](docs/ONE_HUNDRED_PERCENT_BOUNDARY.md)
- [`docs/TEST_MATRIX.md`](docs/TEST_MATRIX.md)
