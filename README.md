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

## Full-precision references and metric directions

All miniature benchmarks below used an **FP32 teacher**: model parameters, activations, teacher training, and evaluation were FP32 on CPU. The low-bit students use exact binary or ternary codes with learned group scales. These tables do not contain a BF16 teacher run; labeling the reference as BF16 would be incorrect.

| Mark | Meaning | Best value |
|---|---|---:|
| `CE ↓` | Cross-entropy against task labels | Lower |
| `Accuracy ↑` | Exact next-token accuracy | Higher |
| `Teacher KL ↓` | Difference from the FP32 teacher output distribution | `0` |
| `Hidden cosine ↑` | Directional agreement with FP32 teacher hidden states | `1.0` / `100%` |
| `Output fidelity ↑` | `exp(-Teacher KL)`, a local distribution-fidelity proxy | `100%` |
| `Code movement ↔ target` | Fraction of low-bit codes changed during recovery | Closest to the stated public target, not simply higher |
| `Target coverage ↔` | Code movement divided by the public target | Closest to `100%`; overshooting is also an error |

The exact FP32 denominators are:

| Benchmark | Seeds | FP32 CE ↓ | FP32 accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL miniature, 2 layers | 3 | 4.60368 +/- 0.03077 | 3.3529% +/- 0.3134% | ~0 | 1.0000 |
| Qwen3-VL miniature, 4 layers | 3 | 4.36467 +/- 0.14874 | 2.1918% +/- 0.3459% | ~0 | 1.0000 |
| Qwen3-VL miniature, 8 layers, 600-step matrix | 3 | 4.70705 +/- 0.02959 | 1.5299% +/- 0.2875% | 0 | 1.0000 |
| Qwen3.6 hybrid miniature, 4 layers | 2 | 5.59156 +/- 0.00719 | 0.3906% +/- 0.1628% | ~0 | 1.0000 |
| Qwen3.6 hybrid miniature, 8 layers | 2 | 5.59657 +/- 0.00601 | 0.4720% +/- 0.0488% | ~0 | 1.0000 |

These are synthetic architecture tests, not full pretrained-Qwen intelligence benchmarks. Bold values in later tables identify the best **low-bit student**, excluding the FP32 reference row.

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

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|
| FP32 teacher reference | 4.60368 | 3.3529% | ~0 | 1.00000 |
| Binary naive | 4.76875 | 1.367% | 0.31115 | 0.59272 |
| Binary hard recovery | **4.49671** | **2.843%** | **0.17731** | 0.68224 |
| Binary categorical recovery | 4.51536 | 2.778% | 0.17838 | **0.68349** |
| Ternary naive | 4.69071 | 1.866% | 0.21809 | 0.74351 |
| Ternary hard recovery | 4.51400 | 3.158% | **0.13002** | 0.78396 |
| Ternary CAT-Q recovery | **4.50444** | **3.190%** | 0.13188 | **0.78771** |

### Four decoder layers, three seeds

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|
| FP32 teacher reference | 4.36467 | 2.1918% | ~0 | 1.00000 |
| Binary naive | 4.47710 | 2.040% | 0.12017 | 0.84206 |
| Binary hard recovery | 4.26599 | 2.387% | **0.05441** | 0.84827 |
| Binary categorical recovery | **4.26291** | **2.658%** | 0.05466 | **0.85057** |
| Ternary naive | 4.46434 | 2.148% | 0.10019 | **0.90964** |
| Ternary hard recovery | **4.26274** | **2.637%** | 0.04448 | 0.87994 |
| Ternary CAT-Q recovery | 4.28870 | 2.561% | **0.04133** | **0.89408** |

Every recovery method beat its naive projection baseline on teacher KL for every tested seed. All final code tensors passed exact-alphabet checks.

Full report and raw data:

- [`docs/OFFICIAL_QWEN3VL_METHOD_MATRIX.md`](docs/OFFICIAL_QWEN3VL_METHOD_MATRIX.md)
- [`experiments/official_qwen3vl_text/results_official_method_matrix.json`](experiments/official_qwen3vl_text/results_official_method_matrix.json)
- [`experiments/official_qwen3vl_text/run_official_method_matrix.py`](experiments/official_qwen3vl_text/run_official_method_matrix.py)

## Long all-method 600-step CPU benchmark

The comprehensive local run now covers every runnable official-architecture recovery matrix on `main`, plus the branch-preserved scale-structure and shared-embedding families: **49 named low-bit configurations** across eight families. Every comparison used three seeds, four layers, 600 FP32 teacher steps, 600 recovery steps, batch 12, and exact g128 codes. Including the missing-baseline replays, the recorded runtime totals 3.82 core-hours-equivalent; the jobs were executed concurrently on this CPU runtime without GitHub Actions.

Rank only within a family because each family has its own deterministic seeds. The main long-run winners are:

| Family | Winner | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Main tradeoff |
|---|---|---:|---:|---:|---|
| Qwen3-VL surrogate, binary | Categorical | 1.6940 | 18.38% | 0.7441 | Slight behavior win; exact-hard matches deployment during training |
| Qwen3-VL surrogate, ternary | CAT-Q to hard | 0.9837 | 37.39% | 0.8572 | Better behavior; fewer code changes than sustained hard recovery |
| Public-profile binary | Categorical + public pressure | 1.5956 | 20.28% | 0.7451 | Best binary behavior in that paired family |
| Ternary schedule | CAT-Q, hard at 75% | 0.9426 | 38.96% | 0.8577 | Best behavior; 10.69% movement versus 17.35% for hard-only |
| Binary loss | KD + hidden MSE | 1.8279 | 16.08% | 0.7279 | Small three-seed improvement over KD-only |
| Shared embedding pair | Frozen shared sign | 2.9283 combined | n/a | 0.7223 / 0.8406 | Exact shared relation; 0.81% better combined KL than independent |
| Qwen3.6 hybrid, binary | Categorical | 3.6803 | 2.52% | 0.3418 | Chance-level teacher limits interpretation |
| Qwen3.6 hybrid, ternary | CAT-Q to hard | 2.9850 | 5.05% | 0.4567 | Chance-level teacher limits interpretation |

This changes the **behavior ranking**, not the geometry conclusion. Categorical binary and late-hardening CAT-Q ternary are the long-run behavior specialists. Sustained exact-hard recovery still moves substantially more codes toward released-checkpoint geometry and has an exact train/deploy forward throughout. The default remains geometry-aware: binary hard is the deployment-faithful default with categorical retained as a behavior option; ternary uses a CAT-Q phase followed by a meaningful hard phase.

The stronger Qwen3-VL teachers reduce the best apparent fidelity proxy from the old weak-teacher ~98% range to 18–39%. That does not mean intelligence collapsed by that amount; it proves that `exp(-teacher KL)` is not an intelligence-retention benchmark and is highly dependent on teacher/data quality.

- [`docs/LONG_ALL_METHODS_600_STEP.md`](docs/LONG_ALL_METHODS_600_STEP.md)
- [`experiments/long_all_methods/results_long_all_methods_summary.json`](experiments/long_all_methods/results_long_all_methods_summary.json)
- [`experiments/long_all_methods/`](experiments/long_all_methods/)

Clean workflow result:

```text
workflow run: 30859347649
artifact: 8873796012
artifact SHA-256: 07bb77f7c03bc8b9740c7f0bb35b56915bf16d11598d6a61e44d9057d460fabe
```

## Local 600-step recovery-budget result

A three-seed, eight-layer run tested 600 recovery steps at learning rate `1.4e-3` entirely on the local CPU runtime. Every method retained an exact legal low-bit alphabet. Ternary hard STE was the strongest joint behavior/geometry result: teacher KL `0.02048`, hidden cosine `0.84974`, and `32.81%` code movement versus the `37.69%` public target. Binary hard STE reached `20.08%` movement versus its `27.89%` target.

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Output fidelity ↑ | Hidden cosine ↑ | Code movement ↔ target |
|---|---:|---:|---:|---:|---:|---:|
| FP32 teacher reference | 4.70705 +/- 0.02959 | 1.5299% +/- 0.2875% | 0 | 100% | 100% | n/a |
| Binary hard STE | **4.69518 +/- 0.02649** | **1.4540% +/- 0.2412%** | 0.02185 +/- 0.00269 | 97.84% | 80.98% | **20.08% / 27.89%** |
| Binary categorical | 4.70742 +/- 0.02519 | 1.3780% +/- 0.3386% | **0.01917 +/- 0.00206** | **98.10%** | **81.59%** | 16.69% / 27.89% |
| Ternary hard STE | **4.70094 +/- 0.02317** | 1.4106% +/- 0.2229% | **0.02048 +/- 0.00279** | **97.97%** | 84.97% | **32.81% / 37.69%** |
| Ternary auto | 4.70200 +/- 0.02654 | **1.5299% +/- 0.1917%** | 0.02130 +/- 0.00217 | 97.89% | **86.82%** | 28.57% / 37.69% |

The apparent `97.84-98.10%` output-fidelity values are `exp(-teacher KL)` proxies, not intelligence-retention scores. The briefly trained synthetic teacher was weak, so the next retention test requires a converged teacher or pretrained checkpoint and full-precision-normalized benchmarks.

- [`docs/RECOVERY_BUDGET_600_MULTI_SEED.md`](docs/RECOVERY_BUDGET_600_MULTI_SEED.md)
- [`experiments/official_qwen3vl_text/results_recovery_budget_600_multiseed_summary.json`](experiments/official_qwen3vl_text/results_recovery_budget_600_multiseed_summary.json)

## Local official Qwen3.6 hybrid validation

This earlier two-seed, 40-step, four/eight-layer matrix also ran locally in FP32 on CPU. Values below are means across both depths and all four teacher instances. The newer 600-step matrix above reverses the behavior choice to binary categorical and ternary CAT-Q-to-hard; this older table remains as budget-dependent evidence.

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|
| FP32 teacher reference | 5.59407 +/- 0.00708 | 0.4313% +/- 0.1269% | ~0 | 1.00000 |
| Binary naive | 5.57544 | 0.3418% | 0.04554 | 0.53775 |
| Binary hard | **5.57448** | **0.3581%** | **0.01839** | **0.79585** |
| Binary categorical | 5.57453 | **0.3581%** | 0.01844 | 0.79043 |
| Ternary naive | **5.56524** | 0.3499% | 0.03134 | 0.69957 |
| Ternary hard | 5.57470 | 0.3825% | **0.01332** | **0.85326** |
| Ternary CAT-Q to hard | 5.57376 | **0.3988%** | 0.01422 | 0.84944 |

The low-bit CE can be slightly lower than the weak FP32 teacher CE because discretization acts as regularization on this synthetic task. That does not mean the low-bit student is more intelligent: teacher KL and hidden cosine still show the information lost relative to the FP32 model.

- [`docs/OFFICIAL_QWEN36_LOCAL_RESULT.md`](docs/OFFICIAL_QWEN36_LOCAL_RESULT.md)
- [`experiments/official_qwen36_text/results_official_qwen36_local_summary.json`](experiments/official_qwen36_text/results_official_qwen36_local_summary.json)

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
