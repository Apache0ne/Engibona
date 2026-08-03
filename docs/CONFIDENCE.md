# Confidence and evidence boundary

## Direct public constraints

- transformation begins from a pretrained Qwen checkpoint;
- model architecture remains unchanged;
- language weights end in exact binary or ternary form;
- contiguous group size 128;
- one FP16 scale per group;
- embeddings and matrix-heavy language components are covered;
- normalization, activations, and sensitive accumulation remain higher precision;
- packed weights are consumed directly by low-bit kernels;
- no disclosed inference-time FP16 language residual.

## Clean official-architecture evidence

A GitHub-hosted runner executed Engibona against Hugging Face's public `Qwen3VLTextModel` at two and four layers, with three independent seeds at each depth. The run used tied embeddings/LM head, official Qwen3-VL attention and MRoPE classes, and exact g128 binary or ternary states.

All unit tests passed. Every low-bit result passed exact-alphabet checks.

### Findings that replicated at both depths and every seed

- functional recovery beat naive projection for binary weights;
- functional recovery beat naive projection for ternary weights;
- exact learned scales and codes remained valid after recovery;
- tied embedding/head quantization worked as one shared state;
- recovery gains persisted as depth increased.

### Binary surrogate result

Exact-hard and categorical binary recovery were nearly tied:

| Depth | Naive KL | Hard KL | Categorical KL |
|---:|---:|---:|---:|
| 2 | 0.31115 | **0.17731** | 0.17838 |
| 4 | 0.12017 | **0.05441** | 0.05466 |

The strongest conclusion is therefore whole-model functional recovery, not that one surrogate gradient is uniquely correct. Exact-hard remains the default because its forward pass matches deployment at every step.

### Ternary surrogate result

| Depth | Naive KL | Hard KL | CAT-Q KL | Categorical KL |
|---:|---:|---:|---:|---:|
| 2 | 0.21809 | **0.13002** | 0.13188 | 0.13517 |
| 4 | 0.10019 | 0.04448 | **0.04133** | 0.05203 |

No universal ternary winner exists under the current budget. Exact-hard was best at two layers; CAT-Q was best at four layers and had the strongest recovered hidden alignment. The selected ternary default remains soft CAT-Q assignment followed by sustained exact-hard recovery.

## Direct miniature evidence beyond the official matrix

The hand-written Qwen3-VL-topology miniature supplied additional controlled tests:

- trained-state export decisively beat post-recovery covariance re-projection;
- empirical-Fisher sign ranking strongly predicted real teacher-loss improvements;
- selected exact Hessian diagonals nearly perfectly predicted measured individual sign-flip changes;
- embeddings were the most sensitive matrix component;
- mixed CE/KD/hidden objectives preserved different aspects of behavior;
- four-layer recovery replicated the two-layer recovery advantage.

These results are useful method-selection evidence but are distinct from the clean official-architecture run.

## Current method confidence

The percentages below describe confidence in the **Engibona public reconstruction choice**, not probability of exact PrismML internal identity.

| Component | Confidence | Status |
|---|---:|---|
| Exact g128 code + FP16 scale | **>99.9%** | Required public format |
| Whole-model functional recovery after initialization | **99%+** | Replicated at 2/4 layers and all official-architecture seeds |
| Preserve trained codes/scales at export | **99%** | Direct finalization ablation |
| Positive learned group scales | **98–99%** | Gradient, recovery, and export tests |
| Tied embedding/head state preservation | **>99%** | Exact alias, gradient, and export tests |
| Exact-hard binary as default | **92–97%** | Strong, but categorical was nearly tied |
| Binary categorical as viable alternative | **85–95%** | Nearly identical official-matrix behavior |
| Hybrid soft-to-hard ternary path | **88–96%** | Depth-dependent official results |
| One universal ternary surrogate | **below 30%** | Falsified by depth-dependent winners |
| Teacher KL as central recovery loss | **95–99%** | Consistent behavior recovery evidence |
| Mixed CE/KD/hidden objective | **87–95%** | Complementary metric ablation |
| Module-sensitive recovery budgets | **95–99%** | Large measured sensitivity spread |
| Empirical-Fisher discrete refinement | **94–98%** | Multi-module predictor and improvement tests |
| Real-loss validation of proposed code moves | **>99%** | Required to prevent approximation error |
| Block/global recovery as depth grows | **94–98%** | 2/4-layer replication |
| Local covariance projection as initializer | **70–88%** | Useful local method, not finalizer |
| Local covariance projection as final export oracle | **below 5%** | Directly degraded recovered behavior |
| Dynamic curriculum as mandatory | **below 50%** | Small mixed evidence only |
| Learned rotations as mandatory | **below 20%** | No Bonsai-specific direct evidence |
| One-pass sign/threshold projection as full method | **below 0.1%** | Rejected by every recovery comparison |
| Inference-time latent/FP residual | **below 0.1%** | Format-incompatible and export-tested absent |

## Clean-run provenance

```text
workflow run: 30859347649
artifact ID: 8873796012
artifact SHA-256: 07bb77f7c03bc8b9740c7f0bb35b56915bf16d11598d6a61e44d9057d460fabe
```

## Irreducible unknowns

Public checkpoints and experiments cannot uniquely identify PrismML's:

- optimizer family;
- learning-rate schedule;
- number of steps or recovery tokens;
- data mixture;
- loss coefficients;
- exact layer ordering;
- distributed implementation;
- private discrete solver;
- whether its internal gradient surrogate matches Engibona.

These require direct source, configuration, logs, optimizer state, or intermediate checkpoints.

## Highest-value remaining tests

1. Longer official-architecture ternary tests with several soft-to-hard transition points.
2. Official-architecture loss matrix: KL, CE, hidden, block-output, and combinations.
3. Student-input versus teacher-input block reconstruction.
4. Sensitivity-proportional versus uniform recovery budgets.
5. Multi-length recovery sequences.
6. Public unpacked Bonsai weight forensics across 1.7B, 4B, and 8B.
7. Packed-kernel parity against dequantized reference execution.

See [`OFFICIAL_QWEN3VL_METHOD_MATRIX.md`](OFFICIAL_QWEN3VL_METHOD_MATRIX.md) and the raw JSON for seed-level results.
