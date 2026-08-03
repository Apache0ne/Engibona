# Direct public Bonsai weight findings

This is the strongest public evidence added so far because it compares the actual released Qwen3-1.7B, binary Bonsai-1.7B-unpacked, and ternary Bonsai-1.7B-unpacked tensors rather than relying only on related literature or miniature-model behavior.

A clean runner sampled 100,864 deterministic g128 groups from 197 tensors: 12,910,592 individual weights. The duplicate base `lm_head.weight` was removed because the Bonsai release ties the output head to token embeddings.

## Exact representation

The unpacked releases follow the expected alphabets to FP16 precision:

| Representation | Maximum normalized alphabet error |
|---|---:|
| Binary | 0.000188 |
| Ternary | 0.000250 |

This confirms one shared magnitude per sampled binary group and `{0,s}` magnitudes per sampled ternary group.

## Binary weights were substantially re-optimized

| Measurement | Result | Tensor-cluster 95% interval |
|---|---:|---:|
| Sign agreement with base `sign(W)` | 72.25% | 71.80–72.73% |
| Sign-flip rate | 27.75% | — |
| Base-magnitude percentile of flipped signs | 35.73% | — |
| Scale correlation with base mean-absolute scale | 0.628 | 0.595–0.660 |
| Median released/base mean-absolute scale ratio | 2.279 | — |
| Released/naive raw-NMSE ratio | 6.642× | 6.220–7.076× |

A one-pass sign projection cannot explain the released transformer weights. More than one quarter of sampled signs changed. The released model is approximately 6.64 times farther from the original Qwen weights in raw MSE than the raw-MSE-optimal naive binary projection.

That result is decisive:

```text
released objective != raw weight reconstruction
```

The codes and scales were optimized for model behavior or another functional criterion that intentionally accepts much larger coordinate-space error.

## Ternary weights were also substantially re-optimized

| Measurement | Result | Tensor-cluster 95% interval |
|---|---:|---:|
| Zero rate | 39.49% | 39.20–39.76% |
| Agreement with naive least-squares threshold projection | 62.43% | 61.93–62.95% |
| Base-magnitude percentile of released zeros | 38.20% | 37.81–38.56% |
| Nonzero-sign agreement with base | 86.80% | — |
| Scale correlation with naive ternary scale | 0.675 | 0.643–0.706 |
| Released/naive raw-NMSE ratio | 4.517× | 4.287–4.749× |

Ordinary magnitude thresholding does not explain the released zero mask. Approximately 37.6% of ternary assignments differ from the naive projection, and released zeros are not restricted to the smallest original weights.

The ternary release is approximately 4.52 times farther from Qwen in raw MSE than the naive ternary projection. This independently supports function-aware code and scale optimization.

## Binary and ternary share a framework, not one final codebook

| Measurement | Result | Tensor-cluster 95% interval |
|---|---:|---:|
| Sign agreement on ternary nonzeros | 90.12% | 89.65–90.60% |
| Group-scale correlation | 0.928 | 0.922–0.934 |

The strong scale correlation and 90% sign agreement support a common initialization and common transformation framework. The remaining approximately 10% sign divergence is too large for the binary checkpoint to be a mechanical encoding of the final ternary checkpoint.

The most plausible relationship is:

```text
shared pretrained model and conversion framework
-> separately optimized binary and ternary recovery runs
```

## Embeddings are a special case

The token embedding behaves very differently from the transformer matrices.

### Binary embedding

- sign agreement with Qwen: 99.927%;
- scale correlation with mean-absolute Qwen scales: 0.99996;
- median scale ratio: 0.99992;
- released/naive NMSE ratio: 1.00002;
- binary/ternary scale correlation: approximately 1.0.

The binary embedding is essentially direct sign plus mean-absolute g128 projection. Its rare sign differences occur at a mean base-magnitude percentile of only 0.25%.

### Ternary embedding

- code agreement with naive projection: 85.70%;
- nonzero signs agree with the base: 100%;
- zero rate: 31.01%;
- scale correlation with naive ternary scales: 0.931;
- released/naive NMSE ratio: 1.394.

The embedding therefore received much less discrete re-optimization than the transformer matrices. This is direct evidence for a module-specific recovery policy.

## Depth pattern

Across the 28 transformer layers:

- layer index versus binary sign agreement: correlation `-0.836`;
- layer index versus binary released/naive NMSE ratio: `+0.832`;
- layer index versus ternary released/naive NMSE ratio: `+0.914`;
- layer index versus ternary zero rate: `+0.673`.

Later layers diverge more strongly from naive projection. The last layer's mean binary sign agreement was approximately 68.3%, compared with approximately 73.1% in the first layer.

This is strong evidence against a uniform independent-layer converter. It supports depth-aware recovery, accumulated-error correction, or a full-model objective that permits later layers to compensate for earlier quantization damage.

## Module differences

| Module | Binary sign agreement | Binary NMSE ratio | Ternary code agreement | Ternary NMSE ratio |
|---|---:|---:|---:|---:|
| Embedding | 99.93% | 1.00× | 85.70% | 1.39× |
| Q projection | 70.57% | 3.20× | 59.43% | 3.09× |
| K projection | 70.03% | 4.12× | 59.48% | 3.44× |
| V projection | 76.13% | 7.87× | 66.84% | 4.41× |
| O projection | 74.05% | 6.07× | 63.90% | 3.96× |
| Gate projection | 70.42% | 9.09× | 60.85% | 5.91× |
| Up projection | 70.02% | 11.25× | 60.22% | 7.55× |
| Down projection | 73.55% | 5.09× | 65.43% | 3.37× |

The MLP up and gate projections show the largest departure from raw-MSE-optimal binary/ternary weights. This implies that recovery effort and code-refinement budgets should be module-sensitive.

## Direct confidence updates

The released-weight evidence raises confidence in:

- substantial code reassignment: extremely high;
- joint code and scale recovery: extremely high;
- functional rather than raw-weight objective: extremely high;
- separate binary and ternary recovery runs: high;
- module-specific treatment: extremely high;
- depth-aware or full-model recovery: high;
- naive embedding projection for binary: extremely high;
- uniform recovery rules for every tensor: strongly rejected;
- scale-only recovery: strongly rejected;
- one-pass sign or threshold conversion: decisively rejected.

It does not identify the exact optimizer, surrogate gradient, dataset, loss coefficients, or training duration.

## Provenance

```text
workflow run: 30861622208
artifact ID: 8874660856
artifact SHA-256: 895756008005ea5200f851bbfee47fe788396aeaeb078c91ae34cc4a28c8b313
runtime: 142.20 seconds
```

The artifact contains per-tensor CSV data, per-module metrics, the global summary, and tensor-cluster confidence intervals.
