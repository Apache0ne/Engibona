# Sign and scale gauge forensics

## Question

Could the large released-code divergence be mostly a simple hidden-channel gauge?

For a matrix, an exact sign gauge has the form

\[
W'_{ij} \approx r_i W_{ij} c_j,
\qquad r_i,c_j\in\{-1,+1\}.
\]

Consistent row and column signs can arise from flipping hidden channels while compensating in adjacent matrices. Such a transformation could lower direct sign agreement without requiring independent code changes.

The experiment selected optimal row signs, input-column signs, combined row-plus-column signs, and row-plus-g128-block signs on structured samples from all released matrices. It also tested whether the ratio between released scales and naive Qwen scales is separable into output-row and input-group-column effects.

## Coverage

```text
tensors:          197
g128 groups:      460,784
weight positions: 58,980,352
```

## Sign gauge result

| Test | Raw agreement | Gauge-adjusted agreement | Gain |
|---|---:|---:|---:|
| Qwen sign vs released binary | 72.356699% | 72.357486% | **0.000787 pp** |
| Qwen sign vs released ternary nonzeros | 87.144117% | 87.144194% | **0.000077 pp** |
| Released binary vs released ternary nonzeros | 90.287546% | 90.287549% | **0.000003 pp** |

The optimizer selected a negative sign for only approximately 0.0104% of output rows and 0.0115% of input columns. A g128-block column gauge improves binary agreement by only 0.000170 percentage points.

These changes are numerically negligible. The released code divergence is not a disguised row/column sign transformation.

## Scale structure

The code gauge is absent, but released group scales have substantial structured change. The experiment modeled

\[
\log\frac{s_{released}(i,g)}{s_{naive}(i,g)}
\approx \mu + a_i + b_g,
\]

where `i` is output row and `g` is input-group column.

| Variant | Row-only R² | Group-column-only R² | Additive in-sample R² | Additive held-out R² |
|---|---:|---:|---:|---:|
| Binary | 0.68397 | 0.06223 | 0.74619 | **0.70784** |
| Ternary | 0.58621 | 0.05523 | 0.64144 | **0.59517** |

Thus, much of the scale recovery is low-rank and output-row dominated, but roughly 29% of binary and 40% of ternary held-out log-scale variance remains unexplained by this separable model.

### Module-specific held-out R²

| Module | Binary | Ternary |
|---|---:|---:|
| Q projection | 0.85457 | 0.75123 |
| K projection | 0.83849 | 0.75127 |
| V projection | 0.65353 | 0.47804 |
| O projection | 0.64430 | 0.59662 |
| Gate projection | 0.84315 | 0.75001 |
| Up projection | 0.74860 | 0.67892 |
| Down projection | 0.55446 | 0.38705 |
| Embedding | -0.10488 | 0.06459 |

Q, K, gate, and up scale changes are especially row-structured. O projection differs: its input-group-column component is unusually strong, consistent with output projection consuming structured attention-head/channel inputs.

Embedding again behaves as a separate policy. Its scale ratios do not benefit from the row/group recovery model, matching the evidence that binary embedding is already essentially direct projection.

## What is now rejected

The following are not major explanations of released transformer codes:

- hidden-channel sign flips;
- output-row sign gauges;
- input-column sign gauges;
- g128-block sign gauges;
- a shared sign gauge between final binary and ternary checkpoints;
- reinterpreting the 27–30% binary code disagreement as a coordinate-choice artifact.

Together with the permutation experiment, this shows that released codes remain in the original head, neuron, row, and column coordinate system while being extensively reassigned inside that system.

## What remains useful

A structured scale prior is supported. A higher-confidence reconstruction can parameterize scale recovery as

```text
initial per-g128 scale
× learned output-row factor
× learned input-group-column factor
× smaller free per-group residual
```

This should be treated as an initialization or regularization structure, not a restriction. The held-out residual is too large to replace free per-group scales completely.

The code variables still require broad behavior-aware recovery; separable scaling alone cannot generate the released code geometry.

## Updated reconstruction family

```text
pretrained Qwen
-> direct embedding projection policy
-> fixed original architecture coordinates
-> sign/threshold code initialization
-> broad transformer code reassignment
-> structured row/group scale recovery plus free residual
-> depth-sensitive whole-model behavioral optimization
-> trained exact codes/scales preserved at export
```

## Provenance

```text
workflow run:          30865843604
artifact ID:           8876115080
artifact ZIP SHA-256:  9f084306f229b83e52631cc09b4fc8acf759c259512681543f4c14a8b22e1133
summary JSON SHA-256:  f998f75a498f4125accfa50de5c082e0e1758b8199fcd76d64b4d42770960893
runtime:               170.61 seconds
```

Machine-readable summary:

- `experiments/public_bonsai_forensics/results_gauge_forensics.json`
