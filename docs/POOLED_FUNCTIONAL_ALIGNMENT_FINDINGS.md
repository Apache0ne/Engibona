# Pooled functional alignment findings

## Scope

The earlier full-model run established a large behavior gap between naive g128 projection and released Bonsai checkpoints. This follow-up uses eight longer prompts, 380 pooled hidden-state tokens, and 64 sampled output positions to determine whether released internal states are related to Qwen by a simple channel gauge.

At hidden-state indices `0, 7, 14, 21, 28`, the experiment measures:

- raw and channel-centered token cosine;
- pooled linear CKA;
- held-out global affine alignment;
- held-out per-channel diagonal affine alignment;
- held-out signed per-channel standardization;
- a 128-dimensional orthogonal Procrustes probe.

Logits are compared on the common vocabulary using teacher KL, centered cosine, affine R², and top-k overlap.

## Output distribution result

| Candidate | Teacher KL | Top-1 agreement | Centered logit cosine | Top-10 overlap | Top-100 overlap |
|---|---:|---:|---:|---:|---:|
| Naive binary | 11.09588 | 0.00% | 0.13673 | 1.25% | 2.25% |
| Released binary | **0.55500** | **65.63%** | **0.62329** | **61.88%** | **60.61%** |
| Naive ternary | 11.28190 | 0.00% | 0.13909 | 3.13% | 5.25% |
| Released ternary | **0.45639** | **65.63%** | **0.71459** | **65.78%** | **66.48%** |

Released binary KL is 5.00% of the naive binary value, or approximately 20.0× lower. Released ternary KL is 4.05% of the naive ternary value, or approximately 24.7× lower.

Raw logit cosine remains sensitive to vocabulary-wide mean structure and is not used as the primary metric. Centering the logits produces the behavior expected from the KL and top-k measurements.

## Hidden-state coordinate alignment

### Binary

| Hidden index | Naive raw cosine | Released raw cosine | Released centered cosine |
|---:|---:|---:|---:|
| 0 | 0.83142 | 0.83138 | 0.82932 |
| 7 | 0.01003 | **0.69101** | 0.61105 |
| 14 | 0.09425 | **0.70986** | 0.56487 |
| 21 | 0.14771 | **0.73371** | 0.59615 |
| 28 | 0.06966 | **0.52910** | 0.57007 |

### Ternary

| Hidden index | Naive raw cosine | Released raw cosine | Released centered cosine |
|---:|---:|---:|---:|
| 0 | 0.91787 | 0.90853 | 0.90717 |
| 7 | 0.31420 | **0.75248** | 0.70392 |
| 14 | 0.43973 | **0.75993** | 0.66609 |
| 21 | 0.36522 | **0.78740** | 0.69483 |
| 28 | 0.20749 | **0.59845** | 0.66617 |

The released models stay in the original residual-stream coordinate system. Naive binary loses nearly all token-wise coordinate alignment by hidden index 7; released binary remains around 0.69–0.73 through the decoder. Ternary retains still stronger alignment.

## Channel sign gauge test

For each hidden dimension, the experiment fits the sign of its teacher/candidate covariance on training tokens and evaluates the relation on held-out tokens.

| Candidate | Index 7 | Index 14 | Index 21 | Index 28 |
|---|---:|---:|---:|---:|
| Naive binary negative channels | 51.27% | 52.20% | 49.07% | 51.95% |
| Released binary negative channels | **0.00%** | **0.00%** | **0.00%** | **0.10%** |
| Naive ternary negative channels | 21.63% | 40.87% | 43.70% | 47.22% |
| Released ternary negative channels | **0.00%** | **0.00%** | **0.00%** | **0.05%** |

This independently rejects a hidden-state sign gauge. Released channels preserve their orientation relative to Qwen even after extensive weight-code reassignment.

The result agrees with the direct weight-gauge experiment, where optimal row/column sign transformations improved code agreement by less than 0.001 percentage points.

## Static diagonal affine test

A channel-wise affine map was fitted on 80% of pooled tokens and evaluated on the remaining 20%.

| Hidden index | Released binary held-out R² | Released ternary held-out R² |
|---:|---:|---:|
| 0 | 0.69740 | 0.82722 |
| 7 | 0.01263 | 0.01917 |
| 14 | 0.01908 | 0.03601 |
| 21 | 0.09268 | 0.10725 |
| 28 | **0.78895** | **0.82415** |

A single static per-channel rescaling and offset does **not** explain the intermediate computation. It accounts for only roughly 1–11% of held-out intermediate variance despite the strong raw and centered cosine. The token-dependent nonlinear computation remains materially different.

At the final hidden state, however, the diagonal affine relation becomes strong: 78.90% for binary and 82.41% for ternary. Signed variance matching alone reaches 68.28% and 75.09%.

This supports a specific picture:

```text
intermediate residual stream:
    same coordinate orientation
    high directional alignment
    token-dependent differences not reducible to a fixed scale gauge

final normalized hidden state:
    strongly recoverable by channel-wise calibration
    then consumed by the tied LM head
```

The likely public reconstruction therefore needs whole-model behavioral recovery plus accurate final-state/channel calibration. It should not add a hidden sign gauge or assume that intermediate differences can be removed by static scale vectors.

## Representation caution

Pooled CKA is high at several depths but unusually low at hidden index 21 despite high token cosine. Transformer hidden states are anisotropic, and CKA can be strongly affected by which token-to-token variation dominates after centering. It is retained as supporting evidence rather than used alone.

The 128-dimensional orthogonal probe also fails to provide a stable improvement over original coordinates at intermediate depths. Because it estimates a relatively high-dimensional map from 304 training tokens, it is treated as a negative indication rather than definitive proof against every possible rotation.

The directly robust results are:

- high released raw and centered cosine;
- near-zero negative-channel fraction;
- low held-out intermediate diagonal R²;
- high final-state diagonal R²;
- low output KL and high top-k overlap.

## Updated highest-confidence reconstruction

```text
pretrained Qwen
-> preserve original head/neuron/channel ordering and orientation
-> direct embedding projection policy
-> broad code reassignment inside fixed coordinates
-> learned structured and free group scales
-> whole-model teacher-distribution recovery
-> token-dependent intermediate compensation
-> strong final hidden-state channel calibration
-> tied low-bit LM head and exact packed export
```

## Provenance

```text
workflow run:          30865992711
artifact ID:           8876266638
artifact ZIP SHA-256:  042f8b79999c0549c1240f3df297e770132dd179e0727bf6395560d6a404d18e
full JSON SHA-256:     a3251466961a1168b3b87dd901727c9ed7173de6f68bbf42b1acdd4e762d22cf
runtime:               468.17 seconds
```

Compact machine-readable summary:

- `experiments/public_bonsai_forensics/results_pooled_functional_alignment.json`
