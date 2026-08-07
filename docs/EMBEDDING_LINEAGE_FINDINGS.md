# Embedding lineage findings

## Central result

The public binary and ternary Bonsai embeddings are not independently recovered embedding matrices. Across both 1.7B and 4B, they are explained to FP16 rounding precision by a shared sign codebook and shared g128 scale, with the ternary checkpoint adding a zero mask:

\[
W_{binary}=s\,b,
\qquad
W_{ternary}=s\,b\,m,
\]

where

\[
b\in\{-1,+1\},
\qquad
m\in\{0,1\}.
\]

The direction of construction remains unknown: final checkpoints cannot determine whether binary signs were recovered first and masked, ternary zeros were filled to produce binary, or both were optimized jointly.

## Coverage

The experiment used remote safetensors byte ranges rather than full model downloads. It sampled deterministic contiguous row blocks over all vocabulary regions plus the final 1,024 rows.

| Model | Sampled rows | Sampled values | Width |
|---|---:|---:|---:|
| 1.7B | 4,992 | 10,223,616 | 2,048 |
| 4B | 5,120 | 13,107,200 | 2,560 |

Both comparisons use the 151,669 vocabulary rows common to Qwen and the released checkpoints.

## Shared binary sign codebook

| Metric | 1.7B | 4B |
|---|---:|---:|
| Binary/ternary sign agreement on ternary nonzeros | **100.000%** | **100.000%** |
| Ternary nonzero sign agreement with Qwen | 100.000% | 99.444% |
| Binary sign agreement with Qwen | 99.940% | 93.197% |

Wherever the ternary embedding retains a nonzero value, its sign always matches the binary embedding sign in the expanded sample. This is exact code-level evidence for one shared sign codebook.

The cross-scale difference is in the relation of that shared codebook to original Qwen:

- at 1.7B it is almost exactly `sign(W_Qwen)`;
- at 4B approximately 6.80% of signs are recovered away from Qwen.

Therefore, the earlier `frozen binary embedding` conclusion was valid for 1.7B but is not a universal Bonsai rule.

## Shared scales

| Metric | 1.7B | 4B |
|---|---:|---:|
| Binary/ternary scale correlation | **0.99999982** | **0.99999869** |
| Bit-exact inferred scale groups | 99.682% | 99.440% |
| Mean relative scale difference | 0.00186% | 0.00427% |
| Maximum relative scale difference | 2.10% | 3.95% |

The small nonexact tail is consistent with inferring scales from separately serialized FP16 unpacked values. It is not evidence for materially different scale states.

## Direct mask reconstruction

The released ternary matrix was reconstructed as:

```text
released_binary_weight * released_ternary_nonzero_mask
```

| Metric | 1.7B | 4B |
|---|---:|---:|
| Bit-exact reconstructed values | 99.782% | 99.616% |
| Mean absolute error | 2.37e-7 | 4.02e-7 |
| Maximum absolute error | 2.44e-4 | 1.83e-4 |

More than 99.6% of all values are bit-exact, and the remaining error is far below normal embedding weight scale. The relation is effectively exact modulo FP16 scale-rounding differences.

This is stronger than correlation evidence. It gives an explicit released-state algebra for the embedding matrices.

## Binary sign changes occur at ternary uncertainty positions

| Conditional metric | 1.7B | 4B |
|---|---:|---:|
| `P(binary flip | ternary zero)` | 0.193% | 20.580% |
| `P(binary flip | ternary nonzero)` | **0.000%** | **0.556%** |
| `P(ternary zero | binary flip)` | **100.000%** | **94.376%** |

At 1.7B, every sampled binary sign change from Qwen occurs at a coordinate removed by the ternary mask. At 4B, 94.38% do.

This makes the ternary mask a direct marker of low-confidence or high-recovery-pressure embedding coordinates. The common representation is likely closer to:

```text
shared binary sign codebook
shared g128 scales
ternary confidence/selection mask
```

than to two separately quantized embeddings.

## Vocabulary-position behavior

The 4B binary sign changes are not confined to special or tail vocabulary rows. Agreement stays near 92.7–94.1% across broad vocabulary deciles. The final vocabulary region is somewhat more modified, but the effect is global.

The ternary zero rate increases toward later vocabulary rows at both scales:

```text
row-index vs zero-rate correlation:
1.7B: +0.4213
4B:   +0.4664
```

Naive ternary agreement also falls in the final vocabulary region. Later and likely rarer tokens require more mask recovery, but the shared-codebook relation persists throughout the vocabulary.

## Corrected embedding implementation

The highest-confidence public reconstruction should represent binary and ternary embeddings jointly at the policy level:

```text
shared embedding state:
    binary sign codebook b
    positive g128 scale s

binary export:
    s * b

ternary export:
    s * b * learned mask m
```

Recommended recovery policy:

```text
initialize b = sign(Qwen embedding)
initialize s = mean_abs(Qwen g128 group)
initialize m from least-squares ternary thresholding

1.7B-like low-pressure case:
    nearly freeze b and s
    recover m

larger/higher-pressure case:
    permit sparse b changes
    concentrate b updates on m=0 or low-confidence coordinates
    recover m and small scale residuals
```

The binary and ternary embedding modules should not maintain unrelated codebooks or unrelated scales.

## Transformer contrast

This exact shared-mask relation is specific to embeddings. Transformer matrices show only approximately 89–90% binary/ternary nonzero sign agreement and scale correlation around 0.93–0.94. Their final binary and ternary states require separate recovery.

Thus the model has two distinct lineage regimes:

```text
embedding:
    one shared binary codebook + scale + ternary mask

transformer matrices:
    shared initialization/framework
    separately recovered final binary and ternary codes/scales
```

## Provenance

```text
workflow run:           30867090328
artifact ID:            8876542034
artifact ZIP SHA-256:   b24480eae019702184562692bbd579219184b35c0a9309210ceda70b36cbdd61
summary JSON SHA-256:   30cfe43e1233e3b5e25a2420b0fb573af6ad0c671c002605dae0b4b5934513a9
row-bin CSV SHA-256:    85d8088e338bfe76073f839243b78afbbf25703901e99d2bf6b687578d1350b0
runtime:                91.70 seconds
```

Machine-readable compact result:

- `experiments/public_bonsai_forensics/results_embedding_lineage_forensics.json`
