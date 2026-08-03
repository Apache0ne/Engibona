# Official Qwen3-VL embedding-policy matrix

Direct released-weight forensics found two unusual embedding fingerprints:

- binary Bonsai embedding is almost exactly frozen sign plus mean-absolute g128 projection;
- ternary Bonsai embedding changes zero assignments and scales while preserving every sampled nonzero source sign.

A clean official-Qwen3-VL experiment tested whether those release-matched policies also maximize behavior on the miniature task.

## Two layers

| Policy | CE | Accuracy | Teacher KL | Hidden cosine | Embedding code change | Nonzero source-sign flip |
|---|---:|---:|---:|---:|---:|---:|
| Binary frozen PTQ | 4.56804 | 0.02724 | **0.16432** | **0.69262** | 0.00% | 0.00% |
| Binary trainable | **4.54837** | 0.02713 | 0.16731 | 0.68203 | 9.27% | 9.27% |
| Ternary sign-locked | **4.54811** | **0.03353** | 0.11259 | 0.79971 | 12.66% | **0.00%** |
| Ternary trainable | 4.55420 | 0.03049 | **0.11087** | 0.80226 | 12.60% | 0.25% |
| Ternary frozen PTQ | 4.56899 | 0.03060 | 0.11685 | **0.80756** | 0.00% | 0.00% |

## Four layers

| Policy | CE | Accuracy | Teacher KL | Hidden cosine | Embedding code change | Nonzero source-sign flip |
|---|---:|---:|---:|---:|---:|---:|
| Binary frozen PTQ | 4.35496 | **0.02648** | 0.04980 | 0.86341 | 0.00% | 0.00% |
| Binary trainable | **4.31627** | 0.02561 | **0.03677** | **0.86559** | 8.89% | 8.89% |
| Ternary sign-locked | 4.32925 | 0.02680 | 0.02834 | 0.90823 | 15.77% | **0.00%** |
| Ternary trainable | **4.32231** | **0.02832** | **0.02440** | 0.91095 | 14.86% | 1.25% |
| Ternary frozen PTQ | 4.36047 | **0.02843** | 0.04917 | **0.91284** | 0.00% | 0.00% |

## What this resolves

There are two distinct optimization targets.

### Release-matched reconstruction

Use the public checkpoint geometry as the primary constraint:

```text
binary embedding: frozen sign + mean-absolute g128 scale
ternary embedding: source signs locked; zero mask and scales recoverable
```

This is the default `EngibonaConfig.release_matched()` profile because the actual Bonsai release provides direct evidence for those policies.

### Behavior-maximizing adaptation

Allow all embedding codes to move:

```text
binary embedding: trainable signs and scales
ternary embedding: trainable signs, zero states, and scales
```

This sometimes improves miniature teacher KL, particularly at four layers, but produces a binary embedding fingerprint that conflicts with the actual released checkpoint. It is exposed as `EngibonaConfig.behavior_maximizing()` rather than silently replacing the release-matched default.

## Important corrected metric

The clean rerun measures final nonzero signs against the original FP embedding signs, including positions that began as ternary zero. Under this correct definition:

- sign-locked ternary recovery produced exactly 0% nonzero source-sign flips;
- fully trainable ternary recovery produced approximately 0.25% flips at two layers and 1.25% at four layers.

## Interpretation

The direct release evidence is stronger for reconstructing PrismML than the miniature task optimum. Therefore the highest-confidence public reconstruction should not optimize one scalar benchmark at the cost of contradicting released tensor geometry.

The embedding appears to be treated as an anchor:

- binary: nearly unchanged discrete projection;
- ternary: sign-preserving sparsity/scale recovery;
- transformer blocks: extensive code and scale re-optimization.

This module-specific split is now represented explicitly in Engibona.

## Provenance

```text
workflow run: 30862873139
artifact ID: 8875103247
artifact SHA-256: be9c8b808d29383fdb85c7bdd6e136536f82c2249f75f0b219fe57b739b7bfb9
```

The corrected aggregate data is stored in `experiments/official_qwen3vl_text/results_official_embedding_policy_summary.json`.
