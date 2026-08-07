# Full-model functional findings

## Scope

This experiment compares five public 1.7B checkpoints on identical prompt tokens:

1. original `Qwen/Qwen3-1.7B` teacher;
2. naive binary g128 projection of Qwen;
3. naive ternary g128 projection of Qwen;
4. released `prism-ml/Bonsai-1.7B-unpacked`;
5. released `prism-ml/Ternary-Bonsai-1.7B-unpacked`.

Models were loaded sequentially in BF16 on a clean GitHub-hosted CPU runner. The naive baselines quantized all 197 matrix tensors, covering 1,720,451,072 weights. The released Bonsai checkpoints expose 151,669 vocabulary rows while Qwen serializes 151,936 padded rows, so released-checkpoint logit comparisons use the renormalized shared vocabulary prefix.

This is direct functional evidence from public final checkpoints. It does not reveal PrismML's private optimizer, data mixture, learning-rate schedule, token count, or intermediate states.

## Aggregate result

| Candidate | Teacher KL | Last-token KL | Top-1 agreement | Hidden cosine |
|---|---:|---:|---:|---:|
| Naive binary | 9.81080 | 8.84887 | 0.00% | 0.11454 |
| Released binary | **0.69486** | **1.01229** | **72.54%** | **0.71731** |
| Naive ternary | 8.34530 | 9.48151 | 2.38% | 0.39261 |
| Released ternary | **0.53843** | **1.00736** | **71.27%** | **0.77170** |

Relative to naive projection:

- released binary teacher KL is **7.08%** as large, or about **14.1× lower**;
- released ternary teacher KL is **6.45%** as large, or about **15.5× lower**;
- released binary gains **72.54 percentage points** of top-1 agreement;
- released ternary gains **68.89 percentage points** of top-1 agreement.

These gaps are too large to attribute to group-scale fitting alone. They are direct evidence that the released checkpoints underwent substantial behavior-aware recovery after initial discretization.

## Layer trajectory

The first hidden state primarily reflects token embedding output. Binary embedding cosine is nearly identical for naive and released binary models:

```text
naive binary first-state cosine:     0.83018
released binary first-state cosine:  0.83015
```

This matches the independent weight-forensic result that binary embeddings are essentially direct sign plus mean-absolute-scale projection.

The trajectories separate sharply as decoder depth increases:

| Candidate | First hidden cosine | Middle hidden cosine | Final hidden cosine |
|---|---:|---:|---:|
| Naive binary | 0.83018 | 0.06636 | 0.10382 |
| Released binary | 0.83015 | **0.69391** | **0.47847** |
| Naive ternary | 0.91754 | 0.37000 | 0.41287 |
| Released ternary | 0.90804 | **0.75998** | **0.54902** |

Naive binary states collapse away from the teacher by the middle of the network. Released binary maintains strong alignment through depth. Ternary shows the same pattern at a higher absolute level.

The released final states retain high linear CKA despite lower coordinate-wise cosine:

```text
released binary final CKA:   0.89678
released ternary final CKA:  0.90898
```

This is consistent with a recovered internal operating point that preserves representational structure without retaining exact teacher coordinates. Because each prompt is short, per-prompt CKA is treated as supporting evidence rather than a standalone ranking metric. The next functional run should pool substantially more tokens before estimating representation-level transformations.

## What this falsifies

The clean full-model result rejects the following as complete explanations:

- one-pass sign or magnitude-threshold projection;
- scale-only recovery;
- minimizing raw weight MSE as the main objective;
- preserving only local Q/K, V/O, or MLP operators while leaving the rest untouched;
- assuming that high local or weight-space similarity is sufficient for model behavior.

Naive projections are locally closer to Qwen in raw weight MSE but catastrophically worse functionally. The released checkpoints are farther away in weight space yet vastly closer in output behavior.

## Strongest supported reconstruction

The combined public evidence now supports this family:

```text
pretrained Qwen checkpoint
-> exact g128 initialization
-> module-specific code and scale policy
-> whole-model or long-window behavioral recovery
-> depth-sensitive compensation
-> preserve trained discrete codes and scales
-> exact packed export with no FP residual path
```

Teacher-distribution matching remains the strongest tested public recovery objective. The released-checkpoint behavior does not prove that PrismML used the exact Engibona surrogate gradient or optimizer.

## Important logit observation

Released models have low KL and high top-1 agreement even though raw full-vector logit cosine is negative in this short run. Raw cosine is sensitive to global mean and scale structure across a 151k-token vocabulary and is not an appropriate sole similarity measure. The next run should add:

- centered logit cosine;
- affine-fit logit correlation;
- pooled-token representation metrics;
- held-out diagonal hidden-state alignment;
- more prompts and longer sequences.

## Provenance

```text
workflow run:            30865043522
artifact ID:             8875836367
artifact ZIP SHA-256:    21cef1851cde957b88a4a208f3c50e5eb29d4e2110d2cbc962829dc7a253f6f2
extracted JSON SHA-256:  6eec15f15d6a90873c969cfbe6797283a5456bf930b6979db528c3e1088810bf
runtime:                 189.60 seconds
```

Compact machine-readable results are stored in:

- `experiments/public_bonsai_forensics/results_full_model_functional_forensics.json`

The workflow artifact remains the authoritative source for prompt-level and complete per-layer arrays.
