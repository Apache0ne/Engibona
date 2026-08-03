# Tiny Qwen3-VL architecture-faithful CPU ablation

## Purpose

This experiment tests the transformation mathematics on the smallest practical decoder that preserves the relevant Qwen3-VL text topology. It is not intended to produce a useful language model or to prove PrismML's private recipe.

The test model has:

- 328,448 parameters in the package smoke configuration;
- hidden width 128;
- two decoder layers;
- four query heads and two key/value heads;
- head dimension 32;
- pre-norm RMSNorm;
- q/k head RMSNorm;
- grouped-query causal attention;
- RoPE theta 5,000,000;
- bias-free q/k/v/o projections;
- SwiGLU gate/up/down projections;
- low-bit embedding and LM-head coverage;
- exact contiguous group size 128.

The task is an autoregressive synthetic recurrence sequence. Quality is intentionally secondary; the test isolates which low-bit transformation math preserves the trained network's behavior.

## Deep three-seed ablation

Three independent FP teachers were trained with seeds 1234, 1235, and 1236. Binary methods were evaluated over sequence lengths 8, 12, 20, and 32.

| Method | CE mean ± std | Accuracy mean ± std | Teacher KL mean ± std | Hidden cosine |
|---|---:|---:|---:|---:|
| FP32 teacher | 3.17618 ± 0.06820 | 0.49853 ± 0.01122 | approximately 0 | 1.00000 |
| Naive sign + absmean | 5.03893 ± 0.05352 | 0.04741 ± 0.00103 | 3.62201 ± 0.01033 | 0.56972 |
| Activation-covariance coordinate PTQ | 5.75938 ± 0.02784 | 0.06158 ± 0.00074 | 4.00645 ± 0.08060 | 0.59454 |
| Smooth KD recovery | 3.46709 ± 0.09771 | 0.28199 ± 0.02059 | 1.95101 ± 0.06829 | 0.65287 |
| Staged KD recovery | 3.24553 ± 0.11357 | 0.29079 ± 0.01581 | 1.86080 ± 0.06440 | 0.65059 |
| Exact-hard STE + KD | 3.12885 ± 0.08439 | 0.33749 ± 0.02087 | 1.79099 ± 0.05307 | 0.65755 |
| Exact-hard STE + KD + Fisher flips | **3.10093 ± 0.08532** | **0.34239 ± 0.01500** | **1.76733 ± 0.05228** | 0.65636 |

## Paired results

Exact-hard STE + KD beat naive projection for every seed:

- CE reduction: 1.785 to 1.989;
- teacher-KL reduction: 1.787 to 1.910;
- accuracy increase: 0.271 to 0.317.

It also beat smooth recovery for every seed:

- CE reduction: 0.318 to 0.350;
- teacher-KL reduction: 0.143 to 0.189;
- accuracy increase: 0.053 to 0.058.

It beat staged recovery for every seed:

- CE reduction: 0.071 to 0.167;
- teacher-KL reduction: 0.040 to 0.104;
- accuracy increase: 0.040 to 0.053.

Fisher-guided exact sign refinement after hard recovery improved CE and teacher KL on all three seeds:

| Seed | CE change | KL change | Accuracy change |
|---:|---:|---:|---:|
| 1234 | -0.03082 | -0.02068 | -0.00532 |
| 1235 | -0.02115 | -0.01644 | +0.01325 |
| 1236 | -0.03179 | -0.03386 | +0.00677 |

The accuracy result was positive for two of three seeds, while behavior losses improved for all three.

## Finalization ablation

A separate seed-1234 test compared three ways to turn a recovered hard-QAT state into released weights:

| Finalization | CE | Teacher KL | Accuracy |
|---|---:|---:|---:|
| Preserve learned exact codes/scales | **3.2743** | **1.8904** | 0.2832 |
| Replace scales with covariance-optimal analytic scales | 3.8967 | 2.4271 | lower |
| Covariance scale plus local coordinate re-projection | 4.9813 | 3.4895 | lower |
| Freeze codes and polish scales for 80 steps | 3.2751 | 1.8890 | **0.2865** |

This falsified the original Engibona default export path. Local activation-covariance projection optimizes its own quadratic layer objective, but after global teacher-guided recovery it can destroy useful cross-layer compensation.

## Strong conclusions from this benchmark

### Raised confidence

- exact hard low-bit forwards during recovery;
- learned positive group scales;
- teacher-behavior recovery;
- meaningful code movement from the original signs;
- preserving the globally trained exact state at export;
- optional Fisher/gradient-informed discrete refinement with calibration line search;
- embeddings and LM head included in the low-bit path.

### Lowered confidence

- smooth relaxation as the universal binary default;
- local activation covariance as the final export objective;
- coordinate projection after global recovery;
- scale recomputation from the latent FP carrier at export;
- treating lower local reconstruction error as proof of better full-model behavior.

### Still unresolved

The experiment cannot identify PrismML's private optimizer, data, training duration, layer schedule, or whether its binary path uses an STE. It demonstrates which candidate mathematics worked on an architecture-faithful miniature and which earlier assumptions failed.

## Current default implied by the evidence

For binary recovery:

```text
PTQ/sign initialization
-> exact hard forward at every recovery step
-> trainable positive g128 scales
-> latent identity gradient for code movement
-> teacher KL plus task loss
-> preserve learned hard codes/scales
-> optional empirical-Fisher sign line search
-> exact packing
```

For ternary recovery, smooth CAT-Q-style assignment remains the default pending an equally deep ternary ablation, followed by a sustained exact-hard recovery phase and trained-state export.

## Reproduction

```bash
python experiments/tiny_qwen3vl/run_cpu_ablation.py \
  --seeds 3 \
  --fp-steps 300 \
  --recovery-steps 220 \
  --batch 24 \
  --threads 4 \
  --output results_package_v2.json
```

The full earlier three-seed output is stored in `results_three_seed_deep_ablation.json`. A short package-integrated smoke result is stored in `results_package_smoke.json`.
