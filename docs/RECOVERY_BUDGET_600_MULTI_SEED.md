# 600-step multi-seed recovery findings

## Scope

This experiment tests whether a longer recovery budget closes the discrete-code movement gap found in the short-budget matrix while retaining teacher behavior.

It ran entirely on the local CPU runtime, not GitHub Actions:

```text
implementation:       transformers.Qwen3VLTextModel
decoder layers:       8
seeds:                16700, 16701, 16702
teacher steps:        80
recovery steps:       600
learning rate:        1.4e-3
batch:                12
CPU threads:          9
runtime:              862.23 seconds
```

The miniature uses Hugging Face's official Qwen3-VL text implementation with width 128, 4 query heads, 2 KV heads, Q/K normalization, RMSNorm, SwiGLU, interleaved MRoPE, and tied embedding/LM-head weights. This preserves the relevant transformer and quantization-aware backpropagation graph, but not full-model capacity or intelligence.

## Target definitions

The experiment intentionally separates three different targets:

1. **representation correctness:** every deployed code must be in the exact binary or ternary alphabet;
2. **released geometry:** code movement should approach the mean public 1.7B layer target, 27.889% binary or 37.691% ternary;
3. **behavior:** teacher KL should approach zero and hidden cosine should approach one.

For readability, `exp(-teacher KL)` is reported as an output-distribution fidelity proxy. It equals 100% at KL zero. It is not an intelligence or benchmark-retention score.

## Results

Values are mean plus or minus population standard deviation over three seeds.

| Method | Exact alphabet | Teacher KL | Output fidelity proxy | Hidden cosine | Code movement | Public target | Target coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Binary hard STE | 100% | 0.02185 +/- 0.00269 | 97.84% | 80.98% +/- 1.19% | 20.08% +/- 0.26% | 27.89% | 71.99% |
| Binary categorical | 100% | **0.01917 +/- 0.00206** | **98.10%** | **81.59% +/- 0.89%** | 16.69% +/- 0.32% | 27.89% | 59.83% |
| Ternary hard STE | 100% | **0.02048 +/- 0.00279** | **97.97%** | 84.97% +/- 1.09% | **32.81% +/- 0.55%** | 37.69% | **87.06%** |
| Ternary auto | 100% | 0.02130 +/- 0.00217 | 97.89% | **86.82% +/- 0.84%** | 28.57% +/- 0.46% | 37.69% | 75.80% |

## Selection

- **Binary behavior:** categorical has the lowest teacher KL and highest hidden cosine.
- **Binary geometry:** hard STE moves substantially more codes and is closer to the public magnitude.
- **Ternary overall:** hard STE is the closest geometry within 10% of the minimum ternary KL.
- **Ternary hidden alignment:** auto has the highest hidden cosine but undershoots public movement more strongly.

The 600-step result confirms that duration/effective learning-rate budget was a real limiting factor. Ternary hard code movement rose to 32.81%, covering 87.06% of the public target while maintaining a 0.02048 mean teacher KL. Binary hard improved but remains materially short at 20.08%, or 71.99% target coverage.

## Intelligence-retention boundary

The full-precision teacher control measured:

```text
validation CE:        4.70705 +/- 0.02959
validation accuracy:  1.52995% +/- 0.28749%
random-vocabulary CE: ln(128) = 4.85203
```

The teacher learned only a small amount of the synthetic recurrence in 80 steps. Consequently, the 97.84-98.10% output-fidelity proxies establish close imitation of this teacher, not 98% retention of a capable Qwen model. These values cannot be compared directly with reported PrismML benchmark-retention values.

The next retention experiment must train the teacher to convergence or load a real pretrained checkpoint, then report full-precision-normalized task scores:

```text
retention = 100 * quantized benchmark score / full-precision benchmark score
```

The near-term candidates carried forward are binary hard STE, for geometry, and ternary hard STE, for the strongest joint geometry/behavior result.

## Reproduction

```bash
PYTHONPATH=src OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 python \
  experiments/official_qwen3vl_text/run_recovery_budget_matrix.py \
  --budgets 600:1.4e-3 \
  --seeds 3 \
  --layers 8 \
  --teacher-steps 80 \
  --batch 12 \
  --threads 9 \
  --output recovery_budget_600_multiseed.json
```

Committed compact result:

- `experiments/official_qwen3vl_text/results_recovery_budget_600_multiseed_summary.json`

Local full-result SHA-256:

```text
0ce57c3e76942b6af46b0c6e25b557032d1258a862e779bb6b00d839e1be42cc
```
