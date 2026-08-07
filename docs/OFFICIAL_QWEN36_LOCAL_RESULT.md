# Official Qwen3.6 hybrid local result

> This is the earlier 40-step result. The newer three-seed, 600-step matrix selects binary categorical and ternary CAT-Q-to-hard on teacher KL. See [`LONG_ALL_METHODS_600_STEP.md`](LONG_ALL_METHODS_600_STEP.md). The reversal is evidence that the preferred surrogate depends on recovery budget.

The official hybrid-architecture method matrix was run locally on the CPU runtime, not through GitHub Actions.

```text
implementation:    transformers qwen3_5_text official hybrid architecture
layers:            4 and 8
seeds:             2 per depth
teacher steps:     20
recovery steps:    40
batch:             8
threads:           9
runtime:           259.97 seconds
result SHA-256:    5bebd3917dfcd614499b8f8180d871aaa37eed7e78e406eefb2152cec8390434
```

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|
| FP32 teacher reference | 5.59407 +/- 0.00708 | 0.4313% +/- 0.1269% | ~0 | 1.00000 |
| Binary naive | 5.57544 | 0.3418% | 0.045539 | 0.53775 |
| **Binary hard** | **5.57448** | **0.3581%** | **0.018387** | **0.79585** |
| Binary categorical | 5.57453 | **0.3581%** | 0.018438 | 0.79043 |
| Ternary naive | **5.56524** | 0.3499% | 0.031343 | 0.69957 |
| **Ternary hard** | 5.57470 | 0.3825% | **0.013318** | **0.85326** |
| Ternary CAT-Q to hard | 5.57376 | **0.3988%** | 0.014220 | 0.84944 |

Exact-hard recovery is the behavior winner for both modes in this local configuration. Binary categorical remains essentially tied, while ternary CAT-Q-to-hard is useful but does not beat sustained hard recovery.

The naive ternary student's label CE is lower than the FP32 teacher's on this weak synthetic task, while its teacher KL and hidden cosine are much worse. This is why CE, KL, and hidden alignment are reported together rather than treating one number as intelligence retention.

This is architecture-level evidence, not a full pretrained 27B intelligence benchmark. The teacher was trained briefly on synthetic data.

Compact result:

- `experiments/official_qwen36_text/results_official_qwen36_local_summary.json`
