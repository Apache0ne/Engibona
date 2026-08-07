# Official Qwen3.6 hybrid local result

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

| Method | Mean teacher KL | Change from naive |
|---|---:|---:|
| Binary naive | 0.045539 | baseline |
| **Binary hard** | **0.018387** | **-59.62%** |
| Binary categorical | 0.018438 | -59.51% |
| Ternary naive | 0.031343 | baseline |
| **Ternary hard** | **0.013318** | **-57.51%** |
| Ternary CAT-Q to hard | 0.014220 | -54.63% |

Exact-hard recovery is the behavior winner for both modes in this local configuration. Binary categorical remains essentially tied, while ternary CAT-Q-to-hard is useful but does not beat sustained hard recovery.

This is architecture-level evidence, not a full pretrained 27B intelligence benchmark. The teacher was trained briefly on synthetic data.

Compact result:

- `experiments/official_qwen36_text/results_official_qwen36_local_summary.json`
