# Long all-method 600-step CPU benchmark

## Contract

Every recovery row used 600 optimizer steps, three seeds, four decoder layers, batch 12, FP32 CPU execution, and exact contiguous g128 codes. Each teacher used 600 FP32 training steps. Qwen3-VL and Qwen3.6 results use different architectures and seed sets, so rank methods only within a table.

The Qwen3-VL miniature uses the official Hugging Face decoder implementation. The Qwen3.6 miniature uses the official `qwen3_5_text` hybrid pattern: three linear-attention layers plus one full-attention layer. These are architecture-real miniature experiments, not pretrained 27B intelligence benchmarks.

`Fidelity proxy = exp(-Teacher KL)`. It is a local distribution-agreement proxy, not an intelligence-retention percentage.

## FP32 teacher references

| Family | Architecture | FP32 CE ↓ | FP32 accuracy ↑ | Teacher KL ↓ | Hidden cosine ↑ |
|---|---|---:|---:|---:|---:|
| surrogates | Qwen3-VL | 5.5517 +/- 0.2190 | 11.14 +/- 0.85% | ~0 | 1.0000 |
| profiles | Qwen3-VL | 5.3130 +/- 0.1885 | 13.68 +/- 0.82% | ~0 | 1.0000 |
| losses | Qwen3-VL | 5.4545 +/- 0.5384 | 13.40 +/- 3.00% | ~0 | 1.0000 |
| ternary_schedules | Qwen3-VL | 5.2970 +/- 0.3360 | 13.77 +/- 2.60% | ~0 | 1.0000 |
| embedding_policies | Qwen3-VL | 5.3201 +/- 0.5933 | 14.96 +/- 2.11% | ~0 | 1.0000 |
| scale_structure | Qwen3-VL | 5.2477 +/- 0.4355 | 15.22 +/- 0.81% | ~0 | 1.0000 |
| shared_embedding | Qwen3-VL | 5.5132 +/- 0.3251 | 13.60 +/- 0.82% | ~0 | 1.0000 |
| qwen36_hybrid | Qwen3.6 hybrid | 9.8425 +/- 0.0335 | 0.39 +/- 0.07% | ~0 | 1.0000 |

Teacher validation quality varies by seed family. The Qwen3.6 teacher remained at chance accuracy despite the longer run, so its KL/cosine rows test architectural recovery mechanics only.

## Recovery surrogate matrix

### Naive, exact-hard, categorical, and CAT-Q

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|---:|
| binary_naive | 4.7932 +/- 0.1878 | 5.14 +/- 0.51% | 2.2998 +/- 0.1836 | 10.03% | 0.6806 +/- 0.0242 |
| binary_hard | 4.6972 +/- 0.1743 | 11.50 +/- 0.88% | 1.7445 +/- 0.1710 | 17.47% | 0.7356 +/- 0.0140 |
| binary_categorical | 4.7937 +/- 0.1892 | 11.30 +/- 0.74% | 1.6940 +/- 0.1656 | 18.38% | 0.7441 +/- 0.0173 |
| ternary_naive | 4.3542 +/- 0.1073 | 7.10 +/- 0.76% | 1.5167 +/- 0.1263 | 21.94% | 0.8056 +/- 0.0139 |
| ternary_hard | 4.7059 +/- 0.1626 | 12.15 +/- 1.01% | 1.2022 +/- 0.1532 | 30.05% | 0.8286 +/- 0.0146 |
| ternary_catq | 4.8189 +/- 0.1725 | 12.02 +/- 1.62% | 0.9837 +/- 0.1318 | 37.39% | 0.8572 +/- 0.0134 |
| ternary_categorical | 4.7891 +/- 0.1551 | 11.84 +/- 0.99% | 1.1867 +/- 0.1435 | 30.52% | 0.8358 +/- 0.0148 |

### Uniform versus public layer/module pressure

| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement ↔ target | Layer RMSE ↓ | Module RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| binary_hard_ste_uniform | 1.6419 +/- 0.2864 | 19.36% | 0.7409 +/- 0.0302 | 11.73 +/- 0.13% | 0.1681 +/- 0.0013 | 0.1629 +/- 0.0015 |
| binary_hard_ste_public | 1.6184 +/- 0.2951 | 19.82% | 0.7427 +/- 0.0331 | 11.92 +/- 0.13% | 0.1658 +/- 0.0013 | 0.1610 +/- 0.0016 |
| binary_categorical_uniform | 1.5995 +/- 0.2966 | 20.20% | 0.7453 +/- 0.0319 | 10.67 +/- 0.14% | 0.1785 +/- 0.0013 | 0.1736 +/- 0.0018 |
| binary_categorical_public | 1.5956 +/- 0.2757 | 20.28% | 0.7451 +/- 0.0297 | 10.86 +/- 0.15% | 0.1763 +/- 0.0015 | 0.1719 +/- 0.0020 |
| ternary_hard_ste_uniform | 1.1154 +/- 0.2233 | 32.78% | 0.8345 +/- 0.0263 | 17.52 +/- 0.29% | 0.2078 +/- 0.0029 | 0.2043 +/- 0.0036 |
| ternary_hard_ste_public | 1.1238 +/- 0.2122 | 32.50% | 0.8341 +/- 0.0238 | 17.73 +/- 0.29% | 0.2053 +/- 0.0029 | 0.2017 +/- 0.0039 |
| ternary_auto_uniform | 0.9573 +/- 0.1995 | 38.39% | 0.8553 +/- 0.0230 | 12.87 +/- 0.34% | 0.2540 +/- 0.0033 | 0.2512 +/- 0.0044 |
| ternary_auto_public | 0.9703 +/- 0.2022 | 37.90% | 0.8551 +/- 0.0234 | 12.96 +/- 0.31% | 0.2530 +/- 0.0031 | 0.2494 +/- 0.0040 |

### Binary recovery loss objectives

| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement |
|---|---:|---:|---:|---:|
| ce_only | 2.4815 +/- 0.1805 | 8.36% | 0.6940 +/- 0.0127 | 11.78% |
| kd_only | 1.8543 +/- 0.1709 | 15.66% | 0.7219 +/- 0.0141 | 11.37% |
| kd_ce | 1.8859 +/- 0.1445 | 15.17% | 0.7196 +/- 0.0125 | 11.41% |
| kd_hidden | 1.8279 +/- 0.2058 | 16.08% | 0.7279 +/- 0.0175 | 11.37% |
| kd_ce_hidden | 1.8477 +/- 0.1406 | 15.76% | 0.7257 +/- 0.0141 | 11.42% |

### Ternary soft-to-hard schedules

| Method | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ | Code movement ↔ target | Zero ratio |
|---|---:|---:|---:|---:|---:|
| hard | 1.2327 +/- 0.0098 | 29.15% | 0.8237 +/- 0.0013 | 17.35 +/- 0.13% | 28.93 +/- 0.04% |
| catq_hard25 | 1.2208 +/- 0.0532 | 29.50% | 0.8272 +/- 0.0028 | 15.98 +/- 0.15% | 29.33 +/- 0.06% |
| catq_hard50 | 1.0738 +/- 0.0239 | 34.17% | 0.8433 +/- 0.0020 | 12.91 +/- 0.23% | 30.22 +/- 0.04% |
| catq_hard75 | 0.9426 +/- 0.0251 | 38.96% | 0.8577 +/- 0.0033 | 10.69 +/- 0.20% | 30.51 +/- 0.04% |
| categorical_hard50 | 1.2118 +/- 0.0432 | 29.76% | 0.8260 +/- 0.0038 | 15.74 +/- 0.18% | 29.31 +/- 0.05% |

### Embedding policies

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|---:|
| binary_frozen_ptq_embedding | 4.5172 +/- 0.4567 | 14.01 +/- 2.67% | 1.8121 +/- 0.3055 | 16.33% | 0.7171 +/- 0.0274 |
| binary_train_embedding | 4.4825 +/- 0.5076 | 14.32 +/- 2.27% | 1.8593 +/- 0.2926 | 15.58% | 0.7156 +/- 0.0280 |
| ternary_sign_locked_embedding | 4.4977 +/- 0.4025 | 15.19 +/- 1.97% | 1.1333 +/- 0.2211 | 32.20% | 0.8376 +/- 0.0227 |
| ternary_train_embedding | 4.4903 +/- 0.3968 | 15.05 +/- 1.50% | 1.1129 +/- 0.2047 | 32.86% | 0.8380 +/- 0.0219 |
| ternary_frozen_ptq_embedding | 4.4498 +/- 0.4114 | 15.69 +/- 2.25% | 1.0914 +/- 0.2213 | 33.57% | 0.8363 +/- 0.0207 |

### Scale-structure regularization

| Mode | Coefficient | Teacher KL ↓ | Paired KL delta ↓ | Improved seeds | Exact p | Additive R2 ↑ | Code movement |
|---|---:|---:|---:|---:|---:|---:|---:|
| binary | 0 | 1.8455 | +0.0000 | 0/3 | 1.000 | 0.5756 | 11.79% |
| binary | 0.03 | 1.8584 | +0.0128 | 1/3 | 0.750 | 0.6001 | 11.78% |
| binary | 0.1 | 1.8263 | -0.0192 | 3/3 | 0.250 | 0.6531 | 11.79% |
| binary | 0.3 | 1.8464 | +0.0008 | 2/3 | 1.000 | 0.7570 | 11.79% |
| binary | 1 | 1.8471 | +0.0016 | 2/3 | 1.000 | 0.9279 | 11.78% |
| ternary | 0 | 1.1451 | +0.0000 | 0/3 | 1.000 | 0.5798 | 12.97% |
| ternary | 0.3 | 1.1584 | +0.0133 | 1/3 | 0.750 | 0.8927 | 12.99% |
| ternary | 1 | 1.1468 | +0.0016 | 1/3 | 1.000 | 0.9734 | 12.97% |
| ternary | 3 | 1.1661 | +0.0209 | 1/3 | 0.500 | 0.9900 | 12.98% |
| ternary | 10 | 1.1434 | -0.0018 | 1/3 | 1.000 | 0.9960 | 12.97% |

### Shared binary-codebook/ternary-mask policies

| Policy | Combined KL ↓ | Binary KL ↓ | Ternary KL ↓ | Binary cosine ↑ | Ternary cosine ↑ | Exact mask relation ↑ | KL / independent ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent | 2.9522 +/- 0.6165 | 1.8060 +/- 0.3642 | 1.1462 +/- 0.2523 | 0.7235 +/- 0.0433 | 0.8378 +/- 0.0315 | 28.75% | 1.0000 |
| shared_frozen_sign | 2.9283 +/- 0.5999 | 1.8249 +/- 0.3485 | 1.1034 +/- 0.2515 | 0.7223 +/- 0.0446 | 0.8406 +/- 0.0302 | 100.00% | 0.9919 |
| shared_mask_focused_sign | 2.9617 +/- 0.5629 | 1.8568 +/- 0.3076 | 1.1049 +/- 0.2561 | 0.7222 +/- 0.0421 | 0.8389 +/- 0.0316 | 100.00% | 1.0032 |
| shared_full_sign | 2.9389 +/- 0.5899 | 1.8515 +/- 0.3383 | 1.0874 +/- 0.2529 | 0.7210 +/- 0.0441 | 0.8399 +/- 0.0310 | 100.00% | 0.9955 |

### Official Qwen3.6 hybrid architecture

| Method | CE ↓ | Accuracy ↑ | Teacher KL ↓ | Fidelity proxy ↑ | Hidden cosine ↑ |
|---|---:|---:|---:|---:|---:|
| binary_naive | 8.2595 +/- 0.0598 | 0.33 +/- 0.03% | 4.5724 +/- 0.0515 | 1.03% | 0.2675 +/- 0.0058 |
| binary_hard | 8.3140 +/- 0.0863 | 0.43 +/- 0.19% | 3.6924 +/- 0.0439 | 2.49% | 0.3322 +/- 0.0099 |
| binary_categorical | 8.3881 +/- 0.0437 | 0.35 +/- 0.11% | 3.6803 +/- 0.0458 | 2.52% | 0.3418 +/- 0.0100 |
| ternary_naive | 8.1356 +/- 0.0375 | 0.35 +/- 0.07% | 3.6872 +/- 0.0208 | 2.50% | 0.4557 +/- 0.0046 |
| ternary_hard | 8.2114 +/- 0.0332 | 0.37 +/- 0.06% | 3.0629 +/- 0.0483 | 4.68% | 0.4418 +/- 0.0100 |
| ternary_catq_hard | 8.1670 +/- 0.0575 | 0.33 +/- 0.05% | 2.9850 +/- 0.0429 | 5.05% | 0.4567 +/- 0.0101 |

## Main findings

- Qwen3-VL behavior winners: binary `binary_categorical` and ternary `ternary_catq`.
- Public pressure helped both binary surrogates slightly; `binary_categorical_public` won the binary profile table. Uniform `ternary_auto_uniform` won ternary behavior, while exact-hard retained more code movement.
- `kd_hidden` had the lowest binary loss-ablation KL. CE-only was clearly worse at teacher retention.
- `catq_hard75` had the best ternary schedule KL/cosine, but moved fewer codes than sustained exact-hard recovery.
- Long embedding behavior favored `binary_frozen_ptq_embedding` and `ternary_frozen_ptq_embedding` within their respective modes.
- Binary scale coefficient `0.1` improved all three paired seeds, but the exact two-sided p-value is `0.25`; retain it as experimental. Ternary coefficient `10` won by only `0.00175` KL and improved one of three seeds, so it is not a reliable default.
- `shared_frozen_sign` produced the best combined shared-pair KL and an exact released-format embedding relation. It improved combined KL by about 0.81% versus independent recovery, but slightly worsened binary KL while improving ternary KL.
- On Qwen3.6, long recovery selected binary `binary_categorical` and ternary `ternary_catq_hard`. The chance-level FP32 teacher prevents intelligence-retention claims.
- The stronger Qwen3-VL teachers drive fidelity proxies far below the earlier weak-teacher ~98% values. That confirms `exp(-KL)` must not be reported as percent intelligence kept.

## Coverage boundary

This suite covers every runnable official-architecture recovery matrix currently present on `main`, plus the branch-preserved scale and shared-embedding runners restored here: 49 named low-bit configurations across eight families. Repeated baselines remain because each family uses its own deterministic seed set.

Not assigned artificial CE/KL rows: hidden gauges, head/neuron permutations, static affine alignment, and checkpoint lineage are forensic hypotheses rather than trainable recovery methods. Network-dependent 1.7B/4B/8B/27B checkpoint studies retain their existing reports. Fisher/metric-projection primitives and inactive config flags do not yet have an official-architecture end-to-end runner; they are not mislabeled as completed long benchmarks.

## Provenance

| Raw file | Runtime | SHA-256 |
|---|---:|---|
| `results_surrogates_600x3.json` | 1399.13s | `5262a14f5827fc2e0c54c9b50f97c45cae1df59ff080c11f4bf4b417e4b77761` |
| `results_profiles_600x3.json` | 1640.23s | `7c8c65a7e20dbaaaf70027188a3247fab897df1a551deab055b60fc4d2fd8766` |
| `results_losses_600x3.json` | 910.24s | `1f1d45b1039dfc2ba71e1cf84b3c0c7da0c7cebef540ddbb0ca120bfeb6397e6` |
| `results_ternary_schedules_600x3.json` | 1373.33s | `d999452f9503a5ea2a0cee7ee2ab4b00c386a79d69394c930ad6544751688c3f` |
| `results_embedding_policies_600x3.json` | 1025.44s | `6f2957dc0c74de2c4bbcaf48fa85a0682364c0b41a177b93505c116d47dd6abb` |
| `results_scale_structure_600x3.json` | 1706.57s | `e254acc1349c01e586c0b53473ef9eda30f278039b38b75b4edf2c528110536b` |
| `results_shared_embedding_600x3.json` | 1301.72s | `9820ab7d8859920702332e908505e98074817354c53496ce477102e0053e6a8b` |
| `results_qwen36_hybrid_600x3.json` | 4169.53s | `6601ff9a1c1b216d9bbc141fba450b8b40c26a889e2bf159e5ef46c3f4e9c3e0` |
| `results_teacher_profiles_600x3.json` | 72.25s | `014007652d22f293e12f075b14d3493b13de69c60ea05620f3eb357f448b7e97` |
| `results_teacher_scale_structure_600x3.json` | 72.46s | `6d2f8458167224caf7b4c195b95db748e547fb487569e990f7290bc1d4c9ba18` |
| `results_teacher_shared_embedding_600x3.json` | 72.21s | `04a56103e0e90816bab6a987e939fe0c648ed4d863e41d6595e836e020590862` |

Total recorded matrix runtime: **3.82 core-hours-equivalent** across concurrently executed CPU processes. No GitHub Actions were used.
