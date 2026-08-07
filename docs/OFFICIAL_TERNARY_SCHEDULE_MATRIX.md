# Official Qwen3-VL ternary hardening schedule matrix

A clean GitHub-hosted run tested five exact-final ternary recovery paths on Hugging Face's public `Qwen3VLTextModel` at two and four decoder layers, with three independent seeds at each depth and 120 recovery steps.

All paths ended with exact `{-1,0,+1}` codes.

## Methods

- `hard`: exact-hard ternary throughout;
- `catq_hard25`: CAT-Q soft phase, hard after 25%;
- `catq_hard50`: CAT-Q soft phase, hard after 50%;
- `catq_hard75`: CAT-Q soft phase, hard after 75%;
- `categorical_hard50`: categorical relaxation, hard after 50%.

## Two layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine | Code change | Zero ratio |
|---|---:|---:|---:|---:|---:|---:|
| Hard | **4.52334** | 0.03309 | 0.13447 | 0.76915 | 17.74% | 28.69% |
| CAT-Q hard 25% | 4.53199 | 0.03570 | 0.13484 | 0.77030 | 17.06% | 28.96% |
| CAT-Q hard 50% | 4.54475 | 0.03624 | **0.13049** | 0.77761 | 15.78% | 29.30% |
| CAT-Q hard 75% | 4.55077 | 0.03168 | 0.13170 | **0.77790** | 13.66% | 29.82% |
| Categorical hard 50% | 4.52602 | **0.03895** | 0.13606 | 0.76828 | 17.30% | 28.61% |

## Four layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine | Code change | Zero ratio |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 4.29735 | 0.02626 | 0.04632 | 0.87466 | 14.39% | 29.13% |
| CAT-Q hard 25% | 4.30117 | 0.02582 | 0.04452 | 0.88051 | 13.89% | 29.22% |
| CAT-Q hard 50% | 4.30724 | 0.02561 | **0.04218** | 0.88687 | 12.78% | 29.38% |
| CAT-Q hard 75% | 4.32534 | 0.02398 | 0.04272 | **0.89319** | 11.01% | 29.79% |
| Categorical hard 50% | **4.29292** | **0.02821** | 0.04607 | 0.87823 | 14.11% | 28.85% |

CAT-Q hard at 50% had lower teacher KL than exact-hard on all three seeds at both depths. It was the strongest teacher-behavior compromise across depth.

## Selected default

```text
CAT-Q soft ternary assignment for the first half
-> exact-hard ternary recovery for the second half
-> preserve trained exact codes and scales
```

The selected `hard_recovery_start` is `0.50`.

Later hardening produced fewer code changes, slightly higher zero ratios, larger scale movement, and stronger hidden alignment, but did not monotonically improve teacher KL or task CE. The midpoint retained enough soft adaptation while leaving a substantial exact-hard phase.

## Provenance

```text
workflow run: 30860011218
artifact ID: 8874139430
artifact SHA-256: ec582eac55f435659a93dc699738c496162368a7e1865e0664e4b13b75aa6eee
runtime: 239.56 seconds
```

This selects the current public reconstruction default. It does not establish that PrismML used CAT-Q or the same transition point.
