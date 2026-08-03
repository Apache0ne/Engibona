# Official Qwen3-VL ternary hardening schedule matrix

A clean GitHub-hosted run tested five exact-final ternary recovery paths on Hugging Face's public `Qwen3VLTextModel` at two and four decoder layers, with three independent seeds at each depth and 120 recovery steps.

All paths ended with exact `{-1,0,+1}` codes.

## Methods

- `hard`: exact-hard ternary forward throughout recovery;
- `catq_hard25`: CAT-Q soft phase, exact-hard after 25% of steps;
- `catq_hard50`: CAT-Q soft phase, exact-hard after 50% of steps;
- `catq_hard75`: CAT-Q soft phase, exact-hard after 75% of steps;
- `categorical_hard50`: categorical relaxation, exact-hard after 50%.

## Two layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine | Code change | Zero ratio |
|---|---:|---:|---:|---:|---:|---:|
| Hard | **4.52334** | 0.03309 | 0.13447 | 0.76915 | 17.74% | 28.69% |
| CAT-Q hard 25% | 4.53199 | 0.03570 | 0.13484 | 0.77030 | 17.06% | 28.96% |
| CAT-Q hard 50% | 4.54475 | 0.03624 | **0.13049** | 0.77761 | 15.78% | 29.30% |
| CAT-Q hard 75% | 4.55077 | 0.03168 | 0.13170 | **0.77790** | 13.66% | 29.82% |
| Categorical hard 50% | 4.52602 | **0.03895** | 0.13606 | 0.76828 | 17.30% | 28.61% |

CAT-Q hard at 50% had lower teacher KL than exact-hard on all three seeds.

## Four layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine | Code change | Zero ratio |
|---|---:|---:|---:|---:|---:|---:|
| Hard | 4.29735 | 0.02626 | 0.04632 | 0.87466 | 14.39% | 29.13% |
| CAT-Q hard 25% | 4.30117 | 0.02582 | 0.04452 | 0.88051 | 13.89% | 29.22% |
| CAT-Q hard 50% | 4.30724 | 0.02561 | **0.04218** | 0.88687 | 12.78% | 29.38% |
| CAT-Q hard 75% | 4.32534 | 0.02398 | 0.04272 | **0.89319** | 11.01% | 29.79% |
| Categorical hard 50% | **4.29292** | **0.02821** | 0.04607 | 0.87823 | 14.11% | 28.85% |

CAT-Q hard at 50% had lower teacher KL than exact-hard on all three seeds. It also beat the 25% transition on two of three seeds and the 75% transition on two of three seeds.

## Selected default

The most stable teacher-behavior choice is:

```text
CAT-Q soft ternary assignment for the first half of recovery
-> exact-hard ternary forward for the second half
-> preserve trained exact codes and scales
```

The default `hard_recovery_start` is therefore changed from `0.55` to `0.50`.

## Mechanistic pattern

Later hardening produced:

- fewer final code changes;
- slightly higher zero ratios;
- larger scale movement;
- stronger hidden alignment;
- but no monotonic improvement in teacher KL or task CE.

This indicates a real tradeoff. Remaining soft longer stabilizes internal geometry, while an adequately long exact-hard phase is needed to adapt the final discrete codes. The midpoint transition was the strongest teacher-behavior compromise at both depths.

## Confidence update

- soft-to-hard ternary continuation: high confidence;
- exact-hard recovery phase: high confidence;
- transition near the middle of recovery: medium-high confidence;
- one universal objective winner across CE, KL, accuracy, and hidden alignment: rejected;
- categorical relaxation as the primary ternary path: lower confidence than CAT-Q under teacher-behavior criteria.

## Provenance

```text
workflow run: 30860011218
artifact ID: 8874139430
artifact SHA-256: ec582eac55f435659a93dc699738c496162368a7e1865e0664e4b13b75aa6eee
runtime: 239.56 seconds
```

Raw results are stored in `experiments/official_qwen3vl_text/results_official_ternary_schedule_matrix.json`.
