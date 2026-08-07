# Behavior and released-code drift findings

## Why behavior alone is insufficient

Teacher KL selects models that imitate the teacher output distribution. It does not establish that the discrete recovery trajectory resembles the public Bonsai converter. Two low-bit students can have similar KL while changing very different codes.

This experiment therefore uses two targets:

1. **behavior:** teacher KL on the official Hugging Face Qwen3-VL text miniature;
2. **released geometry:** final code-change fractions by decoder depth and module type, measured directly from public Qwen3-1.7B and Bonsai checkpoints.

The public geometry target excludes embeddings because they follow a separate shared-codebook/mask regime.

## Methods

At four and eight decoder layers, three paired seeds compare:

- binary exact-hard recovery;
- binary categorical recovery;
- ternary exact-hard recovery;
- ternary CAT-Q soft-to-hard recovery;
- uniform learning rate;
- a mild public-profile multiplier based on released layer/module code-change fractions.

All methods use the same teacher, initialization, and minibatch sequence within each mode.

## Aggregate result

| Method | Teacher KL | Layer-profile RMSE | Module-profile RMSE |
|---|---:|---:|---:|
| Binary hard, uniform | 0.04241 | 0.20654 | 0.20620 |
| **Binary hard, public profile** | 0.04129 | **0.20381** | **0.20369** |
| Binary categorical, uniform | 0.04107 | 0.21027 | 0.21048 |
| **Binary categorical, public profile** | **0.04029** | 0.20858 | 0.20904 |
| Ternary hard, uniform | 0.02571 | 0.25933 | 0.26058 |
| Ternary hard, public profile | 0.02630 | **0.25757** | **0.25847** |
| **Ternary CAT-Q, uniform** | **0.02176** | 0.27935 | 0.28127 |
| Ternary CAT-Q, public profile | 0.02267 | 0.27669 | 0.27804 |

The public-profile multiplier improves released-geometry similarity in every binary pairing and in nearly every ternary pairing.

## Paired public-profile effects

| Surrogate | KL change | Layer-RMSE change | Module-RMSE change | Layer-correlation gain |
|---|---:|---:|---:|---:|
| Binary hard | **-0.00112** | **-0.00273** | **-0.00252** | **+0.0464** |
| Binary categorical | **-0.00078** | **-0.00169** | **-0.00145** | **+0.0410** |
| Ternary hard | +0.00059 | **-0.00176** | **-0.00211** | **+0.0912** |
| Ternary CAT-Q | +0.00091 | **-0.00266** | **-0.00324** | **+0.0729** |

For binary recovery, released-geometry guidance also improves behavior on average. Categorical-public has the lowest aggregate KL; exact-hard-public is the closest tested geometry. The two remain complementary candidates rather than one being decisively established as the private method.

For ternary recovery, CAT-Q remains the behavior winner, while exact-hard changes more codes and is closer to released geometry. This reinforces the selected hybrid path: soft assignment for optimization followed by substantial exact-hard recovery.

## Depth shape is right; magnitude is not

At eight layers, binary exact-hard-public produces:

```text
miniature layer changes:
6.61%, 6.30%, 6.82%, 7.36%, 7.97%, 8.57%, 9.19%, 10.27%

mapped public target:
26.86%, 26.90%, 27.18%, 26.67%, 27.60%, 28.74%, 29.47%, 31.70%
```

The correlation is **0.9318**. The miniature has learned the correct depth ordering and late-layer rise, but its total discrete movement is roughly one quarter of the released checkpoint.

Ternary exact-hard-public shows the same pattern:

```text
miniature:
10.15%, 10.06%, 10.82%, 11.51%, 12.50%, 12.94%, 13.92%, 15.68%

mapped public target:
36.27%, 37.28%, 37.13%, 36.18%, 36.99%, 38.15%, 39.16%, 42.18%
```

Its depth-profile correlation is **0.8752**, but the magnitude remains far below released Bonsai.

## Module shape

The public-profile multiplier correctly raises recovery pressure on Q, K, gate, and up projections relative to V and down. However, all module change fractions remain too small. For example, at eight layers:

```text
binary hard-public:
Q 7.72%, K 7.55%, V 4.83%, gate 8.06%, up 8.02%

public target:
Q 29.43%, K 29.97%, V 23.87%, gate 29.58%, up 29.98%
```

The module hierarchy is approximately correct; the recovery budget is not.

## Main conclusion

The public final checkpoints are not reached by a short stabilization pass after discretization. They require much more cumulative discrete movement than the current 100–120-step miniature runs.

The likely private family now includes:

- long continuation after low-bit initialization;
- repeated code boundary crossings rather than one final projection;
- depth- and module-sensitive optimization pressure;
- behavior preservation throughout the continuation;
- a final hard phase long enough to move roughly 27–30% of binary codes and 36–42% of ternary codes in transformer matrices.

The next experiment must sweep **recovery duration and effective learning-rate budget**. Further surrogate comparisons at the same short budget cannot resolve the remaining gap.

## Selected current implementation path

```text
binary:
    exact-hard and categorical remain viable
    public depth/module multipliers are beneficial
    exact-hard-public selected when geometry fidelity is prioritized
    categorical-public selected when minimum miniature KL is prioritized

ternary:
    CAT-Q soft phase for behavior optimization
    longer exact-hard continuation to increase final code movement
    public depth/module multipliers used as a soft schedule prior
```

## Provenance

```text
workflow run:          30866516677
artifact ID:           8876506254
artifact ZIP SHA-256:  f0b41b2ebf36d804eed1128d2878d5c658fd9185542f3219dda47aec1247b793
full JSON SHA-256:     53f6b27979ba0a1fcb20d7f6bc09f44b273af3b6535a0c1db1eb3de5e12c2b57
runtime:               571.65 seconds
```

Compact result:

- `experiments/official_qwen3vl_text/results_code_drift_profile_matrix.json`
