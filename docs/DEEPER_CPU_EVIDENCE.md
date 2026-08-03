# Deeper CPU evidence suite

This document extends the original three-seed binary ablation with targeted tests for curvature metrics, loss composition, layer depth, module sensitivity, ternary behavior, and dynamic data selection.

These experiments raise confidence in Engibona's implementation choices. They still do not identify PrismML's private training recipe.

## 1. Which metric predicts a useful exact sign flip?

A recovered two-layer binary model was tested on six modules:

- layer 0 q projection;
- layer 0 attention output projection;
- layer 0 MLP gate projection;
- layer 0 MLP down projection;
- layer 1 q projection;
- LM head.

For each module, 64 candidate individual sign flips were evaluated against the real teacher-KL objective. Three predictors were compared:

1. empirical-Fisher diagonal Taylor change;
2. identity/raw-weight reconstruction change;
3. activation-covariance reconstruction change.

### Aggregate predictor quality

| Predictor | Pearson | Spearman | Top-12 actual improvement rate | Mean actual delta of top 12 |
|---|---:|---:|---:|---:|
| Empirical Fisher | **0.92868 ± 0.01691** | **0.91503 ± 0.00600** | **1.00000** | **-0.0007783** |
| Identity reconstruction | 0.23278 ± 0.14358 | 0.14142 ± 0.13425 | 0.43056 | +0.0000853 |
| Activation covariance | 0.28142 ± 0.16849 | 0.20839 ± 0.09518 | 0.51389 | -0.0000277 |

Every empirical-Fisher top-12 candidate set contained only actual improving flips across all six tested modules. Local covariance was much weaker at predicting the globally relevant teacher loss.

### Confidence consequence

- **Raised:** gradient/Fisher-informed discrete refinement.
- **Lowered:** activation covariance as a global code-selection oracle.
- **Kept:** activation covariance for PTQ initialization and local diagnostics.

## 2. Exact Hessian versus empirical Fisher

For 24 selected sign flips in layer-0 q projection, exact diagonal Hessian entries were computed by second-order autograd and compared with empirical Fisher.

| Predictor | Pearson to actual flip delta | Spearman |
|---|---:|---:|
| Exact selected Hessian | **0.999999** | **0.999130** |
| Empirical Fisher | 0.998616 | 0.969565 |

Additional observations:

- selected Hessian diagonal negative fraction: 0.0;
- Hessian/Fisher diagonal correlation: 0.5268;
- exact prediction mean: -0.0009645;
- actual mean: -0.0009632.

### Confidence consequence

The second-order Taylor mathematics is strongly validated locally. Exact Hessian selection is too expensive for a large model, while empirical Fisher retains nearly all useful ranking quality in this test. Engibona therefore implements:

- exact selected Hessian as a tiny-model research oracle;
- empirical Fisher as the scalable discrete-refinement candidate;
- mandatory real-loss prefix validation.

## 3. Recovery-loss composition

Three exact-hard binary recovery objectives were compared over the same three teachers:

| Objective | CE | Accuracy | Teacher KL | Top-1 agreement | Hidden cosine |
|---|---:|---:|---:|---:|---:|
| CE only | **3.11242** | 0.33157 | 1.93557 | 0.34868 | 0.65354 |
| Teacher KD only | 3.12885 | **0.33749** | **1.79099** | **0.39877** | 0.65755 |
| 0.2 CE + KD + 0.1 hidden MSE | 3.11597 | 0.33518 | 1.79533 | 0.38995 | **0.66396** |

### Confidence consequence

- CE best preserves the synthetic task labels.
- Teacher KD best preserves teacher distribution and decisions.
- The mixed loss best preserves internal hidden states while retaining near-KD behavior.

The default full trainer should therefore support a weighted multi-objective loss rather than claiming one universal term. Teacher KD is the central behavior objective; CE and hidden/block losses provide complementary constraints.

## 4. Module sensitivity is highly nonuniform

Each matrix was quantized alone by naive binary projection. Mean teacher-KL damage over three seeds was:

| Rank | Module | Mean KL damage |
|---:|---|---:|
| 1 | token embedding | **2.78742** |
| 2 | layer 0 attention output | 0.57537 |
| 3 | layer 0 value projection | 0.51904 |
| 4 | layer 0 query projection | 0.38109 |
| 5 | layer 0 key projection | 0.36576 |
| 6 | layer 0 MLP down projection | 0.28364 |
| 7 | LM head | 0.27036 |

Later-layer and remaining MLP modules ranged approximately from 0.18 to 0.26.

### Confidence consequence

- Tensor/module sensitivity analysis is necessary.
- Embeddings cannot be ignored merely because their implementation differs from `Linear`.
- The first block was substantially more sensitive than the second in this task.
- A uniform hardening, learning-rate, or refinement budget is unlikely to be optimal.

Engibona now includes embeddings in exact low-bit coverage by default. Sensitivity-dependent allocation remains a trainer-level extension because this experiment establishes heterogeneity, not the exact best scheduling rule.

## 5. Four-layer depth replication

A four-layer, four-query-head, two-KV-head, hidden-128 decoder with 624,000 parameters was tested over three seeds.

| Method | Teacher KL | Accuracy | Hidden cosine |
|---|---:|---:|---:|
| Naive binary | 0.33064 ± 0.01975 | 0.01302 ± 0.00281 | 0.70743 ± 0.00160 |
| Exact-hard KD recovery | **0.16722 ± 0.00820** | **0.02040 ± 0.00771** | **0.76578 ± 0.00152** |

Exact-hard KD recovery reduced teacher KL by approximately 49.4% relative to naive binary projection and improved internal-state alignment across all three seeds.

### Confidence consequence

The hard-recovery result is not restricted to two layers. Cross-layer error grows with depth, while global behavior recovery remains effective. This supports block/global reconstruction rather than independent local projection alone.

## 6. Preliminary ternary comparison

A short one-layer three-seed test compared exact-hard ternary recovery, CAT-Q-style soft-to-hard recovery, and categorical soft-to-hard recovery.

| Ternary path | CE | Teacher KL | Hidden cosine | Zero ratio |
|---|---:|---:|---:|---:|
| Exact-hard STE | 4.95900 | **0.08549** | **0.87758** | 0.26494 |
| CAT-Q soft-to-hard | **4.95477** | 0.09706 | 0.87493 | 0.26427 |
| Categorical soft-to-hard | 4.95482 | 0.09661 | 0.87446 | 0.26432 |

The run was intentionally short and undertrained. It shows:

- exact-hard ternary recovery is competitive and gave the best teacher behavior;
- smooth paths gave a tiny CE advantage;
- zero ratios remained stable around 26.4%;
- no method is yet dominant enough to finalize the ternary default.

### Confidence consequence

CAT-Q remains a reasonable ternary default hypothesis, but its confidence is lower than previously stated. A deeper equal-budget ternary matrix is required, including frozen-code scale polish, zero-ratio controls, and Fisher-style ternary state transitions.

## 7. Dynamic recovery-data selection

A short one-layer three-seed test compared uniform sampling with a dynamic score combining current teacher damage and temporal damage variation.

| Sampling | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Uniform | 5.00888 | 0.00651 | 0.12682 | **0.78651** |
| Dynamic damage + variation | **5.00398** | **0.00859** | **0.12476** | 0.78626 |

The dynamic method improved mean CE, accuracy, and KL, but the per-seed result was mixed and the run was small.

### Confidence consequence

Dynamic selection remains optional. The result raises plausibility but does not justify a mandatory default or a claim that PrismML used GRACE/PADP-style scheduling.

## 8. Updated hierarchy of mathematical picks

### Highest tested confidence

1. Exact grouped binary representation.
2. Exact-hard binary forward during recovery.
3. Learned positive group scales.
4. Teacher-behavior recovery.
5. Trained-state export.
6. Module-sensitive analysis.
7. Empirical-Fisher discrete refinement with real-loss validation.
8. Block/global recovery as depth grows.

### Supported but not fully resolved

1. Mixed CE + KD + hidden/block loss.
2. Sensitivity-dependent budget allocation.
3. Smooth versus exact-hard ternary recovery.
4. Dynamic data curriculum.
5. Scale-only final polish.
6. Recurrent/linear-attention state objectives.

### Rejected as universal defaults

1. Naive sign/threshold projection.
2. Local covariance coordinate optimization as final export.
3. Smooth binary relaxation as automatically superior.
4. Ignoring embeddings.
5. Accepting Taylor-predicted code changes without real-loss validation.

Raw statistics are stored in `experiments/tiny_qwen3vl/results_deeper_suite.json`.
