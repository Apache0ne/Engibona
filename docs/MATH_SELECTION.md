# Mathematical method selection — evidence revision V2

This document records the current selection after comparing published low-bit methods **and** testing competing paths on a tiny Qwen3-VL-text-topology decoder. It does not claim PrismML used the selected implementation.

## 1. Final representation

For every contiguous group of 128 weights:

```text
binary:  q = s c,  c_i in {-1,+1}
ternary: q = s c,  c_i in {-1,0,+1}
```

with one positive FP16 scale `s` per group.

Rejected deployed representations include low-rank binary factors, mixed-precision salient channels, additive codebooks, and inference-time FP16 residuals because they do not match the disclosed direct code-plus-scale form.

## 2. Binary initialization

The Euclidean fixed-scale optimum is:

```text
c_i = sign(w_i)
s = mean_i |w_i|
```

This remains the default initializer, not the full recovery method.

## 3. Binary recovery selected by experiment

The current binary default is an exact hard forward with a trainable positive group scale:

```text
c = sign(u)
q = s c
s = exp(log_s)
```

The training expression is:

```text
q_train = q + u - stop_gradient(u)
```

Forward value:

```text
q_train == q
```

Gradients:

```text
d q_train / d u = 1
d q_train / d log_s = d q / d log_s
```

This preserves an identity surrogate gradient for code movement while keeping real scale gradients. A bounded log-scale trust region is used:

```text
log_s in [log_s0 - r, log_s0 + r]
```

### Why this replaced smooth binary continuation

On three independent tiny Qwen3-VL-topology teachers, exact-hard STE + teacher KD beat smooth and staged recovery for every seed on cross-entropy, teacher KL, and accuracy. Therefore smooth relaxation is no longer treated as the highest-confidence binary default.

This result supports the Engibona implementation choice. It does not prove PrismML used an STE.

## 4. Ternary recovery

Ternary remains less resolved. The current path uses:

- learned positive group scale;
- learned assignment shift;
- learned bounded threshold;
- CAT-Q-style softened assignment early;
- sustained exact-hard recovery later;
- optional zero-ratio regularization;
- trained-state export.

The hard ternary code is:

```text
c_i = -1  if (u_i - shift)/s < -threshold
c_i =  0  if |(u_i - shift)/s| <= threshold
c_i = +1  if (u_i - shift)/s > +threshold
```

A deep ternary version of the CPU ablation is required before this path receives the same confidence as the binary default.

## 5. Teacher-behavior objective

Local weight error is insufficient. The primary global recovery objective is conceptually:

```text
L = lambda_CE * CE(labels, student)
  + lambda_KD * T^2 * KL(teacher || student)
  + lambda_window * ||Y_teacher - Y_student||^2
  + lambda_hidden * L_hidden
  + regularization
```

The tiny benchmark directly supports teacher KL plus task loss. Window, hidden, CKA, and recurrent-state losses remain components to isolate under equal compute.

## 6. Trained-state export

The default export preserves:

```text
codes = hard_codes(recovered_state)
scales = learned_positive_scales(recovered_state)
```

It does not re-project the latent carrier by default.

### Finalization evidence

After global recovery on one seed:

```text
preserve trained codes/scales:       CE 3.2743, KL 1.8904
covariance-optimal scale replacement: CE 3.8967, KL 2.4271
covariance scale + coordinate search: CE 4.9813, KL 3.4895
```

The local covariance objective destroyed cross-layer compensation learned by the global teacher objective. `metric_reproject` therefore remains explicit opt-in behavior.

## 7. Local metric projection

Metric projection is still useful for PTQ initialization, diagnostics, and controlled ablations:

```text
E(c,s) = (w - s c)^T M (w - s c)
```

For fixed code:

```text
s* = (c^T M w) / (c^T M c)
```

After substituting `s*`, code refinement maximizes:

```text
score(c) = (c^T M w)^2 / (c^T M c)
```

For one transition `c' = c + delta e_j`:

```text
numerator'   = numerator + delta (M w)_j
denominator' = denominator + 2 delta (M c)_j + delta^2 M_jj
```

Metrics include:

- identity;
- activation diagonal;
- full within-group activation covariance.

The important boundary is that an exact local improvement is not necessarily a global sequence-model improvement.

## 8. Empirical-Fisher exact sign refinement

After exact-hard binary recovery, Engibona can estimate the local effect of a sign flip.

For:

```text
w = s c
c' = -c
delta_w = -2 s c
```

and empirical-Fisher diagonal `F`, the predicted loss change is:

```text
delta_L ~= g * delta_w + 0.5 * F * delta_w^2
```

Candidate negative deltas are ranked. Prefixes of candidate flips must be evaluated against the actual teacher/task calibration loss; only a real-loss improvement is accepted.

This refinement improved CE and teacher KL on all three tested seeds.

## 9. Scale stability

Unconstrained learned scales can couple pathologically with code thresholds. Engibona therefore uses:

- positive log-parameterized scales;
- finite lower and upper bounds;
- a log-scale trust region around initialization;
- optional scale tethering;
- optional ternary zero-ratio target.

Scale-only polishing with frozen hard codes remains a valid final recovery stage. It matched the original hard state on behavior and slightly improved accuracy in the finalization ablation.

## 10. Coverage

The low-bit wrapper includes by default:

- token embeddings;
- q/k/v/o projections;
- MLP gate/up/down projections;
- LM head.

Normalization parameters and sensitive non-matrix operations remain high precision.

## 11. Optional mechanisms not promoted to defaults

The following remain research options rather than claimed parts of the private Bonsai algorithm:

- ADMM or augmented Lagrangian optimization;
- K-FAC or full gradient covariance;
- learned orthogonal rotations;
- CKA relational geometry;
- dynamic GRACE/PADP-style curriculum;
- recurrent/linear-attention state matching;
- exact Hessian methods.

## 12. Current selected sequence

```text
pretrained model
-> sign/threshold g128 initialization
-> exact-hard binary recovery or soft-then-hard ternary recovery
-> learned positive scale under trust region
-> teacher/task/block behavior optimization
-> preserve trained exact codes and scales
-> optional empirical-Fisher discrete refinement with line search
-> optional frozen-code scale polish
-> exact packing
```

## 13. Next maximum-confidence experiments

1. Repeat all binary methods at four layers.
2. Run a full ternary matrix including threshold, shift, zero-ratio, and scale controls.
3. Compare equal-budget CE, KL, hidden MSE, CKA, block-output, and recurrent-state losses.
4. Compare hard STE, projected gradient, proximal code updates, ADMM, and alternating exact code/scale updates.
5. Compute exact Hessian blocks on the tiny network and compare them with empirical Fisher, activation covariance, diagonal Fisher, and Kronecker approximations.
6. Perform module-by-module ablations for embedding, LM head, attention projections, and MLP projections.
7. Run public Bonsai weight forensics to compare released codes/scales against each candidate algorithm.

Measured results and architecture details are in [`TINY_QWEN3VL_CPU_ABLATION.md`](TINY_QWEN3VL_CPU_ABLATION.md).
