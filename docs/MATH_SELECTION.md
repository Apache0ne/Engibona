# Mathematical method selection

This document records the pre-implementation review requested for Engibona: take each proposed mathematical component, compare it against stronger published alternatives, and select the mechanism that best fits both the public Bonsai representation and the available evidence.

## 1. Final representation constraint

### Required form

For every contiguous group of 128 language weights:

```text
binary:  w_hat = s c,  c_i in {-1,+1}
ternary: w_hat = s c,  c_i in {-1,0,+1}
```

with one FP16 scale `s` per group.

### Rejected alternatives

- NanoQuant low-rank binary factorization: strong compression method, but its deployed `U V^T` factors and channel scales do not match the disclosed single-code-plus-g128-scale representation.
- PB-LLM/BiLLM/SAGE-style mixed salient channels: incompatible with end-to-end binary language weights.
- Additive vector quantization: representation mismatch.

### Pick

Direct exact g128 code and scale tensors.

## 2. Scale estimation

For fixed code `c`, solve

```text
min_s (w - s c)^T M (w - s c).
```

Differentiation gives

```text
s* = (c^T M w) / (c^T M c).
```

### Compared methods

- absmean scale: exact only for binary sign codes under the identity metric;
- learned unconstrained scale: flexible, but can create positive feedback and zero-ratio collapse in ternary QAT;
- direct gradient learning of scale and threshold: CAT-Q reports only modest gains without modulation and softened ternarization;
- metric-optimal analytic scale: exact for fixed codes, stable, and supports identity, diagonal, or full covariance metrics.

### Pick

Analytic metric-optimal positive scale. It is recomputed after every code change. Training-time soft modules use detached absmean for stability; exact export uses the metric-optimal formula.

## 3. Initial codes

### Binary

```text
c_i = sign(w_i)
```

is the Euclidean optimum for fixed positive scale and is the strongest format-compatible initialization.

### Ternary

Initialize by alternating nearest-level assignment and scale fitting:

```text
c_i = sign(w_i) if |w_i| > s/2 else 0
s = sum_{c_i != 0} |w_i| / count(c_i != 0)
```

### Pick

Use these only as initial states, followed by metric-aware code reassignment.

## 4. Code optimization

After substituting the optimal scale, the minimized group objective is

```text
w^T M w - (c^T M w)^2 / (c^T M c).
```

The constant first term means code optimization maximizes

```text
score(c) = (c^T M w)^2 / (c^T M c).
```

For a one-coordinate transition `c' = c + delta e_j`:

```text
numerator'    = numerator + delta (M w)_j
denominator'  = denominator + 2 delta (M c)_j + delta^2 M_jj
```

This scores every binary sign flip or ternary state transition exactly while re-optimizing the group scale in closed form.

### Compared methods

- scale-only fitting: cannot move wrong signs or zero masks;
- STE: indirect and gradient-mismatched;
- full combinatorial search: exact but infeasible;
- ADMM-Q: strong general solver, but a full dense ADMM state is expensive for every g128 group;
- local coordinate search after a smooth/proximal initialization: exact per move, simple, format-matched.

### Pick

Exact metric-aware coordinate refinement. ADMM/operator splitting remains a valid future initializer, not the mandatory deployed representation.

## 5. Reconstruction metric

For a linear layer and calibration matrix `X`:

```text
||XW - XW_hat||_F^2
= Tr((W-W_hat)^T X^T X (W-W_hat)).
```

Therefore:

```text
M = X^T X + lambda I.
```

### Compared methods

- raw weight MSE (`M=I`): ignores actual activations;
- activation diagonal: inexpensive and better than identity;
- full within-group activation covariance: preserves correlations inside each g128 block;
- Kronecker activation/gradient Hessian: theoretically richer, but requires output-gradient covariance and substantially more calibration machinery;
- exact Hessian: infeasible at model scale.

### Pick

Full within-group activation covariance by default, with diagonal and identity fallbacks. Gradient/K-FAC weighting is documented as a high-value extension but is not falsely treated as confirmed.

## 6. Smooth-to-hard transition

### Compared methods

- hard STE from step zero: creates dead zones and gradient mismatch;
- CAT-Q softened ternarization:

```text
f(w;s,d) = [tanh(s(w-d)) + tanh(s(w+d))] / [2 tanh(s)]
```

- Hestia categorical relaxation:

```text
pi_tau(q|w) = exp(-(w/s-q)^2/tau) / sum_k exp(-(w/s-k)^2/tau)
H(w;tau) = s sum_q q pi_tau(q|w)
```

- Hestia dense-to-quantized pressure:

```text
W_eff(t) = (1-p_t)W + p_t H(W;tau).
```

### Pick

- ternary default: CAT-Q transition because it is specifically validated for PTQ ternarization at large scale;
- binary default: categorical relaxation over `{-1,+1}`;
- shared compression-pressure path for both;
- exact hard projection at the end.

## 7. Sensitivity-aware hardening

A uniform hardening rate ignores tensor heterogeneity. Hestia estimates tensor curvature and uses it as a temporal scheduler:

```text
s_i = sigmoid(kappa (log h_i - mean(log h)) / std(log h))
tau_i(t) = tau_bar(t) exp(alpha s_i).
```

Sensitive tensors stay soft longer.

### Pick

Hessian-trace-compatible tensor sensitivity scheduling. The reference includes a simple Hutchinson estimator and a normalization function; a distributed Hutch++ implementation is required for 27B-scale production.

## 8. Reconstruction scope

### Compared methods

- independent weight reconstruction;
- independent layer-output reconstruction;
- sequential block reconstruction with student inputs;
- sliding multi-layer output reconstruction;
- final full-model alignment.

CAT-Q reports a sliding-layer objective:

```text
min ||F(W_window, X) - F(S_window C_window, X)||_2^2.
```

NanoQuant ablations show material gains from error mitigation, block reconstruction, and model reconstruction.

### Pick

Sliding-window or block-output reconstruction followed by global teacher-guided recovery. The package supplies the mathematical pieces and losses; the 27B distributed orchestration is intentionally not fabricated.

## 9. Teacher and representation losses

### Highest-confidence objective components

```text
L = lambda_CE L_CE
  + lambda_KD T^2 KL(p_teacher || p_student)
  + lambda_window ||Y_teacher - Y_student||_2^2
  + lambda_hidden L_hidden.
```

Relational CKA is available:

```text
L_CKA = 1 - CKA(H_teacher, H_student),
```

but remains opt-in because direct evidence for its use in Bonsai is weaker.

### Pick

Teacher KL, normal token loss, and block/window output reconstruction enabled conceptually. CKA and recurrent-state losses are optional.

## 10. Recovery-data selection

GRACE and PADP support adaptive, model-state-aware sample selection. A Bonsai-compatible score is:

```text
damage_t(x) = KL(p_teacher(x) || p_student_t(x))
variation_t(x) = mean_j |damage_j(x) - damage_{j-1}(x)|
```

combined with representation coverage.

### Pick

Implemented as an optional utility, disabled by default. The evidence supports usefulness, not that PrismML specifically used it.

## 11. Packing

- binary: eight signs per byte, `-1 -> 0`, `+1 -> 1`;
- ternary: four two-bit symbols per byte, `-1 -> 0`, `0 -> 1`, `+1 -> 2`;
- FP16 group scales stored separately.

The ternary two-bit format is hardware-friendly but not entropy-optimal. A kernel-specific encoder can replace it after runtime requirements are fixed.

## Final selected sequence

```text
pretrained model
-> activation calibration and tensor sensitivity
-> sign/threshold PTQ initialization
-> smooth dense-to-discrete continuation
-> sliding-window output reconstruction
-> metric-aware exact code/scale refinement
-> global CE + teacher recovery
-> exact code freezing
-> scale/norm polish
-> packed g128 export
```
