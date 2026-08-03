# Confidence and evidence boundary

## Directly constrained by public Bonsai information

- pretrained Qwen checkpoint as the starting point;
- unchanged model architecture;
- exact binary or ternary language weights;
- contiguous group size 128;
- one FP16 scale per group;
- low-bit coverage across embeddings and matrix-heavy language components;
- higher-precision activations, normalization, and sensitive accumulations;
- packed weights consumed directly by custom kernels;
- no disclosed inference-time FP16 language residual.

## Confidence raised by direct Engibona experiments

A two-layer, four-query-head, two-KV-head Qwen3-VL-text-topology model was trained and transformed on CPU over three independent seeds. This does not identify PrismML's private method, but it tests competing math on the same operator structure.

### High confidence for the Engibona binary default

- exact hard binary forward values during recovery;
- trainable positive g128 scales;
- teacher-behavior recovery rather than local weight error alone;
- meaningful code movement from initial signs;
- inclusion of embeddings and LM head;
- preservation of trained exact codes and scales at export;
- optional Fisher/gradient-informed exact sign refinement with real-loss line search.

Exact-hard recovery beat naive projection, covariance-coordinate PTQ, smooth recovery, and staged recovery on CE and teacher KL for all three seeds. Fisher refinement improved CE and KL for all three seeds.

### Explicitly falsified as universal defaults

- smooth continuation as the strongest binary path;
- local activation-covariance projection as the final export objective;
- recomputing scales analytically from the latent carrier after global recovery;
- coordinate re-projection after teacher-guided recovery;
- assuming lower local quadratic reconstruction error means better full-model behavior.

## Current method confidence by component

| Component | Current Engibona status | Evidence level |
|---|---|---|
| Exact g128 code + FP16 scale | Required | Direct public constraint |
| Binary exact-hard forward | Default | Three-seed architecture-faithful experiment |
| Learned positive binary scales | Default | Three-seed experiment and gradient test |
| Teacher KL / functional recovery | Default objective component | Strong experiment + broad low-bit evidence |
| Trained-state export | Default | Direct finalization ablation |
| Fisher discrete sign refinement | Optional | Improved CE/KL on all three seeds |
| PTQ sign initialization | Default initializer | Strong mathematical and empirical support |
| Local metric projection | PTQ/diagnostic only | Useful locally; rejected as finalizer |
| Ternary CAT-Q-style transition | Default ternary hypothesis | Literature-supported; deep ternary test pending |
| Ternary zero-ratio controls | Available | Failure prevention; deep test pending |
| Sliding-window/block reconstruction | Intended trainer objective | Strong literature support; package-scale test pending |
| Dynamic recovery curriculum | Optional | Indirect GRACE/PADP support only |
| Recurrent/linear-attention state loss | Optional | Architecture-motivated, not yet isolated |
| CKA geometry loss | Optional | Plausible, not directly tied to Bonsai |
| ADMM/proximal solver | Research option | Private solver remains unknown |
| Learned rotations | Disabled by default | Insufficient Bonsai-specific evidence |

## Important distinction

The experiment increases confidence in what Engibona should implement. It does **not** make the following PrismML details known:

- optimizer family;
- exact learning rates;
- training steps or token count;
- calibration/recovery corpus;
- loss coefficients;
- whether PrismML used an STE;
- exact layer order or distributed strategy;
- exact private code optimizer.

## Preserved baseline

The pre-ablation implementation is preserved on branch:

```text
baseline-metric-projection-v1
```

The evidence-updated work is developed on:

```text
tiny-qwen3vl-evidence-v2
```

## Highest-value next tests

1. Repeat the full ablation for ternary weights, including zero-ratio collapse and threshold/scale interactions.
2. Expand the miniature from two to four decoder layers to test accumulated error.
3. Isolate q/k/v/o, MLP, embedding, and LM-head sensitivity.
4. Compare teacher KL, hidden MSE, block reconstruction, CKA, and combinations under equal compute.
5. Compare hard STE against projected gradient, proximal updates, and alternating discrete/scale updates under equal budget.
6. Test empirical Fisher diagonal, activation covariance, K-FAC approximations, and exact small-model Hessian-vector products.
7. Run public-weight forensics against unpacked Bonsai checkpoints.
8. Replicate released sign, zero-mask, and scale fingerprints across 1.7B, 4B, 8B, and 27B.
9. Inspect any future PrismML converter, training log, optimizer state, patent, or intermediate checkpoint.

See [`TINY_QWEN3VL_CPU_ABLATION.md`](TINY_QWEN3VL_CPU_ABLATION.md) for the measured statistics and finalization failure analysis.
