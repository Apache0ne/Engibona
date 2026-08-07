# Scale-structure recovery findings

## Public-checkpoint clue

Direct 1.7B Bonsai forensics found that the change from naive Qwen group scales to released scales is partly separable:

\[
\log\frac{s_{released}(i,g)}{s_{naive}(i,g)}
\approx \mu+a_i+b_g+r_{i,g}.
\]

`i` is output row, `g` is the contiguous input-group column, and `r` is a free per-group residual. Held-out additive R² was approximately 0.708 for binary and 0.595 for ternary.

This experiment asks whether softly penalizing `r` improves teacher-behavior recovery on the official Hugging Face Qwen3-VL text architecture miniature. Embeddings are excluded because their released scale lineage follows a separate shared-codebook policy.

## Design

- two and four decoder layers;
- three paired seeds at each depth;
- 100 recovery steps;
- pure teacher KL recovery;
- exact deployed alphabets at completion;
- coefficients `0, 0.1, 1, 10, 100, 1000`;
- identical initialization and minibatch sequence for every coefficient within a mode.

The penalty is:

\[
R_{scale}=\sum_{i,g}
\left(\Delta_{i,g}-\bar\Delta_{i,:}-\bar\Delta_{:,g}+\bar\Delta\right)^2,
\]

where `Δ` is the learned log-scale change from initialization.

## Binary result

| Coefficient | Mean teacher KL | Ratio to free baseline |
|---:|---:|---:|
| 0 | 0.110151 | 1.0000 |
| **0.1** | **0.108697** | **0.9868** |
| 1 | 0.109808 | 0.9969 |
| 10 | 0.109653 | 0.9955 |
| 100 | 0.109211 | 0.9915 |
| 1000 | 0.110703 | 1.0050 |

Coefficient `0.1` improves five of six paired runs. The paired mean KL change is `-0.001454`, a 1.32% pooled improvement. With only six pairs, the exact two-sided sign-randomization probability is `0.1875`, so this is a useful signal but not a high-confidence mandatory-default result.

The structural effect is clear:

| Depth | Free additive R² | Coefficient 0.1 R² | Coefficient 1000 R² |
|---:|---:|---:|---:|
| 2 | 0.6918 | 0.7827 | 0.99983 |
| 4 | 0.7515 | 0.9483 | 0.99973 |

Free recovery spontaneously develops the same broad row/group structure seen in released checkpoints. A mild prior strengthens it and may slightly improve behavior.

## Ternary result

| Coefficient | Mean teacher KL | Ratio to free baseline |
|---:|---:|---:|
| 0 | **0.073154** | 1.0000 |
| 0.1 | 0.073195 | 1.0006 |
| 1 | 0.072972 | 0.9975 |
| 10 | 0.073180 | 1.0004 |
| 100 | 0.073519 | 1.0050 |
| 1000 | 0.073691 | 1.0073 |

Coefficient `1` is the nominal aggregate minimum, but it improves only three of six paired runs. Its paired mean change is `-0.000182`, and the exact sign-randomization probability is `0.90625`. There is no stable ternary behavior advantage in this sample.

Ternary scale structure can still be strongly regularized: coefficient `1` raises additive R² from `0.6603/0.7420` to `0.9716/0.9940` at two/four layers. That does not establish that doing so is behaviorally preferable.

## Interpretation

The experiment supports four conclusions:

1. **Structured scale recovery is real.** It appears without an explicit prior and mirrors direct public-checkpoint geometry.
2. **It is not enough to freeze scales to a separable form.** Public released scales retain substantial residual variation, especially for ternary and down projections.
3. **A mild binary prior may help.** Coefficient `0.1` is the strongest current candidate, but needs more paired seeds and depths.
4. **No ternary default change is justified.** Free per-group scales remain the selected ternary path.

Current implementation choice:

```text
positive free per-g128 scales
+ optional output-row/input-group residual regularizer
+ no hard separability constraint
```

A larger confirmatory matrix should determine whether the mild binary prior is reproducible enough for the release-matched preset.

## Provenance

```text
workflow run:          30867109721
artifact ID:           8876654910
artifact ZIP SHA-256:  79f1b53d89aca6ee7a4cff43566d17a282d99750d8447534a9c9edf00cacc380
full JSON SHA-256:     87be50224c3b44e735e2a165cf2d259f8d728f7428aa6c30856493f3392da46c
runtime:               345.92 seconds
```

Compact result:

- `experiments/official_qwen3vl_text/results_scale_structure_effective.json`
