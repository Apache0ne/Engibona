# Transformer symmetry forensics

## Question

Could the large distance between original Qwen weights and released Bonsai weights be mostly an exact reindexing symmetry rather than learned code recovery?

Two such symmetries preserve transformer function when applied consistently:

1. **Attention-head permutation**
   - reorder Q/K/V head blocks;
   - apply the matching inverse block order to O.
2. **SwiGLU intermediate-neuron permutation**
   - reorder corresponding rows of gate and up projections;
   - apply the same order to columns of the down projection.

The experiment constructed combined code fingerprints for these linked tensors at layers 0, 13, and 27. Candidate permutations were selected using one feature split and evaluated on a disjoint held-out split. This prevents a nearest-neighbour match from being mistaken for a real symmetry.

## Attention heads

Every optimal Q/O and K/V assignment was the identity permutation.

| Variant | Q/O identity assignment | K/V identity assignment | Held-out gain |
|---|---:|---:|---:|
| Binary | **100%** | **100%** | 0.00000 |
| Ternary | **100%** | **100%** | 0.00000 |

This held at every tested layer. There is no evidence that Bonsai reordered attention heads.

## MLP intermediate neurons

| Variant | Identity is nearest train match | Mean identity rank percentile | Best train gain | Held-out matched gain |
|---|---:|---:|---:|---:|
| Binary | **94.60%** | **99.9067%** | +0.00380 | **-0.01588** |
| Ternary | **99.41%** | **99.9961%** | +0.00040 | **-0.00148** |

The alternative match selected on training features is worse on every held-out sample set. The held-out positive-gain fraction is exactly 0 for both binary and ternary runs.

Layer detail:

| Variant | Layer | Identity top-1 | Held-out alternative minus identity |
|---|---:|---:|---:|
| Binary | 0 | 98.05% | -0.00734 |
| Binary | 13 | 97.85% | -0.00926 |
| Binary | 27 | 87.89% | -0.03103 |
| Ternary | 0 | 100.00% | 0.00000 |
| Ternary | 13 | 100.00% | 0.00000 |
| Ternary | 27 | 98.24% | -0.00445 |

Binary layer 27 has the most code drift and therefore the lowest identity top-1 rate. However, the apparent alternative matches fail the held-out test by the largest margin. This is evidence of stronger learned late-layer modification, not a coherent neuron permutation.

## Cross-checkpoint agreement

Binary and ternary choose the same best MLP identity/match for:

```text
layer 0:  98.05%
layer 13: 97.85%
layer 27: 87.89%
```

All attention assignments agree exactly. This indicates that both checkpoints preserve the original architecture index order while independently modifying codes within those coordinates.

## Falsified explanations

The following are rejected as major causes of the released weight divergence:

- global or layer-local attention-head reordering;
- independent Q/O head reordering;
- independent K/V head reordering;
- global or layer-local SwiGLU neuron reordering;
- a converter that first finds an exact permutation-equivalent basis and then performs ordinary scalar quantization.

Small accidental local matches can occur, particularly in late binary layers, but they do not generalize to held-out fingerprint positions.

## Implication for the reconstruction

The released code changes occur primarily **inside the original tensor coordinates**. Engibona should not add learned head or MLP permutations as a default mechanism. Recovery effort should instead target:

- code reassignment at fixed coordinates;
- learned group scales;
- depth-sensitive whole-model compensation;
- possible channel sign/scale gauges, which require a separate direct test;
- behavior-level objectives rather than raw weight reconstruction.

This narrows the likely family substantially: the released checkpoints are not merely reindexed Qwen quantizations.

## Provenance

```text
workflow run:          30865369166
artifact ID:           8875928505
artifact ZIP SHA-256:  39df99344f20321182da05fafc672cf495af011252b0bcb0516b1f4afe8b5b22
summary JSON SHA-256:  7d27fbf655038f1787eb41d71eed077aecbb078460f8684359185826cb150eb4
runtime:               111.57 seconds
```

Machine-readable summary:

- `experiments/public_bonsai_forensics/results_symmetry_forensics.json`
