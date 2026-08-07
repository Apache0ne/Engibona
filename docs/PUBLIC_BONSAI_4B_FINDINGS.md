# Public Bonsai 4B weight findings

## Scope

This run independently repeats the direct checkpoint comparison at 4B using:

- `Qwen/Qwen3-4B`;
- `prism-ml/Bonsai-4B-unpacked`;
- `prism-ml/Ternary-Bonsai-4B-unpacked`.

The public PrismML notice identifies Qwen3-4B as the source model. The experiment uses the 151,669 rows shared by the source and released checkpoints, exact contiguous g128 groups, 1,024 deterministic groups per matrix tensor, and tensor-cluster bootstrap intervals.

Coverage:

```text
tensors:          253
g128 groups:      259,071
weight positions: 33,161,088
```

## Global replication

| Metric | 1.7B | 4B |
|---|---:|---:|
| Binary sign agreement with Qwen | 72.25% | 73.15% |
| Binary released/naive NMSE | 6.64× | 5.82× |
| Ternary agreement with naive thresholding | 62.43% | 62.02% |
| Ternary released/naive NMSE | 4.52× | 4.30× |
| Binary/ternary sign agreement on ternary nonzeros | 90.12% | 88.98% |
| Binary/ternary group-scale correlation | 0.9281 | 0.9442 |

The 4B tensor-cluster 95% intervals are:

```text
binary sign agreement:       72.71%–73.60%
binary released/naive NMSE:  5.52×–6.12×
ternary code agreement:      61.42%–62.62%
ternary released/naive NMSE: 4.15×–4.46×
```

The core fingerprint replicates without ambiguity:

- roughly 27% of binary signs are reassigned;
- roughly 38% of ternary assignments differ from naive thresholding;
- released weights are much farther from Qwen in raw MSE than the MSE-optimal naive projection;
- binary and ternary share strong scale and sign lineage but are not identical final solutions.

This is independent scale replication of behavior-oriented discrete recovery.

## Depth profile replication

The 4B layer trends match the 1.7B trends:

| 4B metric versus layer | Correlation |
|---|---:|
| Binary sign agreement | -0.6924 |
| Binary released/naive NMSE | **+0.9248** |
| Ternary code agreement | -0.3801 |
| Ternary zero rate | +0.6573 |
| Ternary released/naive NMSE | **+0.8826** |

After mapping 28-layer and 36-layer depth to a shared normalized axis, the complete profile correlations are:

| Cross-scale depth profile | Correlation |
|---|---:|
| Binary sign agreement | 0.8391 |
| Binary NMSE ratio | **0.9131** |
| Ternary code agreement | 0.8347 |
| Ternary zero rate | **0.9370** |
| Ternary NMSE ratio | 0.8384 |
| Binary/ternary nonzero sign agreement | **0.9004** |

The same middle-to-late depth region accumulates the greatest raw-weight departure at both scales. This strengthens the case for a shared depth-sensitive whole-model recovery framework rather than unrelated checkpoint-specific quantizers.

## Module structure

Released/naive binary NMSE at 4B:

| Module | Ratio |
|---|---:|
| Q projection | 4.48× |
| K projection | 5.51× |
| V projection | 8.19× |
| O projection | 4.34× |
| Gate projection | 6.74× |
| Up projection | **8.38×** |
| Down projection | 3.23× |

MLP up and gate projections again receive large functional relocation. V is also highly modified. Module-specific recovery budgets remain justified.

## Binary–ternary coupling

The expanded 4B run measures conditional code relationships:

```text
P(binary flip | ternary zero)     = 41.22%
P(binary flip | ternary nonzero)  = 18.13%
P(ternary zero | binary flip)     = 58.56%
```

A binary sign change is more than twice as likely where the ternary solution selects zero. The binary/ternary overlap is meaningful but incomplete: the zero mask identifies many positions at which the binary solution also needs sign reassignment.

This supports a common latent recovery pressure or shared initialization lineage, followed by separately optimized final alphabets. It does not establish that one released checkpoint was mechanically converted from the other.

## Important embedding correction

The 1.7B result suggested an almost frozen binary embedding:

```text
1.7B binary embedding sign agreement: 99.93%
```

The 4B checkpoint falsifies that as a universal rule:

```text
4B binary embedding sign agreement:   93.65%
4B binary embedding scale correlation: 0.9893
```

The scale remains almost the naive mean-absolute scale, but approximately 6.35% of binary embedding signs move.

Those moves are highly structured:

```text
binary/ternary sign agreement on ternary nonzeros: 100.00%
P(binary flip | ternary nonzero):                    0.38%
P(binary flip | ternary zero):                      19.72%
P(ternary zero | binary flip):                      95.83%
```

Nearly every binary embedding sign reassignment occurs at a position that the ternary embedding suppresses to zero. This is a new, high-value clue.

The release-matched embedding policy should therefore become scale-aware:

```text
1.7B:
    binary embedding approximately frozen direct projection

4B:
    preserve direct scales
    lock signs on high-confidence ternary-nonzero positions
    permit binary sign recovery primarily on ternary-zero/uncertain positions
```

A single unconditional `frozen_ptq` binary embedding default is no longer the highest-confidence general rule.

## Updated transformation family

The cross-scale evidence supports:

```text
pretrained Qwen
-> exact g128 sign/threshold initialization
-> fixed original architecture coordinates
-> confidence-aware embedding policy
-> broad transformer code reassignment
-> module- and depth-sensitive recovery
-> structured plus free group-scale recovery
-> whole-model teacher-distribution preservation
-> separately finalized binary and ternary alphabets
-> trained codes/scales preserved in packed export
```

## What 4B rules out again

The independent larger model rejects:

- one-pass binary `sign(W)`;
- one-pass ternary magnitude thresholding;
- raw weight MSE as the main objective;
- scale-only recovery;
- a transformation specific to the 1.7B checkpoint;
- a universal frozen binary embedding rule.

## Provenance

```text
workflow run:             30866053680
artifact ID:              8876224815
artifact ZIP SHA-256:     cac782e1bb7ff261d307dc48cd76bb57fdb82a8318863267ec616cb84278cd08
summary JSON SHA-256:     384e46b53e676546255704533bd4c9b6687d4e87ad1ef8024ddd61d930438e80
confidence JSON SHA-256:  b4bd4bd6fe9aff0e502dca3bd84555fcf331058d962c66558c086865d7e5d7fe
runtime:                  277.96 seconds
```

Machine-readable compact result:

- `experiments/public_bonsai_forensics/results_public_bonsai_forensics_4b.json`
