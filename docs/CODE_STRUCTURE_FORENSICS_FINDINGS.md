# Released-code threshold forensics

## Question

Can the public Bonsai codes be generated from the original Qwen weights by a simple independently fitted threshold inside each contiguous g128 group?

The experiment gives every sampled group its own best possible threshold. This is much more permissive than a global quantizer:

- binary: choose the threshold that maximizes agreement with released `{-1,+1}` codes;
- binary reversed: also allow the orientation to reverse;
- ternary: choose the magnitude threshold that best predicts the released zero mask while locking nonzero signs to the original Qwen signs.

If such a family explains the converter, its best-case agreement should approach 100%.

## Clean result

The run covered 197 matrix tensors and 100,862 sampled groups, or 12,910,336 weight positions.

| Test | Best agreement |
|---|---:|
| Direct binary `sign(W)` | 72.239% |
| Best positive binary threshold per group | **75.147%** |
| Best either-orientation binary threshold | **75.148%** |
| Best sign-locked ternary threshold | **65.778%** |
| Best ternary zero-mask threshold | **68.917%** |

The optimal independently fitted binary threshold recovers only another 2.91 percentage points over plain `sign(W)`. Allowing a reversed orientation changes agreement by only 0.00055 percentage points. Therefore the missing binary structure is not an affine shift or widespread group sign reversal.

The ternary result is even more restrictive: approximately one-third of the released ternary codes cannot be explained even when every g128 group receives its own best magnitude threshold.

## Non-monotonicity

After sorting each original Qwen group by weight value, adjacent released binary codes change state across 36.25% of positions. Released ternary codes change across 47.64% of sorted positions.

A true one-threshold quantizer has at most one main transition in the sorted order. These high transition rates show that released codes are strongly non-monotone functions of the corresponding original scalar weights.

The code distributions are also not degenerate:

```text
binary entropy:  0.99433 bits/group-position
ternary entropy: 1.56025 bits/group-position
```

The result is therefore not caused by groups collapsing to one dominant code.

## Module split

Embedding remains the direct-projection exception:

| Module | Binary direct | Best binary threshold | Best ternary threshold |
|---|---:|---:|---:|
| Embedding | 99.910% | 99.994% | 99.976% |
| Q projection | 70.646% | 73.657% | 62.945% |
| K projection | 70.061% | 73.097% | 62.906% |
| V projection | 76.088% | 78.746% | 69.977% |
| O projection | 74.090% | 76.869% | 67.032% |
| Gate projection | 70.340% | 73.394% | 64.357% |
| Up projection | 70.053% | 73.119% | 63.584% |
| Down projection | 73.411% | 76.264% | 68.428% |

This supports separate policies:

```text
embedding:
    direct or nearly direct scalar projection

transformer matrices:
    function-aware code reassignment and scale recovery
```

## Falsified converter families

The following are rejected as complete explanations of released transformer codes:

- `sign(W - threshold_g)` with one learned threshold per g128 group;
- the same family with a free sign reversal per group;
- sign-preserving ternary magnitude thresholding;
- learning only group scales after scalar threshold initialization;
- any converter whose final code is a monotone scalar function of the original weight inside each group.

This does not rule out thresholding as an initializer. It rules it out as the released finalizer.

## Implication for Engibona

Transformer code variables must remain trainable or discretely revisable after initialization. A reconstruction that freezes `sign(W)` or a magnitude-derived ternary mask cannot match the public code geometry.

The strongest current public path is:

```text
threshold/sign initialization
-> behavior-level whole-model recovery
-> extensive code reassignment
-> learned positive scales
-> exact-hard final state and packed export
```

## Provenance

```text
workflow run:          30865369189
artifact ID:           8875959083
artifact ZIP SHA-256:  cd3a50041f04b6f18cf4f3bba0bdd1cc4f086147a4ec64cf6e7b55feaeee6be9
summary JSON SHA-256:  82348b264a073b4647de53f3adc8adf86ad9bc8f55111a710d26d3cf847956c5
runtime:               220.15 seconds
```

Machine-readable summary:

- `experiments/public_bonsai_forensics/results_code_structure_forensics.json`
