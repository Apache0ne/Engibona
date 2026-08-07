# Paired-operator public Bonsai forensics

The released binary and ternary weights are much farther from Qwen than naive projection in coordinate-space MSE. This experiment tested whether the apparent coordinate damage was recovered by simple within-block compensation among paired matrices.

Full released matrices from layers 0, 13, and 27 were compared with naive binary/ternary projections on fixed random inputs for:

- Q/K attention-score composition;
- V/O composition;
- full gate/up/down SwiGLU output.

The comparison uses the original Qwen coordinate basis and the same input probes for released and naive weights.

## Released versus naive operator NMSE

Values below are the mean ratio across layers 0, 13, and 27. A ratio below 1 would mean the released paired operator is closer to the original Qwen operator than naive projection.

| Mode | Q/K scores | V/O composition | SwiGLU MLP |
|---|---:|---:|---:|
| Binary | **3.48× worse** | **24.87× worse** | **63.85× worse** |
| Ternary | **2.59× worse** | **5.56× worse** | **8.17× worse** |

Even after fitting one scalar to align the candidate operator output, released operators remained worse than naive projection:

| Mode | Q/K aligned ratio | V/O aligned ratio | MLP aligned ratio |
|---|---:|---:|---:|
| Binary | 1.59× | 1.27× | 1.33× |
| Ternary | 2.31× | 1.70× | 1.94× |

## Operator cosine

| Mode | Candidate | Q/K | V/O | MLP |
|---|---|---:|---:|---:|
| Binary | Released | 0.257 | 0.367 | 0.218 |
| Binary | Naive | 0.642 | 0.580 | 0.530 |
| Ternary | Released | 0.432 | 0.529 | 0.370 |
| Ternary | Naive | 0.805 | 0.771 | 0.744 |

## Interpretation

The public release is not explained by a local rule such as:

```text
quantize Q and K jointly to preserve original QK scores
quantize V and O jointly to preserve original VO output
quantize gate/up/down jointly to preserve original MLP output
```

in the original Qwen activation basis.

This strengthens the full-model recovery hypothesis. The released model appears to have moved to a different internal operating point in which:

- intermediate activation distributions changed;
- later layers compensate for earlier changes;
- local operators are not individually required to approximate the original operators on arbitrary original-basis inputs;
- behavior is preserved only through the complete recovered network.

## Rotation and gauge implications

The result does not prove that no function-preserving transformation or learned coordinate change was used. It shows that any such transformation was not canceled locally in a way that restores Qwen's original paired operators on arbitrary inputs.

A hidden rotation-only explanation is weakened because:

- no rotation metadata is present in the released representation;
- simple paired compositions are not recovered in the original basis;
- released binary/ternary scales and signs show depth- and module-dependent optimization.

The next decisive test is full-model functional comparison using real token activations from the original and released checkpoints. Random-input paired-operator error alone cannot identify the recovered model's own activation distribution.

## Confidence changes

- independent local layer reconstruction as the complete objective: strongly reduced;
- whole-model or long-window recovery: strongly increased;
- cross-layer compensation: strongly increased;
- simple local pairwise compensation: strongly rejected;
- rotations as the principal complete explanation: reduced;
- model-internal activation-distribution reshaping: increased.

## Provenance

```text
workflow run: 30862557314
artifact ID: 8874952426
artifact SHA-256: 956196b7c50fde8f0fe30ecc9a79ce5c6dd145333eb37a94795a595c8ab1ec7e
runtime: 110.20 seconds
```

The machine-readable summary is stored in `experiments/public_bonsai_forensics/results_paired_operator_forensics_summary.json`.
