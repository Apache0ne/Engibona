# Maximum-confidence test matrix

The following matrix is the current evidence plan. Each row must have a committed script, fixed seeds, machine-readable output, and an explicit pass/fail or comparison criterion.

| Area | Current test | Next threshold |
|---|---|---|
| Binary alphabet | Unit and export round-trip | zero invalid symbols |
| Ternary alphabet | Unit and export round-trip | zero invalid symbols |
| Group layout | g128 grouping tests | all tensor tails padded and restored exactly |
| Tied parameters | embedding/LM-head alias test | one trained codebook and one parameter state |
| Official architecture | `Qwen3VLTextModel` smoke | official module forward and backward pass |
| Binary recovery | 2- and 4-layer multi-seed tests | beat naive PTQ on every seed |
| Ternary recovery | preliminary one-layer test | deep 2/4-layer equal-budget replication |
| Scale learning | gradient and finalization ablation | learned state no worse than re-projection |
| Discrete refinement | Fisher and exact Hessian oracle | real-loss-validated improvement only |
| Module sensitivity | one-module-at-a-time damage | stable rank across seeds/tasks |
| Loss mixture | CE/KD/hidden comparison | Pareto frontier under equal compute |
| Curriculum | uniform versus adaptive | consistent gain over larger seed count |
| Depth | 2 versus 4 layers | recovery benefit persists with depth |
| Sequence length | 8–32 tokens | recovery benefit persists across lengths |
| Packing | bit-level round trip | decoded codes and reconstructed weights exact |
| Reproducibility | CI artifact | clean runner produces JSON and passing tests |

No component should be promoted solely because it is mathematically elegant or recent. Promotion requires either direct public constraint evidence or a controlled test that survives replication.
