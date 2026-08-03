# What 100% can and cannot mean

## Achievable 100% targets

Engibona can require and test all of the following without access to private PrismML artifacts:

- 100% exact binary or ternary alphabet compliance;
- 100% contiguous group-size and tensor-shape validation;
- 100% packed-code round-trip identity;
- 100% preservation of declared tied parameters;
- 100% absence of inference-time latent or FP residual tensors;
- 100% reproducibility of a specified experiment from a commit and seed;
- 100% pass rate for the implemented invariant tests;
- 100% use of the official public Qwen3-VL text implementation in the architecture-integration smoke.

## Not achievable from public outputs alone

The probability that Engibona is bit-for-bit identical to PrismML's private converter cannot honestly reach 100% without at least one direct private artifact, such as:

- converter or trainer source code;
- complete training configuration;
- optimizer state;
- intermediate checkpoints;
- training logs and data manifest;
- an official algorithm disclosure sufficient for reproduction.

Many distinct training procedures can produce similar final low-bit checkpoints. Final weights do not uniquely identify optimizer, data, step count, or loss schedule.

## Operational definition

Until direct artifacts exist, “100%” in Engibona means:

1. every public representation constraint is implemented exactly;
2. every selected mathematical component is compared under controlled ablations;
3. weaker components are removed when tests falsify them;
4. all reported measurements are reproducible from committed scripts;
5. uncertainty about private details is recorded rather than converted into false precision.
