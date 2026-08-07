# Public Bonsai weight-forensics method

The direct forensic experiment compares the public checkpoints:

- `Qwen/Qwen3-1.7B`;
- `prism-ml/Bonsai-1.7B-unpacked`;
- `prism-ml/Ternary-Bonsai-1.7B-unpacked`.

The unpacked Bonsai repositories retain exact binary or ternary values expanded into FP16 safetensors. Their signs, zero masks, and group scales can therefore be inspected without decoding GGUF.

## Streaming design

Only one checkpoint shard is stored at a time. For every matrix-heavy tensor, the script retains a deterministic sample of up to 512 contiguous g128 groups. This covers millions of individual weights while fitting on a standard clean runner.

The base `lm_head.weight` is excluded when absent from Bonsai because the released model ties the output head to the token embedding; the shared embedding codebook remains included.

## Measurements

### Binary lineage

- agreement between released binary signs and `sign(W_base)`;
- released scale correlation with base mean-absolute and RMS scales;
- location of sign changes within each base-weight magnitude ranking;
- released raw NMSE versus naive sign/absolute-mean projection.

### Ternary lineage

- released zero ratio;
- agreement with alternating least-squares nearest-level projection;
- magnitude percentile of released zero assignments;
- released scale correlation with naive ternary scales;
- released raw NMSE versus naive ternary projection.

### Shared lineage

- binary/ternary sign agreement wherever ternary is nonzero;
- binary/ternary group-scale correlation.

### Exact representation

- maximum deviation from one shared magnitude per binary group;
- maximum deviation from `{0,1}` normalized magnitudes per ternary group.

## Statistical confidence

The report uses a tensor-cluster bootstrap and a module-stratified tensor bootstrap. Individual weights are not treated as independent observations, because millions of weights within one matrix would otherwise produce misleadingly narrow intervals.

## Interpretation boundary

Final checkpoint geometry can distinguish direct projection, scale-only recovery, meaningful code reassignment, simple magnitude thresholding, and shared binary/ternary ancestry. It cannot uniquely identify optimizer name, learning rate, recovery data, step count, or loss schedule.
