# Confidence and evidence boundary

## Directly constrained by public Bonsai information

- pretrained Qwen checkpoint as the starting point;
- unchanged model architecture;
- exact binary or ternary language weights;
- contiguous group size 128;
- one FP16 scale per group;
- low-bit coverage across matrix-heavy language components;
- higher-precision activations, normalization, and sensitive accumulations;
- packed weights consumed directly by custom kernels;
- no disclosed inference-time FP16 language residual.

## High-confidence implementation choices

These are not PrismML disclosures, but multiple independent low-bit results support them and they fit the public representation:

- strong PTQ initialization;
- smooth-to-hard continuation;
- tensor-sensitive hardening;
- activation-aware output reconstruction;
- block/sliding-window reconstruction;
- code as well as scale optimization;
- teacher-guided recovery;
- exact hard projection before export.

## Medium-confidence optional choices

- full activation covariance instead of only its diagonal;
- hidden relational geometry loss;
- dynamic recovery curriculum;
- recurrent/linear-attention state matching;
- ADMM as the exact private solver;
- gradient-covariance/K-FAC weighting;
- learned rotations.

## Deliberately excluded from defaults

- a mixed-precision salient-weight path;
- low-rank binary factors as the released representation;
- function-preserving rotations as a mandatory step;
- a traditional STE as the principal optimizer;
- one-pass sign or threshold projection;
- learned group scales without stability controls;
- inference-time latent weights, LoRA, or FP16 residuals;
- invented optimizer, learning-rate, corpus, or step-count claims.

## What will raise confidence next

1. Run public-weight forensics against the unpacked Bonsai checkpoints.
2. Measure released sign and zero-mask lineage against stock Qwen.
3. Compare released scales to identity, activation-diagonal, and covariance-optimal scales.
4. Test actual Bonsai versus naive projection on captured Qwen activations.
5. Replicate the signature across 1.7B, 4B, 8B, and 27B.
6. Inspect any future PrismML converter, training log, optimizer state, patent, or intermediate checkpoint.
