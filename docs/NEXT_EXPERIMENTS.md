# Next experiments ordered by information gain

1. Clean-run CI of the official Qwen3-VL text smoke.
2. Deep ternary 2-layer and 4-layer equal-budget comparison.
3. Tied versus untied embedding/head recovery under the official architecture.
4. Exact-Hessian versus empirical-Fisher ranking across attention, MLP, embedding, and head modules.
5. Block-output reconstruction versus logits-only KD under matched compute.
6. Student-input versus teacher-input block reconstruction.
7. Sensitivity-proportional recovery budgets versus uniform budgets.
8. Multi-length recovery batches versus one fixed sequence length.
9. Public unpacked Bonsai weight forensics across 1.7B, 4B, and 8B.
10. End-to-end packed runtime parity against dequantized reference weights.

An experiment is promoted only after multiple seeds and a clean-run artifact. A negative result is retained because it reduces the private-method hypothesis space.
