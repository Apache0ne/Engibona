# Architecture fidelity levels

Engibona distinguishes three validation levels.

## Level A — mathematical toy

A minimal tensor or linear layer tests projection, packing, gradients, and exact code alphabets. It does not test transformer interactions.

## Level B — hand-written Qwen3-VL-topology miniature

The CPU ablation preserves grouped-query attention, RMSNorm, Q/K normalization, RoPE, SwiGLU, residual ordering, embeddings, and LM head. It supports rapid controlled experiments, but it is not upstream implementation identity.

## Level C — official public implementation

`experiments/official_qwen3vl_text/run_official_cpu_smoke.py` instantiates Hugging Face's official `Qwen3VLTextModel` with reduced width, depth, vocabulary, and context. The operator implementation, MRoPE handling, decoder layer classes, normalization classes, and attention path come from the upstream public architecture.

A passing Level C run establishes compatibility with the public Qwen3-VL text architecture. It still does not reproduce the private Bonsai transformation algorithm or its training data.
