# Official Qwen3-VL architecture validation

The hand-written tiny decoder is useful for fast mathematical ablations, but it is not a byte-for-byte copy of the upstream implementation. Engibona therefore also includes a CPU smoke test built directly from Hugging Face's official `Qwen3VLTextModel` code.

The tiny official configuration uses:

- hidden size 128;
- intermediate size 256;
- two decoder layers;
- four query heads;
- two key/value heads;
- head dimension 32;
- Qwen3-VL RMSNorm and Q/K normalization;
- grouped-query causal attention;
- SwiGLU MLP;
- interleaved multimodal RoPE configuration with sections `[6, 5, 5]`;
- RoPE theta 5,000,000;
- tied token embedding and LM head;
- exact g128 binary conversion.

The configuration preserves the public operator structure while reducing only width, depth, vocabulary, and context length. It does not instantiate the vision tower, because the current target is the language-weight transformation mathematics.

Run:

```bash
pip install -e ".[hf,test]"
python experiments/official_qwen3vl_text/run_official_cpu_smoke.py \
  --layers 2 \
  --teacher-steps 30 \
  --recovery-steps 30 \
  --output official_qwen3vl_smoke.json
```

The smoke verifies:

1. the official text model initializes without downloading weights;
2. Engibona replaces official embedding, attention, MLP, and LM-head modules;
3. tied embedding/head state remains one quantized codebook;
4. every exported binary code is exactly `-1` or `+1`;
5. teacher and recovered metrics are recorded in a machine-readable artifact.

Passing this smoke establishes compatibility with the public Qwen3-VL architecture code. It does not establish 1:1 identity with PrismML's private converter.
