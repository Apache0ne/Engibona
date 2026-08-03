# Clean official Qwen3-VL method matrix

This experiment was executed on a clean GitHub-hosted Ubuntu runner using Hugging Face's public `Qwen3VLTextModel`, PyTorch 2.13.0, Transformers 5.14.1, and Python 3.11.15.

The reduced architecture retained:

- official Qwen3-VL decoder classes;
- hidden width 128;
- 4 query heads and 2 KV heads;
- head dimension 32;
- official RMSNorm, Q/K normalization, GQA attention, SwiGLU, and interleaved MRoPE implementation;
- tied token embedding and LM head;
- exact contiguous g128 binary or ternary weights.

Three independently initialized teachers were tested at two and four decoder layers. Every low-bit method ended with a valid exact alphabet.

## Two layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Binary naive | 4.76875 | 0.01367 | 0.31115 | 0.59272 |
| Binary exact-hard recovery | **4.49671** | **0.02843** | **0.17731** | 0.68224 |
| Binary categorical recovery | 4.51536 | 0.02778 | 0.17838 | **0.68349** |
| Ternary naive | 4.69071 | 0.01866 | 0.21809 | 0.74351 |
| Ternary exact-hard recovery | 4.51400 | 0.03158 | **0.13002** | 0.78396 |
| Ternary CAT-Q recovery | **4.50444** | **0.03190** | 0.13188 | **0.78771** |
| Ternary categorical recovery | 4.51148 | 0.03179 | 0.13517 | 0.78048 |

Binary exact-hard recovery reduced teacher KL relative to naive binary projection by:

```text
(0.3111549 - 0.1773067) / 0.3111549 = 43.02%
```

Ternary exact-hard recovery reduced teacher KL relative to naive ternary projection by 40.38%. CAT-Q reduced it by 39.53%.

## Four layers

| Method | CE | Accuracy | Teacher KL | Hidden cosine |
|---|---:|---:|---:|---:|
| Binary naive | 4.47710 | 0.02040 | 0.12017 | 0.84206 |
| Binary exact-hard recovery | 4.26599 | 0.02387 | **0.05441** | 0.84827 |
| Binary categorical recovery | **4.26291** | **0.02658** | 0.05466 | **0.85057** |
| Ternary naive | 4.46434 | 0.02148 | 0.10019 | **0.90964** |
| Ternary exact-hard recovery | 4.26274 | **0.02637** | 0.04448 | 0.87994 |
| Ternary CAT-Q recovery | 4.28870 | 0.02561 | **0.04133** | **0.89408** |
| Ternary categorical recovery | **4.25022** | 0.02431 | 0.05203 | 0.87621 |

Binary exact-hard recovery reduced KL relative to naive binary projection by 54.73%. Binary categorical recovery reduced it by 54.51%.

Ternary CAT-Q recovery reduced KL relative to naive ternary projection by 58.75%. Exact-hard ternary reduced it by 55.60%.

## Paired-seed findings

Across all six official-architecture teacher runs:

- every recovered binary method beat naive binary projection on teacher KL;
- every recovered ternary method beat naive ternary projection on teacher KL;
- exact-hard and categorical binary recovery were effectively tied under this budget;
- exact-hard ternary had lower KL than CAT-Q on all three two-layer seeds;
- CAT-Q had lower KL than exact-hard ternary on all three four-layer seeds;
- CAT-Q had lower KL than categorical ternary on all six tested seeds;
- zero ratios remained stable between approximately 29.3% and 30.2% after ternary recovery.

## Corrections to the selected method

### Stronger conclusion

The high-confidence result is not that one surrogate is universally correct. It is:

```text
functional recovery >> one-pass projection
```

under exact deployed alphabets.

### Binary

Exact-hard remains the default because it:

- uses the deployed alphabet during every forward pass;
- was best on mean KL and CE at two layers;
- was statistically indistinguishable from categorical recovery at four layers;
- avoids a train/deploy forward mismatch.

Categorical relaxation remains a first-class ablation rather than a rejected method.

### Ternary

No universal winner was found. The current evidence supports a depth-dependent or hybrid path:

```text
soft CAT-Q assignment early
-> sustained exact-hard recovery
-> trained-state export
```

CAT-Q was the strongest four-layer teacher-behavior method and the strongest hidden-state method at both depths, while exact-hard was strongest for two-layer KL. A production ternary trainer should retain both paths until longer and deeper tests resolve the transition schedule.

### Hidden cosine caveat

Naive ternary occasionally had higher hidden cosine than recovered ternary while having much worse output KL. Raw hidden cosine alone therefore cannot serve as the optimization target. Hidden alignment must be combined with teacher behavior and task loss.

## Reproducibility

- workflow run: `30859347649`;
- result artifact ID: `8873796012`;
- artifact SHA-256: `07bb77f7c03bc8b9740c7f0bb35b56915bf16d11598d6a61e44d9057d460fabe`;
- total matrix runtime: 123.79 seconds;
- all workflow steps and all exact-alphabet assertions passed.

Raw seed-level results are stored in `experiments/official_qwen3vl_text/results_official_method_matrix.json`.

## What this proves

It proves that the Engibona wrappers and recovery methods operate on the public official Qwen3-VL text architecture and that behavioral recovery consistently outperforms naive projection under exact binary and ternary constraints.

It does not prove that PrismML used the same surrogate gradients, optimizer, data, or schedule.
