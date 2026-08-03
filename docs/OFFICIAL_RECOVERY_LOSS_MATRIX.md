# Official Qwen3-VL recovery-loss matrix

A clean GitHub-hosted run compared five equal-budget exact-binary recovery objectives on Hugging Face's public `Qwen3VLTextModel`. Each method used three independent teachers at two and four decoder layers, 100 recovery steps, tied embeddings/LM head, and exact g128 binary weights.

## Objectives

```text
CE only:          CE
KD only:          KL(teacher || student)
KD + CE:          KL + 0.2 CE
KD + hidden:      KL + 0.1 normalized hidden MSE
KD + CE + hidden: KL + 0.2 CE + 0.1 normalized hidden MSE
```

## Two layers

| Objective | CE | Accuracy | Teacher KL | Hidden cosine | Code change |
|---|---:|---:|---:|---:|---:|
| CE only | **4.27375** | **0.03592** | 0.34372 | 0.59558 | **9.99%** |
| KD only | 4.57015 | 0.02376 | **0.15897** | 0.68632 | 8.78% |
| KD + CE | 4.48437 | 0.02680 | 0.17477 | 0.67486 | 9.23% |
| KD + hidden | 4.56221 | 0.02387 | 0.16238 | **0.70119** | 9.05% |
| KD + CE + hidden | 4.49571 | 0.02517 | 0.16827 | 0.69147 | 9.33% |

KD-only had lower teacher KL than KD+CE and KD+CE+hidden on all three seeds. It beat KD+hidden on two of three seeds.

## Four layers

| Objective | CE | Accuracy | Teacher KL | Hidden cosine | Code change |
|---|---:|---:|---:|---:|---:|
| CE only | **3.76168** | **0.04178** | 0.48104 | 0.63366 | **10.92%** |
| KD only | 4.40662 | 0.02322 | **0.04059** | 0.85847 | 6.75% |
| KD + CE | 4.28544 | 0.02875 | 0.05362 | 0.83986 | 7.59% |
| KD + hidden | 4.39974 | 0.02224 | 0.04497 | **0.88466** | 7.75% |
| KD + CE + hidden | 4.29958 | 0.02637 | 0.05548 | 0.87178 | 8.13% |

KD-only had lower teacher KL than every mixed objective on all three four-layer seeds.

## Interpretation

### Teacher behavior preservation

Pure teacher KL is the strongest tested default when the transformation goal is preservation of the pretrained model's behavior. Adding task CE consistently moved the student away from the teacher distribution, even while improving labels on the synthetic recovery task.

### Task adaptation

CE-only generated the largest code movement and the best synthetic-task CE/accuracy, but it also produced by far the largest teacher divergence. It is adaptation, not faithful compression.

### Hidden matching

Hidden MSE consistently improved hidden-state cosine, especially at four layers, but slightly worsened output KL. It should be optional when internal-state preservation or long-rollout stability is explicitly required.

### Selected default

For behavior-preserving conversion:

```text
primary objective: teacher KL
CE weight:         0 by default
hidden weight:     0 by default
optional variants: add CE for domain adaptation; add hidden/block loss for state fidelity
```

This changes Engibona's default objective weights from `KD + 0.2 CE + 0.1 hidden` to pure KD. The other terms remain supported as explicit, measured tradeoffs.

## Confidence update

- teacher/self-distillation as the central recovery objective: very high;
- ordinary next-token CE as a mandatory recovery term: low;
- hidden matching as a mandatory term: medium-low;
- hidden matching as an optional state-fidelity term: high;
- one scalar loss optimizing task accuracy, teacher behavior, and internal geometry simultaneously: rejected;
- multi-objective Pareto reporting: required.

## Provenance

```text
workflow run: 30860408493
artifact ID: 8874195181
artifact SHA-256: f4c4e56ff1b782235ddf801b1350d68ab3a4b443944cffadbeb297fbc94f748f
runtime: 146.09 seconds
```

All final binary states passed exact-alphabet checks. The generating script is `experiments/official_qwen3vl_text/run_official_loss_matrix.py`.
