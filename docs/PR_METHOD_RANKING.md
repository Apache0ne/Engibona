# PR triage and method ranking

## Purpose

This document resolves the thirteen research PRs numbered 4 through 16. It separates:

- evidence worth retaining on `main`;
- methods selected for the default recovery path;
- useful but lower-ranked alternatives;
- falsified or inconclusive ideas that remain preserved on their original branches.

Closing a PR does not delete its branch. Rejected implementations and unvalidated experiment scaffolding remain available under their original branch names.

## Current recovery ranking

| Rank | Method | Status | Evidence and limitation |
|---:|---|---|---|
| 1 | Binary exact-hard STE with learned free g128 scales and public depth/module pressure | **Selected binary default** | Exact train/deploy forward. At 600 steps: KL 0.02185, 20.08% code movement, 71.99% of the public movement target. |
| 1 | Ternary exact-hard STE with learned free g128 scales and public depth/module pressure | **Selected ternary default** | Best joint behavior/geometry result. At 600 steps: KL 0.02048, 32.81% movement, 87.06% target coverage. |
| 2 | Binary categorical relaxation followed by hard export | Behavior specialist | Lowest binary KL, 0.01917, but only 59.83% code-movement target coverage. Retain as an ablation and possible initialization phase, not the geometry default. |
| 2 | Ternary CAT-Q/auto soft-to-hard recovery | Hidden-alignment specialist | Best hidden cosine, 0.86822, but only 75.80% movement target coverage. Retain as a soft phase; final hard recovery remains necessary. |
| 2 | Shared binary embedding codebook, shared scale, and ternary mask | **Selected representation/export option** | Direct checkpoint algebra is effectively exact across sampled embeddings. Jointly forcing this state during recovery was 4-12% worse in combined teacher KL than independent per-mode recovery, so shared joint optimization is not the default. |
| 3 | Mild binary row/group scale-residual prior, coefficient 0.1 | Experimental option | Improved five of six pairs and mean KL by 1.32%, but exact paired p=0.1875. More seeds are required before enabling by default. |
| 4 | Strong or hard-separable scale constraints | Rejected | Strong coefficients reconstruct scale structure but do not reliably improve behavior; coefficient 1000 is worse for both modes. Free per-group residuals remain necessary. |

## Methods ruled out or demoted

| Method | Decision | Evidence |
|---|---|---|
| One-pass sign/threshold projection | **Rejected as final method** | Full-model teacher KL is roughly 8.35-11.28 for naive low-bit projection versus 0.46-0.69 for released Bonsai checkpoints. Recovery is essential. |
| Short 100-120-step recovery | **Rejected as sufficient budget** | It learned the depth ordering but moved only 7.88% binary and 12.20% ternary codes. The 600-step run raised these to 20.08% and 32.81%. |
| Hidden-channel sign gauges | **Rejected** | Optimal row+column gauges improve agreement by less than 0.0008 percentage points over 58.98 million sampled positions. |
| Attention-head permutations | **Rejected** | Every optimal Q/O and K/V assignment was the identity at sampled early, middle, and late layers. |
| SwiGLU neuron permutations | **Rejected** | Identity matching wins on 94.60% binary and 99.41% ternary samples; selected alternatives reduce held-out similarity. |
| Static intermediate-channel affine correction | **Rejected as primary explanation** | Diagonal affine maps explain only 1-11% of held-out intermediate variance. Final-state calibration is strong, but recovery is not merely channel rescaling. |
| Universal frozen binary embedding | **Rejected** | Binary/Qwen embedding sign agreement is 99.94% at 1.7B but only 93.20% at 4B. Sparse sign recovery must remain available. |
| Forced joint shared-embedding recovery | **Demoted** | It obeys the released representation exactly but underperformed independently recovered binary/ternary students on miniature behavior. Use shared archival/export state without assuming one joint optimizer. |
| Ternary scale-structure regularizer | **Not selected** | The nominal coefficient-1 minimum improves only three of six pairs with p=0.90625. Free ternary scales remain the default. |
| KD-versus-CE gradient-direction attribution | **Inconclusive** | PR #12 contains the experiment but no inspected local result. It cannot influence defaults. |

## Architecture and checkpoint evidence retained

1. Released binary and ternary checkpoints are behaviorally recovered models, not naive projections.
2. Hidden channels largely preserve Qwen orientation; recovery changes computations without a hidden basis permutation.
3. Public code-change pressure rises with depth and varies by module family.
4. Embeddings follow an unusually strong shared-codebook/shared-scale/ternary-mask relation.
5. Transformer matrices are less exact but still have high shared-backbone-plus-mask agreement across scales.
6. The official Qwen3.6 hybrid miniature independently favors exact-hard recovery:
   - binary KL: 0.04554 naive to 0.01839 hard;
   - ternary KL: 0.03134 naive to 0.01332 hard.

The four-scale byte-range lineage run covered 1,202 tensors and 19.69 million sampled weights with zero tensor failures. Shared binary-backbone plus ternary-mask agreement was 93.93% at 1.7B, 92.99% at 4B, 94.31% at 8B, and 98.33% at 27B. This supports shared initialization and compact final-state analysis, not a claim that the private trainer used one shared latent state.

## PR disposition

| PR | Branch | Disposition | Reason |
|---:|---|---|---|
| #4 | `ci-full-functional-forensics-v7` | Close as superseded | Its useful full-model evidence is included through the stronger pooled-alignment line in #7. |
| #5 | `ci-gauge-forensics-v9` | Close as known-negative | Sign gauges are decisively falsified; scale-structure clue is retained in the ranking. |
| #6 | `ci-symmetry-forensics-v10` | Close as known-negative | Head and neuron permutations are not the recovery mechanism. |
| #7 | `ci-functional-alignment-v12` | Integrate evidence | Stronger full-model behavior and hidden-alignment validation. |
| #8 | `ci-public-4b-v13` | Close as superseded | 4B findings are integrated; later embedding and lineage work supersedes the branch. |
| #9 | `ci-scale-structure-v14` | Close as experimental | Mild binary signal is retained, but no default change is justified. |
| #10 | `ci-code-drift-profile-v15` | Integrate | Provides the strongest 600-step behavior/geometry selection evidence. |
| #11 | `ci-embedding-lineage-v16` | Partially integrate | Retain exact shared representation/export and forensic evidence; do not select forced joint training. |
| #12 | `ci-gradient-direction-v17` | Close as inconclusive | Experiment scaffolding is preserved on branch; no validated result. |
| #13 | `ci-public-8b-v18` | Close as superseded | Cross-scale lineage in #16 provides later 8B coverage. |
| #14 | `ci-public-27b-v19` | Close as superseded | Actual-Qwen3.6 and cross-scale work in #15/#16 are stronger. |
| #15 | `ci-official-qwen36-v20` | Integrate | Direct local official-hybrid recovery test supports hard recovery. |
| #16 | `ci-transformer-lineage-v21` | Integrate | Strongest cross-scale final-state lineage decomposition. |

## Default path after triage

```text
binary:
    exact g128 {-1,+1}
    exact-hard STE recovery
    free learned positive per-group scales
    public depth/module pressure
    optional categorical comparison, not default

ternary:
    exact g128 {-1,0,+1}
    optional CAT-Q soft phase
    sustained exact-hard recovery
    free learned positive per-group scales
    public depth/module pressure

embedding representation:
    shared binary codebook + shared scale + ternary mask is supported
    independent recovery remains behavior default
    shared pair export is supported
```

This ranking is based on the available public checkpoints and miniature tests. It does not equate miniature teacher KL with full-model intelligence retention.
