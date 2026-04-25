# Experiments Log

One row per experiment. Most recent at the top. **Don't delete rows** — even failed/superseded experiments are evidence. Mark status in the Result column.

Status legend: ✅ confirmed hypothesis | ❌ falsified hypothesis | ⚠️ partial / mixed | 🔁 superseded by later exp | 🛑 invalid (bug, not science)

| # | File | Hypothesis (one sentence) | Tier | Result | Key number | Learning → next |
|---|---|---|---|---|---|---|
| 32 | `exp32_attention_compose.py` | GPT-2 attention heads compose like tensor-logic rules: A_parent ∘ A_parent ≈ A_grandparent. | T1 (laptop, ~2 min) | ❌ falsified, but informative | clean-head Pearson=0.192 (threshold was 0.3); 2.45× baseline ratio | GPT-2 learns parent and grandparent as **separate** dedicated heads (L11H8, L0H6), not compositionally. Suggests transformers don't internally do TL-style composition. |
| 31 | `exp31_surprise_gated.py` | Surprise-gated updates beat uniform for both convergence and retention. | T1 | ⚠️ technically passes, practically weak | uniform: 260 ep / forget A→32; surprise: 300 ep / forget A→30 | Both models forgot phase A almost entirely. Surprise gating barely helps at this scale; needs bigger model or different formulation. |
| 30 | `exp30_init_fix.py` | Initializing absent-edge logits to -LARGE eliminates the sigmoid floor at all T. | T1 | ⚠️ direction confirmed, magnitude matters | init=-10 → F1=1.0 across T∈[0.1, 2.0]; init=-6 leaks at T=2.0 | Floor is essentially an init bug; init magnitude must scale with max T. Updates exp1's verdict. |
| 29 | `exp29_log_odds.py` | Pure log-odds fixpoint eliminates the sigmoid floor problem. | T1 | ❌ but illuminating | log-odds F1=0.485, log-odds+calib F1=0.552, **naive sigmoid with proper -LARGE init: F1=1.000** | The floor problem was an artifact of using logit=0 for "absent" edges, not of sigmoid itself. Fix: init absent-edge logits to a strong negative. Logsumexp soft-OR over-saturates. |
| 28 | `exp28_unified_system.py` | Semantic embeddings beat random ones for relational learning. | T1 | ✅ | random F1=0.077, semantic F1=1.000 | Need transformer-grounded `E_i` for any real KG → motivates exp31 (composed-attention) |
| 27 | `exp27_blocks_world.py` | Tensor-logic rules can plan via BFS over relation states. | T1 | ✅ | finds plans in toy blocks-world | Planning works structurally; representations too thin → exp32 (richer state) |
| 9 | `exp9_tannealing_fix.py` | T-annealing alone fixes the sigmoid floor problem. | T1 | ❌ | annealing keeps halluc=1.0; calibration (margin=0.3) gets F1=0.84 | Need baseline subtraction, not just T → exp29 (log-odds) |
| 8 | `exp8_ssm_tensor_logic.py` | SSM A-matrix as tensor-logic rule outperforms vanilla RNN. | T1 | ❌ | RNN=1.00 acc, TL-SSM=0.547 | Conceptual bridge real, perf win not demonstrated → revisit at scale (T2) |
| 7 | `exp7_violation_of_expectation.py` | Adding a physics rule helps detect physics violations. | T1 | ❌ | half-rule: 5.5× higher loss, *less* surprise on violations | Partial priors backfire → either full prior or none |
| 6 | `exp6_multirelation_kg.py` | Bilinear embeddings can compose relations zero-shot. | T1 | ❌ | symbolic Uncle: F1=1.0; bilinear: F1=0.059 (random) | Composition is free symbolically, not in embedding space → strongest argument for tensor logic |
| 5 | `exp5_rule_sparsification.py` | L1 needed to find which einsum rule is right. | T1 | ⚠️ | even λ=0 → wrong rules go to 0.001 | Gradient already does it; L1 unnecessary on toy scale |
| 4 | `exp4_curiosity_gridworld.py` | Prediction-error curiosity beats random exploration on coverage. | T1 | ❌ | curiosity Gini=0.71 (stuck at boundary), random Gini=0.21 | Noisy-TV problem real → exp33 (count-corrected curiosity already in exp11) |
| 3 | `exp3_generative_replay.py` | A small GMM "dreaming" old data beats EWC for continual learning. | T1 | ✅ | GMM=97.8%, EWC=70.7%, hybrid=99.0% | Dreaming compresses experience extremely well → use for any continual world model |
| 2 | `exp2_semiring_swap.py` | Same einsum + 3 semirings → 3 different computations. | T1 | ✅ | Boolean / shortest-path / reliability all from same loop | Confirms semiring is the meaning, einsum is the structure |
| 1 | `exp1_temperature_phase_transition.py` | There's a sharp phase transition between deductive (T=0) and analogical (T>0). | T1 | ❌ | smooth degradation, no transition; sigmoid floor=0.5 dominates | Floor problem named here → motivates exp9 (annealing) and exp29 (log-odds) |

## How to add a row

When you create `expN_*.py`:
1. Add the row at the top of the table (most recent first).
2. Fill **all** columns including "Learning → next". If you can't fill "Hypothesis" or "Learning → next", the experiment isn't worth running yet.
3. Keep "Key number" to one or two figures — the exact metric that decided the result.
4. After running, update Result and Key number with reality. **Do not edit the hypothesis after seeing the result** — that's how you delude yourself.

## Confirmed truths (so far)

- Tensor logic is a unifying primitive: einsum + nonlinearity covers KG inference, transitive closure, SSMs, and attention with the same shape.
- Symbolic composition is free and exact; bilinear embedding composition is near-random zero-shot.
- Generative replay (small GMM) beats weight regularization for continual learning at this scale.
- Semantic embeddings dramatically outperform random ones for relational learning (F1: 0.08 → 1.00).
- Semiring choice changes the meaning of the same computation, cleanly and without retraining.

## Confirmed failures / dangers

- Sigmoid + threshold=0.5 makes "no evidence" look like 50% confidence; appears in exp1, exp5, exp6, exp9.
- T-annealing alone is not enough; needs paired baseline calibration.
- Naive prediction-error curiosity creates obsession hotspots at boundaries.
- Half-correct physics priors actively hurt — they bias the model in a wrong direction. All-or-nothing for explicit rules.
