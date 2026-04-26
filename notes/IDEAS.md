# Ideas Backlog

Untried directions ranked by **leverage / cost**. Most-bang-for-buck at the top.

Each idea has: hypothesis, falsification criterion, smallest test, what unlocks if it works.

Cost legend: 🟢 hours on CPU | 🟡 day on Colab T4 | 🟠 weekend on A10G | 🔴 multi-day, real GPU spend

---

## High leverage / low cost (do these next)

### 💡 exp56 — River Crossing as TL constraint-graph reachability
- **Hypothesis:** Apple's "Illusion of Thinking" reports frontier LRMs failing River Crossing at N=3 (11 moves). The puzzle is a constraint-satisfaction reachability problem: legal states form a graph, transitions are boat-trips that respect the constraints (wolf-not-with-goat, etc.), goal = is the all-on-far-side state reachable from the all-on-near-side state. TL transitive closure (`exp44`-style 3-scalar recurrence) on the legal-state graph should solve any N where Apple's LRMs collapse.
- **Falsified if:** TL closure cannot find the goal-reachable set for any N≥3 within reasonable iterations, OR the legal-state graph is too large for direct enumeration at N≥6 and a substrate ablation (random-move drift) doesn't show the expected compounding-error collapse.
- **Smallest test:** Encode (people-on-near-side, people-on-far-side, boat-side) as a node. Generate full legal-state graph for N ∈ {3, 4, 5, 6}. Run TL closure. Check goal-reachable. Compare to Apple's reported LRM accuracies.
- **Unlocks:** A second Apple-paper puzzle solved by TL substrate, broadens exp55's "substrate beats reasoning collapse" story across two distinct task shapes (deterministic-execution + constraint-reachability).
- **Cost:** 🟢 ~120 lines, ~10 min wall.

### 💡 exp57 — LM-as-rule-extractor + TL-as-executor hybrid for Hanoi
- **Hypothesis:** Lawsen's rebuttal to Apple shows LRMs can write the recursive Hanoi function correctly when asked for code, just not enumerate the moves into a context window. If we prompt the LM *once* for the schema, parse it, and feed it to the TL substrate (exp55) as the move-generator, we get perfect Hanoi at N=20 *with no manual algorithm input*. This separates "knows the rule" from "has a substrate to execute it" cleanly.
- **Falsified if:** A small open-weights LM (e.g. Qwen 2.5 7B Instruct, Llama 3.2 3B) cannot reliably emit the correct Hanoi schema in ~5/5 sampled completions; OR the LM's emitted schema disagrees with the manual one on more than 0.1% of moves at N=15.
- **Smallest test:** Prompt LM for `def hanoi(n, src, tgt, aux): ...`. Parse + AST-validate. Substitute into exp55. Re-run sweep. Report whether per-N accuracy matches exp55 numbers.
- **Unlocks:** First end-to-end realization of OPENHUMAN_TL_MEMO's SLM+TL architecture. Closes the loop "LM proposes the rule, TL executes it" on a benchmark task.
- **Cost:** 🟡 needs LM, ~half day on Colab T4 or via API.

### 💡 exp58 — Hanoi noise-robustness sweep (drift collapse curve)
- **Hypothesis:** Apple's collapse curve at large N is reproducible by simple compounding execution error: at each step, with probability ε, replace the schema-correct move with a uniformly random *legal* move. Reaches-goal probability should drop as `(1 − ε)^(2^N − 1)` ≈ exp(−ε · 2^N). At ε=0.01 this predicts collapse near N=10, which matches the Apple curve.
- **Falsified if:** The empirical collapse-curve shape is qualitatively different from `exp(−ε · 2^N)` — i.e. the LRM failure pattern is not consistent with simple per-step drift and there's a structurally different failure mode at play.
- **Smallest test:** exp55 with a noise hook. Sweep ε ∈ {0, 0.001, 0.01, 0.05, 0.1} × N ∈ {5, 8, 10, 12, 15}. 30 runs per cell. Plot reaches-goal vs N for each ε.
- **Unlocks:** Quantitative falsification target for "reasoning collapse = compounding execution error" — gives the substrate-side claim a number to defend.
- **Cost:** 🟢 ~50 lines on top of exp55, ~5 min wall.

### 💡 Log-odds tensor logic
- **Hypothesis:** Working in logit space throughout the fixpoint (sigmoid only at the very end) eliminates the `sigmoid(0)=0.5` floor that broke exp1/5/6/9.
- **Falsified if:** Log-odds version still has F1 ≤ 0.85 on the exp1 transitive closure task with threshold=0.5 and no calibration.
- **Smallest test:** Re-run exp1's 5-node graph with logit-space fixpoint; compare F1, hallucination rate at threshold=0.5.
- **Unlocks:** Clean retraining of exp5, exp6, exp9. May fix exp7's calibration issues for free.
- **Cost:** 🟢 ~30 lines, 5 min wall time.

### 💡 Surprise-gated parameter update
- **Hypothesis:** Updating only the dimensions where prediction error exceeds threshold τ (rather than all dimensions uniformly) produces faster learning and less catastrophic forgetting.
- **Falsified if:** Surprise-gated update converges slower than uniform Adam on the object_permanence task, or shows worse retention than EWC alone on a continual learning task.
- **Smallest test:** Object_permanence forward model with surprise gating vs. uniform update; measure (a) wall-time to converge, (b) retention when switching to a new physics regime.
- **Unlocks:** The missing arrow in the world-model loop (perceive → predict → surprise → update only the surprised dim).
- **Cost:** 🟢 ~80 lines.

### 💡 Composed-attention probe in a real transformer
- **Hypothesis:** A real LM's attention matrix `A_i = softmax(Q_i K_i^T)` can be treated as a relation tensor; composed heads (`A_i ∘ A_j`) match meaningful induced relations like "uncle" or "predecessor's successor".
- **Falsified if:** Composed attention matrices show no above-baseline correlation with any held-out relational probe (e.g., LAMA, BATS analogies).
- **Smallest test:** Load GPT-2 small (124M, fits in 1 GB), extract attention from a fixed layer/head pair on a kinship corpus, compute `A_i @ A_j`, probe against ground-truth uncle relation.
- **Unlocks:** Direct bridge from our KG work into how actual LLMs already do reasoning. Highest-leverage idea on the list.
- **Cost:** 🟡 Colab T4, ~2 hours.

---

## Medium leverage / medium cost

### 💡 Energy-based tensor logic
- **Hypothesis:** Defining `E(state)` over the entire relation graph and minimizing it (instead of `sigmoid(score)` per edge) handles global constraints (tree-shape, mutual exclusion) that local scoring can't.
- **Falsified if:** Energy-based version fails to enforce a "Parent must be a tree" constraint when given conflicting facts, where local sigmoid does no better.
- **Smallest test:** 10-node KG with intentional Parent contradictions; compare energy-min reconstruction vs. sigmoid scoring.
- **Unlocks:** Solves exp7's "half-rule backfire" via global consistency; lets us encode logical constraints as energy terms.
- **Cost:** 🟢 ~150 lines.

### 💡 Diffusion over relation matrices
- **Hypothesis:** A denoising diffusion model trained to recover a clean KG from noised versions can sample plausible KGs and impute missing edges better than tensor decomposition.
- **Falsified if:** On link prediction (e.g., FB15k subset), diffusion-imputed edges show worse hits@10 than ComplEx or RotatE baselines.
- **Smallest test:** 50-node kinship KG, mask 20% of edges, compare imputation accuracy (a) tensor decomposition, (b) diffusion.
- **Unlocks:** Generative completion of partial KGs — could be the bridge between LLM hallucination and structured fact retrieval.
- **Cost:** 🟡 Colab T4, ~half day.

### 💡 Two-tower (possible / actual) tensor logic
- **Hypothesis:** Separating "what's possible" `P_ij` from "what's true" `T_ij` (with `T ≤ P` constraint) lets the system reason about counterfactuals without polluting facts.
- **Falsified if:** Counterfactual queries ("what if X were Y's parent?") give the same answers as no separation, or training fails to maintain `T ≤ P`.
- **Smallest test:** Family KG with "what-if" probes; measure whether counterfactual reasoning leaves base facts intact.
- **Unlocks:** Modal logic baked into the tensor structure; needed for any planning system that imagines alternatives.
- **Cost:** 🟢 ~120 lines.

---

## High leverage / high cost (only after T2 wins)

### 💡 Tensor-logic-augmented LM training
- **Hypothesis:** Adding a tensor-logic loss term (rule-consistency on extracted entity relations) during LM fine-tuning improves factual consistency and reduces hallucination on multi-hop QA.
- **Falsified if:** No improvement on a multi-hop QA benchmark (e.g., HotpotQA) over a plain fine-tune baseline, controlling for compute.
- **Smallest test:** Fine-tune GPT-2 medium (355M) on Wikidata triples + a small QA set, with and without rule-consistency loss; evaluate on held-out 2-hop questions.
- **Unlocks:** Whether tensor logic actually improves real LMs or is just academically interesting.
- **Cost:** 🟠 A10G, weekend.

### 💡 World model with the loop closed
- **Hypothesis:** A unified system (perception → semantic embeddings → relation tensors → einsum forward step → decode → surprise → gated update) can learn a navigable, editable world model in a small environment (e.g., Atari Breakout, Minigrid).
- **Falsified if:** Agent's planned trajectories under imagined rollout fail to match actual environment behavior more than 60% of the time after training.
- **Smallest test:** Minigrid with object permanence + simple physics; train end-to-end, evaluate planning accuracy.
- **Unlocks:** The end-to-end thing nobody has built. This is the actual research goal everything else is leading toward.
- **Cost:** 🔴 multi-week.

---

## Speculative (might not be ideas)

- Tensor logic over **time** as a 3rd index (M[r, x, y, t]) — does temporal reasoning fall out for free?
- Replacing `softmax` in transformer attention with **sparsemax + tensor-logic rule** to get hard, interpretable attention.
- Using **Hebbian sleep** (exp25-style) on transformer attention matrices between training batches.

---

## Phase 8: Leveraging the new-models landscape (Apr 2026)

Several recent open releases change which experiments are now cheaply reachable. Listed by leverage, with the TL angle.

### 🌐 NVIDIA Newton physics engine + Isaac GR00T N1.7 + Cosmos Reason VLM
- **What:** Newton is GPU-accelerated, OpenUSD-native, MuJoCo/Isaac-Lab-compatible, *open source* under the Linux Foundation (co-developed with Google DeepMind + Disney Research). GR00T N1.7 is an open vision-language-action model for humanoid robots. Cosmos Reason is an open VLM for "vague instruction → step-by-step plan."
- **Why it matters for TL:** phase 7 failed because TL forward-models don't compose well under sampling-MPC. Newton gives us a *real* forward model for free, which lets TL play its actual sweet spot — the **constraint and goal-reasoning layer on top of someone else's dynamics**. Cosmos Reason is the natural rule-extractor (the SLM-half of the OPENHUMAN_TL_MEMO architecture); GR00T is the natural action layer.
- **exp59 candidate:** TL constraint-checker on top of Newton dynamics. Define safety/goal predicates as TL relations; have GR00T or Cosmos Reason propose action sequences; have TL's tensor closure verify legality / score against goal. Replaces hand-rolled MPC with a deterministic substrate-level verifier.
- **Cost:** 🟠 needs a real GPU box for Newton + Isaac Lab; ~1-2 days to wire up.

### 🧠 Recursive Language Models (DSPy.RLM) + LongCoT benchmark
- **What:** Alex Zhang's RLM idea — LM gets a REPL, recursively sub-queries itself over slices of context. DSPy.RLM hits 45.4% on LongCoT-Mini with Sonnet 4.5 (vs 2.6% without recursion), and Qwen3.5-27B + DSPy.RLM scores >2× GPT-5.2 on the same slice.
- **Why it matters for TL:** RLM is the *orchestration-side* attack on the same problem TL attacks at the *substrate side*. The natural composition: RLM picks which TL relation to query at each recursion step; TL substrate executes the query deterministically. This is potentially the right architecture for openhuman's "agent answers questions about the user's KB."
- **exp60 candidate:** Replace one of DSPy.RLM's tool slots with a TL closure-query tool against a synthetic kinship/ontology KB. Compare end-to-end accuracy on multi-hop kinship queries vs RLM-with-text-only-tools.
- **Cost:** 🟡 ~half day with DSPy on a single H100 or via API.

### 🧊 Microsoft TRELLIS.2 (4B image-to-3D, MIT-licensed)
- **What:** Sparse "field-free" voxel structure (O-Voxel) encoding geometry + PBR materials. ~3-60s per generation on H100.
- **Why it matters for TL:** structured-tensor representation as a *learned* substrate — same family of move TL makes, but for 3D rather than relations. If we ever pivot to multimodal grounding (the README's "Multimodal grounding eventually" line), TRELLIS.2's voxels are the natural feed for a TL relation graph over scene objects (`above`, `inside`, `supports`, etc.).
- **exp61 candidate:** TRELLIS.2-generated scene → extract objects + spatial relations → TL closure for "what would happen if I removed object X?" reasoning. Spelke-style core-knowledge probe at scale.
- **Cost:** 🔴 multi-day; requires GPU for TRELLIS.2 inference and an object-extraction pipeline.

### 👁 Vision Banana (Google DeepMind, generative-pretraining-as-universal-interface)
- **What:** Single generative model handles 2D segmentation/referring-expression + 3D depth/normals via natural-language prompts; SOTA in zero-shot transfer. Argues "image generation is an effective paradigm for visual understanding."
- **Why it matters for TL:** doesn't directly — Vision Banana is the *opposite* architectural bet (one big generative net does everything). But it's a useful baseline for any future multimodal-TL claim: if a TL-grounded scene representation gives better object-permanence or counterfactual reasoning than Vision Banana's generative interface, that's the headline. If not, we should probably stop trying to inject TL into perception.
- **exp62 candidate:** Vision Banana baseline on a small object-permanence / violation-of-expectation video benchmark (Spelke-style). Compare to a Vision Banana → TL hybrid where Banana extracts (object, position, time) tuples and TL maintains a relation graph through occlusion. Replicates exp7 / exp17 at real-data scale.
- **Cost:** 🔴 multi-day; needs Vision Banana access + a small video benchmark.

## Closed / decided against

(none yet — keep this section honest, move things here when we explicitly rule them out)
