# Ideas Backlog

Untried directions ranked by **leverage / cost**. Most-bang-for-buck at the top.

Each idea has: hypothesis, falsification criterion, smallest test, what unlocks if it works.

Cost legend: 🟢 hours on CPU | 🟡 day on Colab T4 | 🟠 weekend on A10G | 🔴 multi-day, real GPU spend

---

## High leverage / low cost (do these next)

### 💡 exp68 — Lazy / top-K proof generation for openhuman scale
- **Hypothesis:** exp67's full proof enumeration is O(branching^depth) — fine at 7-person graphs, infeasible at 10k+ entities. A lazy generator that yields the K simplest proofs (by tree size or rule-chain length) keeps query latency bounded while preserving the auditability story.
- **Falsified if:** The lazy generator's first 5 proofs aren't a strict subset of the full enumeration's first 5 (by depth-first order); OR generation time per proof exceeds the BFS-per-query baseline (1.5 ms at 10k entities) by more than 10×.
- **Smallest test:** Add a `top_k` parameter to `evaluate_with_provenance`; switch from list-construction to generator-yield. Benchmark on the openhuman-scale graph from exp63, query for top-5 derivations of synthetic cousin/uncle queries.
- **Unlocks:** exp67's auditability scales to OPENHUMAN_TL_MEMO's claimed entity counts.
- **Cost:** 🟢 ~80 lines, ~30 min wall.

### 💡 exp69 — Proof ranking by simplicity / Occam preference
- **Hypothesis:** When multiple proofs exist for the same fact, users want the simplest first (fewest rule firings, fewest distinct intermediate entities). Implement a scoring function and return proofs in order. Aligns with the human preference for short explanations and gives the SLM a natural "main reason" to surface to the user.
- **Falsified if:** Proof scores don't differ across the 3 distinct-proof cases in exp67's test set; OR the ranking puts a longer chain ahead of a shorter equivalent.
- **Smallest test:** Build a synthetic graph with at least 3 distinct proofs for some fact (multiple parallel kinship paths). Score proofs by tree-node count and rule-firing count; verify ordering.
- **Unlocks:** OPENHUMAN_TL_MEMO §3 example output (`why did you suggest X?`) gets a single canonical answer instead of N enumerated proofs.
- **Cost:** 🟢 ~50 lines, ~20 min wall.

### 💡 exp70 — Non-binary relations (arity > 2)
- **Hypothesis:** The current harness assumes binary relations (`rel(X, Y)`). Real KBs need ternary+: `meeting(person, person, time)`, `transaction(buyer, seller, amount, time)`. Extend tag protocol to `<tl_rule head="rel(X, Y, Z)" body="...">`; tensor representation becomes 3-d (or higher); einsum chains generalize naturally.
- **Falsified if:** Generalizing einsum to mixed-arity relations breaks the rule-parser (axes don't align); OR memory blows up at openhuman scale (10k entities × 10k × 10k × 4 bytes = 4 TB, infeasible — would need sparse representation from exp63).
- **Smallest test:** Add a `meeting(P, Q, T)` ternary primitive. Define a rule `co_attended(P, Q) :- meeting(P, _, T), meeting(Q, _, T)`. Test on synthetic data.
- **Unlocks:** openhuman's actual KB (calendars, transactions, messages) instead of just kinship.
- **Cost:** 🟡 ~150 lines + sparse extension; needs exp63's BFS substrate at scale.

### ✅ exp67 — Provenance for TL tool-call results (DONE — see EXPERIMENTS.md row 67)

Result: full proof trees built deterministically; 7/7 query types correctly answered with full derivation provenance. Counterfactual retraction (remove `parent(bob, dave)`) propagates 1 → 0 proofs deterministically. OPENHUMAN_TL_MEMO §3 auditability claim landed concretely.

### ✅ exp66 — Datalog negation in tool-call protocol (DONE — see EXPERIMENTS.md row 66)

Result: 9/9 cases pass. Stratified `!atom` negation absorbs cleanly into TL's monoid as element-wise `(1 - neg_tensor)` multiplication. The harness now covers Datalog-with-stratified-negation entirely — no architectural change.

### ✅ exp63 — Sparse closure substrate (DONE — see EXPERIMENTS.md row 63)

Original entry preserved below for reference. Result: BFS-per-query at 10k entities runs at 1.5 ms/query; BFS-per-source 3.6× faster than dense at Hanoi N=8. Per-source closure on a fully-connected graph still hits a memory wall but is dominated by closure-cells, not adjacency size — the substrate is the right shape now.

### 💡 (original) exp63 — Sparse closure substrate (CSR + iterative solver)
- **Hypothesis:** Replacing the dense `R ← ((R @ A + R) > 0)` closure with a CSR / sparse-COO representation + iterative solver (Bellman-Ford-style frontier expansion, or sparse-matrix matmul via `torch.sparse.mm`) lets TL closure scale past the exp59A wall (N=10 dense → ~13 GB) by 1-2 orders of magnitude in state-space size.
- **Falsified if:** Sparse closure on the exp59A Hanoi state-space graph at N=11 (177k states) doesn't fit in <8 GB OR doesn't converge within 5× the dense closure time at N=9; OR sparse closure on the openhuman kinship KB (10k people, ~50k facts) takes >1 s for a single query.
- **Smallest test:** Implement `sparse_closure(adjacency_csr) -> closure_csr`, benchmark against exp59A dense closure at N=7..11 and against the exp44 dense closure at the n=128 import-graph scale. Memory and wall-time per N.
- **Unlocks:** exp59A wall pushed out by 10-100×; OPENHUMAN_TL_MEMO's "10k-entity KB" claim becomes practically defensible at hot-path latency.
- **Cost:** 🟢 ~150 lines, ~1 hour wall.

### ✅ exp64 — Parity re-test (DONE — see EXPERIMENTS.md row 64)

Result: sigmoid caps at majority-class accuracy on every adversarial / non-monotone-parity case; cosine activation at α=π reaches 100% (but unreachable from random init per exp50). Operator + reachable-basin barrier together. Closes exp59C honest miss.

### 💡 (original) exp64 — Parity re-test with irregular counts (close the exp59C honest miss)
- **Hypothesis:** exp59 (C) failed to expose the parity barrier because Hanoi's natural count vector `[2^(N-1), ..., 2, 1]` is sigmoid-separable. With irregular counts (e.g. parity along a randomly-perturbed Hanoi solution where disks move suboptimal numbers of times), TL's monotone sigmoid+iteration recurrence will demonstrably fail.
- **Falsified if:** The new test ALSO doesn't fail TL — i.e. some unanticipated structure in the perturbed-solution count distribution remains sigmoid-separable. Would require revisiting the exp48/50 expressivity claim.
- **Smallest test:** Generate 50 random valid (but not necessarily optimal) Hanoi solutions per N, count moves per disk per solution, target = parity vector. Compare TL_OR3 (monotone sigmoid) vs TL_OR4 (cross-term, exp48) vs cosine activation (exp50).
- **Unlocks:** Closes the exp59C miss honestly; tightens the limitations map.
- **Cost:** 🟢 ~80 lines, ~10 min wall.

### ✅ exp65 — Extend TL tool-call to multi-relation joins (DONE — see EXPERIMENTS.md row 65)

Result: 9/9 cases pass — grandparent, uncle, cousin, great-uncle all evaluate correctly via einsum chains. `<tl_rule head="..." body="...">` tag added to harness. Datalog-class rules now operational without architectural change.

### 💡 (original) exp65 — Extend TL tool-call to multi-relation joins (uncle = parent ∘ sibling)
- **Hypothesis:** The exp60b harness handles single-relation closure; extending it to multi-relation rule chains (`<tl_rule>uncle(X, Y) :- parent(P, Y), sibling(X, P)</tl_rule>`) tests whether the SLM+TL composition scales to actual Datalog-class rules, not just transitive closure.
- **Falsified if:** The extended harness can't handle non-binary join arities, OR the rule-chain provenance trace becomes too long to be useful (>10 facts per derivation), OR rule-conflict resolution (multiple ways to derive the same fact) breaks deterministic semantics.
- **Smallest test:** Add `<tl_rule>` tag to harness; implement 1-step rule application as einsum over relation tensors; test on (parent, sibling) → uncle; (parent, parent) → grandparent; (uncle, parent) → great-uncle.
- **Unlocks:** OPENHUMAN_TL_MEMO §1 claim ("multi-hop relational queries" beyond ancestor) becomes operational, not theoretical.
- **Cost:** 🟢 ~120 lines, ~1 hour wall.

### 🚧 exp60 — TL-as-tool: teach a small instruct LM to invoke TL closure (scaffolded)

Status: a (traces), b (harness), c (rule-based 100% sanity) all done. d (`exp60d_sft.py`) scaffolded — LoRA SFT on Qwen2.5-0.5B-Instruct + 3-way eval (base / SFT-no-tool / SFT+tool). Run: `pip install torch transformers peft datasets accelerate && python3 experiments/exp60d_sft.py`. Falsification gates (≥1.5× deep-hop ratio at hops 3–5, ≥95% tool-call validity) hardcoded into the summary block.

### 💡 exp60 — TL-as-tool: teach a small instruct LM to invoke TL closure
- **Hypothesis:** A small instruction-tuned LM (Qwen 2.5 7B Instruct, Llama 3.2 3B Instruct, or Phi-3 mini) can be SFT'd to emit a structured `<tl_closure>{...}</tl_closure>` call when faced with a multi-hop reachability question, have the call intercepted and executed by the TL substrate (exp44-style 3-scalar closure), and incorporate the result into its final answer. End-to-end accuracy on synthetic kinship / call-graph / dependency-reachability questions should **strictly exceed** the same LM with no tool, especially as hop-count grows.
- **Falsified if:** The LM-with-TL-tool's accuracy at hop ≥ 3 is not ≥ 1.5× the no-tool baseline; OR the LM cannot reliably emit syntactically valid tool-calls (>95%) after SFT on ~1k traces.
- **Smallest test:** Generate 1k synthetic (kinship-graph, query, gold-tool-call, gold-answer) traces; SFT on a small instruct model with LoRA; eval on a held-out set of 200 multi-hop queries with hop ∈ {1, 2, 3, 4}. Compare to: (a) base instruct LM, no tool; (b) base instruct LM + plain text retrieval; (c) SFT'd LM + TL tool.
- **Unlocks:** First end-to-end realization of OPENHUMAN_TL_MEMO's SLM+TL composition. Closes the loop "LM proposes the rule, TL executes it" on a benchmark task. Most likely-to-work member of the integration family.
- **Cost:** 🟡 needs an instruct LM + LoRA training. ~half day on Colab T4 or via Together API.

### 💡 exp61 — TL-as-layer: differentiable closure block inside a transformer
- **Hypothesis:** Insert a TL-closure block between transformer layers L_k and L_{k+1}. Take a learnable projection of the hidden state to a relation-tensor slice `R[i, j] ∈ R^(n×n)`; iterate the closure recurrence `R ← σ(α · R @ R + β · R + γ)` for K steps (3-scalar TL); project back and add residually. Trained end-to-end on a multi-hop relational task (CLUTRR-style kinship), the TL-augmented transformer should beat a same-parameter-count vanilla baseline on deep-hop test instances.
- **Falsified if:** TL-layer transformer matches vanilla on shallow hops (≤2) AND fails to beat it on deep hops (≥4) by ≥5pp at 3 seeds; OR training is unstable (loss diverges) even with Wortsman 2023's qk-layernorm + z-loss interventions, indicating the closure recurrence introduces gradients incompatible with transformer training dynamics.
- **Smallest test:** 50M-param 6-layer GPT-style transformer + a single TL-closure block at layer 3. Train on synthetic kinship sentences with hop ∈ {1, 2, 3} train, evaluate on hop ∈ {1, 2, 3, 4, 5} test. Compare to vanilla 50M baseline (same 6 layers, no TL block). 3 seeds.
- **Unlocks:** Highest research novelty in the family. Direct test of "TL as architectural primitive inside an LM" — exp32/exp37 tried adjacent versions and went null at toy scale; this is the more aggressive form (TL is in the forward pass, not an aux loss). Provides the architectural anchor for any future "TL-augmented small LM" claim.
- **Cost:** 🟠 A10G or H100, weekend. Use Wortsman 2023's qk-layernorm + z-loss + decoupled weight decay from the start to prevent the high-LR instabilities exp37 likely hit silently.

### 💡 exp62 — TL-as-teacher: distill closure into transformer latents
- **Hypothesis:** Generate (graph, ground-truth-closure) pairs at scale (~50k); train a transformer on a *parallel* supervised target — its hidden states at a chosen layer must predict the closure tensor via a small probe head. After distillation, the transformer should internalize the closure operator and answer multi-hop questions correctly *without* invoking TL at inference time.
- **Falsified if:** After distillation, the transformer's deep-hop accuracy on novel graphs is no better than a same-compute vanilla baseline; OR the probe-head MSE plateaus high during training, indicating the closure structure is not learnable in the transformer's representation space.
- **Smallest test:** Same 50M transformer as exp61. Auxiliary loss = MSE(probe(hidden_layer_3), TL_closure(graph_in_input)). Train 5k steps on synthetic kinship; eval deep-hop accuracy. Compare to: (a) vanilla, (b) exp61's TL-as-layer.
- **Unlocks:** Tests whether closure is *learnable* by gradient descent given enough supervision — a question relevant to the bigger Wortsman-style "what can transformers learn vs. what needs substrate" debate. If exp62 succeeds, closure is just a trainable circuit and the substrate is unnecessary; if it fails, the substrate's value is sharper.
- **Cost:** 🟠 A10G, ~1-2 days.

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
