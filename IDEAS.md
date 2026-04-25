# Ideas Backlog

Untried directions ranked by **leverage / cost**. Most-bang-for-buck at the top.

Each idea has: hypothesis, falsification criterion, smallest test, what unlocks if it works.

Cost legend: 🟢 hours on CPU | 🟡 day on Colab T4 | 🟠 weekend on A10G | 🔴 multi-day, real GPU spend

---

## High leverage / low cost (do these next)

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

## Closed / decided against

(none yet — keep this section honest, move things here when we explicitly rule them out)
