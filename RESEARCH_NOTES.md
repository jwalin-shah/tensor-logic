# Research Notes

Dated narrative log. Not a polished doc — a journal. Most recent at the top.

Format: each entry has **Date / Session focus / What we tried / What worked / What surprised us / What we'd do next**.

The point is to make session-to-session continuity possible. If you forget what you were doing, this is the first file you read.

---

## 2026-04-25 — exp33, 34, 35 sprint

**Session focus:** Knock out three queued experiments after the big exp32 finding.

**What we tried:**
- exp33: four candidate soft-OR operators (logsumexp T=1, logsumexp T=0.1, max-pool, noisy-OR-in-probability-space) on the exp1 transitive-closure task.
- exp34: energy-based reconstruction with tree-shape + cycle penalties on a 10-node KG with conflicting facts.
- exp35: two-tower (Possibility, Actuality) tensors with T ≤ P constraint, counterfactual query that should not pollute base facts.

**What worked:**
- exp33: noisy-OR in probability space (1 − ∏(1−p)) gives perfect F1=1.0. All log-space OR operators saturate.
- exp35: cleanly confirmed both falsification axes.

**What surprised us:**
- exp33 inverts the exp29 framing: we thought "stay in logit space" was the fix; turns out **the fix is "use noisy-OR in PROB space" — logit-space OR is fundamentally too loose.** Logsumexp upper-bounds max which upper-bounds true OR. Even at low temperature.
- exp34 was a clean falsification with structure: local sigmoid kept conflicts (FP=4), energy-based killed conflicts AND true edges (FN=4). Same total errors, opposite kinds. Tells us global constraints help only with well-tuned strength — and "well-tuned" is task-dependent. No free lunch from energy methods.
- exp35 worked exactly as designed but the "no pollution" axis is partly trivial (copy = no mutation). The non-trivial finding is that the T ≤ P constraint genuinely held during training (max violation 0.0019) — that's not free, the optimizer had to balance fit vs constraint.

**Takeaway / next:**
- For any future toy-graph reasoning, **use noisy-OR in probability space**, not logit-space tricks. exp29's framing was wrong.
- Energy-based methods need careful λ tuning per problem; not a drop-in fix for half-rule problems (exp7).
- Two-tower architecture is sound; could be useful for any future planning system that needs counterfactuals.
- Next batch options: exp36 (diffusion over relation matrices, needs Colab/Kaggle), or pivot to a unified mini-system that combines the working pieces (noisy-OR + semantic embeddings + generative replay) into one demo.

---

## 2026-04-25 — exp32: attention composition probe

**Session focus:** Test whether GPT-2 small's attention heads compose like tensor-logic rules.

**What we tried:**
- Built 11 kinship sentences ("Story: G's son is P. P's son is C.") with single-token names so causal attention could flow C → P → G.
- Loaded GPT-2 small (124M) via `transformers`, extracted 12×12=144 attention matrices per sentence.
- Found best single heads for each relation (parent: L11H8 score=0.159, grandparent: L0H6 score=0.151).
- Composition test: A_parent @ A_parent → grandparent? Two versions: (a) full search across 144 heads, (b) "clean" search excluding heads already encoding grandparent.

**What worked:**
- The infrastructure works perfectly — 11 usable sentences, attention extracted cleanly, ground-truth correlation computable.
- Clean falsification result rather than ambiguous noise.
- Took ~3 iterations to debug tokenization (multi-token names, sentence-start BPE split, causal directionality).

**What surprised us:**
- **GPT-2 has *dedicated* heads for parent and grandparent separately.** L11H8 fires for parent, L0H6 fires for grandparent. They're not derived from composition; they're learned independently.
- The unrestricted head-pair search hit Pearson 0.471 — but only because it could pick L0H6 (which already does grandparent) and stack it with L11H8. Smuggled signal.
- The clean head-pair search (excluding heads with solo grandparent score > 0.05) drops to Pearson 0.192. Real signal but below our 0.3 threshold.
- Self-composition (A_parent @ A_parent) gives Pearson 0.157 — non-zero but weak.

**Takeaway / next:**
- Transformers probably don't do tensor-logic-style composition internally. They learn many specialized heads and route signals through them.
- This **doesn't** kill the broader research direction — it just means tensor logic and transformer attention are different mechanisms, not the same thing in disguise. The interesting question becomes: can we **add** explicit composition as a training-time loss to *force* the model to internalize it? That would be exp37 territory.
- Two cheaper exps queued first: exp33 (tighter logit-space soft-OR for exp29's saturation issue), and exp34/35 (energy-based and two-tower).

---

## 2026-04-25 — exp29, 30, 31 sprint

**Session focus:** First disciplined-loop experiments after setting up the infrastructure.

**What we tried:**
- exp29: pure log-odds tensor logic (logsumexp soft-OR throughout fixpoint).
- exp30: re-test exp1 with absent-edge logits initialized to -LARGE instead of 0.
- exp31: surprise-gated parameter updates on object_permanence forward model, two-phase continual setup (vel=+1 then vel=+2).

**What worked:**
- exp30 confirmed the diagnostic from exp29: **the sigmoid floor problem is essentially an init bug.** With init=-10, F1=1.0 across all temperatures.
- exp31 technically passed both falsification axes (convergence within 20%, better retention).

**What surprised us:**
- exp29's logsumexp soft-OR over-saturates spectacularly — every cell goes to 1.0. Logsumexp is an upper bound on max, not a tight surrogate for fuzzy-OR. The "log-odds tensor logic" idea needs a different operator entirely.
- exp30 revealed an invariant we hadn't noticed: **init magnitude must scale with max temperature**. At init=-6, T=2.0 leaks because (E+ε)/T amplifies leakage faster than the negative init suppresses it.
- exp31's "wins" were tiny (29.87 vs 31.99 loss on A after B) — both models forgot phase A almost completely. Surprise gating is a real concept but doesn't manifest at toy scale; the per-batch surprise filter just barely changes anything when batches are small and homogeneous.

**What we'd do next:**
- **exp32 (nanoGPT/GPT-2 attention probe)** — most exciting next step, deliberate mode (option B chosen by user).
- exp33 (tighter logit-space soft-OR — fix exp29's saturation issue).
- exp34 (energy-based) and exp35 (two-tower) deferred to next session.

---

## 2026-04-25 — Synthesis & infrastructure

**Session focus:** Stop and figure out what we actually know after 28 experiments + 6 outsourced PRs. Set up real research infrastructure.

**What we tried:**
- Reviewed all 28 exps; categorized into confirmed/falsified/superseded.
- Merged PR #2 (M_compose asymmetric composition fix) and cherry-picked `object_permanence.py` from PR #4. Closed #1, #3, #5, #6 as redundant.
- Built `EXPERIMENTS.md`, `IDEAS.md`, `RESEARCH_NOTES.md` (this file).

**What worked:**
- The honest debrief produced four genuinely confirmed findings (tensor-logic-as-primitive, symbolic > bilinear composition, generative replay > EWC, semantic > random embeddings) and four genuinely refuted approaches (sigmoid-floor, T-annealing alone, naive curiosity, half-rules).
- Cherry-picking + closing redundant PRs was the right call — only 6 lines and 1 file out of ~600 lines were keepers from autonomous agents.

**What surprised us:**
- How little of the autonomous agents' work was usable (~1%) — they're great at coding, terrible at not duplicating each other.
- How many of our exps had no clear hypothesis going in (about half). The ones that did produced sharp results; the ones that didn't produced "interesting" but unactionable observations.

**What we'd do next:**
- **exp29 (log-odds tensor logic).** CPU, hours. Highest leverage / lowest cost item on the backlog. Should retroactively fix exp1, exp5, exp6, exp9.
- After exp29: decide between **exp30 (surprise-gated update)** and **exp31 (composed-attention probe in GPT-2)**. The probe is more exciting but needs Colab; the surprise-gated update stays on CPU.

---

## How to add a session entry

Top of this file. Use the same six fields. Be honest in "What surprised us" — that's the most valuable signal for future-you. If nothing surprised you, the session was probably scripted, not exploratory, and you should ask whether you're still doing research or just executing.

## How this connects to the other docs

- `EXPERIMENTS.md` is the table — one row per run, terse.
- `IDEAS.md` is the backlog — one entry per untried direction, with cost/falsifiability.
- `RESEARCH_NOTES.md` (this) is the narrative — what we were thinking, what shifted.

Read order for resuming work: this file (latest entry) → `EXPERIMENTS.md` (most recent rows) → `IDEAS.md` (top of high-leverage section) → start working.
