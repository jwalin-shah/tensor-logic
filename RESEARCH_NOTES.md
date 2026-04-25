# Research Notes

Dated narrative log. Not a polished doc — a journal. Most recent at the top.

Format: each entry has **Date / Session focus / What we tried / What worked / What surprised us / What we'd do next**.

The point is to make session-to-session continuity possible. If you forget what you were doing, this is the first file you read.

---

## 2026-04-25 — world model phases 1–3b: object permanence via TL + SSM

**Session focus:** Build the smallest possible TL world model end-to-end, following the README's stated research goal. Strip cheats progressively (god-view → identity loss → occlusion → no memory), then add memory back and see if it recovers object permanence.

**What we tried:**
- Phase 1 (`train.py`): TL forward model on 8×8 grid, 2 objects, 4 actions. State `R[obj,x,y]`; transition tensor `W[action,x,y,x',y']`. Random rollouts, MSE loss.
- Phase 2 (`train_phase2.py`): + collision rule (objects can't share cells). Multi-step rollout eval at k=1,2,5,10.
- Phase 3a (`train_phase3.py`): strip identity (collapse to single occupancy channel) and add a 3×3 occluder zone in the center. Train `(obs, action) → true_next`, no memory.
- Phase 3b (`train_phase3b.py`): add Dreamer/RSSM-style belief tensor. `R̂[t+1] = obs[t+1] + (1−obs[t+1])·forward(R̂[t], action[t])`. Train with full BPTT through 20-step trajectories.

**Key results:**
- Phase 1: val_acc 100% — trivial; 16K params for a 16K-cell lookup table.
- Phase 2: val_acc 99.1%; 10-step rollout drift 99.9→97.3%. Per-object factorization can't represent collisions exactly, but only ~6% of steps involve them.
- Phase 3a: P@2 = 87.27%, in-occluder recall 67.55%. The drop characterizes what's lost without memory: the model handles "first step of disappearance" but cannot track over multiple hidden steps.
- Phase 3b: **P@2 = 90.63%, in-occluder recall = 98.84%, stable across 20-step trajectories (88-92% P@2 throughout).** The 31-point jump in object-permanence recall is the headline.

**What surprised us:**
- Phase 3a's failure was less catastrophic than expected. 67% in-occluder recall without any memory means the model handles a lot of cases via "I just saw it disappear into the occluder, predict it's there" — implicit one-step memory via the observation itself.
- Phase 3b's recall jumping to **99%** — not 80%, not 90%, but near-perfect — means the SSM-style recurrence essentially fully recovered the gap to ground-truth permanence.
- The recurrence is stable across 20 steps with no architectural tricks (no gradient clipping, no orthogonal init, no truncation). BPTT through einsum + sigmoid + obs-correction is well-conditioned at this scale.
- exp8's framing ("SSM A-matrix as tensor-logic rule") panned out. We didn't need to *add* an SSM — the einsum forward step IS the recurrence, the action-conditioned W slice IS the per-step A matrix. The thing the project has been building all along was already an SSM.

**Takeaway / next:**
- This is the project's first clean *positive* result on the world-model thread. exp29–37 were 5 nulls and 1 informative falsification chasing TL-as-auxiliary-loss in transformers. This one tested TL-as-substrate and it worked.
- The obs-correction trick `belief = obs + (1−obs)·prior` was the key — pure recurrence with no observation anchoring would likely drift over a long trajectory.
- **Next (phase 3c, exp40): MLP-with-memory baseline at equal param count.** Without this comparison, we can't claim the gain is TL-specific. If MLP+memory ties, then "memory" is the active ingredient and TL is just one possible substrate. If TL+memory wins, then TL's einsum structure matters specifically. This is the experiment that turns 3b from "a working demo" into a research result.
- **Then (phase 3d): add Spelke continuity prior** as an explicit TL constraint — predicted next position must be adjacent (or equal) to a current position. Should improve sample efficiency and tighten the belief.
- Bigger picture: stripping cheats progressively was the right disciplinary move. Phase 1 looked impressive but was trivial; phase 3a's failure mode was informative; phase 3b's success only became interpretable because we'd characterized the failure first.

---

## 2026-04-25 — exp37 (TL inductive bias, NULL result)

**Session focus:** Test whether a tensor-logic auxiliary loss makes a from-scratch transformer generalize better to deeper hops than CE alone.

**What we tried:**
- 858k-param TinyTransformer, 6 epochs × 3 seeds × 2 conditions on synthetic call-graph DAGs.
- Train depth distribution 1–3; test 1–5 (deep bucket = 3–5).
- Vanilla: next-token CE only. +TL: CE + λ · MSE(relation_logits, transitive_closure_target).

**Result:**
- shallow (1–2): vanilla 0.614, +TL 0.606, Δ=-0.008
- deep (3–5): vanilla 0.328, +TL 0.337, Δ=+0.009
- Both deltas well within seed σ (~0.07–0.11). Per-seed deep ranges 0.24–0.45 — bigger than any effect.
- TL auxiliary loss did converge cleanly (0.26 → 0.13), so the relation head learned closure. It just didn't transfer to the LM head.

**What surprised us:**
- Aux loss converging without helping the main task. The encoder is producing closure-respecting features in some subspace, but the LM head doesn't necessarily query that subspace. Gradients flow back through the encoder; they don't *force* the LM head to use the rule-shaped features.
- Both models near chance on deep (33%). The "compositional generalization" comparison is meaningless when neither model has mastered the shallow base case (60%). We diagnosed a loss-function question with a model that couldn't do the task either way.
- Hours of MPS training to learn that the architecture is wrong, not the loss.

**Takeaway / next:**
- "Auxiliary TL loss" at this scale is decoration. To do real work, closure has to be wired into the prediction path: relation head computes closure → output is added/concatenated into the residual stream the LM head reads from. Then the LM head has no choice but to use the closure feature.
- Floor must be lifted before the architecture-vs-loss question is testable. ~5–10M params, more data, more epochs. If vanilla itself can hit ~70% on deep, then the loss/architecture comparison becomes interpretable.
- This makes 5 of the last 6 experiments null/falsified attempts to inject TL into a learning system (exp29, exp31, exp32, exp34, exp37). exp35 was a clean confirmation but trivially so. The pattern is increasingly clear: **TL works as a primitive; every attempt to inject it as auxiliary supervision has come back null at toy scale.** Either toy scale is wrong, or auxiliary supervision is wrong.
- Bigger pivot question now on the table: stop running diagnostic toy experiments and pick a real capability + real data. Code dependency reasoning is the natural target — exp36 was already pointing here.

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
