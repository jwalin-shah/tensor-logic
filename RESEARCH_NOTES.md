# Research Notes

Dated narrative log. Not a polished doc — a journal. Most recent at the top.

Format: each entry has **Date / Session focus / What we tried / What worked / What surprised us / What we'd do next**.

The point is to make session-to-session continuity possible. If you forget what you were doing, this is the first file you read.

---

## 2026-04-25 — exp43 (phase 7): TL planning fails — sharp predictions ≠ MPC-friendly

**Session focus:** Test the actual research ambition: does TL's sample-efficient forward model (exp42's positive finding) translate to better downstream task performance via planning? This is the "teach navigation → easier pickup" question in its simplest form. Build a multi-task planning eval — train forward model on dynamics, then use the model with sampling-MPC to navigate to 4 different corners.

**What we tried:**
- World: 8×8, 1 object, 4 actions, no collision, no occluder, fully observable. The simplest possible planning testbed so the forward-model quality drives the result.
- For each (model, seed, data_size): collect n random-action trajectories → train forward model → run 30 episodes per task with sampling-MPC at horizon 10, 50 random sequences per planning step, max 15 steps per episode.
- Initial run had horizon=4 (too short to reach corners from random start) → bumped to 10. Initial planner used final-step goal-prob → switched to discounted cumulative goal-prob over the horizon (sharp predictions need credit for any horizon step that visits goal).

**Key results:**
- **TL aggregated success rate** (across 4 corner tasks, 3 seeds): 10.6% / 9.7% / 9.4% at n = 50 / 200 / 2000.
- **MLP h=64**: 55.6% / 53.1% / 60.0%.
- **MLP h=128**: 52.8% / 68.6% / 72.8%.
- TL's `avg_steps_to_success` = 1.3 (only "succeeds" when the random start is adjacent to the goal, picks the right action, hits goal in 1 step). It doesn't actually plan navigation from afar.
- MLP's `avg_steps_to_success` = 7.2-7.7 (real planning, taking near-optimal paths).
- MLP IMPROVES with data (53% → 73% from n=50 to n=2000). TL stays flat.

**What surprised us:**
- The gap is HUGE — 45-50pp at every data size. I expected TL to at least be competitive at low data given exp42's positive sample-efficiency finding. Instead TL essentially fails at planning.
- The likely mechanism: **sampling-based MPC under cumulative-goal-prob reward penalizes sharp predictors.** Most random 10-action sequences don't pass through the goal cell. With sharp TL predictions (mass concentrated at one cell per step), almost all sequences score 0 → argmax over zeros is essentially random → the planner doesn't exploit TL's accuracy. MLP's smoother outputs leak mass broadly → cumulative score gives hill-climbing signal even for sub-optimal sequences → planner can find good actions.
- This is partly a *planner-fairness* issue (sampling-MPC isn't designed for sharp predictors) and partly a *real TL limitation* (sigmoid+linear composition under multi-step fuzzy rollouts may also degrade — TL's sample efficiency in 1-step prediction doesn't automatically extend to multi-step rollout quality).
- Either way the practical conclusion stands: **TL's sample-efficiency advantage on point predictions does NOT translate into planning advantage at this scale.**
- TL's avg_steps_to_success of 1.3 is diagnostic. It means TL "successes" are nearly all trivial (start near goal). The model isn't doing meaningful planning at all.

**Takeaway / next:**
- This deepens the falsification of "TL is the right substrate for world-model-based planning." Combined with exp40 (collision ceiling) and exp41 (limited transfer), TL's empirical case for general world-modeling is now weak. The narrowest defensible TL claim from this session: **"TL prior gives sample-efficient point prediction at small data for tasks that match its tensor-product structure (single-step prediction, object permanence)."** That's it.
- To salvage TL on planning, you'd need a different planning interface:
  - **(a) BFS / discrete search**: threshold predicted next-state at 0.5, treat as deterministic next-state, do BFS to goal. Doesn't use sampling, doesn't punish sharp predictions.
  - **(b) Distance-shaped reward**: instead of "prob mass at goal," use "expected L1 distance to goal under prediction." Gives every sequence a continuous score, regardless of sharpness.
  - **(c) Test single-step decisions**: situations where the right action is determined by the current state alone, not multi-step planning. E.g., "predict optimal action given goal" (Q-learning regime, not planning regime).
- Honestly, after 4 nontrivial experiments today (exp40-43), the case for TL on world modeling is now well-mapped:
  - WHERE TL HELPS: 1-step point predictions on small data for sub-tasks matching tensor-product structure (object permanence at low n).
  - WHERE TL DOESN'T HELP: in-distribution overall accuracy under collision, multi-step planning, occluder-position transfer, anything requiring cross-cell interactions.
- For the user's "skills that generalize to new tasks" ambition, TL doesn't deliver in this empirical setup. Two paths forward:
  - Pivot to TL's actual sweet spot per Domingos: compositional symbolic reasoning (rule learning, finite-domain logic) where the tensor-product structure exactly matches the problem. exp1, exp2, exp36 already showed TL works there.
  - Accept the empirical findings and build the world model with whichever architecture works best (MLP+memory) — use TL only where it specifically wins.

**Methodological note:** When testing inductive priors on downstream tasks, the *interface* between the prior and the downstream task matters as much as the prior itself. Sampling-MPC and TL are mismatched: the planner needs hill-climbing signal that TL doesn't provide. A different planner (BFS, distance reward) might give a different result. This is the "horse and harness fit" problem — testing on the wrong setup falsifies a property of the system, not necessarily of the prior alone.

---

## 2026-04-25 — exp42 (phase 6a): TL sample efficiency — bias wins at low data

**Session focus:** Address the strongest open critique of the exp40/41 falsifications: maybe TL's prior matters MOST when data is scarce, and our 2000-trajectory training simply gave MLP enough data to overcome the prior advantage. Test by sweeping data sizes (50, 200, 500, 2000 trajectories) with FIXED compute budget (1000 gradient steps so smaller datasets just see same data more times). 4 models × 3 seeds × 4 sizes = 48 runs.

**What we tried:**
- `train_phase6a_sample_eff.py`: same world as phase 5 (no collision, center occluder, N=2). Each (model, seed) trained at each data size with the SAME 1000-gradient-step compute budget for fair comparison.

**Key results:**
- **Overall P@2**: TL plateaus at 88.0% regardless of data size (50 → 2000). Its structural ceiling is unchanged by more data. MLP h=128 scales 90.5% (n=50) → 92.9% (n=200) → 92.4% (n=2000). **MLP wins overall P@2 at every data size.**
- **In-occluder recall**: at n=50, TL=95.5% beats MLP h=64=91.1% by 4.4pp (TL also has 2.6× lower variance: ±1.9 vs ±5.0). At n=200, TL=97.5% still beats MLP h=64=93.3% by 4.2pp. **Crossover at n=500** (TL=98.2% vs MLP h=64=97.5%, near-tie). At n=2000, MLP h=64 wins 99.9% vs TL 97.9%.

**What surprised us:**
- The bias-vs-capacity tradeoff played out *textbook-precisely*. TL's prior helps at small data (4pp advantage on the structural sub-task at n=50/200), MLP capacity wins at scale (n=2000). Crossover between them. This is exactly what inductive-bias theory predicts and we got the cleanest demonstration possible.
- **TL hits a hard ceiling.** Its P@2 is 88.0% at n=50 AND at n=2000. More data doesn't help — the constraint W[a,x,y,p,q] just can't represent some aspect of the dynamics. (Likely the obs-correction-reliant cells: TL's W gets gradient where mass propagates, but in pure occluder regions where `belief = obs + (1-obs)*prior` and obs=0 always, the gradient flow may be subtly compromised. Worth investigating later.)
- **Variance pattern**: TL has consistently lower std across seeds at small data (±1.9 vs MLP's ±5.0 at n=50). The structural prior stabilizes training when data is scarce, in addition to giving better mean accuracy. This is another classic bias-effect: regularization through inductive constraints.
- **TL's win is NARROW**: it's specifically on in-occluder recall at small data. On overall P@2, TL never wins. So the salvageable claim isn't "TL is better at world modeling" — it's "TL gives sample-efficient generalization for the specific sub-task its structural prior addresses (object permanence under occlusion)."

**Takeaway / next:**
- We now have THREE complementary findings:
  - **exp40 (in-distribution, collision)**: MLP wins. TL has structural ceiling under collision dynamics.
  - **exp41 (transfer, no collision)**: MLP wins absolute, TL has flatter degradation curves.
  - **exp42 (sample efficiency, no collision)**: MLP wins overall P@2 at all sizes; TL wins in-occluder recall at small data (n<500) with lower variance.
- Combined picture: **TL's per-cell prior is a sample-efficiency multiplier on the structural sub-task it models well; it doesn't compete on overall accuracy or under distribution shifts that violate its assumptions.** The honest research contribution is: "TL prior gives 4pp / 2.5× variance reduction for object permanence at n<500."
- This MOTIVATES the multi-task transfer setup the user described ("teach navigation → easier pickup"). Sample efficiency advantages compound across tasks: if TL learns dynamics with less data, every downstream task that reuses those dynamics inherits that efficiency. This is the exp7+ direction.
- Phase 7 design (preliminary): same gridworld dynamics, but instead of just predicting next state, define K different downstream tasks (navigate to (4,4), navigate to (1,1), avoid the occluder, occupy 3 specific cells, etc.). Train forward model ONCE on dynamics, then for each task: do model-predictive control / planning using the learned dynamics. Metric: how quickly each model class achieves each task given the shared dynamics model. TL hypothesis: TL forward model → faster downstream task acquisition because its learned dynamics generalize better with less training.
- An alternative cheap step before that: try TL with translation-equivariant W (`W[a, dx, dy]`, ~1100 params instead of 16K). Should both lift the structural ceiling on P@2 (the dynamics ARE shift-invariant) and fix occluder-position transfer. That's exp43 — small bet, potentially shifts the whole story if it works.

**Methodological note:** The fixed-grad-steps protocol matters. Earlier sweeps used fixed epochs which gives different total compute at different data sizes. Phase 6a's protocol is what makes the bias-vs-capacity comparison fair. For future sample-efficiency experiments, default to fixed gradient steps with batch sampling-with-replacement.

---

## 2026-04-25 — exp41 (phase 5): TL transfer test — flatter curves, lower peaks

**Session focus:** After exp40 falsified "TL+memory > MLP+memory" in-distribution, test the steel-manned version of the inductive-bias claim: *does TL's prior give better OUT-OF-DISTRIBUTION generalization?* Run on the no-collision world so TL's per-cell assumption actually holds. Six transfer conditions: in-distribution sanity check + count transfer (N=1/3/4) + occluder-position transfer (top-left and bottom-right shifted occluders).

**What we tried:**
- `train_phase5_transfer.py`: 4 models (TL + MLP h=64/128/256) × 3 seeds = 12 trainings. Each trained on N=2, center occluder, no collision, 30 epochs. Then evaluated on all 6 transfer conditions.
- Decision rule: TL drops less than MLP under transfer → inductive bias gives generalization.

**Key results:**
- **Absolute P@N (best of each)**: TL 87.8% in-dist, drops to 82.9% at N=4. MLP h=128 92.3% in-dist, drops to 85.6% at N=4. **MLP wins absolute accuracy in every condition.**
- **Transfer DROP (P@N from in-dist)**: TL drops -5.0pp at N=4, MLP h=256 drops -7.9pp. TL drops -5.0pp at occluder-shift-TL, MLP h=256 drops -10.5pp. **TL's drops are smaller across 4 of 5 transfer axes.**
- **Cleanest TL-specific win — in-occluder recall under count transfer**: TL holds 99.8-99.9% at N=3 and N=4. MLP h=128 drops to 91.9% at N=4. MLP h=256 drops to 86.3% at N=4. (MLP h=64 also holds at 99.8% — the smallest MLP doesn't overfit.)
- **Occluder-position transfer fails for ALL models**: in-occluder recall collapses to <25% for TL, <12% for MLPs. Neither has translation invariance built in.

**What surprised us:**
- The result is genuinely mixed in a way I didn't predict. TL's flatter degradation curves are real — across most transfer axes TL drops less than MLP. But MLP wins absolute everywhere, so TL "transfers better" in the relative sense without ever beating MLP in absolute terms. This is subtle: the inductive bias gives a flatter curve at a lower peak.
- **Larger MLPs overfit to N=2 input distribution**. MLP h=256 has 34K params (2x TL); on count transfer to N=4, its in-occluder recall drops 12pp from in-dist while TL's stays flat. That's a clean overfitting-vs-prior tradeoff. The prior wins on transfer specifically because the prior CAN'T overfit (it's structurally constrained).
- **MLP h=64 is the most robust baseline overall.** It matches TL on transfer flatness, beats TL on absolute accuracy, and uses half TL's params. If the goal is "don't overfit, generalize," small MLP+memory is competitive with TL+memory and easier to train.
- **Occluder-position transfer is a setup limitation, not a TL question.** With absolute-position W and a recurrent belief that only fires the prior at observed-empty cells, the model never learns "what to predict in occluder regions" outside the training occluder location. Both TL and MLP fail similarly. To test occluder transfer cleanly, we'd need translation-equivariant W (`W[a, dx, dy]` relative shifts) so the dynamics are spatially uniform. That's exp42 territory.
- **N=1 transfer is the one axis where TL drops MORE than MLP** on in-occluder recall (TL 76% vs MLP h=64 100%). Probable cause: TL has been trained to output ~2 high-confidence cells (the 2 objects); at N=1 it predicts an extra phantom cell. MLP h=64 outputs sparser predictions and handles N=1 cleanly.

**Takeaway / next:**
- The exp40 falsification stands. The exp41 partial confirmation softens it slightly: TL has a real but small inductive-bias-for-generalization signal on count transfer, particularly for in-occluder recall under larger N.
- **Honest scope of the TL claim**: "TL's per-cell prior prevents overfitting to a particular input distribution and gives flatter accuracy curves under count transfer, especially for hidden-object recall. It does NOT give better absolute accuracy on these gridworld tasks. Translation-equivariance is not built in and would require a different parameterization."
- **Two natural next directions:**
  - **(a) exp42: translation-equivariant TL.** Parameterize W as relative shifts `W[a, dx, dy]` (size 4×17×17 = 1156 params instead of 16K). This is a much sharper prior that matches the actual dynamics. Hypothesis: occluder-position transfer should improve dramatically; in-distribution accuracy should also improve (TL was overparameterized for shifts). ~30 lines, ~2 min wall.
  - **(b) Pivot away from gridworld.** TL's actual sweet spot per Domingos's paper is **compositional symbolic reasoning** (finite-domain logic, knowledge graphs, transitive closure). We've already shown TL wins those (exp1, exp2, exp36). Gridworld dynamics may be the WRONG task for TL — they're dense, continuous-ish, spatially structured. Compositional rule-learning tasks (e.g., chained kinship rules with novel combinations) would test TL's actual claim much more directly.
- I lean toward (a) first because it's cheap and would close out the gridworld arc cleanly. If translation-equivariant TL DOESN'T fix occluder transfer, the gridworld really is the wrong testbed and we should pivot. If it DOES fix it, we have a clean positive TL story for spatial dynamics.

**Methodological note for self:** The decision rule I wrote ("TL must drop LESS than MLP") was actually too generous — it's satisfied by 4/5 axes but the absolute-accuracy comparison shows MLP still wins everywhere. A stronger decision rule would have been "TL's transfer-condition accuracy must MEET OR EXCEED MLP's transfer-condition accuracy." Under that rule, TL fails. Picking the framing matters a lot for what the experiment "shows."

---

## 2026-04-25 — exp40 (phase 3c+3d): TL+memory falsified vs MLP+memory baseline

**Session focus:** Run the missing comparison from phase 3b — does TL+memory actually beat MLP+memory at equal param count, or is "memory" the active ingredient and TL incidental? Originally framed as "the experiment that turns the demo into a research result." Then ground the finding against published literature.

**What we tried:**
- Phase 3c (`train_phase3c.py`): single-seed MLP (h=128, ~17K params, slight edge over TL's 16,384). Same world, same belief mechanism, same loss, only the forward model swapped.
- Phase 3d (`train_phase3d_sweep.py`): multi-seed sweep — 3 seeds × 4 configs (TL + MLP h=64/128/256) = 12 runs. Mean ± std on P@2 and in-occluder recall.
- Searched arxiv for current literature: Domingos's TL paper (2510.12269, Oct 2025), object-centric world models (2511.06136, 2511.02225), V-JEPA 2 (2506.09985), C-JEPA, cRSSM.

**Key results:**
- TL+memory (3-seed mean): P@2 = **91.20% ± 0.61**, in-occ recall = 98.64% ± 0.67.
- MLP h=64+memory (8,576 params, **HALF of TL**): P@2 = **95.69% ± 0.55**, in-occ recall = **99.76% ± 0.42**. Beats TL on both metrics.
- MLP h=128+memory: P@2 = 96.12% ± 0.33, in-occ = 95.75% ± 4.65 (high variance).
- MLP h=256+memory: P@2 = 97.34% ± 0.91, in-occ = 96.80% ± 0.89.
- Decision rule: TL beats MLP h=128 and h=256 on in-occ by >1pp; MLP h=64 beats TL on in-occ. Headline: **TL has no robust advantage; smaller MLP wins**.

**What surprised us:**
- **The original phase 3c writeup quoted TL's P@2 as 99.16%, which was wrong.** Phase 3b log shows TL P@2 = 90.63%. The "99%" in our memory was in-occluder recall (98.84%), not P@2 — I misread the number when writing phase 3c's docstring and propagated it forward to my own analysis. The single-seed phase 3c said "TL beats MLP by 3pp/9pp" — that was based on the wrong baseline. Correct picture even at single seed: TL 90.63% vs MLP 96.34%, MLP wins by 6pp.
- The falsification IS the value here. We expected to confirm "TL+memory > MLP+memory." Instead got a clean, reproducible negative result with a clear structural explanation: TL's per-cell einsum factorization can't represent collision dynamics (a cross-cell AND condition), so it plateaus at ~91% while an MLP absorbs the interaction terms.
- **The Nov 2025 OCWM literature found exactly this pattern at scale.** "When Object-Centric World Models Meet Policy Learning" (2511.06136) reports OCWM "underperforms SOTA" precisely because of "representation shift during multi-object interactions." Our toy result mirrors their bigger finding. Independently arrived, same story.
- V-JEPA 2 (2506.09985) demonstrates object permanence "intuitively" at 1M+ hours of video pre-training. We hit it cleanly at 16K params and 30 epochs of belief-tensor recurrence. Memory is enough; you don't need scale (for permanence specifically).

**Takeaway / next:**
- Update `EXPERIMENTS.md` exp39 to flag that the TL-specific framing was falsified by exp40. The recurrence-recovers-permanence claim still stands — just not the TL-is-load-bearing version of it.
- This is a real research contribution, just not the one we expected: **a small, controlled, reproducible falsification of "TL is the active ingredient" for object permanence**, corroborating large-scale OCWM findings.
- **Next (phase 5): transfer test on a no-collision world** where TL's assumptions actually hold. Train at N=2 with center occluder; test at N=1/3/4 (object-count transfer) and shifted occluder positions. Decision rule: if TL's accuracy drops <2pp under transfer while MLP's drops >5pp, the inductive-bias-for-generalization claim survives. If TL transfers no better than MLP, the falsification deepens.
- The natural next-after-that step (if phase 5 doesn't save TL): consider an FIOC-WM-style two-level factorization (per-object + interaction primitives) or simply abandon the TL framing for memory-equipped world-modeling.

**Methodological note:** Always read the raw run-log when comparing across sessions. Don't trust paraphrased numbers in code comments — those are exactly where misreads silently propagate. The phase 3b log was on disk the whole time; one direct read would have prevented the entire phase 3c misframing.

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
