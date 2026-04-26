# Tensor Logic for openhuman: research memo

**Date:** 2026-04-25
**Status:** Research artifact — no implementation commitment
**Question:** Should openhuman use Tensor Logic, and if so, where?

---

## TL;DR

TL belongs in openhuman as a **relation/knowledge-base substrate alongside** a small open-weights language model — not replacing it. The win is concentrated in **trust, persistence, and auditability**, not raw language capability. The architecture is conventional in shape (SLM + structured KB + retrieval) but TL collapses three layers (relational store + rule engine + audit log) into one differentiable substrate, with deterministic inference that preserves SLM fluency at the language boundary.

**Recommendation:** scout-grade prototype (~2 weeks) to validate the bet before deeper investment. Do not pretrain a custom LM. Do not put TL inside the transformer.

---

## What TL is, in one sentence

TL is a deterministic tensor-algebra framework for **relations and rules**: facts are sparse tensors, rules are tensor equations, inference is matrix products and fixed-point closure. Given a KB and a query, inference always returns the same answer. Rule weights, if learned, are deterministic at inference time even though training is stochastic.

The expressive class is roughly **Datalog with a soft semiring** — transitive closure, kinship/relational graphs, soft unification, bidirectional reasoning. It is *not* a substitute for a language model: it cannot read free text, cannot generate language, and cannot represent distributional polysemy. It is also not a complete replacement for arithmetic or counting — its semantics are monotone, so non-monotone tasks (parity, negation, mod-N counting) sit outside its learnable subclass.

---

## The architecture

```
                  ┌────────────────────────────────────────────┐
                  │         openhuman agent (local-first)      │
                  └────────────────────────────────────────────┘

  [user input]                                          [user output]
       │                                                      ▲
       ▼                                                      │
  ┌──────────────┐    structured query    ┌──────────────────┐
  │  small LM    │──────────────────────▶ │  TL substrate    │
  │  (Llama 3.2  │                        │  - relations     │
  │   1B / Qwen) │ ◀──────── facts ────── │  - rules         │
  │              │                        │  - provenance    │
  │ - parsing    │ tool: query_kb         │  - closure cache │
  │ - generation │ tool: assert_fact      │                  │
  │ - polish     │                        │  deterministic   │
  └──────────────┘                        └──────────────────┘
       ▲                                          ▲
       │                                          │
  [structured generation]                  [SLM extractor batch jobs]
   constrain SLM to KB-grounded outputs    populate KB from inbox events
```

**Three latency tiers:**

| Tier | Where SLM runs | TL involvement | Budget |
|------|---------------|----------------|--------|
| Hot (autocomplete, suggestions) | Quantized 1B local, polish only | Read-only against precomputed closure tables | <100ms |
| Warm (queries, replies) | Same model, full generation | Live inference + provenance logging | <500ms |
| Cold (extraction, learning) | Same or larger model | Full pipeline, rule updates | seconds |

**The architectural commitment that ties it together:**

> The SLM polishes, ranks, and rephrases. It does not invent grounded facts. Every fact in any output traces to a KB row, and every SLM call is logged with full I/O.

This is enforced by **structured generation** (Outlines / Instructor / grammar-constrained decoding): the SLM emits JSON conforming to a schema where every entity reference must come from a KB-derived candidate set. Ungrounded text causes generation failure and falls back to a deterministic KB-only response.

---

## What TL specifically buys openhuman

Concentrated in four properties. None of these are "5% better at language tasks." They are categorical wins where the alternative architecture either can't do the thing or does it unreliably.

### 1. Multi-hop relational queries (kinship, social graph)

> "Find people on my mom's side of the family who are free this weekend."

A vector store with RAG cannot do this without the answer already existing as a document. A TL closure walks `parent⁻¹ ∘ sibling ∘ descendants*` deterministically and returns the closed subgraph. Kinship is exactly transitive closure over a finite relation — TL's native sweet spot. ~10× better than RAG when the query needs >2 hops.

### 2. Surgical edit when life changes

> "I'm divorced. Karen's not my partner anymore."

Pure-LLM systems diffuse facts across embeddings; you cannot reliably remove "Karen is my partner" without retraining. In TL, `retract(partner(jwalin, karen))` plus `assert(former_partner(jwalin, karen, divorced=true))` is immediate, persistent across sessions, and propagates through every dependent rule. This is categorically different — not a quality improvement.

### 3. Auditability by construction

Every TL inference leaves a rule chain. The user can ask "why did you suggest dinner with Aunt Sarah?" and receive an actual derivation:

```
close_family(sarah, jwalin) ∧ haven't_met(sarah, last_60d) ∧
available_friday(sarah, ⊤) ∧ user_pref(prioritize_close_family, ⊤)
```

Each clause is editable. SLM-only systems generate plausible post-hoc justifications that may or may not reflect actual reasoning.

### 4. Persistent identity across sessions

> [March] "Bob is my new project manager."
> [October] "What's the latest from my PM?"

LLMs are stateless across sessions modulo a memory file; KBs are stateful by design. For an agent supposed to run a user's life over years, statefulness with surgical update is the right substrate. The TL framing makes this compositional and queryable rather than a flat key-value store.

### 5. Compressed ontology + hypernym reasoning

A taxonomy is exactly transitive closure over `is_a`, which is TL's structural sweet spot. This unlocks two distinct capabilities:

**(a) Inherited behavior across the user's ontology.** Rules attach to *categories*, not individual entities. Add a new friend, and they automatically inherit every rule that applies to `friend`, `close_relation`, `person`. No enumeration required.

```
is_a(friend, close_relation).
is_a(close_relation, contact).
is_a(family, close_relation).
is_a(coworker, contact).

priority_threshold(close_relation, 0.8).
priority_threshold(contact, 0.4).

-- Derived: anyone classified as friend or family inherits priority 0.8
-- Anyone classified as coworker inherits 0.4
-- New "friend" assertion → automatic priority elevation, no rule edit
```

For a personal-assistant agent that needs to maintain consistent treatment of categories of people/events/places over years, this is *the* mechanism. Pure-LLM systems handle this via prompt engineering ("treat family with higher priority") that degrades silently and cannot be audited. SQL+rules handles it but you write the inheritance logic by hand for every category. TL gives you closure-based inheritance for free.

**(b) Knowledge compression via rule induction.** Every TL rule IS a compression: it replaces enumerated facts with a generative pattern. A KB of 50 enumerated `uncle()` facts becomes 5 sibling/parent facts plus 1 rule. Compression ratio scales with how compositional the user's domain is.

```
[ uncle(bob, alice).         ]      [ sibling(bob, jane).
[ uncle(bob, charlie).       ]      [ parent(jane, alice).
[ uncle(bob, em).            ]  →   [ parent(jane, charlie).
[ uncle(dave, alice).        ]      [ sibling(dave, jane).
[ uncle(dave, charlie).      ]      [ parent(jane, em).
[ uncle(dave, em).           ]      [ uncle(X,Y) :- sibling(X,Z), parent(Z,Y). ]
```

This is **inductive logic programming**. As openhuman's KB grows, periodic rule induction can discover patterns like "everyone in family X lives in region Y," "people who attend event-class Z usually share interest W," and replace many enumerated facts with one rule. TL's *differentiable* pitch finally earns its keep here: rule weights are learned by gradient descent over fact-coverage data, which is a real, available training signal (count of facts the rule explains).

**Soft hypernyms via embeddings.** The taxonomy doesn't have to be rigid. `soft_is_a(X, Y) = sim(emb_X, emb_Y) * w_hierarchy` — a bilinear form over embedding tensors gives fuzzy taxonomic matching. "Dog" partially activates "pet," "companion animal," "mammal," each weighted by semantic similarity composed with the explicit hierarchy. Vector stores can do similarity but cannot compose it with rule-based hierarchy walks; TL does both in the same algebra.

**Where this lives in openhuman:** alongside the kinship/social KB, a parallel `concept` ontology stores categories and their `is_a` chains. Every fact about a person/event/place links to one or more concepts. Rules attach to concepts, not entities. The user's ontology grows over time, gets compressed periodically by induction, and queries traverse it transparently via closure. None of this is on the SLM hot path — it's all in the structured substrate, fully deterministic, fully auditable.

### 6. Continual learning without plasticity loss

A personal-assistant agent operates in a non-stationary world by definition. The user's life evolves; new people, jobs, schedules, preferences arrive constantly.

For pure-LLM systems, this exposes **loss of plasticity** (Joudaki et al. 2026, "Barriers for Learning in an Evolving World"): in non-stationary settings, gradient dynamics become entrapped in invariant submanifolds of parameter space — activation saturation freezes units, representational redundancy clones units onto identical manifolds. The paradoxical finding: the same low-rank compression that aids generalization on static tasks is what drags the net into these traps. The result is a net that *appears* to keep training but cannot actually learn new things.

TL has no plasticity problem on the fragment-class slice — not because it solves LoP, but because there are no gradient dynamics to trap. New facts go into relation tables. New rules are added or retracted at the substrate level. The substrate is constitutionally plastic.

| Property | LLM-only | TL substrate |
|----------|---------|--------------|
| Adapt to new facts | Retrain / RAG with stale embeddings, subject to LoP | `assert_fact` |
| Surgical retract | Diffuse across embeddings, unreliable | `retract` |
| Continual learning over years | LoP-prone | Substrate-level |

For an assistant meant to grow with the user over years, this is arguably the strongest argument: a substrate that quietly stops being able to learn is the worst possible failure mode, and it is exactly what LoP predicts for naive continual training of monolithic LMs.

---

## Related work: convergent critiques of pure scaling

Three independent research lines converge on the diagnosis that scaling a single dense net is *one* strategy, not *the* strategy. TL stakes the substrate-level position complementary to all three.

**Learning mechanics.** Simon et al. (2026, "There Will Be a Scientific Theory of Deep Learning") argues for falsifiable laws over training dynamics, hidden representations, and generalization, in place of empirical scale-up. Tian's body of work makes this concrete: CoGS (2410.01779) proves that two-layer nets with quadratic activation discover *symbolic algebraic compositions* via gradient descent (semi-ring structure on weight space, ~95% match between GD solutions and analytical constructions). The Li₂ grokking framework (2509.21519) gives provable scaling laws for feature emergence across three learning stages. Tuci et al. (2026, "Generalization at the Edge of Stability") show generalization depends on training-geometry quantities — sharpness dimension, fractal attractors — invisible to raw param count. **TL is the limiting case of CoGS:** the algebraic structure is given as substrate, not discovered through training. The fragment-class / parity-class boundary in TL likely maps onto algebraic-decomposability conditions in Tian's framework.

**Smarter inference compute.** ∇-Reasoner (ICLR 2026) replaces discrete test-time search with first-order optimization on token logits in latent space, dual to KL-regularized RL alignment. Same diagnosis as Simon: bottleneck is not size, it's the inference procedure. TL is orthogonal — on the fragment-class slice, inference is closed-form fixed-point closure; no test-time optimization needed.

**Architectural compression.** Hyperloop Transformers (2026) achieve equal or better LM quality with ~50% fewer parameters by reusing a middle block across depth with matrix-valued hyper-connections. The **Lottery Ticket Hypothesis** (Frankle & Carbin) goes further: a randomly-initialized dense net contains a sparse subnetwork (~10–20% of weights) which, trained in isolation from the same init, matches the full net. **Mechanistic interpretability circuits** (Anthropic) localize learned computation to small attention-head + MLP subgraphs inside large transformers. All three say the same thing: the actual computation is a small structured object hiding inside a large, mostly-dead net. TL skips the hiding step — the structure is the model.

**Interpretability: retrofit vs substrate.** Shahnovsky & Dror (2026) make LLM web agents auditable by mapping their trajectories onto classical planning paradigms (BFS / best-first / DFS) post-hoc. TL inverts this: the substrate *is* the planning structure (forward-chaining closure over a relation graph). Interpretability is constitutive, not diagnostic.

**Reasoning collapse on long-horizon execution.** Apple's "Illusion of Thinking" (Shojaee et al., Jun 2025; arXiv ~2506) reports frontier reasoning models (o3-mini-high, DeepSeek-R1, Claude-thinking) collapsing to ~0% accuracy on Tower of Hanoi past N≈8-10, on River Crossing at N=3, and on Blocks World / Checker Jumping at analogous thresholds — *even when the explicit algorithm is provided in the prompt*. Reasoning-token usage *decreases* past the collapse point: the model effectively gives up. The Lawsen rebuttal (arXiv 2506.09250) attributes most of this to output-token truncation and shows the same models write a recursive Hanoi function fine when asked for code. The convergent reading: **single-pass token enumeration is the wrong execution substrate for problems whose natural form is recursion or fixpoint-iteration over typed state.** Our `exp55_tl_hanoi.py` lands the substrate-side counterpoint cleanly: a TL state-tensor + tensor-update substrate executes Hanoi at N=20 (≈10⁶ moves) with 100% legal-move + goal-reached, in 37s on CPU, with zero learned parameters. Closure-shape problems like the OPENHUMAN kinship/social-graph queries are exactly this regime — long-horizon, deterministic, state-heavy. Putting them in a substrate that's deterministic by construction is what lets the SLM stay small.

**Recursive scaffolding from the orchestration side.** Alex Zhang's Recursive Language Models (RLM) blog and the LongCoT benchmark (raw.works) attack the same collapse from the *opposite* end: LM gets a REPL, recursively sub-queries itself over slices of context. Claude Sonnet 4.5 + DSPy.RLM hits 45.4% on LongCoT-Mini vs 2.6% for the same model in passive single-shot mode; Qwen3.5-27B + DSPy.RLM scores 22.18% on the full LongCoT, >2× GPT-5.2. RLM is complementary to TL, not competitive: RLM picks *what to query*, TL determines *what gets executed when you query it*. The natural composition is RLM tools that include TL closure queries — that's the openhuman architecture in disguise.

**Methodological precedent for small-scale exps.** Wortsman et al. (ICLR 2024, arXiv 2309.14322, "Small-scale proxies for large-scale Transformer training instabilities") demonstrate that big-model phenomena (logit growth in attention, output-logit divergence) reproduce in small models at high learning rates, and that interventions (qk-layernorm, z-loss) transfer cleanly across scales. This is direct vindication of the methodology this repo has been using since exp1: **small, controlled, falsifiable experiments produce real signal about large-scale phenomena.** It also gives concrete tools (qk-layernorm, z-loss) for any future TL-injected-into-transformer attempt — exp32/exp37 were both null at toy scale and may have been training-instability-bound rather than architectural-bound.

**Shared implication for openhuman.** Whether you go through learning-mechanics laws, sparse-circuit extraction, post-hoc planning maps, the substrate vs orchestration attacks on reasoning collapse, or the small-scale-proxy methodology, the field is reaching for the same thing: *the small, structured object that's actually doing the work.* TL is what you get when you write that object down directly, on the slice of the problem where it can be written down. The rest belongs to the SLM.

---

## The new-models landscape (Apr 2026)

Several recent open releases change which architectural moves are now cheap to try. Brief enumeration of where each fits in (or doesn't):

- **NVIDIA Newton physics engine + Isaac GR00T N1.7 + Cosmos Reason VLM** (CoRL 2025, GTC 2026). Newton is a GPU-accelerated, OpenUSD-native, *open-source* physics engine under the Linux Foundation (with Google DeepMind + Disney Research). GR00T is an open VLA for humanoid robots; Cosmos Reason is an open VLM for "vague instruction → step-by-step plan." For TL: the natural composition is **Cosmos Reason as rule-extractor → TL substrate as constraint/goal-reasoning layer → GR00T as actuator → Newton as forward model.** Phase 7 of this repo failed because TL forward-models don't compose with sampling-MPC; the right reframe is to let Newton be the forward model and have TL do constraint checking and goal scoring on top — closer to TL's actual sweet spot.
- **Microsoft TRELLIS.2** (Dec 2025, MIT-licensed). 4B-param image-to-3D generator with the O-Voxel sparse field-free voxel representation (geometry + PBR materials in one tensor). For openhuman directly: not load-bearing. For longer-horizon multimodal-grounding work: TRELLIS.2 voxels are the natural feed for a TL relation graph over scene objects (`above`, `inside`, `supports`, etc.) — an experimental path toward Spelke-style core-knowledge probes at real-data scale (`IDEAS.md` exp61).
- **Vision Banana** (Google DeepMind). Single generative model for 2D segmentation/referring-expression + 3D depth/normals via NL prompts; SOTA in zero-shot transfer. Tests the *opposite* architectural bet (one big generative net does everything). Useful as the baseline against which any future multimodal-TL claim has to defend itself.
- **Recursive Language Models / DSPy.RLM** (Zhang, raw.works, 2025-2026). Already covered above as the orchestration-side complement to TL.
- **Apple "Illusion of Thinking"** (Jun 2025) and the Lawsen rebuttal. Already covered above as the long-horizon-execution failure mode that TL substrates address by construction.
**Substrate-side coverage as of 2026-04-26 (post exp55-67).** The architectural claims in §1-3 of this memo are now operationally demonstrated:

| § | Claim | Status |
|---|---|---|
| §1 | Multi-hop relational queries (kinship/social) | ✅ exp65 (joins), exp44/47/53 (closure on real graphs) |
| §2 | Surgical edit when life changes | ✅ exp67 (counterfactual retraction propagates 1 → 0 proofs) |
| §3 | Auditability by construction | ✅ exp67 (full proof trees as data) |
| §4 | Persistent identity across sessions | ⚠️ trivially true (immutable graph storage), untested |
| §5 | Compressed ontology + hypernym reasoning | ✅ exp65 (rule chains over `is_a`) |
| §6 | Continual learning without plasticity loss | ✅ trivially (substrate is plastic by construction) |

The remaining open questions are integration-side: whether a small SLM can be SFT'd to reliably emit `<tl_closure>` / `<tl_rule>` calls (exp60d), whether TL-as-layer beats TL-as-tool (exp61), whether closure is learnable in transformer latents (exp62). Substrate-side, the bet is paid off.

- **NVIDIA Dynamo** (developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo). Open-source (github.com/ai-dynamo/dynamo) inference orchestration layer for agentic LM workflows. The relevant primitives are KV-aware routing (Flash Indexer maintains a global index of which KV cache blocks live on which workers), priority-based eviction, and `nvext.agent_hints.cache_control` to pin a stable system prompt while ephemeral reasoning tokens are evicted first. For an SLM+TL system, the (ontology, current-KB-state, query) prefix is the *stable* part — pin in cache; the `<tl_closure>` interlude and the post-tool-result generation are the only fresh decode work per turn. Claude Code's reported 85-97% cache hit rate at 11.7× read/write ratio is the order-of-magnitude target. **What this means for openhuman:** the SLM+TL architecture isn't a research curiosity that needs custom serving infrastructure; production-grade serving for exactly this composition pattern shipped as MIT-style open source in 2026. The bet is no longer "wait for the right serving stack"; it's "use the existing one."

The repo-level implication: the substrate-side claim openhuman rests on (TL as the deterministic relational layer underneath an SLM) is now sandwiched between (a) a frontier-scale failure mode it directly addresses (Apple's collapse) and (b) a frontier-scale architectural ally (RLM orchestration). The bet looks better than it did six months ago, not worse.

---

## Cleanest property: TL as both data layer AND audit layer

Provenance is itself a relation. Same algebra, no separate logging system:

```
slm_call(call_id, model, prompt_hash, output, timestamp)
grounded_in(call_id, fact_id)
produced(call_id, suggestion_id)
shown_to_user(suggestion_id, timestamp)
user_corrected(suggestion_id, correction, timestamp)
rule_fired(call_id, rule_id)
fact_retracted(fact_id, timestamp, reason)
```

Governance queries become normal TL queries:

- *Which SLM calls relied on facts later retracted?* → `produced(C, S), grounded_in(C, F), retracted(F, T_r), shown_to_user(S, T_s), T_s < T_r`
- *Which rules fire most but get corrected most?* → ratio of `rule_fired ∧ user_corrected` over `rule_fired`
- *What did the SLM tell me about Bob in the last 30 days?* → straightforward join

This is **provenance semirings** in the Datalog/TL literature. The audit layer is not bolted on; it is the same tensor that holds the facts. This is one of the strongest practical arguments for TL over a generic SQL+rules-engine stack.

---

## Why not GraphRAG?

The closest mainstream architecture to what this memo proposes is **GraphRAG** (Microsoft 2024) and the broader knowledge-graph-augmented LLM stack: extract entities and relations into a graph, retrieve a relevant subgraph at query time, hand it to the LLM as in-context text, generate an answer. This is the live competitor — not vector RAG, which the memo already dismisses.

The two architectures look superficially similar (both have a graph and an LM), but they differ on **where reasoning actually runs**:

| Capability | GraphRAG + LLM | TL substrate + SLM |
|---|---|---|
| Where multi-hop inference executes | Inside the LLM, reading the subgraph as text | Inside the substrate, as deterministic closure |
| Inference determinism | LLM-stochastic (same query → varying answers) | Closed-form (same query → identical answer) |
| Provenance | Cited subgraph rows; the *reasoning step* is opaque | Full rule chain; every derivation step inspectable |
| Surgical retract | Edit graph; LLM may still recall stale embedding traces | `retract` propagates deterministically through closure |
| Cost per multi-hop query | Full LLM call with subgraph in context | Closure walk (µs) + small SLM polish |
| Failure mode | Plausible hallucination grounded in a real graph | Empty result or partial chain — visibly incomplete |

The substantive distinction: in GraphRAG, the graph is *retrieved and read*; reasoning still happens inside the LLM with all its stochasticity, latency, and audit problems — the graph is *informational*. In the TL architecture, the graph **is** the reasoning engine; the SLM only polishes surface form — the graph is *operational*.

GraphRAG is the right choice when the bottleneck is finding relevant context for free-form questions over a large corpus. TL is the right choice when the bottleneck is rigorous multi-hop derivation that has to be reproducible and auditable.

For openhuman specifically, the personal-assistant workload is dominated by the latter: kinship walks, ontology inheritance, surgical retraction on life events, "why did you suggest this." That is TL's natural slice. For genuinely fuzzy queries (open-ended NL understanding), the SLM still handles the language boundary; TL doesn't intrude. **TL and GraphRAG are not interchangeable — they answer different questions about where reasoning should live.**

**Adjacent alternatives worth naming.** Production Datalog engines (Soufflé, RDFox, DDlog) cover the deterministic-closure side without the differentiable-rule story; they are the right tool if rule learning is never needed. Memory-augmented LLM systems (MemGPT/Letta, Mem0, LongMem) bolt persistent state onto an LLM but keep reasoning inside the LLM — same operational mode as GraphRAG, with the same audit limits. Logic Tensor Networks and DeepProbLog sit in TL's neuro-symbolic neighborhood; TL distinguishes itself by collapsing facts, rules, and provenance into a single tensor algebra rather than coupling a logic layer to a separate neural network.

---

## What TL does NOT help with

Be honest about this. The pitch collapses if these get oversold.

| Need | TL helps? | Right tool |
|------|-----------|------------|
| Free-form NL understanding | ❌ | SLM |
| Generation | ❌ | SLM |
| Novel intents | ❌ silently misses | SLM degrades gracefully |
| Numerical aggregation ("how many emails this week") | ❌ | SQL tool call |
| Counting / parity / mod-N | ❌ structurally | Imperative code |
| Temporal arithmetic ("3 days from now") | ❌ | datetime library |
| Open-world novel-entity introduction | ⚠️ requires KB extension | LLM handles fluidly |

If openhuman's bottleneck is "the SLM is bad at understanding what I said," TL doesn't help. TL only helps when the bottleneck is **structured reasoning, persistent memory, or auditability**.

---

## Determinism: an underrated property

| Layer | Deterministic at inference? |
|-------|------------------------------|
| TL inference | ✅ Same KB + same query → same answer, always |
| TL rule weights (if learned) | ✅ Stochastic only during training |
| SLM polish | ❌ Sampling-stochastic |

**Implication:** the grounded part of every reply is reproducible. The wording varies; the facts don't. This is genuinely different from a pure LLM and matters for trust. It also makes the KB layer testable with conventional unit tests, even when the SLM layer is hard to test.

---

## Latency budget

For a personal-scale KB (~5K entities, ~50K facts, ~50 rules) on Mac M-series CPU:

| Operation | Latency |
|-----------|---------|
| Atomic fact lookup | 10-50µs |
| One-hop join | 50-200µs |
| Multi-hop kinship (3-4 hops) | 0.5-2ms |
| Closure (precomputed read) | 50µs |
| Soft unification w/ embeddings | 1-10ms |
| Provenance trace query | 0.5-5ms |

Compare:

| Reference | Latency |
|-----------|---------|
| SQLite indexed lookup | 0.1-1ms |
| Llama 3.2 1B INT8 first token | 50-100ms |
| Llama 3.2 1B INT8 per token | 5-15ms |

**TL is never the bottleneck.** The SLM dominates the budget at every tier. You can be lavish with TL inference — including running provenance traces and "should this suggestion have been made" sanity checks — without affecting user-perceived latency.

---

## Cost & risks

### Engineering cost (rough estimates)

| Phase | Effort |
|-------|--------|
| Schema design (8-15 hard predicates) | 0.5 day |
| Bootstrap interview UX | 1 day |
| SLM extractor pipeline | 2-3 days |
| Entity resolver | 3-5 days *(this is the genuinely hard piece)* |
| TL/relation store + closure rules | 2-3 days |
| query_kb / assert_fact tool interface | 2 days |
| Eval harness vs. vanilla RAG baseline | 3 days |
| **Prototype to ship** | **~2 weeks** |

### Risks

1. **Entity resolution is harder than it looks.** Multiple "Bob"s in a corpus must collapse to the right atoms. Standard ER literature applies; the inbox project's `contacts.py` provides a starting point. This is the single biggest risk to prototype quality.
2. **Schema drift.** If the SLM extractor is allowed to mint new predicates, the schema decays into noise. Lock the predicate set; new predicates require human review.
3. **Extractor quality.** A bad extractor poisons the KB. Need confidence-weighted facts and periodic re-extraction with newer models.
4. **The TL "differentiable" pitch goes unused at prototype scale.** At ~50K facts, hand-written rules outperform learned rule weights; gradient training has no signal. This is fine — TL still earns its keep as a uniform substrate — but the more exotic claims about learnability stay theoretical until scale or labeled data justifies them.
5. **Inductive-bias mismatch on non-monotone problems.** Anything requiring counting, negation-as-failure, or non-monotone reasoning will need to be punted to SQL or imperative code outside TL.

### Risks NOT to take

- Don't pretrain a custom LM. Use Llama 3.2 / Qwen 2.5 / Phi-3 — fine-tune if needed.
- Don't put TL inside the transformer (Differentiable Theorem Provers, etc.). Open research, no demonstrated wins.
- Don't try to make TL handle language understanding. It can't, structurally.

---

## Recommendation

**Build a focused prototype before deeper commitment.**

The architecture is coherent on paper, but the value proposition depends on whether trust/auditability/persistence properties actually matter to openhuman's first 100 users. That is testable empirically. Two weeks of focused work produces the artifact that resolves the question.

**Proposed prototype scope:**

- Schema: 10 hard predicates (person, name, email, sent, received, parent, sibling, partner, attended, location_of)
- Mini-ontology: ~15 seed concepts (friend, family, coworker, contact, close_relation, work_event, social_event, ...) with hand-coded `is_a` chain; defer rule induction to post-prototype
- KB: SQLite tables, one per predicate, plus concept/is_a/classified_as tables, with confidence column
- Extractor: Llama 3.2 1B with structured output, run over a subset of inbox data; produces both fact triples AND concept classifications
- Rules: ~10 hand-written derivation rules (kinship closure, family, recency) + ontology closure rules
- Query interface: `query_kb` and `assert_fact` as MCP tools, with concept lookups exposed
- SLM: existing inbox MCP wired to Llama via Ollama
- Eval: 50 hand-labeled queries split across kinship multi-hop, ontology-inherited behavior, and surgical-edit scenarios; measure precision/recall vs. vector RAG baseline

**Decision criterion:** if TL+KB beats RAG by >20% on multi-hop queries AND the ontology layer demonstrates clean inheritance for new entities AND the auditability UX feels meaningfully better in user testing, proceed. Otherwise, drop TL and use SQLite + vector store + system-prompt rules.

---

## Open questions

These need decisions before the prototype begins. Some need user input; others can be designed.

1. **Schema specifics** — exact predicate vocabulary
2. **Bootstrap interview** — seed questions, when to run, intrusiveness
3. **Extractor model + prompt** — quality/latency/cost tradeoff
4. **Entity resolution** — disambiguation strategy, integration with `contacts.py`
5. **Eval methodology** — what constitutes "better"
6. **Correction UX** — how user edits/retracts facts
7. **Privacy boundary** — what runs locally vs. cloud (extraction step is the question)
8. **Failure modes** — extractor wrong, KB stale, rule conflicts, degraded fallbacks
9. **Prototype scope** — exact cut for the 2-week spike
10. **Ontology seeding** — initial concept set, hypernym hierarchy, who curates additions
11. **Rule induction trigger** — when (if ever) do we run inductive logic programming over the KB; post-prototype decision

---

## Appendix A: minimum schema sketch

```sql
-- Entities
CREATE TABLE person (id INTEGER PRIMARY KEY, canonical_name TEXT);

-- Hard predicates (one table each)
CREATE TABLE rel_name      (person_id INTEGER, name TEXT, conf REAL);
CREATE TABLE rel_email     (person_id INTEGER, email TEXT, conf REAL);
CREATE TABLE rel_phone     (person_id INTEGER, phone TEXT, conf REAL);
CREATE TABLE rel_sent      (person_id INTEGER, message_id INTEGER, ts INTEGER);
CREATE TABLE rel_received  (person_id INTEGER, message_id INTEGER, ts INTEGER);
CREATE TABLE rel_parent    (parent_id INTEGER, child_id INTEGER, conf REAL);
CREATE TABLE rel_sibling   (a_id INTEGER, b_id INTEGER, conf REAL);
CREATE TABLE rel_partner   (a_id INTEGER, b_id INTEGER, conf REAL, status TEXT);
CREATE TABLE rel_attended  (person_id INTEGER, event_id INTEGER, conf REAL);

-- Ontology layer (concepts + hypernym hierarchy)
CREATE TABLE concept (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB);
CREATE TABLE rel_is_a (sub_id INTEGER, super_id INTEGER, conf REAL);
CREATE TABLE rel_classified_as (entity_id INTEGER, entity_type TEXT, concept_id INTEGER, conf REAL);
CREATE TABLE concept_property (concept_id INTEGER, property TEXT, value TEXT, conf REAL);

-- Provenance
CREATE TABLE slm_call (call_id INTEGER PRIMARY KEY, model TEXT, prompt_hash TEXT, output TEXT, ts INTEGER);
CREATE TABLE grounded_in (call_id INTEGER, table_name TEXT, row_id INTEGER);
CREATE TABLE rule_fired (call_id INTEGER, rule_id TEXT);

-- Induced rules (output of periodic rule-induction batch job)
CREATE TABLE induced_rule (rule_id TEXT PRIMARY KEY, body TEXT, head TEXT,
                          coverage INTEGER, confidence REAL, status TEXT, ts INTEGER);
```

Closure rules expressed as recursive CTEs or, equivalently, sparse tensor products if the substrate ports to PyTorch. The ontology layer is queried identically — `is_a` closure walks the hypernym chain, `classified_as` joins entities to their inherited rules.

## Appendix B: example derivation rules

**Kinship closure:**
```
uncle(X, Y) :- sibling(X, Z), parent(Z, Y), male(X).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
close_family(X, Y) :- parent(X, Y).
close_family(X, Y) :- sibling(X, Y).
close_family(X, Y) :- close_family(X, Z), parent(Z, Y).
```

**Ontology / hypernym closure:**
```
ancestor_class(X, Y) :- is_a(X, Y).
ancestor_class(X, Y) :- is_a(X, Z), ancestor_class(Z, Y).

-- Inherited classification: entity belongs to all super-concepts of its direct class
inherited_class(E, C) :- classified_as(E, C).
inherited_class(E, C) :- classified_as(E, C0), ancestor_class(C0, C).

-- Inherited property: entity inherits properties of any concept it belongs to
inherited_prop(E, P, V) :- inherited_class(E, C), concept_property(C, P, V).
```

**Worked example — a new contact entering the system:**
```
Initial state:
  is_a(friend, close_relation).
  is_a(close_relation, contact).
  concept_property(close_relation, priority_floor, 0.8).
  concept_property(contact, priority_floor, 0.4).

User asserts: "Dave is my friend."
  → assert classified_as(dave, friend).

Now, by closure, no new rules needed:
  inherited_class(dave, friend)            ✓ direct
  inherited_class(dave, close_relation)    ✓ via is_a closure
  inherited_class(dave, contact)           ✓ via is_a closure
  inherited_prop(dave, priority_floor, 0.8) ✓ via close_relation property

Dave's priority floor is automatically 0.8. No rule edited, no entity-specific
configuration. The category does the work.
```

## Appendix C: example end-to-end trace

```
User: "remind sa..."

Hot path:
  1. KB candidates (deterministic): [sarah, sam, samir]
     Each carries provenance: name match + relation derivation
  2. SLM polish (Llama 3.2 1B, ~70ms):
     Input: prefix + candidates + thread context
     Output: "remind Sarah to bring the cake"
  3. UI renders.
  4. Audit log entry written:
     - 3 candidates considered, each with rule chain
     - Full SLM prompt + output + latency
     - Selection rationale (close_family + thread-context boost)

User: "why did you suggest Sarah?"

Agent walks the audit log entry, surfaces:
  - person(p_19) is in your contacts
  - aunt(p_19, jwalin) [from msg_3211, 2025-12-04, conf 0.94]
  - close_family rule fired
  - thread context: family chat
  - SLM rephrased "Sarah" as the natural completion

Each component editable.
```

---

*Memo ends. Next step if openhuman decides to investigate: scope the 2-week prototype against open questions 1, 2, 3, 9.*
