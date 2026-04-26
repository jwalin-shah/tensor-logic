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
