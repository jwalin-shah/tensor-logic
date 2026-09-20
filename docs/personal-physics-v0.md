# Personal Physics v0

Personal Physics is a replayable reasoning layer over LifeOps evidence.

It is not a memory summary and it is not a personality clone.

## Core contract

Every consequential conclusion must be reconstructable from:

1. source evidence,
2. explicitly admitted primitive facts,
3. versioned rules or deterministic adapters,
4. the exact Tensor Logic engine version,
5. a replayable derivation graph.

Candidate model output is never a premise merely because a model is confident.

## Rule classes

### 1. Invariants

Deterministic domain truths inside a declared scope.

Example:

- if two admitted time intervals overlap, the overlap relation is admitted.

These should be changed only by versioning the adapter or rule.

### 2. Source-derived computations

Deterministic calculations over authoritative observations.

Examples:

- calendar interval overlap,
- travel time versus available gap,
- dependency closure,
- test status,
- account balance arithmetic.

The numeric calculation is performed by a deterministic adapter. Its result is
an evidence-backed primitive relation that Tensor Logic composes with other
relations.

### 3. Human policies

Explicit user-authored or user-admitted preferences.

Examples:

- protect a declared focus block,
- do not auto-schedule over a protected commitment,
- surface a conflict when an optional event displaces a high-priority goal.

These are not objective truths. They are versioned policy rules whose authority
comes from the human.

### 4. Learned hypotheses

Rules or predicates proposed by an LLM, Jev-like model, BDH-like pathway model,
or representation learner.

Examples:

- a recurring latent pattern may predict follow-up risk,
- a learned anonymous relation may support a compositional rule.

Learned hypotheses remain candidates until an evaluator and admission process
promote them for a declared scope.

## Epistemic states

Personal Physics does not use absence as negation.

Facts have explicit states:

- candidate
- admitted
- rejected
- contradicted
- retracted

UNKNOWN is represented by the absence of an admitted positive or explicit
negative fact. A rule that needs a negative premise must depend on an explicit
negative relation, not negation-as-failure.

## Tool and API boundary

Tool calls are part of the world model.

A call should have:

- tool identity,
- input identity or hash,
- output identity or hash,
- execution receipt,
- observed time,
- source authority,
- evidence object.

Example:

routing API -> response and receipt -> admitted insufficient-travel-gap primitive
-> Tensor Logic rule -> infeasible-transition conclusion.

The language model may explain the trace, but it does not manufacture the
routing duration.

## What counts as proof

A Personal Physics proof is an external machine-checkable derivation graph, not
the hidden chain-of-thought of a model.

For a conclusion such as an infeasible transition between two events, the trace
must identify:

- primitive fact IDs,
- evidence refs,
- deterministic adapter refs where relevant,
- exact rule ID and rule digest,
- world snapshot digest,
- derivation digest.

If a premise is retracted, replay must remove any conclusion that depended on
that premise.

## Adaptability

Rules are not assumed to be universally correct forever.

Every rule has a version and scope. Adaptation means:

1. observe a counterexample or changed human policy,
2. preserve the old rule and its historical derivations,
3. propose a new rule or version,
4. test it against held-out evidence,
5. admit the new version only for an explicit scope,
6. recompute affected derived state.

This gives adaptability without silently rewriting history.

## Relationship to learned systems

- GLiNER-like systems: propose structured extractions from raw text.
- Jev-like systems: make cheap typed probabilistic decisions.
- BDH or representation learners: discover latent pathways or candidate structure.
- Tensor Logic: deterministically composes admitted structure.
- LifeOps: owns evidence, provenance, admission, contradiction, and durable state.
- LLM researchers: propose hypotheses, rules, adapters, experiments, and explanations.

No one component is the authority over its own output.

## v0 proof

The first proof must cover:

- calendar conflict,
- travel feasibility,
- unresolved follow-up with explicit negative evidence,
- project-to-goal composition,
- tool-call-to-claim provenance,
- candidate fact exclusion,
- fact retraction,
- deterministic replay digest.

The fixture must contain no personal data.
