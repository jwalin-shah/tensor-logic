# Canonical Rule Language Shape

**Date:** 2026-05-04
**Status:** Approved for implementation planning
**Scope:** Shared Atom and Rule representation for TL program rules, LM-emitted `tl_rule` tags, proof search, and negative-proof search.

---

## Decision

The canonical internal Rule language is the `tensor_logic.program` shape:

```python
@dataclass(frozen=True)
class Atom:
    relation: str
    args: tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom, ...]
```

This shape is the only shared representation implementation slices should target. The current duplicate `tensor_logic.rules.Atom` / `tensor_logic.rules.Rule` shape is an adapter-local parser/evaluator artifact and should be migrated to produce the canonical `program.Atom` / `program.Rule` values.

---

## Syntax Surfaces

### TL Program Rules

Program rules keep the current human-authored expression syntax:

```tl
grandparent(x,z) := (parent(x,y) * parent(y,z)).step()
ancestor(x,z) := (parent(x,z) + ancestor(x,y) * parent(y,z)).step()
```

`Program.rule()` / `RuleParser` remain responsible for parsing this syntax into two outputs:

- the tensor expression assigned to the destination relation, for current eager tensor evaluation;
- one or more canonical `Rule` values, one per OR-disjunct, for proof and negative-proof search.

The canonical `Rule` does not store expression operators such as `+`, `*`, `.step()`, or `.sigmoid()`. Those are syntax/evaluation concerns. The canonical rule body is only the logical conjunction of atoms in one derivation alternative.

### LM-Emitted `tl_rule` Tags

LM-emitted tags keep the compact XML-like surface:

```xml
<tl_rule head="grandparent(X, Y)" body="parent(X, Z), parent(Z, Y)"></tl_rule>
<tl_rule head="eligible(X, Y)" body="applicant(X, Y), !blocked(X, Y)"></tl_rule>
```

The `tl_rule` adapter should parse tag extraction, attribute validation, comma-separated body atoms, and `!` negation, then return canonical `Rule` values. It should not define its own durable Rule dataclasses.

Tag parsing may stay stricter than the canonical language while the LM path is binary-only. For example, it can reject malformed tags, missing body atoms, unknown arity for the current evaluator, or negated heads.

---

## Negated Atoms

Negation belongs on shared `Atom`, not in syntax adapters:

```python
Atom("blocked", ("X", "Y"), negated=True)
```

Rules may contain negated body atoms. Rule heads must be positive for now; adapters should reject or normalize away negated heads before canonical construction.

Shared invariants:

- `Atom.negated == False` by default.
- Positive and negated body atoms use the same `relation` and `args` fields.
- Negation is semantic data available to proof, negative-proof, tensor, and future search consumers.

Adapter responsibilities:

- TL program syntax decides what source spelling, if any, denotes negation in program rules.
- `tl_rule` syntax uses `!rel(X, Y)` in body atoms.
- Adapters reject unsupported negation forms for their surface syntax.

Consumer responsibilities:

- Positive proof search treats positive atoms as required derivations.
- Positive proof search treats negated atoms as required non-derivations.
- Negative-proof search treats failed rule bodies against the same invariant: a positive body atom fails when it cannot be proven; a negated body atom fails when the positive atom can be proven.

---

## Arity Boundary

The canonical Atom shape is variadic because `args: tuple[str, ...]` carries arity structurally. Do not bake binary-only fields such as `left` and `right` into the shared language.

Current consumers are still binary:

- `tensor_logic.proofs.prove()` and `prove_negative()` accept `(relation_name, src, dst)` and bind two head variables.
- Current proof helpers return `invalid_arity` or `None` for non-binary rule atoms.
- `tensor_logic.rules.evaluate_rule()` builds 2D relation tensors and has binary negation alignment assumptions.
- Existing `tl_rule` parsing only recognizes `rel(A, B)` atoms.

These binary assumptions should remain explicit at consumer boundaries until a dedicated future arity slice expands them. Future arity work should land in:

- proof APIs and proof tree head shapes, replacing fixed `(relation, src, dst)` tuples with relation plus argument tuple;
- relation tensor evaluation, including einsum equation generation for arbitrary arity;
- syntax adapters, after the canonical shape and proof/evaluator consumers can accept the wider arity;
- tests that lock mixed-arity parsing, proof binding, and negative-proof witness enumeration.

---

## Shared Rule Invariants

All syntax adapters should construct canonical values that satisfy these invariants before consumers see them:

- `Rule.head.negated` is false.
- `Rule.head.args` and each body `Atom.args` are non-empty tuples of variable or symbol names.
- `Rule.body` is a tuple. An empty body is allowed only if the consuming surface explicitly supports fact-like rules.
- Each `Atom.relation` is a relation name, not an expression method or operator.
- OR alternatives are represented as separate `Rule` values with the same head.
- Adapter-specific syntax artifacts are not stored on `Rule`.

Consumers may add their own boundary checks, such as "proof search currently requires head arity 2" or "`tl_rule` evaluation currently requires every atom arity 2". Those checks should report consumer limitations, not redefine the canonical language.

---

## Proof and Negative-Proof Semantics

Positive proof and negative-proof search should consume the same canonical `Rule` invariants.

For a positive proof of `head(args)`:

1. Bind the canonical rule head variables to the query arguments.
2. Search witnesses for unbound body variables.
3. For each candidate binding, evaluate body atoms left-to-right.
4. A positive atom succeeds when `prove()` can derive it.
5. A negated atom succeeds when `prove()` cannot derive its positive counterpart.
6. The rule succeeds when every body atom succeeds under one witness binding.

For a negative proof of `head(args)`:

1. Check the base fact first. If the queried fact is present, no negative proof exists.
2. Check every applicable rule; stopping at the first failed rule is invalid.
3. A positive body atom contributes a failure when it cannot be proven.
4. A negated body atom contributes a failure when its positive counterpart can be proven.
5. Witness enumeration must prove that no candidate binding makes the full rule body succeed.
6. The query has a negative proof only when no base fact and no rule derivation succeeds.

This keeps "why true?" and "why not?" explanations on one rule language. The difference is in the proof search algorithm, not in separate rule ASTs.

---

## Implementation Consequences

- Dependent parser work should make `tensor_logic.rules.parse_rule()` return canonical `program.Rule` values while preserving the existing `<tl_rule>` surface behavior.
- Program rule work should add `negated` to `program.Atom` and ensure existing parser outputs default to positive atoms.
- Proof work should add negated-body handling using `Atom.negated`, then keep current binary checks local to proof helpers.
- Future arbitrary-arity work should change proof/query APIs and tensor evaluators after this shared shape is in place.
