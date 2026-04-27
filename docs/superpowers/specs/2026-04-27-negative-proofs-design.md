# Negative Proofs Design

**Date:** 2026-04-27  
**Status:** Approved for implementation  
**Scope:** Add "why not?" query capability to tensor_logic proof engine

---

## Overview

Currently, `prove()` returns a `Proof` object (showing how a fact IS derivable) or `None` (showing it isn't). When a query fails, users have no insight into *why* it failed.

This design adds **negative proofs** — structured explanations of why a query cannot be proven. A negative proof shows:
1. The query is not a base fact
2. Every applicable rule fails to derive it (with recursive explanations of why each rule body couldn't be satisfied)

---

## Data Structure

### `NegativeProof`

```python
@dataclass(frozen=True)
class NegativeProof:
    head: tuple[str, str, str]  # (relation, src, dst)
    body: tuple["NegativeProof", ...] = ()  # failures from applicable rules
    reason: str = ""  # "not_a_fact", "rule_body_failed", "no_rules", etc.
```

**Design rationale:**
- **Mirrors `Proof` structure** — same recursive tree shape for consistency
- **Deep explanations** — body contains sub-failures showing why each rule/atom couldn't be proven
- **Reason field** — brief explanation of failure type (optional, used for formatting)

**Semantics:**
- `head`: The failed query (relation, src, dst)
- `body`: For each applicable rule, a `NegativeProof` showing why that rule's body couldn't be satisfied
- `reason`: One of:
  - `"not_a_fact"` — query not in base relation
  - `"no_rules"` — no rules derive this relation (and it's not a fact)
  - `"rule_body_failed"` — rule head matched, but body couldn't be proven
  - `"atom_not_provable"` — specific atom in rule body failed

---

## API

### `prove_negative()`

```python
def prove_negative(
    program: Program,
    relation_name: str,
    src: str,
    dst: str,
    recursive: bool = False
) -> NegativeProof | None:
    """
    Explain why a relation is NOT derivable.
    
    Returns:
        NegativeProof: detailed explanation of failure (query is provably false)
        None: query is provably true (can't give negative proof)
    """
```

**Behavior:**
1. Check if query is a base fact → if yes, return `None` (can't prove negative of a fact)
2. Check applicable rules:
   - If no rules exist → return `NegativeProof` with reason `"no_rules"`
   - For each rule: attempt to prove body atoms; collect failures
3. If any rule succeeds → return `None` (query is true)
4. If all rules fail → return `NegativeProof` with body containing failures from each rule
5. If `recursive=True` and no facts/rules help: run BFS-based recursive search; if no path found, explain

**Key design: Check ALL rules**

A negative proof is only logically valid if it shows that **no applicable derivation path succeeds**. If a relation has multiple rules:
- Stopping at the first failed rule is incorrect (another rule might succeed)
- We must check every rule and show why each one fails
- Output is bounded by the number of rules and rule body complexity

---

## Output Formatting

### `fmt_negative_proof_tree()`

```python
def fmt_negative_proof_tree(neg_proof: NegativeProof, indent: int = 0) -> str:
    """Format negative proof as human-readable tree."""
```

Example output:

```
depends_on(models, worker) = False

├─ Not a base fact
│  └ fact depends_on(models, worker) not in relation
│
├─ Checked rule: depends_on(X, Y) := imports(X, Y)
│  └ Rule body failed:
│     └─ imports(models, worker) not provable
│        └ Not a base fact
│           └ fact imports(models, worker) not in relation
│
└─ Checked rule: depends_on(X, Y) := imports(X, Z) * depends_on(Z, Y)
   └ Rule body failed:
      ├─ imports(models, Z) not provable (for any witness Z)
      │  └ checked Z in {worker, models, ...}
      │     └ all Z failed: [details...]
      └─ (didn't evaluate depends_on due to prior failure)
```

---

## CLI Integration

### Flag: `--why-not`

```bash
python -m tensor_logic prove <file.tl> <relation> <src> <dst> [--why-not] [--recursive]
```

**Behavior:**
- If query is true: show positive proof (tree or JSON format)
- If query is false and `--why-not` is set: show negative proof tree
- If query is false and `--why-not` is NOT set: show "False" or exit code 1

**JSON output with `--format json --why-not`:**
```json
{
  "answer": false,
  "explanation": {
    "head": ["depends_on", "models", "worker"],
    "reason": "no_applicable_rules",
    "body": [...]
  }
}
```

---

## Implementation Plan

### Phase 1: Core Engine (proofs.py)
1. Add `NegativeProof` dataclass
2. Implement `prove_negative()` with all-rules checking
3. Implement `fmt_negative_proof_tree()` formatter
4. Handle cycles/recursion (track visited nodes)

### Phase 2: Testing
1. Unit tests for negative proofs
2. Tests on example .tl files
3. Edge cases: no rules, multiple rules, recursive relations

### Phase 3: CLI Integration (__main__.py)
1. Add `--why-not` flag to `prove` command
2. Wire `prove_negative()` output to formatter
3. Add JSON serialization for negative proofs

---

## Scope & Non-Goals

### In Scope
- Explain why a query is NOT provable (depth-first, all rules)
- Format as tree and JSON
- CLI support with `--why-not`
- Tests on existing examples

### Out of Scope (Future)
- Model training / seeding negative proofs into LLMs
- Optimized output for large failure trees (bounded explanations)
- Confidence/probability scores on negative proofs
- Performance optimization for wide rule sets

---

## Edge Cases & Handling

| Case | Handling |
|------|----------|
| Query is a base fact | Return `None` (can't prove negative of a fact) |
| No rules for relation | Return `NegativeProof` with reason `"no_rules"` |
| Multiple rules, all fail | Return `NegativeProof` with body containing all rule failures |
| Recursive relation, no path | Run BFS search, explain why no path exists |
| Circular rules (A → B → A) | Track visited (relation, bindings) to prevent infinite recursion |
| Deep failure trees | Truncate output after N levels (future: add `--explain-depth` flag) |

---

## Success Criteria

- [ ] `NegativeProof` dataclass added
- [ ] `prove_negative()` correctly checks all rules
- [ ] All 11 existing tests still pass
- [ ] 5+ new tests for negative proofs (false queries, no rules, rule failures)
- [ ] `fmt_negative_proof_tree()` produces readable output
- [ ] CLI `--why-not` flag works end-to-end
- [ ] JSON serialization of negative proofs works
- [ ] Example .tl files produce sensible negative proofs

---

## Questions for Implementation

1. **Recursion depth limit:** Should we cap explanation depth to prevent huge outputs?
2. **Witness enumeration in negatives:** When explaining why an existential rule failed (e.g., "for any witness Z, imports(models, Z) failed"), should we enumerate all Z or summarize?
3. **JSON nesting:** Should JSON negative proofs follow exact same structure as positive proofs, or slightly different for readability?
