# tensor-logic — Wayfinder Map

Label: `wayfinder:map`
Created: 2026-07-17

## What It Is

Tensor logic proof system for program verification. Working through Pedro
Domingos' tensor logic paper. Goal: prove component contracts via tensor
operations, feed into Bridge's architecture compiler for verification gates.

## Current State

- **Branch:** `main` — clean
- **Build:** ✅ go build, go test, go vet all pass
- **Tests:** Minimal — prover, compose, counter packages have zero test coverage
- **Last commit:** `f9fc714` (2026-07-16)
- **Prototype ladder:** P1 (pattern library, 9 patterns) → P2 (tensor logic for leasing) planned

## Architecture

- 9 patterns built (daemon lifecycle, bounded channel, etc.)
- Pattern matcher for mapping problems to patterns
- LLM extraction pipeline for deriving patterns from natural language
- 23 Codex agent branches on remote for parallel development

## Prototype ladder

| Phase | Scope | Status |
|-------|-------|--------|
| P1 | Pattern library (9 patterns) | ✅ Done |
| P2 | Tensor logic for leasing | Not started |
| P3 | Composition (patterns → compound proofs) | Not started |
| P4 | Bridge integration (proof artifacts → ARCHITECTURE_VALIDATED gate) | Not started |
| P5 | Lean specs generation | Not started |

## Known gaps

- Bounded-channel and event-counter patterns referenced but not built
- Z2 only for MVP — composition hits 2^64 states at 8x8-bit components
- Linear invariants only — nonlinear needs piecewise or SMT fallback
- Encoding discipline is manual and error-prone

## Tickets

### 🔴 Active

1. **Add tests for prover, compose, counter** — zero test coverage in these packages.

2. **P2: Tensor logic for leasing** — prove a real Bridge component (worktree leasing) with tensor operations.

### 🟡 Next

3. **Bridge integration (P4)** — wire tensor-logic as a Go library that Bridge imports. Currently bridge references tensor-logic in comments but doesn't import it.

### 🔵 Future

4. **P5: Lean specs** — generate Lean specifications from verified tensor proofs.
