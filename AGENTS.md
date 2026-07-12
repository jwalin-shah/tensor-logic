# tensor-logic

Tensor logic representation and proof system for program verification.
Encodes program state as tensors, state transitions as tensor contractions,
and invariants as subspace constraints. Proofs compose by tensor product —
prove components independently, compose proofs automatically.

## Purpose

Bridge's architecture compiler uses tensor-logic to:

1. **Represent** program state as vectors in finite-dimensional spaces over Z₂
   (MVP), ℝ (later), and finite fields (later).
2. **Encode** state transitions as linear transformations (tensor contractions).
3. **Prove** invariants: does transition T preserve invariant subspace I?
4. **Compose** proofs: if A is proven and B is proven, A ⊗ B is proven
   automatically. No re-proving. No state explosion.
5. **Counterexample** extraction: when a proof fails, produce the exact state
   vector v ∈ I such that T·v ∉ I — the error that tells you what to fix.

## Stack

- Go 1.26, single module `tensor-logic`.
- `cmd/tlogic` — CLI entrypoint (`tlogic prove`, `tlogic compose`, `tlogic check`).
- `internal/tensor` — core tensor types: Vector, Matrix (transition), Subspace
  (invariant). Operations: contraction, tensor product (Kronecker), norm.
- `internal/space` — state space definitions. Z₂ (bits) for MVP. Finite fields
  GF(2^k) later. ℝ later.
- `internal/invariant` — invariant encoding. Linear constraints (hyperplanes).
  Affine constraints. Piecewise linear for nonlinear later.
- `internal/prover` — proof checker. Exhaustive for Z₂ (finite). Sampling for
  large spaces. SMT fallback for ℝ.
- `internal/compose` — tensor product composition. Given two proven components,
  construct and verify the composed proof. Pure Kronecker product + verification.
- `internal/counter` — counterexample extraction. When prover returns FAIL,
  compute v ∈ I such that T·v ∉ I. For linear invariants: linear programming.
  For others: SMT or search.
- `patterns/` — pre-proven architecture patterns. Each pattern is a component
  with a state space, transition tensor, invariant subspace, and proof artifact.
  Start with: leasing, fencing, event-counting, bounded-channel.
- `verification/` — proofs that tensor-logic itself is correct. Z3 proofs for
  the proof checker. Property-based tests for tensor operations.

## Build

```sh
go build ./cmd/tlogic
```

## Test

```sh
go test ./...
```

## Lint

```sh
go vet ./...
```

## Architecture

```
Brain dump (from Bridge create pipeline)
  │
  ├─→ Problem signature extraction
  │     Boundaries, concurrency model, state shape, lifetime
  │
  ├─→ Pattern matching
  │     Match problem signature against pattern library.
  │     "event watcher with bounded buffer" → EventWatcherPattern
  │
  ├─→ Component selection
  │     Select pre-proven components that satisfy the pattern:
  │     buffer → bounded-channel component (proven)
  │     stop → atomic-flag component (proven)
  │     handler → guarded-callback component (proven)
  │
  ├─→ Composition
  │     Compose selected components via tensor product.
  │     I = I_buffer ⊗ I_stop ⊗ I_handler
  │     T = T_buffer ⊗ T_stop ⊗ T_handler
  │     Verify: does T preserve I? (Automatic — composition theorem)
  │
  └─→ Proof artifact
        Emit: proven architecture + tensor encodings + verification.
        Bridge ARCHITECTURE_VALIDATED gate passes when all proofs hold.
```

## Verification

- **Z3 proofs**: `verification/z3/` — tensor-logic's own proof checker is correct.
  Proves: the composition theorem holds, the counterexample extractor is sound,
  the Kronecker product preserves linear independence.
- **Property-based tests**: gopter random state vectors against known invariants.
  Generate random tensors, random states, verify invariant preservation.
- **Golden tests**: known patterns (leasing, fencing, counting) with expected
  proof outcomes and expected counterexamples. Immutable — if a golden test
  breaks, something fundamental changed.

## Pattern Matcher

`internal/matcher` connects natural-language problem descriptions to pre-proven
components. It is deterministic — no ML, no embeddings. Every match cites which
keyword or domain triggered it.

```
brain dump ("file watcher daemon")
  → ExtractSignature: boundaries, concurrency, state shape, lifetime, domains, keywords
  → MatchComponents: keyword matching + domain defaults + lifetime defaults + concurrency defaults
  → ComposeMatched: compose matched components into a single system
  → output: component list + composed state space + proof status
```

CLI: `tlogic match "file watcher daemon with graceful shutdown"`

### Domain defaults

When a brain dump is too sparse for keyword extraction (e.g. just "file watcher
daemon"), domain defaults fill in the standard architecture from OSS precedent:

| Domain | Standard components | Language | Why |
|--------|-------------------|----------|-----|
| file-watcher | bounded-channel, stop-signal, event-counter, fence | Rust | kernel boundary, GC-free callbacks |
| api-server | mutex, stop-signal, event-counter, fence | Go | stdlib HTTP, goroutines per request |
| daemon | stop-signal, event-counter | Go | single binary, LaunchAgent compatible |
| cli-tool | (none) | Go | stateless transform, fast startup |
| pipeline | bounded-channel, stop-signal | Go | channels = pipeline stages |

### Catalog gaps

The pattern library has 8 components. Known gaps — components referenced by
domain defaults but not yet built:

- **bounded-channel** — referenced by file-watcher, pipeline domains
- **event-counter** — referenced by all domains except cli-tool

These need non-trivial Z₂ transitions (not identity matrices). The pattern
worker in the first spawn correctly skipped them — they require nonlinear
invariants (counter comparison) or multi-bit state (channel capacity).

## Conventions

- A `Component` is the unit of proof. It has: a `Space` (the state vector space),
  a `Transition` (the matrix/tensor), and an `Invariant` (the subspace).
- `Compose(A, B)` returns a new `Component` whose proof is automatically valid
  if A and B are individually proven.
- `Prove(C)` returns `(passed bool, counterexample *Vector)`. If `passed` is
  false, `counterexample` is the exact state that violates the invariant.
- The `counterexample` is the error message. It should be human-readable: "state
  [closed=true, buffered=3] after transition leads to [closed=true, buffered=2].
  Invariant `no_event_lost` violated: event dropped without processing."
- Tensor operations over Z₂ use `uint64` as the backing word. Bitwise AND/OR/XOR
  are the primitive operations. This is fast and exact — no floating point.
- All public types implement `fmt.Stringer` for human-readable output.
- Proof artifacts are JSON-serializable for Bridge's ledger.

## Sharp edges

- Z₂ only for MVP. Moving to larger finite fields changes the representation
  from `uint64` to `big.Int` or a polynomial ring. Plan for this but don't
  build it yet.
- Composition by tensor product produces state spaces of size |A| × |B|.
  For Z₂, this is `2^(bits_A + bits_B)`. At 8 components with 8 bits each,
  that's 2^64 states — exhaustive checking becomes impossible. Switch to
  randomized checking at some threshold (TBD: 2^20 states?).
- Linear invariants only. Many real invariants are nonlinear ("if closed then
  buffered=0"). For MVP, approximate nonlinear invariants as piecewise linear
  (conjunction of linear constraints). Fall back to Z3/SMT for genuinely
  nonlinear cases.
- The encoding discipline problem: mapping a real program's state to a tensor
  is manual and error-prone. Mitigation: the pattern library provides
  pre-verified encodings. New encodings require adversarial review.
- This is a tool for Bridge's architecture compiler. It is not a standalone
  product. The CLI exists for development and debugging. Production use is
  through Bridge's `internal/create/arch_validate.go` calling tensor-logic
  as a library or subprocess.

## Prototype ladder

### P1: Pattern library (weeks 1-2)
- `patterns/` directory with 3-5 pre-proven patterns
- Each pattern has: state space definition, transition, invariant, proof
- Proofs are manual (hand-computed) for now
- CLI: `tlogic list-patterns`, `tlogic show-pattern <name>`

### P2: Tensor logic for leasing (weeks 3-5)
- Core tensor types and operations (Z₂)
- Prover: exhaustive checking for small state spaces
- Counterexample extraction
- One pattern (leasing) fully proven end-to-end
- CLI: `tlogic prove --pattern lease`

### P3: Composition (weeks 6-7)
- Composition engine: `Compose(A, B)` via Kronecker product
- Automatic proof: if A and B proven separately, the composition holds
- Test with leasing + fencing composed

### P4: Bridge integration (weeks 8-9)
- ARCHITECTURE_VALIDATED phase in Bridge's create pipeline
- Bridge calls tensor-logic for proof checking
- Proof artifacts in the ledger

### P5: Lean specifications (weeks 10-12)
- Lean specs for the 3 most-violated invariants
- Generate Go property-based tests from Lean specs
- Integrate into Bridge's verification commands
