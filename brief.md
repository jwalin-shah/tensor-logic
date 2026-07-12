# pattern-library-extraction

## task

Build the pattern extraction pipeline in Bridge that populates tensor-logic's
pattern library from real evidence. Target 8 proven patterns.

## code references

- `patterns/patterns.go` — existing Component type, ComposeParallel, 3 stub patterns
- `internal/tensor/tensor.go` — Z₂ tensor primitives (Vector, Matrix, ConstraintMatrix)
- `internal/prover/prover.go` — exhaustive proof checker
- `internal/counter/counter.go` — counterexample extraction with fix suggestions
- `AGENTS.md` — full prototype ladder and architecture

## approach

1. Query githits for OSS implementations of each primitive (mutual exclusion,
   bounded channels, stop signaling, state machines, fencing tokens,
   event counting, circuit breakers, work stealing).

2. For each primitive: extract state variables, transition logic, invariants.

3. Encode as tensor-logic Components. Linear invariants → Z₂ exhaustive proof.
   Nonlinear invariants → Z3 delegation with tensor composition for encoding.

4. Write adversarial tests for each pattern.

5. Register in AllPatterns(). Each pattern cites provenance.

## reasoning scaffold

### SCOPE
- Files: patterns/*.go, internal/tensor/*.go, internal/prover/*.go
- New: patterns from githits extraction
- Non-goals: nonlinear prover, Lean proofs, automatic discovery without curation

### RISK
- Encoding errors: wrong state bits or transition matrix → vacuous proofs
- Mitigation: adversarial counterexample tests, Z3 cross-validation

### ROUTING
- ct (routine task — mechanical extraction from existing standards)

### VERIFICATION
- `go test ./patterns/` — all Verify() pass
- `tlogic prove <pattern>` — each succeeds
- Adversarial fixtures: deliberately broken transitions caught

### ACCEPTANCE
- 8 patterns proven with provenance
- All tests pass, no vet warnings
