// Package patterns provides pre-proven architecture patterns.
// Each pattern is a Component with a state space, transition tensor,
// invariant subspace, and proof artifact. Patterns are the building
// blocks the architecture compiler composes.
//
// A Component is the unit of proof:
//   - Space: the state vector space (dimension, bit assignments)
//   - Transition: the transition matrix over Z₂
//   - Invariant: the constraint matrix defining the safe subspace
//   - Proof: verification that T preserves I
//
// Patterns are pre-proven — their proofs are verified at build time
// via Go tests or external proof checkers (Z3, tensor-logic prover).
// When composed, the composition theorem guarantees the composite
// preserves its invariants without re-proving.
package patterns

import (
	"fmt"

	"tensor-logic/internal/tensor"
)

// ComponentRole describes the role a component plays within a pattern.
type ComponentRole string

const (
	RoleBuffer         ComponentRole = "buffer"
	RoleStopSignal     ComponentRole = "stop_signal"
	RoleGuard          ComponentRole = "guard"
	RoleLease          ComponentRole = "lease"
	RoleFence          ComponentRole = "fence"
	RoleCounter        ComponentRole = "counter"
	RoleLock           ComponentRole = "lock"
)

// Component is a proven unit of architecture. Each Component has:
//   - A human-readable Name for identification
//   - A Role describing its architectural function
//   - A description of its StateSpace (which bits mean what)
//   - A Transition matrix over Z₂
//   - An Invariant (constraint matrix) defining the safe subspace
//   - A mapping from bit positions to human-readable state variable names
type Component struct {
	Name       string
	Role       ComponentRole
	Dim        int                // number of state bits
	Variables  []string           // var[i] = name of bit i
	Transition tensor.Matrix      // transition over Z₂
	Invariant  tensor.ConstraintMatrix // safe subspace
	Proven     bool               // has this component been proven?
	ProofNote  string             // how the proof was established
}

// Verify checks that the component's transition preserves its invariant
// for all reachable states. Returns nil if the invariant holds, or an
// error describing the counterexample.
func (c *Component) Verify() error {
	if c.Dim > 20 {
		return fmt.Errorf("state space too large for exhaustive check: %d bits = %d states", c.Dim, 1<<c.Dim)
	}

	maxState := uint64(1) << c.Dim
	for v := uint64(0); v < maxState; v++ {
		state := tensor.Vector(v)
		if c.Invariant.Contains(state) {
			next := c.Transition.Apply(state)
			if !c.Invariant.Contains(next) {
				return &CounterexampleError{
					Component:  c.Name,
					PreState:   state,
					PostState:  next,
					Invariant:  c.Invariant,
				}
			}
		}
	}
	return nil
}

// CounterexampleError is a structured error from invariant verification.
// It contains the pre-state (inside the invariant), the post-state (outside
// the invariant), and the constraint matrix that was violated.
type CounterexampleError struct {
	Component  string
	PreState   tensor.Vector
	PostState  tensor.Vector
	Invariant  tensor.ConstraintMatrix
}

func (e *CounterexampleError) Error() string {
	return fmt.Sprintf(
		"%s: invariant violated\n  pre-state:  %s (%#x)\n  post-state: %s (%#x)\n  post-state violates invariant constraints",
		e.Component, e.PreState, uint64(e.PreState), e.PostState, uint64(e.PostState),
	)
}

// ComposeParallel composes two components in parallel (no interaction).
// The composed component's state space is the direct sum: state = [stateA | stateB].
// The transition is the block diagonal matrix.
// The invariant is the direct sum: I = I_A ⊕ I_B.
//
// If both components are individually proven, the composition is automatically
// proven — the composition theorem guarantees it.
func ComposeParallel(A, B *Component, label string) *Component {
	dim := A.Dim + B.Dim

	// Compose variables: A's variables keep their bit positions,
	// B's variables shift by A.Dim.
	vars := make([]string, dim)
	copy(vars, A.Variables)
	for i, name := range B.Variables {
		vars[A.Dim+i] = name
	}
	// Prepend component label to disambiguate.
	for i := range vars {
		vars[i] = label + "." + vars[i]
	}

	// Block diagonal transition.
	trans := tensor.BlockDiagonal(A.Transition, B.Transition, A.Dim, B.Dim)

	// Direct sum of invariants: constraints on A + constraints on B.
	C := tensor.NewConstraintMatrix(A.Invariant.Constraints() + B.Invariant.Constraints())
	for i, row := range A.Invariant {
		C[i] = row // bits are already at correct positions for A
	}
	offset := A.Invariant.Constraints()
	for i, row := range B.Invariant {
		C[offset+i] = row << A.Dim // shift B's constraints into B's bit range
	}

	proven := A.Proven && B.Proven
	note := "composition of proven components"
	if !proven {
		note = "composition of unproven components — verify required"
	}

	return &Component{
		Name:       label,
		Role:       ComponentRole("composite." + string(A.Role) + "+" + string(B.Role)),
		Dim:        dim,
		Variables:  vars,
		Transition: trans,
		Invariant:  C,
		Proven:     proven,
		ProofNote:  note,
	}
}

// --- Known patterns ---

// LeasePattern models a mutual-exclusion lease.
//
// State bits:
//   0: held — 1 if this component holds the lease, 0 otherwise
//   1: requested — 1 if a request to acquire is pending
//
// Transition (acquire if requested and not held; release always possible):
//   request:  held=0, requested=1 → held=1, requested=0
//   release:  held=1 → held=0
//   no-op:    otherwise
//
// Invariant: held=1 → requested=0 (can't be both held and requesting).
//   Constraint: held & requested = 0 → row = [1 1] → parity constraint.
func LeasePattern() *Component {
	c := &Component{
		Name:      "lease",
		Role:      RoleLease,
		Dim:       2,
		Variables: []string{"held", "requested"},
	}

	// Transition matrix (2×2 over Z₂):
	// held_next = requested & !held   (acquire)
	// requested_next = !requested & held (release clears request) | external request
	//
	// Simplified: the transition models the acquire/release protocol.
	// T[0] (held'): held' = requested & !held
	// T[1] (req'):  req' = requested & !held (request persists until acquired)
	//
	// In Z₂ matrix form:
	// held'  = 0*held + 1*requested  (but only when held=0, so held'=requested & !held)
	// request' = 0*held + 1*requested
	//
	// Actually for a linear transition over Z₂, we model the protocol as:
	// held' = requested (if !held), 0 (if held) — nonlinear.
	//
	// For the linear approximation (MVP): we use a simpler model.
	// The invariant is the important part. Even with the linear approximation,
	// the LeasePattern serves as the architectural spec. The full nonlinear
	// model is proven via Z3 separately, and the linear model is used for
	// composition compatibility.
	c.Transition = tensor.IdentityMatrix(2) // placeholder — full encoding in P2
	c.Invariant = tensor.NewConstraintMatrix(1)
	c.Invariant[0] = (1 << 0) | (1 << 1) // held & requested must have even parity
	// This constraint allows {00, 11} and forbids {01, 10}.
	// But we want to forbid {11}. We use a DIFFERENT encoding:
	// The actual invariant is: !(held & requested) which means
	// bit 0 AND bit 1 are never both 1.
	//
	// Linear invariant (parity): held XOR requested ∈ {0, ?}. Hmm.
	// For the linear subspace model: {00, 01, 10} is not a subspace over Z₂!
	// {00, 01, 10} is missing 11. Linear subspaces over Z₂ have size 2^k.
	// We need 2 constraints to exclude 11: held=0 OR requested=0. Both are
	// linear (single-bit constraints), but it's a union of subspaces, not one.
	//
	// Correct: use two constraint rows — one for held, one for requested —
	// but that would force BOTH to be 0. Not what we want either.
	//
	// The honest encoding: for non-subspace invariants, we approximate.
	// This is a documented limitation addressed in P2.
	c.Proven = false
	c.ProofNote = "linear approximation — nonlinear invariant (mutual exclusion) requires piecewise encoding; full proof in Z3"
	return c
}

// FencePattern models a fencing token for idempotency.
//
// State bits:
//   0: token_valid — 1 if the token matches, 0 if stale
//
// Transition: token_valid stays valid once set (monotonic).
//
// Invariant: token_valid never transitions from 1 to 0.
//   This is a linear invariant!
func FencePattern() *Component {
	c := &Component{
		Name:      "fence",
		Role:      RoleFence,
		Dim:       1,
		Variables: []string{"token_valid"},
	}

	// Transition: identity (token_valid persists).
	c.Transition = tensor.IdentityMatrix(1)

	// Invariant: token_valid is monotonic. Encoded as identity transition
	// preserving the subspace of ALL states (no constraint).
	// The monotonicity invariant is: if token_valid=1, then token_valid'=1.
	// In Z₂ linear form: token_valid' = token_valid (the transition).
	// This trivially preserves any subspace.
	c.Invariant = tensor.NewConstraintMatrix(0) // all states safe
	c.Proven = true
	c.ProofNote = "trivial — identity transition preserves all subspaces"
	return c
}

// EventCounterPattern models an event counter invariant.
//
// State bits:
//   0-7: received_count (8-bit counter)
//   8-15: processed_count (8-bit counter)
//
// Transition: events arrive (received++) and are processed (processed++).
// This is nonlinear (addition mod 256 is nonlinear over Z₂).
//
// Invariant: processed ≤ received (counter comparison).
//   Also nonlinear.
//
// The linear approximation: model the DIFFERENCE as a separate state variable.
// This is the P2 encoding work.
func EventCounterPattern() *Component {
	c := &Component{
		Name:      "event-counter",
		Role:      RoleCounter,
		Dim:       16,
		Variables: make([]string, 16),
	}
	for i := 0; i < 8; i++ {
		c.Variables[i] = fmt.Sprintf("received_%d", i)
	}
	for i := 0; i < 8; i++ {
		c.Variables[8+i] = fmt.Sprintf("processed_%d", i)
	}

	c.Transition = tensor.IdentityMatrix(16) // placeholder
	c.Proven = false
	c.ProofNote = "counters are nonlinear over Z₂ — requires finite field or Z3 encoding; P2 work item"
	return c
}

// AllPatterns returns all known architecture patterns.
func AllPatterns() map[string]*Component {
	return map[string]*Component{
		"lease":        LeasePattern(),
		"fence":        FencePattern(),
		"event-counter": EventCounterPattern(),
	}
}
