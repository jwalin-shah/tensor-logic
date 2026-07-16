// Package prover implements the proof checker for tensor-logic.
// It verifies that a transition matrix preserves an invariant subspace:
// for all v ∈ I, T·v ∈ I.
//
// Exhaustive checking for small state spaces (≤20 bits).
// Randomized checking for larger spaces (later).
// SMT fallback for nonlinear invariants (later).
package prover

import (
	"fmt"

	"tensor-logic/internal/tensor"
)

// Result is the outcome of a proof attempt.
type Result struct {
	Passed           bool
	StatesChecked    uint64
	StatesInSubspace uint64
	Counterexample   *Counterexample
}

// Counterexample captures the exact state that violates an invariant.
type Counterexample struct {
	PreState    tensor.Vector
	PostState   tensor.Vector
	Constraint  tensor.ConstraintMatrix
	ViolatedRow int // which constraint was violated (-1 if unknown)
}

func (c *Counterexample) Error() string {
	if c.ViolatedRow >= 0 {
		return fmt.Sprintf(
			"invariant violation: pre=%s post=%s violates constraint[%d]",
			c.PreState, c.PostState, c.ViolatedRow,
		)
	}
	return fmt.Sprintf(
		"invariant violation: pre=%s post=%s is outside invariant subspace",
		c.PreState, c.PostState,
	)
}

// ProveExhaustive checks the invariant for all 2^dim states.
// Returns a Result with the counterexample if the invariant fails.
func ProveExhaustive(T tensor.Matrix, I tensor.ConstraintMatrix, dim int) Result {
	if dim > 20 {
		return Result{
			Passed: false,
			Counterexample: &Counterexample{
				PreState:    0,
				PostState:   0,
				Constraint:  I,
				ViolatedRow: -1,
			},
		}
	}

	r := Result{}
	maxState := uint64(1) << dim

	for v := uint64(0); v < maxState; v++ {
		r.StatesChecked++
		state := tensor.Vector(v)
		if I.Contains(state) {
			r.StatesInSubspace++
			next := T.Apply(state)
			if !I.Contains(next) {
				// Find which constraint was violated.
				violatedRow := -1
				for i, row := range I {
					dot := uint64(row) & uint64(next)
					parity := dot & 1 // simplified — need popcount
					_ = parity
					if !checkConstraint(row, next) {
						violatedRow = i
						break
					}
				}
				r.Passed = false
				r.Counterexample = &Counterexample{
					PreState:    state,
					PostState:   next,
					Constraint:  I,
					ViolatedRow: violatedRow,
				}
				return r
			}
		}
	}

	r.Passed = true
	return r
}

// checkConstraint returns whether v satisfies constraint row c: c·v = 0 (over Z₂).
func checkConstraint(c uint64, v tensor.Vector) bool {
	dot := c & uint64(v)
	// popcount parity: the sum of bits mod 2.
	parity := uint8(0)
	for dot != 0 {
		parity ^= uint8(dot & 1)
		dot >>= 1
	}
	return parity == 0
}

// String returns a human-readable summary of the proof result.
func (r *Result) String() string {
	if r.Passed {
		return fmt.Sprintf(
			"PROVED: invariant holds for all %d states (%d in subspace)",
			r.StatesChecked, r.StatesInSubspace,
		)
	}
	return fmt.Sprintf(
		"FAILED: %s\n  checked %d states, %d in subspace",
		r.Counterexample.Error(), r.StatesChecked, r.StatesInSubspace,
	)
}
