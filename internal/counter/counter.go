// Package counter extracts counterexamples from failed proofs.
// Given a failed proof — T·v ∉ I — it computes:
//  1. Which constraint was violated
//  2. The exact pre-state and post-state
//  3. A human-readable explanation
//
// The counterexample is the error message. It tells the developer
// (or AI worker) exactly what went wrong and what to fix.
package counter

import (
	"fmt"
	"strings"

	"tensor-logic/internal/tensor"
)

// Report is a structured counterexample report.
type Report struct {
	Component    string
	PreState     tensor.Vector
	PostState    tensor.Vector
	Dim          int
	Variables    []string
	Invariant    tensor.ConstraintMatrix
	ViolatedRows []int
}

// Extract builds a Report from a failed proof. It finds every constraint
// row that the post-state violates and annotates them with variable names.
func Extract(
	component string,
	pre, post tensor.Vector,
	C tensor.ConstraintMatrix,
	dim int,
	vars []string,
) *Report {
	var violated []int
	for i, row := range C {
		dot := uint64(row) & uint64(post)
		parity := parity(dot)
		if parity == 1 {
			violated = append(violated, i)
		}
	}

	return &Report{
		Component:    component,
		PreState:     pre,
		PostState:    post,
		Dim:          dim,
		Variables:    vars,
		Invariant:    C,
		ViolatedRows: violated,
	}
}

func parity(v uint64) int {
	p := 0
	for v != 0 {
		p ^= int(v & 1)
		v >>= 1
	}
	return p
}

// String returns a human-readable counterexample explanation.
func (r *Report) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "=== COUNTEREXAMPLE: %s ===\n\n", r.Component)

	b.WriteString("Pre-state (inside invariant):\n")
	r.writeState(&b, r.PreState)
	b.WriteString("\n")

	b.WriteString("Post-state (outside invariant):\n")
	r.writeState(&b, r.PostState)
	b.WriteString("\n")

	b.WriteString("Violated constraints:\n")
	for _, rowIdx := range r.ViolatedRows {
		row := r.Invariant[rowIdx]
		fmt.Fprintf(&b, "  constraint[%d]: ", rowIdx)
		r.writeConstraint(&b, row)
	}
	b.WriteString("\n")

	b.WriteString("Fix suggestion:\n")
	r.writeFixSuggestion(&b)

	return b.String()
}

func (r *Report) writeState(b *strings.Builder, v tensor.Vector) {
	for i := 0; i < r.Dim; i++ {
		name := fmt.Sprintf("bit_%d", i)
		if i < len(r.Variables) && r.Variables[i] != "" {
			name = r.Variables[i]
		}
		fmt.Fprintf(b, "  %-20s = %d\n", name, v.Bit(i))
	}
}

func (r *Report) writeConstraint(b *strings.Builder, row uint64) {
	terms := []string{}
	for i := 0; i < r.Dim; i++ {
		if (row>>i)&1 == 1 {
			name := fmt.Sprintf("bit_%d", i)
			if i < len(r.Variables) && r.Variables[i] != "" {
				name = r.Variables[i]
			}
			terms = append(terms, name)
		}
	}
	joined := strings.Join(terms, " ⊕ ")
	if joined == "" {
		joined = "0"
	}
	fmt.Fprintf(b, "%s = 0\n", joined)
}

func (r *Report) writeFixSuggestion(b *strings.Builder) {
	if len(r.ViolatedRows) == 0 {
		b.WriteString("  No constraints violated — this is unexpected. Check the invariant encoding.\n")
		return
	}

	for _, rowIdx := range r.ViolatedRows {
		row := r.Invariant[rowIdx]

		// Find which pre-state bits differ from post-state bits.
		changed := uint64(r.PreState) ^ uint64(r.PostState)

		// Bits that changed AND are in the constraint: those are the suspect bits.
		suspectBits := changed & row

		if suspectBits == 0 {
			fmt.Fprintf(b, "  constraint[%d]: violated but no bit in the constraint changed\n", rowIdx)
			b.WriteString("    → suspect: the transition matrix is incorrect for this state\n")
			continue
		}

		b.WriteString("  The following bits changed AND appear in the violated constraint:\n")
		for i := 0; i < r.Dim; i++ {
			if (suspectBits>>i)&1 == 1 {
				name := fmt.Sprintf("bit_%d", i)
				if i < len(r.Variables) && r.Variables[i] != "" {
					name = r.Variables[i]
				}
				from := r.PreState.Bit(i)
				to := r.PostState.Bit(i)
				fmt.Fprintf(b, "    %s: %d → %d\n", name, from, to)
			}
		}
		b.WriteString("    → fix: update the transition matrix so the constraint holds for this state change\n")
	}
}
