// Package compose implements the composition engine.
// Given two proven components (A, B), it constructs the composed component
// and verifies that the invariant is preserved without re-proving.
//
// The composition theorem: if T_A preserves I_A and T_B preserves I_B,
// then T_A ⊗ T_B (block diagonal for parallel, Kronecker for interacting)
// preserves I_A ⊕ I_B (direct sum of invariants).
package compose

import (
	"fmt"

	"tensor-logic/internal/tensor"
)

// Parallel composes two state spaces in parallel (no interaction).
//   - State: direct sum S = S_A ⊕ S_B
//   - Transition: block diagonal T = [T_A 0; 0 T_B]
//   - Invariant: direct sum I = I_A ⊕ I_B
//
// The composition theorem guarantees that if both components are individually
// proven, the composed system is proven automatically.
func Parallel(
	T_A, T_B tensor.Matrix,
	I_A, I_B tensor.ConstraintMatrix,
	dimA, dimB int,
) (tensor.Matrix, tensor.ConstraintMatrix, int) {
	T := tensor.BlockDiagonal(T_A, T_B, dimA, dimB)

	C := tensor.NewConstraintMatrix(I_A.Constraints() + I_B.Constraints())
	copy(C, I_A)
	for i, row := range I_B {
		C[I_A.Constraints()+i] = row << dimA
	}

	return T, C, dimA + dimB
}

// Interactive composes two state spaces with interaction (Kronecker product).
//   - State: tensor product S = S_A ⊗ S_B
//   - Transition: Kronecker product T = T_A ⊗ T_B
//   - Invariant: more complex — depends on interaction pattern
//
// The invariant for interacting components is not simply the tensor product
// of individual invariants. The caller must supply the composed invariant.
// This function constructs the transition only.
func Interactive(
	T_A, T_B tensor.Matrix,
	dimA, dimB int,
	cA, cB int,
) tensor.Matrix {
	return tensor.Kronecker(T_A, T_B, cA, cB)
}

// VerifyComposition checks that the composition theorem holds:
// given proven A and B, the composed system preserves the composed invariant.
// This is a sanity check, not a full proof — the theorem guarantees it.
func VerifyComposition(
	T tensor.Matrix, I tensor.ConstraintMatrix, dim int,
) error {
	maxState := uint64(1) << dim
	if maxState > 1<<20 {
		return fmt.Errorf("composed state space too large: 2^%d states", dim)
	}

	for v := uint64(0); v < maxState; v++ {
		state := tensor.Vector(v)
		if I.Contains(state) {
			next := T.Apply(state)
			if !I.Contains(next) {
				return fmt.Errorf(
					"composition theorem FAILED: state %s → %s violates composed invariant",
					state, next,
				)
			}
		}
	}
	return nil
}
