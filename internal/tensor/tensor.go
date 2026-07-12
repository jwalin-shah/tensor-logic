// Package tensor implements the core tensor logic primitives over Z₂
// (the field of two elements). A state vector is a bitstring (uint64).
// A transition is a linear transformation (matrix over Z₂, []uint64).
// An invariant is a subspace defined as the nullspace of a constraint matrix.
//
// Operations over Z₂: addition is XOR, multiplication is AND.
// Matrix-vector multiplication: (M·v)[i] = parity of (M[i] & v).
package tensor

import (
	"fmt"
	"math/bits"
)

// Vector is a state vector over Z₂. Bits 0..n-1 represent state variables.
// The zero value is the zero vector. Operations are bitwise.
type Vector uint64

// Matrix is a linear transformation over Z₂ with m rows and n columns.
// Row i is a bitmask; bit j is set iff M[i][j] = 1.
// Matrix applications use Z₂ arithmetic: XOR for addition, AND for multiplication.
type Matrix []uint64

// ConstraintMatrix defines an invariant subspace as its nullspace.
// A state vector v satisfies the constraints iff C·v = 0 (over Z₂).
// In geometric terms: I = ker(C), the subspace of all vectors orthogonal
// to every row of C.
type ConstraintMatrix []uint64

// --- Vector operations ---

// Dim returns the maximum bit position actively used in this vector.
// Returns 0 for the zero vector.
func (v Vector) Dim() int {
	if v == 0 {
		return 0
	}
	return 64 - bits.LeadingZeros64(uint64(v))
}

// Bit returns the value of the j-th bit (0 or 1).
func (v Vector) Bit(j int) uint8 {
	return uint8((uint64(v) >> j) & 1)
}

// SetBit returns a new Vector with bit j set to b.
func (v Vector) SetBit(j int, b uint8) Vector {
	if b == 1 {
		return v | (1 << j)
	}
	return v &^ (1 << j)
}

// String formats the vector as a bitstring, least significant bit first.
func (v Vector) String() string {
	if v == 0 {
		return "[0]"
	}
	d := v.Dim()
	buf := make([]byte, 0, d*2+2)
	buf = append(buf, '[')
	for i := 0; i < d; i++ {
		if i > 0 {
			buf = append(buf, ' ')
		}
		buf = append(buf, '0'+v.Bit(i))
	}
	buf = append(buf, ']')
	return string(buf)
}

// --- Matrix operations ---

// NewMatrix creates an m×n zero matrix over Z₂.
func NewMatrix(m, n int) Matrix {
	// ponytail: n is documented; rows are bitmasks of width n.
	_ = n
	return make(Matrix, m)
}

// IdentityMatrix creates an n×n identity matrix over Z₂.
func IdentityMatrix(n int) Matrix {
	m := make(Matrix, n)
	for i := 0; i < n; i++ {
		m[i] = 1 << i
	}
	return m
}

// Rows returns the number of rows.
func (m Matrix) Rows() int { return len(m) }

// Apply computes M·v over Z₂. Result[i] = parity of (M[i] & v).
func (m Matrix) Apply(v Vector) Vector {
	var result Vector
	for i, row := range m {
		dot := uint64(row) & uint64(v)
		parity := bits.OnesCount64(dot) & 1
		if parity == 1 {
			result |= (1 << i)
		}
	}
	return result
}

// String formats the matrix row by row.
func (m Matrix) String() string {
	if len(m) == 0 {
		return "[]"
	}
	s := ""
	for i, row := range m {
		if i > 0 {
			s += "\n"
		}
		s += fmt.Sprintf("row %d: %s", i, Vector(row))
	}
	return s
}

// BlockDiagonal returns the block diagonal composition [A 0; 0 B].
// The composed matrix operates on the direct sum of the state spaces:
// state = [state_A | state_B], where state_A occupies bits 0..dimA-1
// and state_B occupies bits dimA..dimA+dimB-1.
func BlockDiagonal(A, B Matrix, dimA, dimB int) Matrix {
	m := NewMatrix(A.Rows()+B.Rows(), dimA+dimB)

	// A block: rows 0..rA-1, columns 0..dimA-1 (no shift needed).
	for i := 0; i < A.Rows(); i++ {
		m[i] = A[i]
	}

	// B block: rows rA..rA+rB-1, columns dimA..dimA+dimB-1.
	for i := 0; i < B.Rows(); i++ {
		m[A.Rows()+i] = B[i] << dimA
	}

	return m
}

// Kronecker returns the Kronecker (tensor) product A ⊗ B.
// If A is rA × cA and B is rB × cB, the result is (rA·rB) × (cA·cB).
// The Kronecker product models interacting components where the combined
// state is the tensor product of individual state spaces.
//
// For a Z₂ matrix, each element A[i][j] is either 0 or 1, so the product
// block is either the zero matrix or B itself.
func Kronecker(A, B Matrix, cA, cB int) Matrix {
	rA, rB := A.Rows(), B.Rows()
	result := NewMatrix(rA*rB, cA*cB)

	for i := 0; i < rA; i++ {
		for j := 0; j < cA; j++ {
			if (A[i]>>j)&1 == 1 {
				// Block (i,j) = B, placed at row i*rB, col j*cB.
				for bi := 0; bi < rB; bi++ {
					result[i*rB+bi] |= B[bi] << (j * cB)
				}
			}
		}
	}

	return result
}

// --- ConstraintMatrix (invariant subspace) operations ---

// NewConstraintMatrix creates an empty constraint matrix with c constraints.
// Each constraint is a row: v ∈ I iff C[i]·v = 0 for all i.
func NewConstraintMatrix(c int) ConstraintMatrix {
	return make(ConstraintMatrix, c)
}

// Constraints returns the number of constraints.
func (c ConstraintMatrix) Constraints() int { return len(c) }

// Contains checks whether v satisfies all constraints: C·v = 0.
func (c ConstraintMatrix) Contains(v Vector) bool {
	for _, row := range c {
		dot := uint64(row) & uint64(v)
		if bits.OnesCount64(dot)&1 == 1 {
			return false
		}
	}
	return true
}

// Dimension returns the dimension of the subspace (nullity of C) for a
// state space of the given number of bits. For Z₂, a k-dimensional subspace
// has exactly 2^k elements, so dim = log2(|I|).
//
// Only call this for small dim (≤20). For larger spaces, use Gaussian
// elimination over Z₂ instead of enumeration.
func (c ConstraintMatrix) Dimension(dim int) int {
	maxState := uint64(1) << dim
	count := uint64(0)
	for v := uint64(0); v < maxState; v++ {
		if c.Contains(Vector(v)) {
			count++
		}
	}
	if count == 0 {
		return 0
	}
	// count = 2^k, find k.
	k := 0
	for (uint64(1) << k) < count {
		k++
	}
	return k
}

// String formats the constraint matrix as equations.
func (c ConstraintMatrix) String() string {
	s := ""
	for i, row := range c {
		if i > 0 {
			s += "\n"
		}
		s += fmt.Sprintf("  c%d·v = 0, where c%d = %s", i, i, Vector(row))
	}
	return s
}
