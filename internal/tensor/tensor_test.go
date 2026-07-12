package tensor

import (
	"testing"

	"math/bits"
)

func TestVectorDim(t *testing.T) {
	tests := []struct {
		v    Vector
		want int
	}{
		{0, 0},
		{1, 1},
		{2, 2},    // bit 1 set
		{0xFF, 8}, // bits 0-7 set
	}
	for _, tt := range tests {
		if got := tt.v.Dim(); got != tt.want {
			t.Errorf("Vector(%d).Dim() = %d, want %d", tt.v, got, tt.want)
		}
	}
}

func TestVectorBitOps(t *testing.T) {
	v := Vector(0)
	v = v.SetBit(0, 1)
	if v.Bit(0) != 1 {
		t.Error("bit 0 should be 1")
	}
	v = v.SetBit(3, 1)
	if v.Bit(3) != 1 {
		t.Error("bit 3 should be 1")
	}
	v = v.SetBit(0, 0)
	if v.Bit(0) != 0 {
		t.Error("bit 0 should be 0 after clear")
	}
	if v != Vector(8) { // only bit 3 set
		t.Errorf("expected Vector(8), got %d", v)
	}
}

func TestMatrixApply(t *testing.T) {
	// Identity matrix: should return the same vector.
	I := IdentityMatrix(4)
	v := Vector(5) // bits 0 and 2 set
	if got := I.Apply(v); got != v {
		t.Errorf("Identity·[0101] = %s, want %s", got, v)
	}

	// Zero matrix: should return zero vector.
	Z := NewMatrix(4, 4)
	if got := Z.Apply(v); got != 0 {
		t.Errorf("Zero·[0101] = %s, want [0]", got)
	}

	// Permutation matrix: swaps bits 0 and 2.
	P := NewMatrix(4, 4)
	P[0] = 1 << 2 // output[0] = input[2]
	P[1] = 1 << 1 // output[1] = input[1]
	P[2] = 1 << 0 // output[2] = input[0]
	P[3] = 1 << 3 // output[3] = input[3]

	got := P.Apply(v)
	// v = [0101], meaning bits 0 and 2 set.
	// After swap (0↔2): bits 2 and 0 set = same! Actually let's use a different vector.
	v2 := Vector(1) // only bit 0 set
	got2 := P.Apply(v2)
	if got2.Bit(2) != 1 || got2.Bit(0) != 0 {
		t.Errorf("Perm·[1000] = %s, want bit 2 set only", got2)
	}
	_ = got
}

func TestMatrixApplyParity(t *testing.T) {
	// Matrix where row 0 = bits 0 and 1 (their XOR goes to output bit 0).
	M := NewMatrix(2, 4)
	M[0] = (1 << 0) | (1 << 1) // output[0] = input[0] XOR input[1]
	M[1] = (1 << 2)            // output[1] = input[2]

	// v has bits 0 and 1 set → XOR = 0.
	v := Vector((1 << 0) | (1 << 1))
	got := M.Apply(v)
	if got.Bit(0) != 0 {
		t.Errorf("parity of bits 0,1 when both set = 0 (XOR), got %d", got.Bit(0))
	}
	if got.Bit(1) != 0 {
		t.Errorf("bit 2 not set in input, output[1] should be 0")
	}

	// v has only bit 0 set → XOR = 1.
	v2 := Vector(1 << 0)
	got2 := M.Apply(v2)
	if got2.Bit(0) != 1 {
		t.Errorf("parity of bit 0 alone = 1, got %d", got2.Bit(0))
	}
}

func TestBlockDiagonal(t *testing.T) {
	// A: 2×2 identity, dimA=2.
	A := IdentityMatrix(2)
	// B: 2×2 identity, dimB=2.
	B := IdentityMatrix(2)

	BD := BlockDiagonal(A, B, 2, 2)

	if BD.Rows() != 4 {
		t.Errorf("expected 4 rows, got %d", BD.Rows())
	}

	// The composed vector is [a0 a1 b0 b1] (bits 0-3).
	// Block diagonal identity should return the same vector.
	v := Vector((1 << 0) | (1 << 3)) // a0=1, b1=1
	got := BD.Apply(v)
	if got != v {
		t.Errorf("BlockDiag(I,I)·%s = %s, want %s", v, got, v)
	}
}

func TestKronecker(t *testing.T) {
	// A: 2×2, B: 1×2.
	A := NewMatrix(2, 2)
	A[0] = 1 << 0 // A[0][0] = 1
	A[1] = 1 << 1 // A[1][1] = 1
	B := NewMatrix(1, 2)
	B[0] = (1 << 0) | (1 << 1) // B[0][0] = B[0][1] = 1

	K := Kronecker(A, B, 2, 2)

	// A ⊗ B should be:
	// [ B  0 ]   where 0 is the 1×2 zero matrix.
	// [ 0  B ]
	if K.Rows() != 2 {
		t.Errorf("expected 2 rows (rA*rB = 2*1), got %d", K.Rows())
	}

	// Row 0 should have B in columns 0-1: bits 0,1 set.
	// Row 1 should have B in columns 2-3: bits 2,3 set.
	if K[0] != ((1<<0)|(1<<1)) || K[1] != ((1<<2)|(1<<3)) {
		t.Errorf("Kronecker product:\nrow 0: %s\nrow 1: %s", Vector(K[0]), Vector(K[1]))
	}
}

func TestConstraintMatrix(t *testing.T) {
	// Constraint: bit 1 must be 0. (c = [0 1 0 ...], so c·v = 0 means bit 1 = 0).
	C := NewConstraintMatrix(1)
	C[0] = 1 << 1

	// Vector with bit 1 = 0 → should pass.
	v1 := Vector(1 << 0) // only bit 0 set
	if !C.Contains(v1) {
		t.Error("v1 with bit 1 clear should pass constraint")
	}

	// Vector with bit 1 = 1 → should fail.
	v2 := Vector(1 << 1)
	if C.Contains(v2) {
		t.Error("v2 with bit 1 set should fail constraint")
	}

	// Vector with bit 1 = 1 and others set → should fail.
	v3 := Vector((1 << 0) | (1 << 1))
	if C.Contains(v3) {
		t.Error("v3 with bit 1 set should fail constraint")
	}
}

func TestConstraintMatrixParity(t *testing.T) {
	// Constraint: bit 0 XOR bit 1 = 0 (even parity).
	C := NewConstraintMatrix(1)
	C[0] = (1 << 0) | (1 << 1)

	// {00, 11} pass; {01, 10} fail.
	pass := []Vector{0, (1 << 0) | (1 << 1)}
	fail := []Vector{1 << 0, 1 << 1}

	for _, v := range pass {
		if !C.Contains(v) {
			t.Errorf("%s should pass parity constraint", v)
		}
	}
	for _, v := range fail {
		if C.Contains(v) {
			t.Errorf("%s should fail parity constraint", v)
		}
	}
}

func TestSubspaceDimension(t *testing.T) {
	// Unconstrained 3-bit space: nullspace of 0 constraints = all 8 states.
	C := NewConstraintMatrix(0)
	dim := C.Dimension(3)
	if dim != 3 {
		t.Errorf("unconstrained 3-bit space: dim = %d, want 3", dim)
	}

	// One independent constraint: dimension should be 2.
	C2 := NewConstraintMatrix(1)
	C2[0] = 1 << 0 // bit 0 must be 0
	dim2 := C2.Dimension(3)
	if dim2 != 2 {
		t.Errorf("single constraint: dim = %d, want 2", dim2)
	}
}

// TestInvariantPreservation is the core proof pattern:
// does transition T preserve invariant subspace I?
// I is preserved iff for all v ∈ I, T·v ∈ I.
func TestInvariantPreservation(t *testing.T) {
	// Identity transition on 3-bit space.
	// Invariant: bit 0 = 0.
	T := IdentityMatrix(3)
	C := NewConstraintMatrix(1)
	C[0] = 1 << 0 // v[0] must be 0

	// Enumerate: check T·v ∈ I for all v ∈ I.
	preserved := true
	maxState := uint64(1) << 3
	for v := uint64(0); v < maxState; v++ {
		vec := Vector(v)
		if C.Contains(vec) {
			if !C.Contains(T.Apply(vec)) {
				t.Errorf("invariant violated: T·%s = %s ∉ I", vec, T.Apply(vec))
				preserved = false
			}
		}
	}
	if !preserved {
		t.Error("invariant should be preserved by identity transition")
	}
}

// Benchmark matrix-vector multiply for 16-bit state space (16×16 matrix).
func BenchmarkMatrixApply(b *testing.B) {
	M := NewMatrix(16, 16)
	for i := range M {
		M[i] = uint64(i*7+3) & 0xFFFF
	}
	v := Vector(0xAAAA)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		M.Apply(v)
	}
}

// Ensure popcount parity isn't optimized away in benchmarks.
var _ = bits.OnesCount64
