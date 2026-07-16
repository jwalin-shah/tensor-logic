package patterns

import (
	"strings"
	"testing"

	"tensor-logic/internal/tensor"
)

// --- Individual pattern tests ---

func TestFencePattern(t *testing.T) {
	fence := FencePattern()
	if !fence.Proven {
		t.Error("fence pattern should be proven by construction")
	}
	if fence.Dim != 1 {
		t.Errorf("fence dim = %d, want 1", fence.Dim)
	}
	if err := fence.Verify(); err != nil {
		t.Errorf("fence verification failed: %v", err)
	}
}

func TestStopSignalPattern(t *testing.T) {
	ss := StopSignalPattern()
	if !ss.Proven {
		t.Error("stop-signal pattern should be proven")
	}
	if ss.Dim != 1 {
		t.Errorf("stop-signal dim = %d, want 1", ss.Dim)
	}
	if ss.Variables[0] != "stopped" {
		t.Errorf("stop-signal var[0] = %q, want stopped", ss.Variables[0])
	}
	if err := ss.Verify(); err != nil {
		t.Errorf("stop-signal verification failed: %v", err)
	}
}

func TestMutexPattern(t *testing.T) {
	mutex := MutexPattern()
	if !mutex.Proven {
		t.Error("mutex pattern should be proven")
	}
	if mutex.Dim != 1 {
		t.Errorf("mutex dim = %d, want 1", mutex.Dim)
	}
	if err := mutex.Verify(); err != nil {
		t.Errorf("mutex verification failed: %v", err)
	}
}

func TestLockStepPattern(t *testing.T) {
	ls := LockStepPattern()
	if !ls.Proven {
		t.Error("lockstep pattern should be proven")
	}
	if ls.Dim != 2 {
		t.Errorf("lockstep dim = %d, want 2", ls.Dim)
	}
	if err := ls.Verify(); err != nil {
		t.Errorf("lockstep verification failed: %v", err)
	}

	// Verify the subspace is {00, 11}.
	if !ls.Invariant.Contains(0) {
		t.Error("lockstep invariant should contain state 00")
	}
	if !ls.Invariant.Contains(3) { // 0b11
		t.Error("lockstep invariant should contain state 11")
	}
	if ls.Invariant.Contains(1) { // 0b01
		t.Error("lockstep invariant should NOT contain state 01")
	}
	if ls.Invariant.Contains(2) { // 0b10
		t.Error("lockstep invariant should NOT contain state 10")
	}
}

func TestRoundRobinPattern(t *testing.T) {
	rr := RoundRobinPattern()
	if !rr.Proven {
		t.Error("round-robin pattern should be proven")
	}
	if rr.Dim != 2 {
		t.Errorf("round-robin dim = %d, want 2", rr.Dim)
	}
	if err := rr.Verify(); err != nil {
		t.Errorf("round-robin verification failed: %v", err)
	}

	// Verify swap transition.
	// T·[1,0] (token_a=1, token_b=0) = [0,1] (token_a=0, token_b=1).
	post := rr.Transition.Apply(1) // 0b01: token_a=1, token_b=0
	if post != 2 {                 // 0b10: token_a=0, token_b=1
		t.Errorf("round-robin swap: T·01 = %s, want 10", post)
	}
	// T·[0,1] = [1,0].
	post = rr.Transition.Apply(2) // 0b10
	if post != 1 {                // 0b01
		t.Errorf("round-robin swap: T·10 = %s, want 01", post)
	}
	// T·[1,1] = [1,1].
	post = rr.Transition.Apply(3) // 0b11
	if post != 3 {
		t.Errorf("round-robin swap: T·11 = %s, want 11", post)
	}
}

func TestTogglePairPattern(t *testing.T) {
	tp := TogglePairPattern()
	if !tp.Proven {
		t.Error("toggle-pair pattern should be proven")
	}
	if tp.Dim != 2 {
		t.Errorf("toggle-pair dim = %d, want 2", tp.Dim)
	}
	if err := tp.Verify(); err != nil {
		t.Errorf("toggle-pair verification failed: %v", err)
	}

	// Verify both-flip transition: T·[a,b] = [a⊕b, a⊕b].
	post := tp.Transition.Apply(0) // 00
	if post != 0 {
		t.Errorf("toggle-pair: T·00 = %s, want 00", post)
	}
	post = tp.Transition.Apply(3) // 11
	if post != 0 {                // 1⊕1=0 for both bits
		t.Errorf("toggle-pair: T·11 = %s, want 00", post)
	}
}

func TestUnanimousVotePattern(t *testing.T) {
	uv := UnanimousVotePattern()
	if !uv.Proven {
		t.Error("unanimous-vote pattern should be proven")
	}
	if uv.Dim != 3 {
		t.Errorf("unanimous-vote dim = %d, want 3", uv.Dim)
	}
	if err := uv.Verify(); err != nil {
		t.Errorf("unanimous-vote verification failed: %v", err)
	}

	// Verify subspace: {000, 111}.
	if !uv.Invariant.Contains(0) {
		t.Error("unanimous-vote should contain state 000")
	}
	if !uv.Invariant.Contains(7) { // 0b111
		t.Error("unanimous-vote should contain state 111")
	}
	// Check that dissent states are excluded.
	for _, v := range []tensor.Vector{1, 2, 3, 4, 5, 6} {
		if uv.Invariant.Contains(v) {
			t.Errorf("unanimous-vote should NOT contain state %s", v)
		}
	}
}

func TestLeasePattern(t *testing.T) {
	lease := LeasePattern()
	if !lease.Proven {
		t.Error("lease pattern with linear approximation should be proven (the Z₂ encoding is correct)")
	}
	if lease.Dim != 2 {
		t.Errorf("lease dim = %d, want 2", lease.Dim)
	}
	if err := lease.Verify(); err != nil {
		t.Errorf("lease linear approximation failed verification: %v", err)
	}

	// The linear approximation allows {00, 11}. Confirm.
	if !lease.Invariant.Contains(0) {
		t.Error("lease invariant should contain state 00")
	}
	if !lease.Invariant.Contains(3) {
		t.Error("lease invariant should contain state 11 (linear approximation)")
	}
	// The semantically valid state 01 (held, not requested) is excluded
	// by the linear approximation. This is the documented gap.
	if lease.Invariant.Contains(1) {
		t.Log("lease invariant contains 01 — linear approximation is exact?")
	}

	// Verify the proof note mentions the approximation.
	if !strings.Contains(lease.ProofNote, "linear approximation") {
		t.Error("lease proof note should document linear approximation")
	}
}

// --- Adversarial tests ---

// modifyTransition returns a copy of c with its transition matrix replaced.
// Used to test that broken transitions are caught by Verify.
func modifyTransition(c *Component, newTrans tensor.Matrix) *Component {
	cp := *c // shallow copy
	cp.Transition = newTrans
	return &cp
}

func TestAdversarial_LockStep_BrokenTransition(t *testing.T) {
	ls := LockStepPattern()
	// Identity preserves {00, 11}. Use a projection that breaks it:
	// T = [[1,0],[0,0]] maps 11 → 10 (outside subspace).
	broken := modifyTransition(ls, tensor.Matrix{1 << 0, 0}) // b0'=b0, b1'=0
	// Verify: T·11 = [1, 0] = 01, which violates a⊕b=0.
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken lockstep transition should fail Verify")
	}
	cerr, ok := err.(*CounterexampleError)
	if !ok {
		t.Fatalf("expected CounterexampleError, got %T: %v", err, err)
	}
	if cerr.PreState != 3 { // state 11
		t.Errorf("counterexample pre-state = %s, want 11", cerr.PreState)
	}
	if cerr.PostState != 1 { // state 01
		t.Errorf("counterexample post-state = %s, want 01", cerr.PostState)
	}
	t.Logf("correctly caught: %v", err)
}

func TestAdversarial_RoundRobin_BrokenTransition(t *testing.T) {
	rr := RoundRobinPattern()
	// Valid transition is swap. Replace with a projection that breaks the subspace:
	// T = [[1,0],[0,0]] maps 11 → 10 (outside {00, 11}).
	broken := modifyTransition(rr, tensor.Matrix{1 << 0, 0})
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken round-robin transition should fail Verify")
	}
	cerr, ok := err.(*CounterexampleError)
	if !ok {
		t.Fatalf("expected CounterexampleError, got %T", err)
	}
	if cerr.PreState != 3 {
		t.Errorf("counterexample pre-state = %s, want 11", cerr.PreState)
	}
	t.Logf("correctly caught: %v", err)
}

func TestAdversarial_TogglePair_BrokenTransition(t *testing.T) {
	tp := TogglePairPattern()
	// Valid is both-flip [[1,1],[1,1]]. Replace with identity — but wait,
	// identity ALSO preserves {00,11}. Use a projection instead:
	// T = [[1,0],[0,0]] maps 11→10.
	broken := modifyTransition(tp, tensor.Matrix{1 << 0, 0})
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken toggle-pair transition should fail Verify")
	}
	t.Logf("correctly caught: %v", err)
}

func TestAdversarial_UnanimousVote_BrokenTransition(t *testing.T) {
	uv := UnanimousVotePattern()
	// Identity(3) preserves {000, 111}. Use a transition that clears bit 1:
	// T = identity but with row 1 = [0,0,0]. T·111 → 101 (outside {000,111}).
	broken := modifyTransition(uv, tensor.Matrix{1 << 0, 0, 1 << 2})
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken unanimous-vote transition should fail Verify")
	}
	cerr, ok := err.(*CounterexampleError)
	if !ok {
		t.Fatalf("expected CounterexampleError, got %T", err)
	}
	if cerr.PreState != 7 { // state 111
		t.Errorf("counterexample pre-state = %s, want 111", cerr.PreState)
	}
	// T[0] = bit0, T[1] = 0, T[2] = bit2.
	// T·111: result[0]=1, result[1]=0, result[2]=1 = 101 = 5.
	if cerr.PostState != 5 {
		t.Errorf("counterexample post-state = %s (%#x), want 101 (0x5)", cerr.PostState, uint64(cerr.PostState))
	}
	t.Logf("correctly caught: %v", err)
}

func TestAdversarial_Lease_BrokenTransition(t *testing.T) {
	lease := LeasePattern()
	// Identity preserves {00, 11}. Broken: projection like LockStep.
	broken := modifyTransition(lease, tensor.Matrix{1 << 0, 0})
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken lease transition should fail Verify")
	}
	t.Logf("correctly caught: %v", err)
}

// TestAdversarial_BrokenInvariant_TooPermissive verifies that an invariant
// that is too permissive (all states) would NOT catch a broken transition.
// This confirms that the nontrivial invariants are actually doing work.
func TestAdversarial_BrokenInvariant_TooPermissive(t *testing.T) {
	// Take LockStep with its real transition (identity) but a vacuous invariant.
	ls := LockStepPattern()
	vacuous := *ls
	vacuous.Invariant = tensor.NewConstraintMatrix(0) // all states safe
	vacuous.Transition = tensor.Matrix{1 << 0, 0}     // broken transition

	// With all states safe, the "broken" transition trivially passes.
	err := vacuous.Verify()
	if err != nil {
		t.Errorf("vacuous invariant should not catch any transition, but got: %v", err)
	} else {
		t.Log("confirmed: vacuous invariant does not catch broken transitions")
	}
}

// --- Composition tests ---

func TestComposeParallel(t *testing.T) {
	fence := FencePattern()
	lockstep := LockStepPattern()

	composite := ComposeParallel(fence, lockstep, "test-composite")

	wantDim := fence.Dim + lockstep.Dim
	if composite.Dim != wantDim {
		t.Errorf("composite dim = %d, want %d", composite.Dim, wantDim)
	}

	// Variables should be prefixed.
	if composite.Variables[0] != "test-composite.token_valid" {
		t.Errorf("var[0] = %q, want test-composite.token_valid", composite.Variables[0])
	}
	// lockstep variables start at index fence.Dim=1.
	if composite.Variables[1] != "test-composite.phase_a" {
		t.Errorf("var[1] = %q, want test-composite.phase_a", composite.Variables[1])
	}

	// Both proven → composite should be proven.
	if !composite.Proven {
		t.Error("composite of two proven components should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("proven composite failed verification: %v", err)
	}
}

func TestComposeParallel_ThreeComponents(t *testing.T) {
	fence := FencePattern()
	lockstep := LockStepPattern()
	vote := UnanimousVotePattern()

	// Compose in two steps: (fence ∥ lockstep) ∥ vote.
	tmp := ComposeParallel(fence, lockstep, "tmp")
	composite := ComposeParallel(tmp, vote, "full")

	wantDim := fence.Dim + lockstep.Dim + vote.Dim
	if composite.Dim != wantDim {
		t.Errorf("3-component composite dim = %d, want %d", composite.Dim, wantDim)
	}
	if !composite.Proven {
		t.Error("3-component composite should be proven (all components proven)")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("3-component composite verification failed: %v", err)
	}

	// Check subspace size: fence (1 bit, all states), lockstep (2 bits, dim 1),
	// vote (3 bits, dim 1) → composite has constraint rows from lockstep + vote.
	wantConstraints := lockstep.Invariant.Constraints() + vote.Invariant.Constraints()
	if composite.Invariant.Constraints() != wantConstraints {
		t.Errorf("constraints = %d, want %d", composite.Invariant.Constraints(), wantConstraints)
	}
}

// TestComposeParallel_ApproximateComponent verifies that composing an
// approximately-proven component (Lease, with linear approximation) with
// a proven component still marks the composite as proven. The Proven field
// is purely about Z₂ correctness; the semantic caveat is in ProofNote.
func TestComposeParallel_ApproximateComponent(t *testing.T) {
	fence := FencePattern()
	lease := LeasePattern() // proven via linear approximation

	composite := ComposeParallel(fence, lease, "fence+lease")
	if !composite.Proven {
		t.Error("composite of proven + approximate-proven should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("fence+lease composite verification failed: %v", err)
	}
}

// --- AllPatterns meta-test ---

func TestAllPatterns_Count(t *testing.T) {
	all := AllPatterns()
	if len(all) != 9 {
		t.Errorf("expected 9 patterns, got %d", len(all))
	}
}

func TestAllPatterns_AllProven(t *testing.T) {
	all := AllPatterns()
	for name, c := range all {
		if !c.Proven {
			t.Errorf("pattern %q is not proven", name)
		}
	}
}

func TestAllPatterns_AllVerifyPass(t *testing.T) {
	all := AllPatterns()
	for name, c := range all {
		if err := c.Verify(); err != nil {
			t.Errorf("pattern %q claims proven but fails Verify: %v", name, err)
		}
	}
}

func TestAllPatterns_UniqueNames(t *testing.T) {
	all := AllPatterns()
	seen := make(map[string]bool)
	for name := range all {
		if seen[name] {
			t.Errorf("duplicate pattern name: %q", name)
		}
		seen[name] = true
	}
}

func TestAllPatterns_AllHaveVariables(t *testing.T) {
	all := AllPatterns()
	for name, c := range all {
		if len(c.Variables) != c.Dim {
			t.Errorf("pattern %q: variables count %d != dim %d", name, len(c.Variables), c.Dim)
		}
		for i, v := range c.Variables {
			if v == "" {
				t.Errorf("pattern %q: variable %d is empty", name, i)
			}
		}
	}
}

func TestAllPatterns_AllHaveProofNote(t *testing.T) {
	all := AllPatterns()
	for name, c := range all {
		if c.ProofNote == "" {
			t.Errorf("pattern %q: proof note is empty", name)
		}
	}
}

// --- Counterexample format test ---

func TestCounterexampleError_Format(t *testing.T) {
	err := &CounterexampleError{
		Component: "test-pattern",
		PreState:  3, // 0b11
		PostState: 1, // 0b01
		Invariant: tensor.ConstraintMatrix{0b11},
		Variables: []string{"a", "b"},
	}
	msg := err.Error()
	if !strings.Contains(msg, "test-pattern") {
		t.Error("error message should contain component name")
	}
	if !strings.Contains(msg, "a=1") {
		t.Error("error message should contain variable values")
	}
	if !strings.Contains(msg, "b=1") {
		t.Error("error message should contain all variable values")
	}
}

// --- Transition behavior tests ---

func TestSwapTransition_Properties(t *testing.T) {
	T := tensor.Matrix{1 << 1, 1 << 0} // swap
	// Double swap is identity: T² = I.
	for v := uint64(0); v < 4; v++ {
		once := T.Apply(tensor.Vector(v))
		twice := T.Apply(once)
		if twice != tensor.Vector(v) {
			t.Errorf("T²·%s = %s, want %s", tensor.Vector(v), twice, tensor.Vector(v))
		}
	}
}

func TestBothFlipTransition_Properties(t *testing.T) {
	T := tensor.Matrix{(1 << 0) | (1 << 1), (1 << 0) | (1 << 1)} // both-flip
	// Applying twice: first gives [a⊕b, a⊕b], second gives [0, 0].
	for v := uint64(0); v < 4; v++ {
		once := T.Apply(tensor.Vector(v))
		twice := T.Apply(once)
		if twice != 0 {
			t.Errorf("T²·%s = %s, want 00", tensor.Vector(v), twice)
		}
	}
}

// --- Benchmark for exhaustive verification ---

func BenchmarkVerify_LockStep(b *testing.B) {
	ls := LockStepPattern()
	for b.Loop() {
		ls.Verify()
	}
}

func BenchmarkVerify_UnanimousVote(b *testing.B) {
	uv := UnanimousVotePattern()
	for b.Loop() {
		uv.Verify()
	}
}

func BenchmarkVerify_Fence(b *testing.B) {
	fence := FencePattern()
	for b.Loop() {
		fence.Verify()
	}
}
