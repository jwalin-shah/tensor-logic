package patterns

import (
	"fmt"
	"strings"
	"testing"

	"tensor-logic/internal/tensor"
)

// --- Basic property tests ---

func TestDaemonLifecyclePattern_Basic(t *testing.T) {
	dl := DaemonLifecyclePattern()
	if !dl.Proven {
		t.Error("daemon-lifecycle pattern should be proven")
	}
	if dl.Dim != 3 {
		t.Errorf("daemon-lifecycle dim = %d, want 3", dl.Dim)
	}
	if len(dl.Variables) != 3 {
		t.Errorf("daemon-lifecycle variables = %d, want 3", len(dl.Variables))
	}
	if dl.Variables[0] != "lockdir_held" {
		t.Errorf("var[0] = %q, want lockdir_held", dl.Variables[0])
	}
	if dl.Variables[1] != "daemon_running" {
		t.Errorf("var[1] = %q, want daemon_running", dl.Variables[1])
	}
	if dl.Variables[2] != "health_check_due" {
		t.Errorf("var[2] = %q, want health_check_due", dl.Variables[2])
	}
	if dl.Role != RoleDaemon {
		t.Errorf("role = %q, want daemon", string(dl.Role))
	}
}

func TestDaemonLifecyclePattern_Verify(t *testing.T) {
	dl := DaemonLifecyclePattern()
	if err := dl.Verify(); err != nil {
		t.Errorf("daemon-lifecycle verification failed: %v", err)
	}
}

// --- Invariant subspace tests ---

func TestDaemonLifecycle_InvariantSubspace(t *testing.T) {
	dl := DaemonLifecyclePattern()
	I := dl.Invariant

	// Linear subspace: daemon_running=0 (bit 1 = 0).
	// States in subspace: {0, 1, 4, 5}
	// Binary: {000, 100, 001, 101}
	allowed := []tensor.Vector{0, 1, 4, 5}
	for _, v := range allowed {
		if !I.Contains(v) {
			t.Errorf("invariant should contain state %d (%s)", v, stateShort(v))
		}
	}

	// States excluded from linear subspace:
	//   {2, 6} — daemon_running=1, lockdir_held=0 (semantically invalid)
	//   {3, 7} — daemon_running=1, lockdir_held=1 (semantically valid, excluded by
	//             linear approximation since daemon_running=1)
	disallowed := []tensor.Vector{2, 3, 6, 7}
	for _, v := range disallowed {
		if I.Contains(v) {
			t.Errorf("invariant should NOT contain state %d (%s)", v, stateShort(v))
		}
	}
}

// stateShort returns a human-readable 3-bit string for a Vector.
func stateShort(v tensor.Vector) string {
	bits := uint64(v)
	b2 := (bits >> 2) & 1
	b1 := (bits >> 1) & 1
	b0 := bits & 1
	return fmt.Sprintf("{%d%d%d}", b2, b1, b0)
}

// --- Semantic transition tests ---
//
// Each semantic transition is tested explicitly by pre/post state pair.
// These transitions form a daemon lifecycle cycle:
//
//           000 ──acquire_lock──→ 100 ──validate_imports──→ 100
//            ↑                                                |
//            |                                          start_daemon
//       clean_stale                                           |
//           000  ←──── daemon_crash ────── 110 ←─────────────┘
//
// Bit mapping: bit 0 = lockdir_held, bit 1 = daemon_running, bit 2 = health_check_due
// State values: lock=1 (1<<0), daemon=2 (1<<1), health=4 (1<<2)
//
// Values for common states:
//   000 = 0       100 = 1       010 = 2       110 = 3
//   001 = 4       101 = 5       011 = 6       111 = 7

const (
	lockOnly   = 1 << 0              // 100 (binary) = lockdir_held=1
	daemonOnly = 1 << 1              // 010 (binary) = daemon_running=1
	lockDaemon = (1 << 0) | (1 << 1) // 110 (binary) = 3
)

func acquireLock() (pre tensor.Vector, post tensor.Vector) {
	pre = 0         // 000
	post = lockOnly // 100
	return
}

func validateImports() (pre tensor.Vector, post tensor.Vector) {
	pre = lockOnly  // 100
	post = lockOnly // 100 (idempotent)
	return
}

func startDaemon() (pre tensor.Vector, post tensor.Vector) {
	pre = lockOnly    // 100 (lock held, daemon stopped)
	post = lockDaemon // 110 (lock held, daemon running)
	return
}

func daemonCrash() (pre tensor.Vector, post tensor.Vector) {
	pre = lockDaemon // 110 (lock held, daemon running)
	post = lockOnly  // 100 (lock held, daemon stopped)
	return
}

func cleanStale() (pre tensor.Vector, post tensor.Vector) {
	pre = lockOnly // 100 (stale lock)
	post = 0       // 000 (clean)
	return
}

// realI returns the 6 semantically valid states of the real invariant:
// I = {000, 100, 110, 001, 101, 111} = {0, 1, 3, 4, 5, 7}
func realI() []tensor.Vector {
	return []tensor.Vector{0, 1, 3, 4, 5, 7}
}

func inSet(v tensor.Vector, set []tensor.Vector) bool {
	for _, x := range set {
		if v == x {
			return true
		}
	}
	return false
}

func TestDaemonLifecycle_AcquireLock(t *testing.T) {
	pre, post := acquireLock()
	I := realI()

	if !inSet(pre, I) {
		t.Errorf("acquire_lock: pre-state %d %s not in I", pre, stateShort(pre))
	}
	if !inSet(post, I) {
		t.Errorf("acquire_lock: post-state %d %s not in I", post, stateShort(post))
	}
	if pre != 0 {
		t.Errorf("acquire_lock: pre-state should be 000 (0), got %d", pre)
	}
	if post != lockOnly {
		t.Errorf("acquire_lock: post-state should be 100 (%d), got %d", lockOnly, post)
	}
}

func TestDaemonLifecycle_ValidateImports(t *testing.T) {
	pre, post := validateImports()
	I := realI()

	if !inSet(pre, I) {
		t.Errorf("validate_imports: pre-state %d %s not in I", pre, stateShort(pre))
	}
	if !inSet(post, I) {
		t.Errorf("validate_imports: post-state %d %s not in I", post, stateShort(post))
	}
	if pre != post {
		t.Errorf("validate_imports: expected idempotent (%d == %d)", pre, post)
	}
}

func TestDaemonLifecycle_StartDaemon(t *testing.T) {
	pre, post := startDaemon()
	I := realI()

	if !inSet(pre, I) {
		t.Errorf("start_daemon: pre-state %d %s not in I", pre, stateShort(pre))
	}
	if !inSet(post, I) {
		t.Errorf("start_daemon: post-state %d %s not in I", post, stateShort(post))
	}
	if pre != lockOnly || post != lockDaemon {
		t.Errorf("start_daemon: expected 100 (%d) → 110 (%d), got %d → %d", lockOnly, lockDaemon, pre, post)
	}
}

func TestDaemonLifecycle_DaemonCrash(t *testing.T) {
	pre, post := daemonCrash()
	I := realI()

	if !inSet(pre, I) {
		t.Errorf("daemon_crash: pre-state %d %s not in I", pre, stateShort(pre))
	}
	if !inSet(post, I) {
		t.Errorf("daemon_crash: post-state %d %s not in I", post, stateShort(post))
	}
	if pre != lockDaemon || post != lockOnly {
		t.Errorf("daemon_crash: expected 110 (%d) → 100 (%d), got %d → %d", lockDaemon, lockOnly, pre, post)
	}
}

func TestDaemonLifecycle_CleanStale(t *testing.T) {
	pre, post := cleanStale()
	I := realI()

	if !inSet(pre, I) {
		t.Errorf("clean_stale: pre-state %d %s not in I", pre, stateShort(pre))
	}
	if !inSet(post, I) {
		t.Errorf("clean_stale: post-state %d %s not in I", post, stateShort(post))
	}
	if pre != lockOnly || post != 0 {
		t.Errorf("clean_stale: expected 100 (%d) → 000 (0), got %d → %d", lockOnly, pre, post)
	}
}

// --- Exhaustive semantic invariants check ---

func TestDaemonLifecycle_AllFiveTransitionsPreserveI(t *testing.T) {
	I := realI()

	transitions := []struct {
		name string
		fn   func() (tensor.Vector, tensor.Vector)
	}{
		{"acquire_lock", acquireLock},
		{"validate_imports", validateImports},
		{"start_daemon", startDaemon},
		{"daemon_crash", daemonCrash},
		{"clean_stale", cleanStale},
	}

	for _, tr := range transitions {
		pre, post := tr.fn()

		if !inSet(pre, I) {
			t.Errorf("%s: pre-state %d %s must be in I", tr.name, pre, stateShort(pre))
			continue
		}
		if !inSet(post, I) {
			t.Errorf("%s: post-state %d %s NOT in I (invariant violated)", tr.name, post, stateShort(post))
		}
	}
}

// --- Identity transition preserves linear approximation ---

func TestDaemonLifecycle_IdentityPreservesLinearApproximation(t *testing.T) {
	dl := DaemonLifecyclePattern()
	if err := dl.Verify(); err != nil {
		t.Errorf("identity transition should preserve the linear approximation: %v", err)
	}
}

// --- Start_daemon transition as matrix ---
//
// The start_daemon action sets bit 1 (daemon_running) to match bit 0
// (lockdir_held). This is a linear transformation over Z₂:
//   bit_0' = bit_0
//   bit_1' = bit_0
//   bit_2' = bit_2
//
// Matrix T = {1, 1, 4} (row 0 = bit0, row 1 = bit0, row 2 = bit2).
// This matrix preserves the real invariant I = {0, 1, 3, 4, 5, 7}
// but NOT the linear approximation {0, 1, 4, 5} (since 1→3 has bit1=1).

func TestDaemonLifecycle_StartDaemonAsMatrix(t *testing.T) {
	// T: bit1' = bit0, others unchanged (bit0' = bit0, bit2' = bit2).
	T := tensor.Matrix{1 << 0, 1 << 0, 1 << 2} // {1, 1, 4}

	// Expected mapping on states in I:
	//   T·000 = 000
	//   T·100 = 110 (start_daemon: sets bit1 from bit0)
	//   T·110 = 110 (already running — stays running)
	//   T·001 = 001 (bit0=0, bit1 stays 0, bit2 stays 1)
	//   T·101 = 111 (sets bit1 from bit0=1)
	//   T·111 = 111 (already running — stays running)
	expected := map[tensor.Vector]tensor.Vector{
		0: 0,
		1: 3, // 100 → 110
		3: 3, // 110 → 110
		4: 4, // 001 → 001
		5: 7, // 101 → 111
		7: 7, // 111 → 111
	}

	I := realI()
	for _, v := range I {
		got := T.Apply(v)
		want, ok := expected[v]
		if ok && got != want {
			t.Errorf("start_daemon T·%d (%s) = %d (%s), want %d (%s)",
				v, stateShort(v), got, stateShort(got), want, stateShort(want))
		}
		if !inSet(got, I) {
			t.Errorf("start_daemon T·%d (%s) = %d (%s) NOT in real invariant I",
				v, stateShort(v), got, stateShort(got))
		}
	}
}

// --- Daemon_crash transition as matrix ---
//
// daemon_crash clears bit 1 (daemon stops) while preserving bits 0 and 2.
// Matrix T = {1, 0, 4}: identity on bits 0 and 2, zero on bit 1.

func TestDaemonLifecycle_DaemonCrashAsMatrix(t *testing.T) {
	// T: bit1' = 0, others unchanged.
	T := tensor.Matrix{1 << 0, 0, 1 << 2} // {1, 0, 4}

	// Expected mapping on states in I:
	//   T·000 = 000
	//   T·100 = 100 (no daemon to stop)
	//   T·110 = 100 (daemon_crash: clears bit 1)
	//   T·001 = 001 (no daemon to stop)
	//   T·101 = 101 (no daemon to stop)
	//   T·111 = 101 (clears bit 1)
	expected := map[tensor.Vector]tensor.Vector{
		0: 0,
		1: 1, // 100 → 100
		3: 1, // 110 → 100
		4: 4, // 001 → 001
		5: 5, // 101 → 101
		7: 5, // 111 → 101
	}

	I := realI()
	for _, v := range I {
		got := T.Apply(v)
		want, ok := expected[v]
		if ok && got != want {
			t.Errorf("daemon_crash T·%d (%s) = %d (%s), want %d (%s)",
				v, stateShort(v), got, stateShort(got), want, stateShort(want))
		}
		if !inSet(got, I) {
			t.Errorf("daemon_crash T·%d (%s) = %d (%s) NOT in real invariant I",
				v, stateShort(v), got, stateShort(got))
		}
	}
}

// --- Proof note tests ---

func TestDaemonLifecycle_ProofNote(t *testing.T) {
	dl := DaemonLifecyclePattern()
	if dl.ProofNote == "" {
		t.Error("daemon-lifecycle proof note should be non-empty")
	}
	if !strings.Contains(dl.ProofNote, "linear approximation") {
		t.Error("daemon-lifecycle proof note should mention linear approximation")
	}
	if !strings.Contains(dl.ProofNote, "gap") {
		t.Error("daemon-lifecycle proof note should mention the gap")
	}
}

// --- Adversarial tests ---

func TestAdversarial_DaemonLifecycle_BrokenTransition(t *testing.T) {
	dl := DaemonLifecyclePattern()

	// Broken: use the start_daemon matrix (bit1' = bit0). This breaks the
	// linear approximation {0, 1, 4, 5} because T·1 = 3, which has bit1=1.
	broken := modifyTransition(dl, tensor.Matrix{1 << 0, 1 << 0, 1 << 2})
	err := broken.Verify()
	if err == nil {
		t.Fatal("adversarial: broken daemon-lifecycle transition should fail Verify for the linear approximation")
	}
	cerr, ok := err.(*CounterexampleError)
	if !ok {
		t.Fatalf("expected CounterexampleError, got %T: %v", err, err)
	}
	if cerr.PreState == cerr.PostState {
		t.Errorf("counterexample should have different pre/post states")
	}
	t.Logf("correctly caught: %v", err)
}

func TestAdversarial_DaemonLifecycle_ZeroTransition(t *testing.T) {
	// The zero matrix maps all states to 000. Since 000 is in the linear
	// approximation, the zero matrix trivially preserves the subspace.
	dl := DaemonLifecyclePattern()
	zero := modifyTransition(dl, tensor.Matrix{0, 0, 0})
	err := zero.Verify()
	if err != nil {
		t.Errorf("zero matrix should trivially preserve the linear approximation: %v", err)
	}
}

// --- Composition tests ---

func TestComposeParallel_DaemonLifecycle_Fence(t *testing.T) {
	dl := DaemonLifecyclePattern()
	fence := FencePattern()

	composite := ComposeParallel(dl, fence, "daemon+fence")

	wantDim := dl.Dim + fence.Dim
	if composite.Dim != wantDim {
		t.Errorf("composite dim = %d, want %d", composite.Dim, wantDim)
	}

	if composite.Variables[0] != "daemon+fence.lockdir_held" {
		t.Errorf("var[0] = %q, want daemon+fence.lockdir_held", composite.Variables[0])
	}
	if composite.Variables[3] != "daemon+fence.token_valid" {
		t.Errorf("var[3] = %q, want daemon+fence.token_valid", composite.Variables[3])
	}

	if !composite.Proven {
		t.Error("composite of two proven components should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("proven composite failed verification: %v", err)
	}
}

func TestComposeParallel_DaemonLifecycle_StopSignal(t *testing.T) {
	dl := DaemonLifecyclePattern()
	ss := StopSignalPattern()

	composite := ComposeParallel(dl, ss, "daemon+stop")

	wantDim := dl.Dim + ss.Dim
	if composite.Dim != wantDim {
		t.Errorf("composite dim = %d, want %d", composite.Dim, wantDim)
	}
	if !composite.Proven {
		t.Error("composite should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("composite verification failed: %v", err)
	}
}

func TestComposeParallel_DaemonLifecycle_ThreeComponents(t *testing.T) {
	dl := DaemonLifecyclePattern()
	fence := FencePattern()
	stop := StopSignalPattern()

	tmp := ComposeParallel(dl, fence, "dl+fence")
	composite := ComposeParallel(tmp, stop, "dl+fence+stop")

	wantDim := dl.Dim + fence.Dim + stop.Dim
	if composite.Dim != wantDim {
		t.Errorf("3-component composite dim = %d, want %d", composite.Dim, wantDim)
	}
	if !composite.Proven {
		t.Error("3-component composite should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("3-component composite verification failed: %v", err)
	}

	wantConstraints := dl.Invariant.Constraints() + fence.Invariant.Constraints() + stop.Invariant.Constraints()
	if composite.Invariant.Constraints() != wantConstraints {
		t.Errorf("constraints = %d, want %d", composite.Invariant.Constraints(), wantConstraints)
	}
}

// --- State space enumeration test ---

func TestDaemonLifecycle_ExhaustiveInvalidStateCheck(t *testing.T) {
	for v := uint64(0); v < 8; v++ {
		lockHeld := (v>>0)&1 == 1      // bit 0
		daemonRunning := (v>>1)&1 == 1 // bit 1

		isInvalid := !lockHeld && daemonRunning // bit_0=0 ∧ bit_1=1

		// Invalid states: {2, 6} = {010, 011} in bit terms
		//   2 = b010 = lock=0, daemon=1, health=0
		//   6 = b110 = lock=0, daemon=1, health=1
		wantInvalid := (v == 2 || v == 6)

		if isInvalid != wantInvalid {
			t.Errorf("state %d (%s): isInvalid=%v, want %v (lockHeld=%v, daemonRunning=%v)",
				v, stateShort(tensor.Vector(v)), isInvalid, wantInvalid, lockHeld, daemonRunning)
		}
	}
}

// --- Benchmark ---

func BenchmarkVerify_DaemonLifecycle(b *testing.B) {
	dl := DaemonLifecyclePattern()
	for b.Loop() {
		dl.Verify()
	}
}
