// Pattern 9: DaemonLifecycle — daemon process lifecycle with lockdir guard
//
// State bits:
//   0: lockdir_held — 1 if the lock directory (pidfile) is acquired
//   1: daemon_running — 1 if the daemon process is actively running
//   2: health_check_due — 1 when a periodic health check is due
//
// Total: 3 bits = 8 states
//
// Transitions (semantic operations, verified in tests):
//   acquire_lock:    (0,0,0) → (1,0,0)  — acquire the pidfile lock
//   validate_imports:(1,0,0) → (1,0,0)  — idempotent readiness check
//   start_daemon:    (1,0,0) → (1,1,0)  — fork/exec the daemon process
//   daemon_crash:    (1,1,0) → (1,0,0)  — daemon exits; lockdir is stale
//   clean_stale:     (1,0,0) → (0,0,0)  — health check recovers stale lock
//
// Invariant (semantic): "not (lockdir_held=0 ∧ daemon_running=1)"
//   A running daemon must hold the lock. The 6 valid states are:
//   I = {000, 001, 100, 101, 110, 111}. States {010, 011} (daemon
//   running without lock) are forbidden.
//
// Linear approximation over Z₂:
//   The real invariant I has 6 states — not a power of 2, therefore not
//   a valid Z₂ subspace. The largest linear subspace contained in I is
//   the set where daemon_running = 0 (bit 1 = 0), giving the 4-state
//   subspace {000, 001, 100, 101}.
//
//   Constraint: daemon_running = 0  (mask 0b010 = 2)
//
//   Gap: states {110, 111} (lock held, daemon running) are semantically
//   valid but excluded from the linear subspace. All 5 semantic transitions
//   map I → I; the gap is purely a Z₂ linearity artifact.
//
// Transition: identity. The daemon state persists between external actions.
// The identity trivially preserves the linear approximation.
//
// Provenance:
//   - Unix daemon convention: /var/run/*.pid lockdir pattern
//     (man pidfile, man daemon, libslack pidfile example)
//     Before starting, a daemon acquires a pidfile lock; after crash,
//     the health check detects the stale lockdir and cleans it up.
//   - systemd service lifecycle
//     (systemd/src/core/service.c:150-350)
//     service_start → service_running → service_exited → service_reload
//     Activate → running → stop → cleanup; lock-based process supervision.
//   - Homebrew lockfile pattern
//     (Library/Homebrew/utils/lock.sh:20-80)
//     Lock directory acquired before formula operations; released after.
//   - PostgreSQL postmaster lifecycle
//     (src/backend/postmaster/postmaster.c:500-800)
//     Lock file (postmaster.pid) acquired on startup; stale lock cleaned
//     by subsequent start attempts.
package patterns

import "tensor-logic/internal/tensor"

// DaemonLifecyclePattern returns the daemon lifecycle component with
// a 3-bit state space and an identity transition. The component captures
// the invariant that a daemon process must never run without holding its
// lock directory.
func DaemonLifecyclePattern() *Component {
	c := &Component{
		Name:       "daemon-lifecycle",
		Role:       RoleDaemon,
		Dim:        3,
		Variables:  []string{"lockdir_held", "daemon_running", "health_check_due"},
		Transition: tensor.IdentityMatrix(3),
	}
	// Linear approximation: daemon_running = 0 (bit 1 = 0).
	// Subspace {000, 001, 100, 101} is the largest Z₂ subspace in I.
	c.Invariant = tensor.NewConstraintMatrix(1)
	c.Invariant[0] = 1 << 1 // constraint: bit 1 = 0
	c.Proven = true
	c.ProofNote = "identity transition preserves daemon_running=0 subspace {000, 001, 100, 101}; real invariant I = {000, 001, 100, 101, 110, 111} (6 states) is not a Z₂ subspace; gap: {110, 111} are semantically valid but excluded from linear approximation"
	return c
}
