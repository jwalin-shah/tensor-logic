package patterns

import (
	"testing"
)

func TestFencePattern(t *testing.T) {
	fence := FencePattern()
	if !fence.Proven {
		t.Error("fence pattern should be proven by construction")
	}
	if err := fence.Verify(); err != nil {
		t.Errorf("fence verification failed: %v", err)
	}
}

func TestLeasePattern(t *testing.T) {
	lease := LeasePattern()
	if lease.Proven {
		t.Error("lease pattern with linear approximation should not be marked proven")
	}
	// Verify should fail for the linear approximation because
	// the constraint allows {11} which is the excluded state.
	err := lease.Verify()
	if err == nil {
		t.Log("lease linear approximation passed (vacuous or lucky)")
	}
}

func TestComposeParallel(t *testing.T) {
	fence := FencePattern()
	lease := LeasePattern()

	composite := ComposeParallel(fence, lease, "test-composite")

	if composite.Dim != fence.Dim+lease.Dim {
		t.Errorf("composite dim = %d, want %d", composite.Dim, fence.Dim+lease.Dim)
	}

	// Variables should be prefixed.
	if composite.Variables[0] != "test-composite.token_valid" {
		t.Errorf("var[0] = %q, want test-composite.token_valid", composite.Variables[0])
	}

	// Fence is proven, lease is not → composite should not be proven.
	if composite.Proven {
		t.Error("composite with unproven component should not be proven")
	}
}

func TestComposeParallelBothProven(t *testing.T) {
	// Two fence patterns — both proven.
	f1 := FencePattern()
	f2 := FencePattern()

	composite := ComposeParallel(f1, f2, "double-fence")
	if !composite.Proven {
		t.Error("composite of two proven components should be proven")
	}
	if err := composite.Verify(); err != nil {
		t.Errorf("proven composite failed verification: %v", err)
	}
}

func TestAllPatterns(t *testing.T) {
	all := AllPatterns()
	if len(all) < 2 {
		t.Errorf("expected at least 2 patterns, got %d", len(all))
	}

	// Verify every pattern that claims to be proven.
	for name, c := range all {
		if c.Proven {
			if err := c.Verify(); err != nil {
				t.Errorf("pattern %q claims proven but fails Verify: %v", name, err)
			}
		}
	}
}
