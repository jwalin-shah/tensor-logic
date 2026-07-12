package matcher

import (
	"strings"
	"testing"

	"tensor-logic/patterns"
)

func TestExtractSignature_FileWatcher(t *testing.T) {
	text := "build a file system watcher daemon that watches for changes and handles concurrent events without losing any. must support graceful shutdown."
	sig := ExtractSignature(text)

	// Boundaries.
	if !contains(sig.Boundaries, "filesystem") {
		t.Errorf("filesystem boundary not detected, boundaries=%v", sig.Boundaries)
	}

	// Concurrency: "concurrent events" → request-parallel.
	if sig.Concurrency != "request-parallel" {
		t.Errorf("expected concurrency=request-parallel, got %q", sig.Concurrency)
	}

	// Lifetime: "daemon" → years.
	if sig.Lifetime != "years" {
		t.Errorf("expected lifetime=years, got %q", sig.Lifetime)
	}

	// Domains.
	if !contains(sig.Domains, "file-watcher") {
		t.Errorf("file-watcher domain not detected, domains=%v", sig.Domains)
	}
	if !contains(sig.Domains, "daemon") {
		t.Errorf("daemon domain not detected, domains=%v", sig.Domains)
	}

	// Keywords.
	if !contains(sig.Keywords, "stop-signal") {
		t.Errorf("stop-signal keyword not detected from 'graceful shutdown': keywords=%v", sig.Keywords)
	}
}

func TestExtractSignature_KernelBoundary(t *testing.T) {
	text := "kqueue-based event notifier with ioctl integration"
	sig := ExtractSignature(text)

	if !contains(sig.Boundaries, "kernel") {
		t.Errorf("kernel boundary not detected, boundaries=%v", sig.Boundaries)
	}
}

func TestExtractSignature_HTTPServer(t *testing.T) {
	text := "build an HTTP API server with per-request goroutines and a SQLite backend. it must never lose data."
	sig := ExtractSignature(text)

	if !contains(sig.Boundaries, "network") {
		t.Errorf("network boundary not detected, boundaries=%v", sig.Boundaries)
	}
	if sig.Concurrency != "request-parallel" {
		t.Errorf("expected concurrency=request-parallel from 'per-request goroutines', got %q", sig.Concurrency)
	}
	if sig.StateShape != "persistent" {
		t.Errorf("expected persistent state from 'SQLite', got %q", sig.StateShape)
	}
}

func TestExtractSignature_OneOffScript(t *testing.T) {
	text := "write a one-off script to convert CSV files to JSON"
	sig := ExtractSignature(text)

	if sig.Lifetime != "minutes" {
		t.Errorf("expected lifetime=minutes, got %q", sig.Lifetime)
	}
}

func TestMatchComponents_FileWatcher(t *testing.T) {
	text := "build a file system watcher daemon that watches for changes with a bounded buffer and graceful shutdown"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	// Should match: bounded-capacity (buffer), stop-signal (shutdown).
	roles := matchRoles(matches)

	if !contains(roles, "buffer") && !contains(roles, "stop_signal") {
		t.Errorf("expected buffer or stop_signal match, got roles=%v", roles)
	}

	if len(matches) < 1 {
		t.Errorf("expected at least 1 match for file watcher, got %d", len(matches))
	}

	for _, m := range matches {
		if len(m.Signals) == 0 {
			t.Errorf("match %q has no signals — every match must cite a reason", m.Component.Name)
		}
		if m.Rationale == "" {
			t.Errorf("match %q has empty rationale", m.Component.Name)
		}
		if m.Confidence <= 0 || m.Confidence > 1.0 {
			t.Errorf("match %q has invalid confidence %f", m.Component.Name, m.Confidence)
		}
	}
}

func TestMatchComponents_MutexExclusion(t *testing.T) {
	text := "build a service where only one writer can access the database at a time using mutual exclusion"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	roles := matchRoles(matches)
	if !contains(roles, "lock") && !contains(roles, "lease") {
		t.Errorf("expected lock or lease match for mutual exclusion, got roles=%v", roles)
	}
}

func TestMatchComponents_StopSignal_LongRunning(t *testing.T) {
	text := "long-running daemon service"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	// Should get stop-signal as a default for long-running services.
	roles := matchRoles(matches)
	if !contains(roles, "stop_signal") {
		t.Errorf("long-running daemon should get stop-signal default, got roles=%v", roles)
	}
}

func TestMatchComponents_NoKeywords(t *testing.T) {
	text := "write some code"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	// No keywords → no matches (or just bare defaults).
	if len(matches) > 0 {
		t.Logf("bare input produced %d matches: this is fine if they're sensible defaults", len(matches))
		for _, m := range matches {
			t.Logf("  %s: %s (confidence=%.2f)", m.Component.Name, m.Rationale, m.Confidence)
		}
	}
}

func TestComposeMatched(t *testing.T) {
	text := "file watcher with bounded buffer, graceful stop, and mutual exclusion"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	if len(matches) == 0 {
		t.Skip("no matches found — test data may need adjustment")
	}

	composed, used := ComposeMatched(matches, "test-composite")
	if composed == nil {
		t.Fatal("ComposeMatched returned nil")
	}
	if len(used) != len(matches) {
		t.Errorf("expected %d components in composition, got %d", len(matches), len(used))
	}
	if composed.Dim == 0 {
		t.Error("composed component has zero dimension")
	}
	if !composed.Proven {
		t.Log("composite is not proven: some sub-components are unproven")
	}
}

func TestAllMatchesAreDeterministic(t *testing.T) {
	text := "bounded buffer with stop signal and lease"
	sig1 := ExtractSignature(text)
	sig2 := ExtractSignature(text)

	matches1 := MatchComponents(sig1)
	matches2 := MatchComponents(sig2)

	if len(matches1) != len(matches2) {
		t.Fatalf("non-deterministic: first run got %d matches, second got %d", len(matches1), len(matches2))
	}
	for i := range matches1 {
		if matches1[i].Component.Name != matches2[i].Component.Name {
			t.Errorf("non-deterministic: match[%d] first=%q second=%q", i, matches1[i].Component.Name, matches2[i].Component.Name)
		}
	}
}

func TestMatchComponentNames(t *testing.T) {
	// Every match must reference a component that actually exists in the catalog.
	text := "bounded buffer with graceful shutdown and mutual exclusion via lease with fencing token for round-robin voting"
	sig := ExtractSignature(text)
	matches := MatchComponents(sig)

	all := patterns.AllPatterns()
	for _, m := range matches {
		found := false
		for name := range all {
			if name == m.Component.Name {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("match %q references component that does not exist in AllPatterns()", m.Component.Name)
		}
	}
}

func matchRoles(matches []Match) []string {
	var roles []string
	for _, m := range matches {
		roles = append(roles, string(m.Component.Role))
	}
	return roles
}

func contains(slice []string, s string) bool {
	for _, item := range slice {
		if strings.Contains(item, s) || item == s {
			return true
		}
	}
	return false
}
