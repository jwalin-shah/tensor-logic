package extract

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testKB loads the knowledge base from the project root.
func testKB(t *testing.T) *KnowledgeBase {
	t.Helper()
	// Find knowledge/domains.json relative to the test file.
	// Walk up from the current directory to find the tensor-logic root.
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		path := filepath.Join(dir, "knowledge", "domains.json")
		if _, err := os.Stat(path); err == nil {
			kb, err := LoadKnowledgeBase(path)
			if err != nil {
				t.Fatal(err)
			}
			return kb
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find knowledge/domains.json")
		}
		dir = parent
	}
}

func TestLoadKnowledgeBase(t *testing.T) {
	kb := testKB(t)
	if kb == nil {
		t.Fatal("kb is nil")
	}
	if len(kb.Ontology.Boundaries) == 0 {
		t.Error("no boundaries in knowledge base")
	}
	if len(kb.Domains) == 0 {
		t.Error("no domains in knowledge base")
	}
	if len(kb.AllowedBoundaries()) == 0 {
		t.Error("AllowedBoundaries() returned empty")
	}

	// Check a known domain exists.
	if _, ok := kb.Domains["file-watcher"]; !ok {
		t.Error("file-watcher domain not found in knowledge base")
	}
	if _, ok := kb.Domains["api-server"]; !ok {
		t.Error("api-server domain not found in knowledge base")
	}
}

func TestKeywordExtractionWithKB(t *testing.T) {
	kb := testKB(t)

	tests := []struct {
		text     string
		check    func(*ProblemSignature) bool
		describe string
	}{
		{
			"build a file system watcher daemon",
			func(s *ProblemSignature) bool {
				return contains(s.Boundaries, "filesystem") && contains(s.Domains, "file-watcher")
			},
			"file watcher detects filesystem boundary and file-watcher domain",
		},
		{
			"build an HTTP API server with SQLite",
			func(s *ProblemSignature) bool {
				return contains(s.Boundaries, "network") && contains(s.Domains, "api-server")
			},
			"api server detects network boundary and api-server domain",
		},
		{
			"distributed lock service with fencing tokens",
			func(s *ProblemSignature) bool {
				return contains(s.Domains, "distributed-lock")
			},
			"lock service detects distributed-lock domain",
		},
		{
			"raft consensus protocol for leader election",
			func(s *ProblemSignature) bool {
				return contains(s.Domains, "consensus")
			},
			"raft detects consensus domain",
		},
		{
			"event sourcing system with append-only log",
			func(s *ProblemSignature) bool {
				return contains(s.Domains, "event-sourcing")
			},
			"event sourcing detects event-sourcing domain",
		},
	}

	for _, tt := range tests {
		t.Run(tt.describe, func(t *testing.T) {
			sig := extractWithKeywords(tt.text, kb)
			if !tt.check(sig) {
				t.Errorf("extraction check failed for %q\nsig: boundaries=%v domains=%v concurrency=%q state=%q lifetime=%q",
					tt.text, sig.Boundaries, sig.Domains, sig.Concurrency, sig.StateShape, sig.Lifetime)
			}
		})
	}
}

func TestExtractionIsDeterministic(t *testing.T) {
	kb := testKB(t)
	text := "file watcher daemon with graceful shutdown"

	sig1 := extractWithKeywords(text, kb)
	sig2 := extractWithKeywords(text, kb)

	if sig1.Boundaries[0] != sig2.Boundaries[0] {
		t.Error("extraction is not deterministic")
	}
}

func TestPromptGeneration(t *testing.T) {
	kb := testKB(t)
	prompt := buildExtractionPrompt("test description", kb)

	if !strings.Contains(prompt, "file-watcher") {
		t.Error("prompt missing domain file-watcher")
	}
	if !strings.Contains(prompt, "api-server") {
		t.Error("prompt missing domain api-server")
	}
	if !strings.Contains(prompt, "kernel") {
		t.Error("prompt missing boundary kernel")
	}
	if !strings.Contains(prompt, "request-parallel") {
		t.Error("prompt missing concurrency request-parallel")
	}
	if !strings.Contains(prompt, "test description") {
		t.Error("prompt missing the input text")
	}
}

func TestKnowledgeBaseDomainDefaults(t *testing.T) {
	kb := testKB(t)

	// File watcher should recommend Rust with specific components.
	fw, ok := kb.Domains["file-watcher"]
	if !ok {
		t.Fatal("file-watcher domain missing")
	}
	if fw.RecommendedLanguage != "rust" {
		t.Errorf("file-watcher language = %q, want rust", fw.RecommendedLanguage)
	}
	if len(fw.StandardComponents) == 0 {
		t.Error("file-watcher has no standard components")
	}
	if len(fw.OSSReferences) == 0 {
		t.Error("file-watcher has no OSS references")
	}
	if len(fw.Principles) == 0 {
		t.Error("file-watcher has no design principles")
	}

	// API server should recommend Go.
	as, ok := kb.Domains["api-server"]
	if !ok {
		t.Fatal("api-server domain missing")
	}
	if as.RecommendedLanguage != "go" {
		t.Errorf("api-server language = %q, want go", as.RecommendedLanguage)
	}
}

func contains(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}
