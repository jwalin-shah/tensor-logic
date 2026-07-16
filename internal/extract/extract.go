// Package extract bridges unstructured brain dumps to structured problem
// signatures. It uses an LLM for extraction (one call, structured output)
// and falls back to keyword scanning if the LLM is unavailable.
//
// The LLM's job is narrow: map free text to a fixed schema. It does NOT
// decide architecture. The deterministic matcher does that. This split
// keeps the LLM boundary small and testable.
//
// The ontology (allowed boundaries, concurrency models, domains, etc.)
// is loaded from knowledge/domains.json. Adding a new domain is a JSON
// edit — no code change required.
package extract

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

// KnowledgeBase is the full ontology loaded from knowledge/domains.json.
type KnowledgeBase struct {
	Ontology         Ontology          `json:"ontology"`
	Domains          map[string]Domain `json:"domains"`
	DesignPrinciples Principles        `json:"design_principles"`
	LedgerLearnings  LedgerLearnings   `json:"ledger_learnings"`
}

type Ontology struct {
	Boundaries  map[string]EnumValue `json:"boundaries"`
	Concurrency map[string]EnumValue `json:"concurrency"`
	StateShape  map[string]EnumValue `json:"state_shape"`
	Lifetime    map[string]EnumValue `json:"lifetime"`
}

type EnumValue struct {
	Description       string   `json:"description"`
	Keywords          []string `json:"keywords"`
	Languages         []string `json:"languages,omitempty"`
	AntiLanguages     []string `json:"anti_languages,omitempty"`
	Rationale         string   `json:"rationale,omitempty"`
	DefaultComponents []string `json:"default_components,omitempty"`
}

type Domain struct {
	Description         string   `json:"description"`
	Keywords            []string `json:"keywords"`
	StandardComponents  []string `json:"standard_components"`
	RecommendedLanguage string   `json:"recommended_language"`
	LanguageRationale   string   `json:"language_rationale"`
	OSSReferences       []OSSRef `json:"oss_references"`
	CommonFailures      []string `json:"common_failures"`
	Principles          []string `json:"principles"`
}

type OSSRef struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	Language string `json:"language"`
	Note     string `json:"note"`
}

type Principles struct {
	General       []string `json:"general"`
	Concurrency   []string `json:"concurrency"`
	ErrorHandling []string `json:"error_handling"`
	Observability []string `json:"observability"`
}

type LedgerLearnings struct {
	Description string   `json:"description"`
	Entries     []string `json:"entries"`
}

// LoadKnowledgeBase reads the ontology from a JSON file.
func LoadKnowledgeBase(path string) (*KnowledgeBase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load knowledge base %s: %w", path, err)
	}
	var kb KnowledgeBase
	if err := json.Unmarshal(data, &kb); err != nil {
		return nil, fmt.Errorf("parse knowledge base: %w", err)
	}
	return &kb, nil
}

// AllowedBoundaries returns the list of allowed boundary values from the ontology.
func (kb *KnowledgeBase) AllowedBoundaries() []string {
	keys := make([]string, 0, len(kb.Ontology.Boundaries))
	for k := range kb.Ontology.Boundaries {
		keys = append(keys, k)
	}
	return keys
}

// AllowedConcurrency returns the list of allowed concurrency values.
func (kb *KnowledgeBase) AllowedConcurrency() []string {
	keys := make([]string, 0, len(kb.Ontology.Concurrency))
	for k := range kb.Ontology.Concurrency {
		keys = append(keys, k)
	}
	return keys
}

// AllowedStateShapes returns the list of allowed state shape values.
func (kb *KnowledgeBase) AllowedStateShapes() []string {
	keys := make([]string, 0, len(kb.Ontology.StateShape))
	for k := range kb.Ontology.StateShape {
		keys = append(keys, k)
	}
	return keys
}

// AllowedLifetimes returns the list of allowed lifetime values.
func (kb *KnowledgeBase) AllowedLifetimes() []string {
	keys := make([]string, 0, len(kb.Ontology.Lifetime))
	for k := range kb.Ontology.Lifetime {
		keys = append(keys, k)
	}
	return keys
}

// AllowedDomains returns the list of allowed domain values.
func (kb *KnowledgeBase) AllowedDomains() []string {
	keys := make([]string, 0, len(kb.Domains))
	for k := range kb.Domains {
		keys = append(keys, k)
	}
	return keys
}

// ProblemSignature is a structured description of what a system needs.
// This is the canonical input to the deterministic matcher. Every field
// is a constrained enum — the LLM prompt restricts output to these values.
type ProblemSignature struct {
	// What boundaries does this system cross?
	Boundaries []string `json:"boundaries"` // kernel, network, filesystem, human, database, inter-process

	// What concurrency shape?
	Concurrency string `json:"concurrency"` // request-parallel, pipeline, event-loop, fork-reap, none

	// What state does it hold?
	StateShape string `json:"state_shape"` // stateless, in-flight, persistent, shared-mutable

	// How long must it live?
	Lifetime string `json:"lifetime"` // minutes, days, months, years, decades

	// What problem domains does it touch?
	Domains []string `json:"domains"` // file-watcher, api-server, daemon, cli-tool, pipeline, distributed-lock, consensus, event-sourcing, stream-processor

	// What's the core task, distilled to one sentence?
	TaskSummary string `json:"task_summary"`

	// What specific constraints does the captain specify?
	Constraints []string `json:"constraints"` // "must not lose events", "must handle 10k conn/s", "must be in Go"

	// What is explicitly out of scope?
	NonGoals []string `json:"non_goals"`

	// How confident is the LLM in each field? (0.0-1.0)
	// Low confidence → flag for captain review.
	Confidence float64 `json:"confidence"` // aggregate across all fields
}

// LLMConfig configures the LLM extraction call.
type LLMConfig struct {
	// Command to invoke the LLM (e.g. "ca" for Claude direct).
	Adapter string
	// Model to use. Empty means adapter default.
	Model string
	// Timeout for the extraction call. Default: 30s.
	TimeoutSeconds int
}

// Extract uses an LLM to extract a ProblemSignature from free-form text.
// Falls back to the keyword scanner if the LLM is unavailable or fails.
// The knowledge base constrains allowed values for all enum fields.
func Extract(text string, kb *KnowledgeBase, cfg LLMConfig) (*ProblemSignature, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("extract: text must not be empty")
	}

	// Try LLM extraction first.
	sig, err := extractWithLLM(text, kb, cfg)
	if err == nil && sig != nil && sig.Confidence >= 0.3 {
		return sig, nil
	}

	// Fall back to keyword scanner (uses KB keywords).
	sig = extractWithKeywords(text, kb)
	sig.Confidence = 0.2 // keyword scanner is low-confidence by design
	return sig, nil
}

// extractWithLLM calls the configured adapter with the extraction prompt.
// The prompt constrains output to the ProblemSignature JSON schema.
func extractWithLLM(text string, kb *KnowledgeBase, cfg LLMConfig) (*ProblemSignature, error) {
	adapter := cfg.Adapter
	if adapter == "" {
		adapter = "ca"
	}

	adapterPath, err := exec.LookPath(adapter)
	if err != nil {
		return nil, fmt.Errorf("adapter %q not found: %w", adapter, err)
	}

	prompt := buildExtractionPrompt(text, kb)

	// ponytail: ca takes stdin as the prompt, outputs JSON to stdout.
	// Other adapters may differ — extend Config for adapter-specific flags.
	cmd := exec.Command(adapterPath)
	cmd.Stdin = strings.NewReader(prompt)

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("llm extraction failed: %w", err)
	}

	var sig ProblemSignature
	if err := json.Unmarshal(output, &sig); err != nil {
		return nil, fmt.Errorf("parse llm output: %w\nraw: %s", err, string(output)[:200])
	}

	return &sig, nil
}

// buildExtractionPrompt constructs the one-shot extraction prompt.
// The prompt lists every allowed value for every enum field. No creativity.
func buildExtractionPrompt(text string, kb *KnowledgeBase) string {
	// Build enum lists from the knowledge base so adding a new value is a
	// JSON edit, not a prompt change.
	boundaries := quoteList(kb.AllowedBoundaries())
	concurrencies := quoteList(kb.AllowedConcurrency())
	stateShapes := quoteList(kb.AllowedStateShapes())
	lifetimes := quoteList(kb.AllowedLifetimes())
	domains := quoteList(kb.AllowedDomains())

	// Build boundary descriptions for the prompt.
	var boundaryHelp strings.Builder
	for name, v := range kb.Ontology.Boundaries {
		fmt.Fprintf(&boundaryHelp, "- %s: %s\n", name, v.Description)
	}

	// Build domain descriptions.
	var domainHelp strings.Builder
	for name, d := range kb.Domains {
		fmt.Fprintf(&domainHelp, "- %s: %s\n", name, d.Description)
	}

	return fmt.Sprintf(`Extract a structured problem signature from this description.
Output ONLY valid JSON. No commentary, no markdown, no code fences.

DESCRIPTION:
%s

SCHEMA (every field constrained — use ONLY the listed values):

{
  "boundaries": [LIST FROM: %s],
  "concurrency": [ONE OF: %s],
  "state_shape": [ONE OF: %s],
  "lifetime": [ONE OF: %s],
  "domains": [LIST FROM: %s],
  "task_summary": "one sentence distilling the core task",
  "constraints": ["list", "of", "explicit", "constraints"],
  "non_goals": ["what", "is", "explicitly", "out", "of", "scope"],
  "confidence": 0.85
}

BOUNDARY DEFINITIONS:
%s
DOMAIN DEFINITIONS:
%s
RULES:
- NEVER invent values outside the listed enums.
- If the description doesn't mention a field, leave it empty.
- boundaries: what does this system TALK TO? Pick from the boundary definitions above.
- concurrency: how does it handle MULTIPLE things at once? Pick from: %s.
- state_shape: what kind of STATE does it hold? Pick from: %s.
- lifetime: how long will this code LIVE? Pick from: %s.
- domains: what PROBLEM CLASS does this belong to? Pick from: %s.
- confidence: 0.9+ for clear descriptions, 0.5-0.8 for ambiguous, <0.5 for guessing.

JSON:`, text, boundaries, concurrencies, stateShapes, lifetimes, domains,
		boundaryHelp.String(), domainHelp.String(),
		concurrencies, stateShapes, lifetimes, domains)
}

func quoteList(items []string) string {
	quoted := make([]string, len(items))
	for i, item := range items {
		quoted[i] = fmt.Sprintf("%q", item)
	}
	return strings.Join(quoted, ", ")
}

// extractWithKeywords is the fallback keyword scanner. It reads keywords
// from the knowledge base — every boundary, concurrency model, state shape,
// lifetime, and domain is defined there. No hardcoded keywords.
func extractWithKeywords(text string, kb *KnowledgeBase) *ProblemSignature {
	lower := strings.ToLower(text)
	sig := &ProblemSignature{}

	// Boundary detection — from KB.
	for name, v := range kb.Ontology.Boundaries {
		for _, kw := range v.Keywords {
			if strings.Contains(lower, kw) {
				sig.Boundaries = append(sig.Boundaries, name)
				break
			}
		}
	}

	// Concurrency — first-match wins.
	for name, v := range kb.Ontology.Concurrency {
		for _, kw := range v.Keywords {
			if strings.Contains(lower, kw) {
				sig.Concurrency = name
				goto concurrencyDone
			}
		}
	}
concurrencyDone:

	// State shape — first-match wins.
	for name, v := range kb.Ontology.StateShape {
		for _, kw := range v.Keywords {
			if strings.Contains(lower, kw) {
				sig.StateShape = name
				goto stateDone
			}
		}
	}
stateDone:

	// Lifetime — first-match wins.
	for name, v := range kb.Ontology.Lifetime {
		for _, kw := range v.Keywords {
			if strings.Contains(lower, kw) {
				sig.Lifetime = name
				goto lifetimeDone
			}
		}
	}
lifetimeDone:

	// Domain detection — from KB domain keywords.
	for name, d := range kb.Domains {
		matched := false
		// Check explicit keywords first.
		for _, kw := range d.Keywords {
			if strings.Contains(lower, kw) {
				sig.Domains = append(sig.Domains, name)
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		// Fallback: check if the domain name or key description words appear.
		if strings.Contains(lower, name) {
			sig.Domains = append(sig.Domains, name)
			continue
		}
		// Check description words (at least 2 must match).
		descWords := strings.Fields(strings.ToLower(d.Description))
		matchCount := 0
		for _, w := range descWords {
			if len(w) > 3 && strings.Contains(lower, w) {
				matchCount++
			}
		}
		if matchCount >= 2 {
			sig.Domains = append(sig.Domains, name)
		}
	}

	// Task summary: first 100 chars.
	if len(text) > 100 {
		sig.TaskSummary = text[:100] + "..."
	} else {
		sig.TaskSummary = text
	}

	return sig
}
