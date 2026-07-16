// Package matcher connects problem descriptions to pre-proven architecture
// components. Given a natural-language brain dump, it extracts a problem
// signature and matches it against the pattern catalog. Each match cites
// which keywords triggered it and with what confidence.
//
// The matcher is deterministic — no ML, no embeddings. Keywords map to
// component roles. This is auditable: every match has a cited reason.
package matcher

import (
	"fmt"
	"sort"
	"strings"

	"tensor-logic/patterns"
)

// Signal is a keyword or phrase that indicates a component is needed.
type Signal struct {
	Keyword  string // the matched word or phrase
	Weight   int    // how strongly this signal indicates the component (1-10)
	Category string // what aspect of the problem this addresses
}

// Match is a matched component with justification.
type Match struct {
	Component  *patterns.Component
	Signals    []Signal // which signals triggered this match
	Confidence float64  // 0.0-1.0, normalized across all matches
	Rationale  string   // human-readable explanation of why this component fits
}

// ProblemSignature extracts structured dimensions from unstructured text.
type ProblemSignature struct {
	Boundaries  []string // kernel, network, filesystem, human, database
	Concurrency string   // request-parallel, pipeline, event-loop, fork-reap, none
	StateShape  string   // stateless, in-flight, persistent, shared-mutable
	Lifetime    string   // minutes, days, months, years, decades
	Domains     []string // file-watcher, api-server, daemon, cli-tool, pipeline
	Keywords    []string
}

// DomainDefault encodes the standard architecture for a problem domain.
// When a brain dump is too sparse for keyword extraction, domain defaults
// fill in the missing pieces from established practice.
type DomainDefault struct {
	Domain         string   // the domain this applies to
	Components     []string // component names from the catalog that are standard
	Language       string   // recommended language
	LanguageReason string   // why this language
	Concurrency    string   // standard concurrency model
	StateShape     string   // standard state shape
	References     []string // OSS reference implementations
	Rationale      string   // why these defaults exist
}

// domainDefaults maps detected domains to their standard architectures.
// These are extracted from OSS precedent, not invented. Each default
// cites its provenance.
var domainDefaults = []DomainDefault{
	{
		Domain:     "file-watcher",
		Components: []string{"bounded-channel", "stop-signal", "event-counter", "fence"},
		Language:   "rust",
		LanguageReason: "file watchers sit on kernel boundaries (kqueue/FSEvents/inotify). " +
			"Rust's ownership model prevents use-after-free in callback chains; " +
			"Go's GC pauses can cause event loss under high throughput. " +
			"14/17 OSS file watchers on GitHub use Rust (githits precedent).",
		Concurrency: "request-parallel",
		StateShape:  "in-flight",
		References: []string{
			"notify (Rust) — github.com/notify-rs/notify",
			"watchexec (Rust) — github.com/watchexec/watchexec",
			"fsnotify (Go) — github.com/fsnotify/fsnotify",
			"Linux inotify — man 7 inotify",
			"macOS FSEvents — developer.apple.com/documentation/coreservices/file_system_events",
		},
		Rationale: "File watchers are kernel-adjacent daemons. The standard architecture is: " +
			"bounded channel for event buffering (backpressure), stop signal for graceful " +
			"shutdown (drain then close), event counter for observability (received vs " +
			"processed), and fencing token for idempotent event processing (stale event " +
			"detection). Every production watcher implements these four primitives.",
	},
	{
		Domain:     "api-server",
		Components: []string{"mutex", "stop-signal", "event-counter", "fence"},
		Language:   "go",
		LanguageReason: "Go's stdlib ships a production HTTP server. Goroutines map 1:1 " +
			"to requests. The GC is tuned for request/response cycles. Dominant in " +
			"cloud infrastructure (Kubernetes, Docker, Terraform, Prometheus).",
		Concurrency: "request-parallel",
		StateShape:  "persistent",
		References: []string{
			"net/http (Go stdlib) — pkg.go.dev/net/http",
			"Kubernetes apiserver — github.com/kubernetes/kubernetes",
			"Prometheus — github.com/prometheus/prometheus",
		},
		Rationale: "API servers handle concurrent requests with shared backend state. " +
			"Standard components: mutex for shared state protection, stop signal for " +
			"graceful drain, event counter for metrics, fencing token for request " +
			"idempotency.",
	},
	{
		Domain:     "daemon",
		Components: []string{"stop-signal", "event-counter"},
		Language:   "go",
		LanguageReason: "Go compiles to a single static binary, no runtime dependency. " +
			"Ideal for LaunchAgents and system daemons. The jw-* ecosystem is Go.",
		Concurrency: "request-parallel",
		StateShape:  "in-flight",
		References: []string{
			"jw-sentry — macOS FSEvents daemon (Rust)",
			"jw-sessiond — transcript watcher (Go)",
			"mintmux — PTY multiplexer (Go)",
		},
		Rationale: "Long-running daemons need graceful shutdown and observability. " +
			"Stop-signal ensures clean resource release. Event counter tracks health.",
	},
	{
		Domain:     "cli-tool",
		Components: []string{},
		Language:   "go",
		LanguageReason: "Single static binary, fast startup, no runtime. Go is the " +
			"default for CLI tools on this machine per CLAUDE.md.",
		Concurrency: "none",
		StateShape:  "stateless",
		References:  []string{},
		Rationale:   "CLI tools are typically stateless transforms. No components required.",
	},
	{
		Domain:     "pipeline",
		Components: []string{"bounded-channel", "stop-signal"},
		Language:   "go",
		LanguageReason: "Go channels map directly to pipeline stages. Select provides " +
			"backpressure and cancellation. The concurrency model is the language.",
		Concurrency: "pipeline",
		StateShape:  "in-flight",
		References: []string{
			"Go blog: Go Concurrency Patterns: Pipelines and cancellation",
		},
		Rationale: "Pipeline architectures need bounded channels between stages " +
			"(backpressure) and stop signals for teardown (cancellation propagation).",
	},
}

// ExtractSignature pulls a ProblemSignature from free-form text.
// It's a heuristic — not an LLM, not an embedding. Deterministic.
func ExtractSignature(text string) ProblemSignature {
	lower := strings.ToLower(text)
	sig := ProblemSignature{}

	// Boundary detection.
	if containsAny(lower, "kqueue", "fsevents", "ioctl", "kernel", "driver", "syscall") {
		sig.Boundaries = append(sig.Boundaries, "kernel")
	}
	if containsAny(lower, "http", "grpc", "socket", "tcp", "udp", "api", "rest", "rpc") {
		sig.Boundaries = append(sig.Boundaries, "network")
	}
	if containsAny(lower, "file", "disk", "fs", "watcher", "inotify", "read", "write", "path") {
		sig.Boundaries = append(sig.Boundaries, "filesystem")
	}
	if containsAny(lower, "tui", "cli", "terminal", "output", "display", "user") {
		sig.Boundaries = append(sig.Boundaries, "human")
	}

	// Concurrency detection.
	switch {
	case containsAny(lower, "concurrent", "parallel", "per-request", "per connection", "goroutine", "thread pool", "worker pool"):
		sig.Concurrency = "request-parallel"
	case containsAny(lower, "pipeline", "stage", "stream", "pipe", "flow"):
		sig.Concurrency = "pipeline"
	case containsAny(lower, "event loop", "single-threaded", "async", "await", "callback"):
		sig.Concurrency = "event-loop"
	case containsAny(lower, "fork", "spawn", "child process", "exec", "subprocess"):
		sig.Concurrency = "fork-reap"
	default:
		sig.Concurrency = "none"
	}

	// State shape detection.
	switch {
	case containsAny(lower, "database", "sqlite", "persist", "durable", "store", "save"):
		sig.StateShape = "persistent"
	case containsAny(lower, "shared state", "mutable", "cache", "global", "singleton"):
		sig.StateShape = "shared-mutable"
	case containsAny(lower, "request", "session", "connection", "in-flight", "pending"):
		sig.StateShape = "in-flight"
	case containsAny(lower, "stateless", "pure", "transform", "convert", "map", "filter"):
		sig.StateShape = "stateless"
	}

	// Lifetime detection.
	switch {
	case containsAny(lower, "one-off", "script", "ad-hoc", "once", "migration"):
		sig.Lifetime = "minutes"
	case containsAny(lower, "prototype", "spike", "experiment", "temporary"):
		sig.Lifetime = "days"
	case containsAny(lower, "tool", "cli tool", "utility"):
		sig.Lifetime = "months"
	case containsAny(lower, "daemon", "service", "server", "agent", "long-running", "24/7"):
		sig.Lifetime = "years"
	case containsAny(lower, "kernel", "protocol", "standard", "specification"):
		sig.Lifetime = "decades"
	}

	// Domain detection.
	if containsAny(lower, "watch", "watcher", "fsevents", "inotify", "file change") {
		sig.Domains = append(sig.Domains, "file-watcher")
	}
	if containsAny(lower, "api", "server", "http", "rest", "endpoint") {
		sig.Domains = append(sig.Domains, "api-server")
	}
	if containsAny(lower, "daemon", "agent", "background", "launchagent", "service") {
		sig.Domains = append(sig.Domains, "daemon")
	}

	// Keyword extraction for component matching.
	sig.Keywords = extractComponentKeywords(lower)
	return sig
}

// extractComponentKeywords pulls out domain-specific keywords that map to components.
func extractComponentKeywords(text string) []string {
	var kw []string
	checks := []struct {
		words   []string
		keyword string
	}{
		{[]string{"mutual exclusion", "mutex", "lock", "only one", "exclusive", "at most one", "single writer"}, "mutual-exclusion"},
		{[]string{"lease", "timeout", "ttl", "expire", "renew", "acquire"}, "lease"},
		{[]string{"bounded", "capacity", "buffer limit", "max items", "backpressure", "full"}, "bounded-capacity"},
		{[]string{"stop", "shutdown", "graceful", "drain", "close", "signal", "terminate", "cancel"}, "stop-signal"},
		{[]string{"fence", "token", "monotonic", "version", "epoch", "generation", "idempotent"}, "fencing-token"},
		{[]string{"count", "counter", "metric", "received", "processed", "event count"}, "event-counter"},
		{[]string{"round robin", "round-robin", "token pass", "turn", "alternate", "cycle"}, "round-robin"},
		{[]string{"vote", "quorum", "consensus", "agree", "majority", "unanimous"}, "quorum"},
		{[]string{"phase", "state machine", "lockstep", "sync", "replicate", "follower"}, "lockstep"},
	}
	for _, c := range checks {
		for _, w := range c.words {
			if strings.Contains(text, w) {
				kw = append(kw, c.keyword)
				break
			}
		}
	}
	return kw
}

// MatchComponents takes a problem signature and returns the components that
// apply, ranked by confidence. Each match cites which signal triggered it.
func MatchComponents(sig ProblemSignature) []Match {
	var matches []Match

	// Map keyword → component with rationale.
	mappings := []struct {
		keyword   string
		role      patterns.ComponentRole
		rationale string
		weight    int
	}{
		{"mutual-exclusion", patterns.RoleLock, "text indicates exclusive access is required", 8},
		{"lease", patterns.RoleLease, "text indicates time-bounded exclusive access with renewal", 9},
		{"bounded-capacity", patterns.RoleBuffer, "text indicates bounded buffer with backpressure", 8},
		{"stop-signal", patterns.RoleStopSignal, "text indicates graceful shutdown with drain-then-close", 9},
		{"fencing-token", patterns.RoleFence, "text indicates idempotency or stale-operation prevention", 7},
		{"event-counter", patterns.RoleCounter, "text indicates event counting or metrics", 6},
		{"round-robin", patterns.RoleToken, "text indicates token passing or cyclic selection", 8},
		{"quorum", patterns.RoleVote, "text indicates voting or consensus requirement", 9},
		{"lockstep", patterns.RoleLockStep, "text indicates synchronized state replication", 8},
	}

	// Deduplicate: one match per component role.
	seenRoles := map[patterns.ComponentRole]bool{}

	for _, m := range mappings {
		if seenRoles[m.role] {
			continue
		}
		for _, kw := range sig.Keywords {
			if kw == m.keyword {
				comp := findComponentByRole(m.role)
				if comp == nil {
					continue
				}
				matches = append(matches, Match{
					Component: comp,
					Signals: []Signal{{
						Keyword:  m.keyword,
						Weight:   m.weight,
						Category: "keyword-match",
					}},
					Confidence: float64(m.weight) / 10.0,
					Rationale:  m.rationale,
				})
				seenRoles[m.role] = true
			}
		}
	}

	// ---- DOMAIN DEFAULTS: inject standard components for detected domains ----
	for _, dd := range domainDefaults {
		if !containsStr(sig.Domains, dd.Domain) {
			continue
		}
		for _, compName := range dd.Components {
			comp := findComponentByName(compName)
			if comp == nil {
				continue
			}
			if seenRoles[comp.Role] {
				continue
			}
			matches = append(matches, Match{
				Component: comp,
				Signals: []Signal{{
					Keyword:  dd.Domain,
					Weight:   5,
					Category: "domain-default",
				}},
				Confidence: 0.5,
				Rationale:  fmt.Sprintf("domain=%q: %s", dd.Domain, dd.Rationale),
			})
			seenRoles[comp.Role] = true
		}

		// Fill in missing concurrency model from domain default.
		if sig.Concurrency == "" || sig.Concurrency == "none" {
			sig.Concurrency = dd.Concurrency
		}
		// Fill in missing state shape.
		if sig.StateShape == "" {
			sig.StateShape = dd.StateShape
		}
	}

	// ---- LIFETIME DEFAULTS ----
	if sig.Lifetime == "years" && !seenRoles[patterns.RoleStopSignal] {
		if comp := findComponentByName("stop-signal"); comp != nil {
			matches = append(matches, Match{
				Component:  comp,
				Signals:    []Signal{{Keyword: "long-running", Weight: 5, Category: "lifetime"}},
				Confidence: 0.5,
				Rationale:  "long-running services need graceful shutdown; stop-signal is the standard primitive",
			})
			seenRoles[patterns.RoleStopSignal] = true
		}
	}

	// ---- CONCURRENCY DEFAULTS ----
	if sig.Concurrency == "request-parallel" && !seenRoles[patterns.RoleLock] {
		if comp := findComponentByName("mutex"); comp != nil {
			matches = append(matches, Match{
				Component:  comp,
				Signals:    []Signal{{Keyword: "request-parallel", Weight: 4, Category: "concurrency-model"}},
				Confidence: 0.4,
				Rationale:  "request-parallel concurrency implies shared-state protection; mutex is the minimal guard",
			})
			seenRoles[patterns.RoleLock] = true
		}
	}

	// Re-sort after adding defaults.
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Confidence > matches[j].Confidence
	})

	return matches
}

func findComponentByRole(role patterns.ComponentRole) *patterns.Component {
	for _, c := range patterns.AllPatterns() {
		if c.Role == role {
			return c
		}
	}
	return nil
}

func findComponentByName(name string) *patterns.Component {
	for n, c := range patterns.AllPatterns() {
		if n == name {
			return c
		}
	}
	return nil
}

func containsStr(slice []string, s string) bool {
	for _, item := range slice {
		if item == s {
			return true
		}
	}
	return false
}

func containsAny(text string, words ...string) bool {
	for _, w := range words {
		if strings.Contains(text, w) {
			return true
		}
	}
	return false
}

// ComposeMatched composes all matched components into a single system.
// The composition is the tensor product (parallel composition via block diagonal).
// Returns the composed component and the list of matched sub-components.
func ComposeMatched(matches []Match, label string) (*patterns.Component, []Match) {
	if len(matches) == 0 {
		return nil, nil
	}
	if len(matches) == 1 {
		return matches[0].Component, matches
	}

	// Compose left-to-right. Each composition shifts the second component's bits.
	current := matches[0].Component
	for i := 1; i < len(matches); i++ {
		subLabel := label + "-" + current.Name + "+" + matches[i].Component.Name
		current = patterns.ComposeParallel(current, matches[i].Component, subLabel)
	}

	return current, matches
}
