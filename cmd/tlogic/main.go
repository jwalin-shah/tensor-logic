// tlogic is the CLI for the tensor-logic proof system.
// Commands:
//   tlogic prove <component>     — prove an invariant holds for a component
//   tlogic compose <A> <B>       — compose two components and verify
//   tlogic list-patterns         — list all known patterns
//   tlogic show-pattern <name>   — show a pattern's details
//   tlogic match <description>   — match problem description against pattern catalog
package main

import (
	"fmt"
	"os"
	"strings"

	"tensor-logic/internal/matcher"
	"tensor-logic/internal/prover"
	"tensor-logic/patterns"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	switch cmd {
	case "list-patterns":
		listPatterns()
	case "show-pattern":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: tlogic show-pattern <name>")
			os.Exit(1)
		}
		showPattern(os.Args[2])
	case "prove":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: tlogic prove <component>")
			os.Exit(1)
		}
		proveComponent(os.Args[2])
	case "compose":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "usage: tlogic compose <A> <B>")
			os.Exit(1)
		}
		composeComponents(os.Args[2], os.Args[3])
	case "match":
		text := strings.Join(os.Args[2:], " ")
		if strings.TrimSpace(text) == "" {
			fmt.Fprintln(os.Stderr, "usage: tlogic match <problem description>")
			os.Exit(1)
		}
		matchProblem(text)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Println("tlogic — tensor logic proof system")
	fmt.Println()
	fmt.Println("commands:")
	fmt.Println("  list-patterns              list all known architecture patterns")
	fmt.Println("  show-pattern <name>        show pattern details")
	fmt.Println("  prove <component>          prove an invariant (from patterns)")
	fmt.Println("  compose <A> <B>            compose two components and verify")
	fmt.Println("  match <description>        match problem description against pattern catalog")
}

func listPatterns() {
	all := patterns.AllPatterns()
	for name, c := range all {
		status := "UNPROVEN"
		if c.Proven {
			status = "PROVEN"
		}
		fmt.Printf("%-20s  %-8s  %d bits  %s\n", name, status, c.Dim, c.ProofNote)
	}
}

func showPattern(name string) {
	all := patterns.AllPatterns()
	c, ok := all[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "pattern %q not found\n", name)
		os.Exit(1)
	}

	fmt.Printf("Name:    %s\n", c.Name)
	fmt.Printf("Role:    %s\n", c.Role)
	fmt.Printf("Dim:     %d bits\n", c.Dim)
	fmt.Printf("Proven:  %v\n", c.Proven)
	fmt.Printf("Note:    %s\n", c.ProofNote)
	fmt.Println()
	fmt.Println("Variables:")
	for i, v := range c.Variables {
		fmt.Printf("  bit %d: %s\n", i, v)
	}
	fmt.Println()
	fmt.Println("Transition matrix:")
	fmt.Println(c.Transition)
	fmt.Println()
	fmt.Println("Invariant constraints:")
	if c.Invariant.Constraints() == 0 {
		fmt.Println("  (none — all states are safe)")
	} else {
		fmt.Println(c.Invariant)
	}
}

func proveComponent(name string) {
	all := patterns.AllPatterns()
	c, ok := all[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "pattern %q not found\n", name)
		os.Exit(1)
	}

	fmt.Printf("Proving %s (%d bits, %d states)...\n", name, c.Dim, 1<<c.Dim)
	err := c.Verify()
	if err != nil {
		fmt.Printf("FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("PROVED: all %d states preserve the invariant\n", 1<<c.Dim)
}

func composeComponents(aName, bName string) {
	all := patterns.AllPatterns()
	A, ok := all[aName]
	if !ok {
		fmt.Fprintf(os.Stderr, "component %q not found\n", aName)
		os.Exit(1)
	}
	B, ok := all[bName]
	if !ok {
		fmt.Fprintf(os.Stderr, "component %q not found\n", bName)
		os.Exit(1)
	}

	label := fmt.Sprintf("%s+%s", A.Name, B.Name)
	composite := patterns.ComposeParallel(A, B, label)

	fmt.Printf("Composed: %s\n", composite.Name)
	fmt.Printf("Dim:      %d bits (%d states)\n", composite.Dim, 1<<composite.Dim)
	fmt.Printf("Proven:   %v (%s)\n", composite.Proven, composite.ProofNote)

	if composite.Proven {
		fmt.Println("Composition theorem guarantees the invariant holds.")
		fmt.Println("(Both components are individually proven, so no re-proving needed.)")
	} else {
		fmt.Println("One or both components are unproven. Running verification...")
		// Run exhaustive check only for small composites.
		if composite.Dim <= 20 {
			result := prover.ProveExhaustive(composite.Transition, composite.Invariant, composite.Dim)
			fmt.Println(result)
		} else {
			fmt.Printf("State space too large (%d states). Verify components individually.\n", 1<<composite.Dim)
		}
	}
}

func matchProblem(text string) {
	sig := matcher.ExtractSignature(text)
	matches := matcher.MatchComponents(sig)

	fmt.Println("=== Problem Signature ===")
	if len(sig.Boundaries) > 0 {
		fmt.Printf("Boundaries:   %v\n", sig.Boundaries)
	} else {
		fmt.Println("Boundaries:   (none detected)")
	}
	if sig.Concurrency != "" && sig.Concurrency != "none" {
		fmt.Printf("Concurrency:  %s\n", sig.Concurrency)
	}
	if sig.StateShape != "" {
		fmt.Printf("State:        %s\n", sig.StateShape)
	}
	if sig.Lifetime != "" {
		fmt.Printf("Lifetime:     %s\n", sig.Lifetime)
	}
	if len(sig.Domains) > 0 {
		fmt.Printf("Domains:      %v\n", sig.Domains)
	}
	fmt.Println()

	if len(matches) == 0 {
		fmt.Println("No matching components found in the catalog.")
		fmt.Println("This problem may require a novel architecture. Run:")
		fmt.Println("  bridge create \"<your description>\"")
		return
	}

	fmt.Println("=== Matched Components ===")
	for i, m := range matches {
		status := "UNPROVEN"
		if m.Component.Proven {
			status = "PROVEN"
		}
		fmt.Printf("%d. %s (%s) [%s]\n", i+1, m.Component.Name, status, m.Component.Role)
		fmt.Printf("   Confidence: %.0f%%\n", m.Confidence*100)
		fmt.Printf("   Rationale:  %s\n", m.Rationale)
		for _, s := range m.Signals {
			fmt.Printf("   → matched: %q (weight=%d, category=%s)\n", s.Keyword, s.Weight, s.Category)
		}
		fmt.Println()
	}

	// Compose them.
	composed, _ := matcher.ComposeMatched(matches, "sys")
	fmt.Println("=== Composed System ===")
	if composed != nil {
		fmt.Printf("Total state space: %d bits (%d states)\n", composed.Dim, 1<<composed.Dim)
		fmt.Printf("Proven:            %v\n", composed.Proven)
		if composed.Proven {
			fmt.Println("All sub-components are individually proven. Composition theorem applies.")
			fmt.Println("No re-proving needed for parallel composition.")
		} else {
			fmt.Println("Some components are unproven. Verify individually, then re-compose.")
		}
	} else {
		fmt.Println("Nothing to compose (0 components matched).")
	}
}
