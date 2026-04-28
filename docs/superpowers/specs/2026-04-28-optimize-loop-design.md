# Design: optimize loop + exp81 + memjuice reason

**Date:** 2026-04-28
**Status:** Approved for implementation

## Motivation

Tensor Logic's proof engine produces structured symbolic feedback — proof trees and negative proof explanations (`why-not`) — that tell a proposer exactly *why* a candidate failed. This is Actionable Side Information (ASI) in the sense of the `optimize_anything` framework. Most symbolic systems can't do this; TL can.

Two applications share the same loop structure: LLM-guided rule induction (exp81) and relational recall over session history (`memjuice reason`). The shared primitive is `tensor_logic/optimize.py`.

Paper grounding: AI Hippocampus survey (arXiv 2601.09113) — TL KB = explicit relational memory tier, optimize loop = structured recall with provenance, memjuice extraction rules = episodic capture.

---

## Layer 1: `tensor_logic/optimize.py`

### Interface

```python
@dataclass
class EvalResult:
    artifact: str                          # the candidate (format is caller-defined)
    score: float                           # primary Pareto dimension
    secondary_score: float = 0.0          # secondary Pareto dimension (default = same as score → single-axis)
    asi: str = ""                          # human-readable TL proof tree or why-not explanation
    asi_kind: Literal["proof", "why_not", "engine_error"] = "proof"

def optimize(
    propose: Callable[[str], str],         # feedback str → new artifact str
    evaluate: Callable[[str], EvalResult], # artifact → EvalResult
    accept: Callable[[EvalResult], bool],  # stopping criterion (caller-defined)
    pareto_axes: tuple[str, str],          # names of the two Pareto dimensions (for logging)
    max_steps: int = 50,
    frontier_size: int = 10,
    stagnation_k: int = 5,                 # halt if frontier unchanged for K consecutive steps
) -> list[EvalResult]:                     # returns Pareto frontier at termination
```

### Artifact format
The `artifact` field is a caller-defined string. The loop does not parse it. For exp81 it is a JSON array of relation strings; for memjuice reason it is a Datalog query string. The proposer and evaluator agree on the format; the loop does not care.

### Feedback string
On iteration 1, `propose` receives an empty string — the caller is responsible for baking any initial prompt into the proposer closure. On subsequent iterations, `propose` receives only the ASI from the *previous* evaluation (not accumulated history). For 0.5B models, full history concatenation degrades performance quickly. Callers that need history must compress it before passing it in.

### Pareto frontier
- Maintained as a list of up to `frontier_size` EvalResults, non-dominated on (score, secondary_score).
- Secondary score is caller-supplied via a second `evaluate` return value — or callers can encode both dimensions into `score` (e.g., F1 as primary, precision as secondary via `EvalResult` subclass). Default: degenerate single-axis frontier (secondary_score = score).
- Tiebreak within the frontier: shorter artifact string (Occam). Callers may override by wrapping `evaluate`.

### Stagnation exit
If the set of artifact strings in the frontier is identical for `stagnation_k` consecutive steps, halt early and return the current frontier. This prevents burning inference budget when the proposer has converged to a local basin.

### Error handling
- Invalid artifact (proposer emits unparseable output): evaluator should return `EvalResult(score=0.0, asi="...", asi_kind="engine_error")`. The loop continues — bad proposals score zero and fall off the frontier naturally.
- Max steps reached without `accept` firing: return current frontier, do not raise.

---

## Layer 2: exp81 — LLM-guided rule induction

**File:** `experiments/exp81_optimize_rule_induction.py`

### What it replaces
The brute-force `enumerate_rules` in exp78/exp79, which exhausts all relation-pair templates up to `max_len`. On hard mode schemas with 16 primitives this generates 4,368+ candidates.

### Proposer: Qwen2.5-0.5B as pruner, not synthesizer

The proposer's role is to *narrow the search space*, not to synthesize complete rules. This keeps the validated 97% accuracy figure relevant — the model is doing selection/pruning, which is the task it was SFT'd for.

```
feedback (ASI from last eval) → LM emits: {"relations": ["r1", "r2", ...], "max_len": 2}
```

The optimize loop then runs `induce_from_examples` on the reduced relation set rather than all primitives. On each iteration: prune → brute-force on reduced space → evaluate best candidate → ASI → prune again.

### Evaluator
1. Run `induce_from_examples` on the pruned relation set.
2. Take the best-scoring rule body from that sub-search.
3. Compute precision + recall on the current world.
4. For each missed positive: run `prove_negative` → append why-not tree to ASI string.
5. Return `EvalResult(score=f1, asi=formatted_failures, asi_kind="why_not" if any_missed else "proof")`.

### Pareto axes
- Primary: F1
- Secondary: precision (prefer precise rules when F1 ties)
- Tiebreak: rule body length (Occam)

### Frontier size
- Easy/medium: 5
- Hard: 15 (matches the density of near-F1 candidates seen in exp79)

### Success criterion
`accept` fires when F1 = 1.0 on the current world. Generalization measured post-hoc on 5 held-out worlds (same protocol as exp79).

### Test schemas
Same 3 as exp78: transitive closure, grandparent, sibling. Same easy/medium/hard modes as exp79 (noise, distractors). Primary metric: steps-to-F1=1.0 and total LLM calls vs brute-force template count.

### Falsification
- If exp81 requires *more* steps than brute force on easy mode → proposer is not helping; stop.
- If hard mode fails to recover F1=1.0 within max_steps → loop does not outperform exp79 baseline.

---

## Layer 3: `memjuice reason` command

**Depends on:** exp81 validated, optimize loop proven.

### JSONL → TL facts decomposition

Each memjuice observation is flattened to binary EAV facts. The `id` is the 0-indexed line number within the ledger file (stable, since the ledger is append-only).

```
obs_kind(42, "decision")
obs_project(42, "tensor")
obs_file(42, "tensor_logic/rules.py")
obs_ts(42, "2026-04-28T07:10:00Z")
obs_text(42, "switch to Borda voting")
obs_harness(42, "ClaudeCode")
obs_sha(42, "d570543")
```

Not all fields are present on every observation; absent fields simply produce no fact (open-world assumption, consistent with TL's default).

`ts` is stored as an ISO string fact; TL does not do datetime arithmetic, so temporal queries ("decisions in the last week") are answered by pre-filtering in Rust before loading facts, not inside TL.

### Optimize loop role
1. User runs `memjuice reason "what decisions changed the voting logic?"`
2. LM proposes an initial Datalog query over the EAV schema.
3. TL executes query against loaded facts.
4. Proof tree or why-not explanation → ASI.
5. LM refines query → iterate until `accept` fires (non-empty result) or max_steps.
6. Return: matched observation IDs + proof trees + source pointers (file:line).

### Python↔Rust boundary
Subprocess call to `python -m tensor_logic` (same pattern as `web_workbench/server.py`). Timeout: 30 seconds per call. If the subprocess hangs, memjuice reason prints a timeout error and exits cleanly.

### Non-goals (v1)
- No temporal arithmetic inside TL (pre-filter in Rust).
- No multi-project joins (one project's ledger per invocation).
- No streaming results (return full frontier when done).

---

## Build order

```
tensor_logic/optimize.py   — no external deps, pure library
       ↓
experiments/exp81_...      — validates loop on known-good task with ground truth
       ↓
memjuice reason            — consumer; Python↔Rust integration
```

If exp81 does not outperform brute force on easy mode, the loop design needs revision before the memjuice integration is worth building.

---

## Open questions (deferred)

- Does Qwen2.5-0.5B need additional fine-tuning to output valid JSON pruning responses, or does few-shot prompting suffice?
- Should `memjuice reason` load facts into a persistent in-process TL KB (faster for multi-query sessions) or rebuild per call (simpler, matches current stateless API pattern)?
- Should the optimize loop support multi-task mode (cross-transfer across schemas) as in `optimize_anything`? Deferred to v2.
