# tensor-logic implementation-readiness review

Date: 2026-05-07
Queue item: `tensor-logic-implementation-readiness`
Branch: `codex/goal-tensor-logic-implementation-readiness`
Initial HEAD: `c9e2cfce206bb279a4348189726908fcb436d3dc`

## Scope and validation

This is a read-only review and queue-prep pass. I did not edit product code or touch external trackers, deploys, pushes, or PRs. The only intended repository change is this report under `docs/overnight/2026-05-07-whole-portfolio-review/`.

Required validation command for this queue item:

```bash
git status --short
```

Additional evidence commands used for this review:

```bash
git status --short --branch
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
llm-tldr tree .
fd -H -t f '(^package.json$|^pnpm-lock.yaml$|^yarn.lock$|^package-lock.json$|^vite.config|^vitest.config|^tsconfig|^README|^AGENTS.md$|^CLAUDE.md$|^\.github|docs|test|spec|__tests__|playwright|eslint|prettier)' .
fd -H -t f . docs/overnight 2>/dev/null || true
fd -H -t f 'result.json|handoff.md' runs 2>/dev/null || true
python3 tools/code_index.py --dump
jq '{experiment, quick, gates, tl, mlp, deepsets}' experiments/exp87_support_data/results.json
```

Local note: this worktree's shell has `python3` but not `python`; CI and repo docs use `python`.

## Repository state

- `git status --short --branch` started clean on `codex/goal-tensor-logic-implementation-readiness`.
- No `docs/overnight/` report files existed before this pass.
- No local `runs/*/result.json` or `runs/*/handoff.md` files were present in this worktree.
- No `CONTEXT.md`, `CONTEXT-MAP.md`, or `docs/adr/` files were present, even though `docs/agents/domain.md` describes that optional domain-doc layout.
- No PR was created; external service access is out of scope for this queue item.

## Concrete file-path observations

1. `README.md` frames the repo as both a learning project and a shared `tensor_logic` library dependency, then records the headline import-closure result and the limits on parity/code-closure tasks. This makes claim hygiene important before turning new experiment results into public-facing work.

2. `pyproject.toml` defines a Python 3.11 package named `tensor-logic`, depends only on `torch>=2.0`, and puts `numpy`, `matplotlib`, and `pytest` in the `dev` extra. There is no lockfile in the repo, so reproducible worker setup relies on environment discipline rather than pinned dependency resolution.

3. `.github/workflows/ci.yml` runs one CI job on Python 3.11: install `-e ".[dev]"` and run `python -m pytest tests/ -v`. There is no separate lint, format, type-check, or quick experiment result validation job.

4. `tests/test_packaging_ci.py` explicitly asserts that `pyproject.toml`, `.github/workflows/ci.yml`, `README.md`, and `docs/SYMPHONY_RUN_PROTOCOL.md` preserve the worker validation contract. This is a good guardrail for packaging and CI drift.

5. `CLAUDE.md` requires `python tools/code_index.py --lookup <RelevantSymbol>` before implementation plans that touch `tensor_logic/`. `python3 tools/code_index.py --dump` showed exported APIs across `tensor_logic.closure`, `execution`, `file_format`, `http_api`, `language`, `optimize`, `program`, `proofs`, `reason`, `repo_graph_view`, `rules`, and `semirings`.

6. `tensor_logic/__init__.py` exports a broad public surface: proof APIs, file loading, HTTP helpers, repo ingest, proof-tree rendering, provenance helpers, and semiring/rule utilities. Changes under `tensor_logic/` should be treated as shared-library changes, not experiment-only edits.

7. `tests/test_tensor_logic_core.py` is the main integration test for core behavior: dense/BFS closure, rule parsing, stratified negation, provenance ranking, GF(2), named-index recursion, `.tl` file loading, positive and negative proofs, source-backed facts, disjunctive rule splitting, repo graph helpers, CLI/HTTP parity, and workbench sample validity.

8. `tensor_logic/proofs.py` has tabled recursion and cycle guards for positive and negative proof construction, plus a fallback BFS-style recursive proof path. The tests cover cycles and false recursive queries, so proof work can be queued safely if it is narrow and test-first.

9. `web_workbench/server.py` implements the browser workbench by writing temporary `.tl` files and invoking `python -m tensor_logic` via `subprocess.run`; `tests/test_web_workbench.py` covers query smoke behavior and why-not parity with `tensor_logic.http_api.prove_source`. This is executable but still duplicates CLI plumbing instead of using the in-process execution API.

10. `docs/superpowers/plans/2026-05-05-support-stability.md` is a strong implementation contract for the physics support/stability lane: it defines V1 non-goals, primitive relations, baselines, evaluation splits, falsification gates, testing contracts, and vertical slices exp84-exp87.

11. `experiments/exp84_support_data.py` and `tests/test_exp84_support_data.py` provide deterministic object-table generation plus label oracle checks for ID/OOD scenes, removal retraction, branching support, aliasing, and invalid interventions.

12. `experiments/exp85_support_tl.py` and `tests/test_exp85_support_tl.py` provide the TL stability engine, primitive extraction, proof-bearing labels, tolerance handling, generator parity, and removal retraction coverage.

13. `experiments/exp86_support_baselines.py` and `tests/test_exp86_support_baselines.py` provide MLP and DeepSets baselines with padding/masking, per-object logits, masked accuracy, and a tiny loss-reduction check. These are enough for smoke validation but not a statistical robustness story.

14. `experiments/exp87_support_eval.py` writes full V1 ID/OOD/counterfactual results and raises if TL deterministic or counterfactual accuracy falls below 100%. `experiments/exp87_support_data/results.json` records `v1_passed: true`, TL 100% across ID, larger OOD, deeper OOD, branching OOD, and counterfactual, with best-neural OOD margins of about 16.3 points on larger OOD and 30.8 points on deeper OOD.

15. `experiments/exp88_support_noisy_relations_data/results.json` shows the first sharp brittleness boundary: in XY geometry noise, hard TL first falls below 100% at delta `0.0001` with aggregate accuracy about `0.527`; tolerant geometry first falls below 100% at delta `0.001` with aggregate accuracy about `0.842`.

16. `experiments/exp91_interval_support_uncertainty.py` and `tests/test_exp91_interval_support_uncertainty.py` implement coordinate-band feasibility as an abstain/recover signal. `experiments/exp91_interval_support_uncertainty_data/results.json` shows interval feasibility can preserve 100% accepted accuracy at useful coverage for point predictions that degrade under noise.

17. `experiments/exp92_pixel_abstain_recover.py` and `tests/test_exp92_pixel_abstain_recover.py` move one step upstream to a synthetic pixel renderer/detector stub. Full results show XY delta `0.01` hard accuracy around `0.549`, while interval plus repair keeps accepted accuracy at `1.0` with about `0.800` coverage and zero accepted wrong cases.

18. `experiments/exp93_detector_calibration_stress.py` and `tests/test_exp93_detector_calibration_stress.py` introduce coordinate, missing, merge, and false-positive detector stress. Full XY results at uncertainty multiplier `1.0` show structural modes require a guard: guarded structural abstain has zero accepted wrong but zero coverage for structural failures, while naive routes have many false-stable cases, especially false positives.

19. `experiments/exp94_object_hypothesis_layer.py` and `tests/test_exp94_object_hypothesis_layer.py` add an oracle object-hypothesis layer that uses simulated structural anomaly metadata. `experiments/exp94_object_hypothesis_layer_data/results.json` shows this upper-bound route gets 100% accepted accuracy across missing, merge, and false-positive structural modes, with high but not full coverage.

20. `experiments/exp95_scored_object_hypotheses.py` and `tests/test_exp95_scored_object_hypotheses.py` remove oracle structural identity metadata and rank non-oracle candidates. This is the active implementation frontier: `experiments/exp95_scored_object_hypotheses_data/results.json` shows scored non-oracle still has large recovery gaps versus oracle and nonzero false-stable cases. For example, XY delta `0.01` has scored accuracy about `0.598` on missing with `592` false-stable records, and scored accuracy about `0.722` on false-positive with `82` false-stable records.

## Implementation-readiness assessment

The repo is ready for narrow executable work, especially in the support/stability experiment chain. Exp84-exp87 have a clean vertical slice with tests, CI coverage, and committed full results. Exp91-exp95 expose the next concrete frontier: moving from perfect object tables to pixel/detector uncertainty and then from oracle structural metadata to non-oracle object hypotheses.

The safest next work should not start by claiming a new research result. It should first improve measurement, failure-case auditability, and result-contract tests around exp93-exp95. Exp95 in particular is not claim-ready, but it is highly task-ready: the failing metrics are concrete, the owned files are localized, and the validations can be small.

## Risks and blockers

- Full local test validation was not required by this queue item and was not run. The required validation for this report is `git status --short`.
- The local shell does not provide `python`; worker docs and CI commands use `python`. Future local workers in this exact shell should use a venv or `python3`.
- No lockfile is present, so `torch>=2.0` plus the dev extras can resolve differently across worker machines.
- The full committed result JSONs for exp93-exp95 are large, and current tests mostly validate quick-run schemas rather than persisted full-result claims.
- Exp95 is not ready for an external claim: non-oracle scoring still accepts wrong structural repairs and has large recovery gaps versus the exp94 oracle upper bound.
- No previous overnight reports or local `runs/*` artifacts were available in this worktree, so this pass could not cross-check runner handoffs.

## Implementation-ready follow-up tasks

### 1. Add exp95 failure-case dumps for ranked candidate debugging

Owned files:
- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_exp95_scored_object_hypotheses.py`

Acceptance criteria:
- Add an optional `--dump-failures <path>` argument.
- When provided, write JSONL records for accepted-wrong and false-stable scored predictions.
- Each JSONL row includes split, failure mode, localization mode, delta, object id, expected label, observed label, oracle label, scored label, selected hypothesis, selected repair kind, candidate count, selected score, and score parts.
- Existing default result JSON schema remains backward compatible.

Smallest useful validation:

```bash
python -m pytest tests/test_exp95_scored_object_hypotheses.py -v
python experiments/exp95_scored_object_hypotheses.py --quick --dump-failures /tmp/exp95_failures.jsonl
```

### 2. Add explicit exp95 gates for non-oracle claim readiness

Owned files:
- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_exp95_scored_object_hypotheses.py`
- `experiments/exp95_scored_object_hypotheses_data/results_quick.json`
- `experiments/exp95_scored_object_hypotheses_data/results.json`

Acceptance criteria:
- Add a top-level `gates` object to exp95 results.
- Gates include zero accepted false-stable records, maximum recovery gap versus oracle, and minimum scored accepted accuracy per structural failure mode.
- Current full results should mark the non-oracle claim as not passed instead of relying on prose interpretation.
- Add `--enforce-gates` so future result-generation jobs can fail loudly when configured thresholds are missed.

Smallest useful validation:

```bash
python -m pytest tests/test_exp95_scored_object_hypotheses.py -v
python experiments/exp95_scored_object_hypotheses.py --quick
```

### 3. Add persisted results contract tests for exp87-exp95

Owned files:
- `tests/test_support_results_contract.py`
- `experiments/exp87_support_data/results.json`
- `experiments/exp91_interval_support_uncertainty_data/results.json`
- `experiments/exp92_pixel_abstain_recover_data/results.json`
- `experiments/exp93_detector_calibration_stress_data/results.json`
- `experiments/exp94_object_hypothesis_layer_data/results.json`
- `experiments/exp95_scored_object_hypotheses_data/results.json`

Acceptance criteria:
- Add lightweight tests that read committed full result JSONs without rerunning experiments.
- Assert exp87 V1 gates pass and split totals are nonzero.
- Assert exp91 and exp92 accepted-accuracy routes remain 100% for the documented XY uncertainty rows.
- Assert exp93 structural guarded route has zero accepted wrong for structural failures.
- Assert exp94 oracle object-hypothesis accepted accuracy is 100% for structural modes.
- Assert exp95 currently records non-oracle gaps or false-stable counts, so stale "passed" claims cannot slip into committed results.

Smallest useful validation:

```bash
python -m pytest tests/test_support_results_contract.py -v
```

### 4. Convert the web workbench server to in-process execution APIs

Owned files:
- `web_workbench/server.py`
- `tests/test_web_workbench.py`
- `web_workbench/static/app.js` if UI output options need a JSON/tree toggle

Acceptance criteria:
- Replace temporary-file subprocess execution in `run_tensor_logic_action()` with `tensor_logic.execution` or `tensor_logic.http_api` helpers where possible.
- Preserve current `run`, `query`, `prove`, and `why-not` behavior.
- Keep parse and arity errors as HTTP 400 responses with useful stderr/error text.
- Add a test proving why-not JSON semantics can be requested or that tree output remains semantically equivalent to `prove_source(..., why_not=True)`.

Smallest useful validation:

```bash
python -m pytest tests/test_web_workbench.py tests/test_tensor_logic_core.py::TensorLogicCoreTest::test_web_workbench_sample_is_valid_tl -v
```

### 5. Document and test the exp95 non-oracle ranking frontier

Owned files:
- `docs/superpowers/plans/2026-05-05-support-stability.md`
- `notes/EXPERIMENTS.md`
- `tests/test_exp95_scored_object_hypotheses.py`

Acceptance criteria:
- Add a short "current frontier" section stating that exp94 is an oracle upper bound and exp95 is a non-oracle ranking attempt with known gaps.
- Include the exact exp95 full-result metrics that block a claim: recovery gap versus oracle and false-stable counts for missing, merge, and false-positive modes.
- Add one focused regression test for the most dangerous exp95 failure class: a scored candidate must not accept a stable label for an object whose expected label is falls in a hand-built false-positive support scene.
- Keep this as documentation plus a targeted guard, not a broad ranking rewrite.

Smallest useful validation:

```bash
python -m pytest tests/test_exp95_scored_object_hypotheses.py -v
```

## Bottom line

`tensor-logic` is implementation-ready for small, well-owned tasks. The strongest immediate queue items are result-contract tests and exp95 failure audit tooling. The repo should not promote exp95 as a solved detector-structural-repair result until non-oracle gates are explicit and false-stable accepted cases are driven to zero or intentionally framed as an abstention tradeoff.
