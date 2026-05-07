# tensor-logic Risk And Validation Review

Date: 2026-05-07
Queue item: `tensor-logic-risk-and-validation-review`
Branch: `codex/goal-tensor-logic-risk-and-validation-review`
Reviewed HEAD: `c9e2cfce206bb279a4348189726908fcb436d3dc`

## Scope

This is a read-only risk and validation review for `tensor-logic`. Product code was not edited. The only intended repository change from this queue item is this report.

No repo-local prior overnight artifacts were found: `rg --files --hidden | rg '(^runs/|^docs/overnight/|result\.json$|handoff\.md$)'` found committed experiment result JSONs, but no `runs/*/result.json`, no `runs/*/handoff.md`, and no existing `docs/overnight/*` report for this pass.

## Commands Run

- Repo structure: `llm-tldr tree .`
- Hidden CI/overnight scan: `rg --files --hidden -g '.github/**' -g 'docs/overnight/**' -g 'runs/**' -g 'CODEX_WORKPAD.md' -g 'ISSUE.md' -g 'AGENTS.md' -g 'CLAUDE.md' -g 'README.md'`
- Current state: `git status --porcelain=v1 --branch`
- HEAD: `git rev-parse HEAD`
- API index lookups required before planning `tensor_logic/` changes: `python3 tools/code_index.py --lookup Program`, `python3 tools/code_index.py --lookup prove`, `python3 tools/code_index.py --lookup load_tl`
- Smoke validation run during review: `python3 -m pytest tests/test_packaging_ci.py tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -q` -> `20 passed in 0.06s`
- Required queue validation to run after writing this report: `git status --short`

Note: `python tools/code_index.py --dump` failed locally because `python` is not on PATH in this shell; `python3` worked. CI and docs use `python`, so local worker setup should either provide a `python` shim or workers should use `python3` consistently when running locally.

## Concrete Observations

1. `README.md:85-102` documents the clean-checkout worker validation contract: `python -m pip install -e ".[dev]"`, `python -m pytest tests/ -v`, the support fast path for `tests/test_exp84_support_data.py tests/test_exp85_support_tl.py`, and `python experiments/exp86_support_baselines.py --quick`.

2. `.github/workflows/ci.yml:18-30` runs Python 3.11 on pull requests and `main`, installs `".[dev]"`, and runs `python -m pytest tests/ -v`. This is a real CI contract, not only README text.

3. `pyproject.toml:12-21` keeps install scope small: runtime dependency is only `torch>=2.0`; dev extras are `matplotlib`, `numpy`, and `pytest`. Experiments with optional LM backends (`transformers`, `mlx_lm`) are not part of the default test/install contract.

4. `tests/test_packaging_ci.py:16-54` protects package metadata, README validation text, CI command strings, and the Symphony protocol's PR/validation contract. This makes drift in the basic worker instructions test-visible.

5. `CLAUDE.md:5-11` requires `tools/code_index.py` lookups before implementation planning that touches `tensor_logic/`. `tools/code_index.py` writes `tools/index.json`, and `.gitignore:18-19` correctly ignores `.cocoindex_code/` and `tools/index.json`, so the lookup can be used without adding generated artifacts.

6. `docs/RUN_PROTOCOL.md:16-24` requires remote experiment outputs to include a manifest with config, metrics, git SHA, timestamps, and failure logs; `docs/RUN_PROTOCOL.md:65-85` requires committed result pointers. Current support/stability scripts such as `experiments/exp87_support_eval.py:366-383` write config and `results_path`, but no git SHA or run timestamps.

7. `docs/superpowers/plans/2026-05-05-support-stability.md:111-119` defines the V1 falsification gate: TL must be 100% on deterministic labels, 100% on counterfactual retraction, and at least 10 percentage points better than the best neural baseline on larger/deeper OOD. `experiments/exp87_support_eval.py:286-334` computes all three gates.

8. `experiments/exp87_support_eval.py:361-364` raises when TL deterministic or counterfactual gates fail, but it does not raise when `tl_ood_margin_vs_best_neural_at_least_10pp` fails. It can therefore write a falsified result with exit code 0 even though the plan treats the OOD margin as part of the pass gate.

9. `experiments/exp87_support_data/results_quick.json` currently records a passing quick V1 result: TL accuracy is 1.0 on ID, larger OOD, deeper OOD, branching OOD, and counterfactual splits; larger OOD margin is about 20.8 percentage points over DeepSets, and deeper OOD margin is about 38.4 percentage points over DeepSets.

10. `tests/test_exp84_support_data.py` covers deterministic generation, object ranges, non-overlap, removal-chain retraction, branch support, non-aliasing, and unknown remove targets. `tests/test_exp85_support_tl.py` covers hand-built near misses, stable/falling cases, branching support, generator parity, removal retraction, and tolerance recovery. These are strong deterministic contracts for the support substrate.

11. `tests/test_exp86_support_baselines.py:51-71` checks loss reduction and metric schema for neural baselines, but it intentionally avoids a research-accuracy threshold. That is appropriate for CI stability, but it means claims about baseline strength depend on committed result JSONs and notes, not on a failing unit gate.

12. `notes/EXPERIMENTS.md:9-11` records the latest support/pixel-facing conclusions. Exp93 says structural detector failures break naive routing; exp94 says oracle object hypotheses recover them; exp95 is explicitly mixed, with false-positive ranking useful but missing/merge still needing identity evidence.

13. `experiments/exp95_scored_object_hypotheses.py:740-747` accurately labels the current non-oracle hypothesis layer as having remaining identity/cardinality gaps. The quick result JSON mirrors the risk: for missing-object stress at `delta=0.0`, observed naive accepted accuracy is about 70.8%, but scored non-oracle is about 68.1% with 65 scored false-stable accepts; oracle remains 100%.

14. `tests/test_exp95_scored_object_hypotheses.py:120-146` only checks schema and presence of oracle/scored fields for the aggregate run. It does not gate the safety property implied by `notes/EXPERIMENTS.md:9`: accept false-positive drops when confident, but abstain missing/merge repairs until identity evidence is available.

15. `tensor_logic/proofs.py:348-378` implements recursive proof fallback by choosing the first binary relation with the same domains as the recursive relation. That is convenient for examples, but it is ambiguous if a program has multiple same-domain base relations and a recursive rule should only follow one of them.

16. `web_workbench/server.py:44-71` writes user source to a temp file and shells out to `python -m tensor_logic` without a timeout or source-size guard. Existing tests cover happy-path query/prove behavior, but not runaway recursive rules or oversized source payloads.

## Risks And Blockers

- No prior overnight `runs/*` handoff artifacts were available in this worktree, so this review could not compare runner-created PRs or handoffs against current repo state.
- Local command drift exists: docs and CI use `python`, while this shell only has `python3`. That is not a repo bug by itself, but it is a worker reliability risk.
- Support/stability results are well documented, but committed result JSONs lack the run provenance required by the older experiment run protocol: git SHA, timestamps, command, and failure-output pointers.
- Exp87 computes the OOD margin gate but does not fail the script when that gate fails. This is the sharpest validation hole because the falsification gate is central to the current research claim.
- Exp95 is correctly documented as mixed, but its tests do not yet encode a safety gate that prevents missing/merge hypothesis scoring from becoming an accepted recovery claim.
- Recursive proof fallback can silently use the wrong base relation in ambiguous same-domain programs.
- The web workbench subprocess path can hang or consume resources on pathological input because it has no timeout or size limit.

## Implementation-Ready Follow-Up Tasks

### 1. Make the exp87 OOD margin gate fail loudly

Owned files:
- `experiments/exp87_support_eval.py`
- `tests/test_exp87_support_eval.py`

Acceptance criteria:
- `run_evaluation()` raises a clear `RuntimeError` when `tl_ood_margin_vs_best_neural_at_least_10pp.passed` is false.
- Tests cover the negative path by constructing or monkeypatching a gate result where deterministic and counterfactual TL pass but the OOD margin fails.
- Positive-path tests still assert the current quick schema and gate fields.

Smallest useful validation:

```bash
python3 -m pytest tests/test_exp87_support_eval.py -v
```

### 2. Add result provenance metadata for support/stability result JSONs

Owned files:
- `experiments/exp87_support_eval.py`
- `experiments/exp88_support_noisy_relations.py`
- `experiments/exp89_support_primitive_confidence.py`
- `experiments/exp90_support_repair_sweep.py`
- `experiments/exp91_interval_support_uncertainty.py`
- `experiments/exp92_pixel_abstain_recover.py`
- `experiments/exp93_detector_calibration_stress.py`
- `experiments/exp94_object_hypothesis_layer.py`
- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_support_result_metadata.py`

Acceptance criteria:
- Every support/stability runner from exp87 through exp95 writes `git_sha`, `started_at`, `finished_at`, and `command` or `argv` into its result JSON.
- Tests verify the metadata exists when a runner writes to a temp output path.
- The metadata helper must not require network access or external services.

Smallest useful validation:

```bash
python3 -m pytest tests/test_support_result_metadata.py -v
```

### 3. Encode exp95 non-oracle safety gates

Owned files:
- `experiments/exp95_scored_object_hypotheses.py`
- `tests/test_exp95_scored_object_hypotheses.py`
- `notes/EXPERIMENTS.md`

Acceptance criteria:
- Results include explicit `safety_gates` per failure mode for `scored_non_oracle`.
- False-positive stress must show fewer false-stable accepts than observed naive at the configured gate point.
- Missing and merge stress must not be reported as recovery successes unless scored non-oracle is at least as safe as observed naive on accepted-wrong and false-stable counts; otherwise the result should be labeled `needs_identity_evidence` or equivalent.
- Tests include one aggregate quick run assertion for the safety-gate fields and one focused unit test for a missing-object case that should abstain rather than accept an unsafe synthetic support.

Smallest useful validation:

```bash
python3 -m pytest tests/test_exp95_scored_object_hypotheses.py -v
```

### 4. Disambiguate recursive proof fallback base relations

Owned files:
- `tensor_logic/proofs.py`
- `tests/test_proof_recursion.py` or `tests/test_tensor_logic_core.py`

Acceptance criteria:
- Recursive proof fallback uses the base relation implied by the rule body, or refuses fallback when multiple same-domain base relations are plausible.
- A new regression test builds a program with two same-domain base relations and proves that `reachable` does not traverse the unrelated relation.
- Existing recursive proof and `.tl` file proof tests continue to pass.

Smallest useful validation:

```bash
python3 -m pytest tests/test_proof_recursion.py tests/test_tensor_logic_core.py -k "recursive or proof" -v
```

Before planning this task, run:

```bash
python3 tools/code_index.py --lookup prove
```

### 5. Add timeout and input-size guard to the web workbench subprocess wrapper

Owned files:
- `web_workbench/server.py`
- `tests/test_web_workbench.py`

Acceptance criteria:
- `run_tensor_logic_action()` rejects oversized `source` payloads with a clear HTTP error before writing a temp file.
- The subprocess call has a bounded timeout and returns a structured timeout error instead of hanging.
- Existing query/prove/why-not behavior and CLI parity tests continue to pass.

Smallest useful validation:

```bash
python3 -m pytest tests/test_web_workbench.py -v
```

## Handoff Notes

- Product code changes made: none.
- Report path: `docs/overnight/2026-05-07-whole-portfolio-review/tensor-logic-risk-and-validation-review.md`
- Local commit: not created. `git add docs/overnight/2026-05-07-whole-portfolio-review/tensor-logic-risk-and-validation-review.md` failed because the sandbox could not create `/Users/jwalinshah/projects/tensor/tensor-logic/.git/worktrees/tensor-logic-risk-and-validation-review/index.lock`.
- PR created: none; external pushes and PR creation are out of scope for this queue item.
- Required final validation: `git status --short`
