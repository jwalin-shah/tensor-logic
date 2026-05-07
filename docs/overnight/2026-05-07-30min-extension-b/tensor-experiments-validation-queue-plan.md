# tensor-experiments Validation Queue Plan

Date: 2026-05-07
Branch: `codex/goal-tensor-experiments-validation-queue-plan`
Repo: `tensor-experiments` / `tensor-logic`
Scope: read-only validation and queue-readiness audit; no product code edited.

## Summary

The repo is queue-ready for normal implementation work if workers use the CI-equivalent `python3 -m pytest tests/ -v` command in this local environment. The full suite passed under `/usr/local/bin/python3` with Python 3.12.8: `140 passed in 32.74s`.

There are two handoff risks to fix before broadening the queue:

1. The repo docs and CI assert `python -m ...`, but this local shell has no `python` executable. `python3` works locally; GitHub Actions setup-python normally provides `python`.
2. The `rtk pytest tests/ -v` wrapper used a different interpreter path and reported `139 passed, 1 failed` on `tests/test_exp81.py::test_make_proposer_returns_valid_json`, while the same test and the full suite passed with `python3`. Treat this as an environment-selection risk, not a confirmed product failure.

No first-pass extension report, `runs/*/result.json`, `runs/*/handoff.md`, `CODEX_WORKPAD.md`, or repo-local queue `ISSUE.md` was present in this worktree. The JSON queue item named `items/tensor-experiments-validation-queue-plan/ISSUE.md`, but that path is missing here; the issue body was available from the Goal Pack prompt.

## Validation Commands Run

| Command | Result | Notes |
|---|---:|---|
| `git status --short --branch` | pass | Initial output only showed branch `codex/goal-tensor-experiments-validation-queue-plan`. |
| `rtk pytest tests/ -v` | fail | Wrapper reported `139 passed, 1 failed`; failing test was `tests/test_exp81.py::test_make_proposer_returns_valid_json`, with `httpx` / connection-pool stack in output. |
| `python -m pytest tests/test_exp81.py::test_make_proposer_returns_valid_json -vv -s` | blocked | Exit 127: `/opt/homebrew/bin/bash: line 1: python: command not found`. |
| `python3 -m pytest tests/test_exp81.py::test_make_proposer_returns_valid_json -vv -s` | pass | `1 passed in 5.62s` under Python 3.12.8. |
| `python3 -m pytest tests/ -v` | pass | `140 passed in 32.74s` under Python 3.12.8. |
| `git status --short --ignored` | pass with ignored artifacts | After tests: only ignored `.pytest_cache/` and `tools/index.json`. |

Queue validation command for this queue item:

```bash
git status --short
```

## Concrete File-Path Observations

1. `README.md` documents the worker validation contract as:
   `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`. Locally, `python` is missing, so worker instructions should either require setup-python-like environments or include a `python3` fallback.
2. `.github/workflows/ci.yml` runs on Ubuntu with Python 3.11, installs `.[dev]`, and runs only `python -m pytest tests/ -v`; there is no lint, type check, docs artifact check, or experiment-result consistency check.
3. `pyproject.toml` declares runtime dependency `torch>=2.0` and dev extras `matplotlib`, `numpy`, and `pytest`; it does not pin versions or declare `transformers`/`mlx_lm`, even though LM-backed experiment paths import them opportunistically.
4. `pyproject.toml` only configures pytest `testpaths = ["tests"]`; the repo has no `tool.ruff`, `tool.mypy`, `tox`, `nox`, or Makefile validation surface.
5. `tests/test_packaging_ci.py` usefully locks the README, CI, and pyproject validation contract, but it asserts command strings rather than proving the local shell can run `python`.
6. `tests/test_code_index.py` rebuilds `tools/index.json` during validation. `tests/test_code_index.py::test_index_json_is_gitignored` guards that this generated artifact stays ignored, which matched the observed `git status --short --ignored`.
7. `CLAUDE.md` says to run `python tools/code_index.py --lookup <RelevantSymbol>` before plans that touch `tensor_logic/`; this report does not plan a direct `tensor_logic/` code edit, but future implementation tasks touching that package should follow it.
8. `experiments/exp78_rule_induction.py` falls back to all relations when `transformers` is unavailable, but if `transformers` is installed it may load `Qwen/Qwen2.5-0.5B-Instruct` via `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`; that path can need network/cache and is not isolated in CI.
9. `experiments/exp81_optimize_rule_induction.py` wraps `lm_prune()` in `make_proposer()`, so `tests/test_exp81.py::test_make_proposer_returns_valid_json` is sensitive to whether the default model is locally cached under the interpreter used by the test runner.
10. `tests/test_exp81.py` only smoke-tests proposer JSON shape; it does not pin offline mode, model-cache expectations, or deterministic fallback behavior for environments without model access.
11. `docs/RUN_PROTOCOL.md` requires remote experiment runs to persist `manifest.json`, `failures.jsonl`, adapter pointers, and committed `experiments/expN_data/results.json`; this repo currently has committed results JSON for exp78, exp79, and exp83 only.
12. `experiments/exp78_data/results.json` supports the README/notes claim for rule induction: all listed targets have `f1: 1.0`, `semantic_equiv: 1.0`, and `rule_found: true`.
13. `experiments/exp79_data/results.json` records easy and medium modes at 10/10 answered, hard at 4/10 answered, very_hard at 7/10 answered, and adversarial attempts correctly rejected; this is good queue context but not enforced by pytest.
14. `experiments/exp83_slot_data/results.json` has `gates.probe: false`, `gates.tl_only: true`, and `gates.e2e: true`; any future claim that exp83 passed all gates would be stale.
15. `notes/EXPERIMENTS.md` says exp80 external official worked-example validation is still pending, despite internal synthetic invariants passing; this is a product-judgment blocker for using FAFSA numbers as an external benchmark.
16. `experiments/exp80_validate_synthetic.py` writes `experiments/exp80_spot_check_cases.json`, and that JSON is committed with named cases. Tests cover 25 seeded families, not the full 1,015-family synthetic run or external aid-estimator spot checks.
17. `docs/SYMPHONY_RUN_PROTOCOL.md` asks workers to record PR URL, commit SHA, validation command, and blockers. This queue item forbids pushes/PR creation, so the correct handoff is this committed report plus local commit SHA.
18. `items/tensor-experiments-validation-queue-plan/ISSUE.md` is absent from the worktree. If future workers are launched from repo files rather than Goal Pack prompts, this queue item cannot be reproduced from local files alone.

## Validation Gaps

- Local docs/CI portability gap: `python` is assumed but not available in this shell. Use `python3` locally or ensure a managed environment supplies `python`.
- Interpreter drift gap: `rtk pytest` selected a different Python stack than `python3 -m pytest`; one stack failed the LM-proposer smoke while Python 3.12.8 passed.
- Offline determinism gap: LM-backed paths in `exp78`/`exp81` can depend on local model caches, `transformers`, and network availability.
- Research artifact gap: committed result JSON is present for exp78/79/83, but the run protocol expects richer remote manifests/failure logs/adapter pointers for remote jobs.
- Claim enforcement gap: pytest enforces API behavior and packaging/CI strings, but not README headline tables, notes claims, exp79 JSON summaries, or exp83 gate status.
- External validation gap: exp80 remains internally validated only; official FAFSA worked examples or aid-estimator spot checks are still pending.
- Handoff gap: the queue issue file referenced by the runner is missing from this worktree.

## Known Blockers

- No external services, pushes, PR creation, or tracker updates were allowed for this queue item.
- `python -m ...` cannot run locally because `python` is missing from PATH.
- `rtk pytest` currently does not provide a reliable pass/fail signal for this repo unless its interpreter selection is pinned or documented.
- The absence of `items/tensor-experiments-validation-queue-plan/ISSUE.md` means this worktree is not self-contained for future replay.
- Exp80 cannot be promoted to an external benchmark without human/product judgment on official validation sources.

## Safe Next Implementation Tasks

1. **Make local validation command portable**
   - Owned files: `README.md`, `CLAUDE.md`, `.github/workflows/ci.yml`, `tests/test_packaging_ci.py`.
   - Acceptance criteria: docs mention the canonical CI command and a local fallback when `python` is absent; packaging test asserts both without weakening CI.
   - Smallest useful validation: `python3 -m pytest tests/test_packaging_ci.py -v`.

2. **Pin or document the `rtk pytest` interpreter path**
   - Owned files: `CLAUDE.md`, `README.md`, optionally a repo-local script such as `tools/validate.py` if the repo wants one command.
   - Acceptance criteria: `rtk pytest` ambiguity is documented or replaced by a deterministic local command; no product behavior changes.
   - Smallest useful validation: run `rtk pytest tests/test_exp81.py::test_make_proposer_returns_valid_json -v` and `python3 -m pytest tests/test_exp81.py::test_make_proposer_returns_valid_json -v`, then record expected behavior.

3. **Add offline fallback coverage for LM proposer paths**
   - Owned files: `experiments/exp78_rule_induction.py`, `experiments/exp81_optimize_rule_induction.py`, `tests/test_exp81.py`.
   - Acceptance criteria: tests prove `make_proposer()` returns valid JSON when `transformers` is missing or model loading fails, without needing network/model cache.
   - Smallest useful validation: `python3 -m pytest tests/test_exp81.py -v`.

4. **Add experiment-result consistency checks**
   - Owned files: `tests/test_experiment_results.py`, `experiments/exp78_data/results.json`, `experiments/exp79_data/results.json`, `experiments/exp83_slot_data/results.json`.
   - Acceptance criteria: tests assert the minimum gates that docs rely on, including exp78 all targets found, exp79 easy/medium success, adversarial rejection, and exp83 `probe` gate status documented as false.
   - Smallest useful validation: `python3 -m pytest tests/test_experiment_results.py -v`.

5. **Make queue handoff replayable from repo-local files**
   - Owned files: `docs/overnight/2026-05-07-30min-extension-b/`, optionally `docs/overnight/README.md` or a queue manifest if this repo adopts one.
   - Acceptance criteria: future workers can find queue reports, validation commands, blockers, and source issue text without the hidden Goal Pack prompt.
   - Smallest useful validation: `git status --short` plus a grep check such as `rg "tensor-experiments-validation-queue-plan" docs/overnight`.

## Handoff

- Files changed: `docs/overnight/2026-05-07-30min-extension-b/tensor-experiments-validation-queue-plan.md`.
- Product code changed: none.
- Current base commit before this report: `5565791a8c888511bbbff1107d2d357164f88baa`.
- Local commit: not created. `git add docs/overnight/2026-05-07-30min-extension-b/tensor-experiments-validation-queue-plan.md` failed because the git worktree index is outside the writable sandbox: `fatal: Unable to create '/Users/jwalinshah/projects/tensor/experiments/.git/worktrees/tensor-experiments-validation-queue-plan/index.lock': Operation not permitted`.
- PR URL: none; PR creation is out of scope for this queue item.
- Final validation command: `git status --short`.
- Final validation result: exit 0 with output `?? docs/overnight/`, because git staging/commit is sandbox-blocked and the report remains untracked.
