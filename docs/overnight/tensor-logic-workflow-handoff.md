# tensor-logic workflow-handoff audit

Queue item: `tensor-logic-workflow-handoff`  
Focus area: `workflow-handoff`  
Worktree: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-logic-workflow-handoff`  
Branch: `codex/goal-tensor-logic-workflow-handoff`  
HEAD at audit time: `c9e2cfce206bb279a4348189726908fcb436d3dc`  
Remote: `origin https://github.com/jwalin-shah/tensor-logic.git`

## Purpose and state

`tensor-logic` is a Python research repo that has grown into a reusable `tensor_logic` package plus many experiment scripts. The README frames it as "Tensor Logic -> Cognition", a learning project, and also says the repo is now the shared `tensor_logic` library dependency for sibling projects such as `fafsa-engine`.

Initial state was clean:

- `git status --short --branch` printed `## codex/goal-tensor-logic-workflow-handoff`.
- `git status --short` printed no tracked or untracked changes.
- `git log --oneline -5` showed recent merged support/object-hypothesis work, with `c9e2cfc Merge pull request #46 from jwalin-shah/codex/SYM-220-scored-object-hypotheses` at HEAD.

This audit intentionally changed only this report file under `docs/overnight/`.

## Local evidence

- `README.md` describes the public story, headline transitive-closure result, limits on XOR/parity and non-reachability tasks, worker validation commands, and CPU demo commands.
- `pyproject.toml` declares package `tensor-logic`, Python `>=3.11`, runtime dependency `torch>=2.0`, dev extras `matplotlib`, `numpy`, and `pytest`, and setuptools package discovery for `tensor_logic*`.
- `.github/workflows/ci.yml` installs `python -m pip install -e ".[dev]"` and runs `python -m pytest tests/ -v` on PRs and pushes to `main`.
- `CLAUDE.md` says to run `python tools/code_index.py --lookup <RelevantSymbol>` before planning changes that touch `tensor_logic/`, but this shell has no `python` executable.
- `docs/SYMPHONY_RUN_PROTOCOL.md` defines branch, validation, PR, and result-recording expectations; it still says to treat a Linear issue as the final task contract.
- `docs/agents/issue-tracker.md` says this repo uses GitHub Issues and `gh`, which conflicts with the Linear wording in the Symphony protocol.
- `docs/agents/domain.md` expects optional `CONTEXT.md`, `CONTEXT-MAP.md`, and `docs/adr/`; `fd -H 'CONTEXT.md|CONTEXT-MAP.md|docs/adr' .` found none.
- `tensor_logic/__init__.py` re-exports a broad public API: closure, language, program, file loading, command execution, HTTP helpers, ingest, proof results, proof trees, provenance, repo-graph view, and rule parsing/evaluation.
- `tensor_logic/__main__.py` is the CLI entrypoint for `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`.
- `tensor_logic/file_format.py` parses `.tl` files with `domain`, `relation`, `fact`, `rule`, `query`, `prove`, and `include` statements; include cycles raise `ValueError`.
- `tensor_logic/execution.py` is the command execution layer and currently requires binary query/proof args.
- `tensor_logic/http_api.py` exposes local POST endpoints for `/ingest-python`, `/run`, `/query`, and `/prove` using `ThreadingHTTPServer`.
- `web_workbench/server.py` shells out with `sys.executable -m tensor_logic`, which is safer than the README text that says `python -m tensor_logic`.
- `tests/test_packaging_ci.py` explicitly guards the packaging, README validation commands, CI workflow, and PR handoff protocol.
- `tests/test_code_index.py` covers `tools/code_index.py`, but `test_cli_status_exits_0_when_fresh` rebuilds `tools/index.json` in the real repo root. That file is gitignored, but this is still a local side effect.
- `fd -e py -d 2 . tensor_logic tests experiments demos phase_training tools web_workbench | wc -l` found 167 Python files; `experiments/` alone has 98 Python files and `tests/` has 23 Python files.
- `fd -H 'requirements.*|uv.lock|poetry.lock|Pipfile|environment.yml|setup.cfg|tox.ini|noxfile.py|Makefile|Dockerfile|\\.env|\\.python-version|ruff.toml|mypy.ini|pytest.ini' .` found no lockfile, Makefile, tox/nox config, `.python-version`, or dedicated lint/type config.
- `.gitignore` excludes `.venv/`, `.uv/`, `.pytest_cache/`, `.ruff_cache/`, weight files (`*.pt`, `*.pth`), `/.cocoindex_code/`, and `tools/index.json`.
- `python --version` and `python tools/code_index.py --dump` both failed with exit 127: `/opt/homebrew/bin/bash: line 1: python: command not found`.
- `python3 --version` printed `Python 3.12.8`, even though project metadata says `>=3.11` and CI uses Python 3.11.
- `python3 tools/code_index.py --status` failed with exit 1 and printed `stale: index.json is missing or older than source files`.
- `git check-ignore -v tools/index.json` confirmed `tools/index.json` is ignored by `.gitignore:19`.
- `python3 -m pytest tests/test_packaging_ci.py -q` passed: `4 passed in 0.02s`.

## Handoff boundaries

Use these as the repo's current implementation boundaries:

- Public library API: `tensor_logic/__init__.py`.
- CLI: `tensor_logic/__main__.py`.
- File parser and include semantics: `tensor_logic/file_format.py`.
- Command execution shared by CLI, HTTP, and workbench: `tensor_logic/execution.py`.
- Local HTTP API: `tensor_logic/http_api.py`.
- Web shell: `web_workbench/server.py` and `web_workbench/static/`.
- Code-index planning rail: `tools/code_index.py` plus ignored `tools/index.json`.
- Support/stability research lane: `experiments/exp84_support_data.py`, `experiments/exp85_support_tl.py`, `experiments/exp86_support_baselines.py`, and tests `tests/test_exp84_support_data.py`, `tests/test_exp85_support_tl.py`, `tests/test_exp86_support_baselines.py`.
- Long-running or external-compute experiment discipline: `docs/RUN_PROTOCOL.md`, especially for Kaggle output persistence and adapter/result pointers.
- Worker and PR handoff discipline: `docs/SYMPHONY_RUN_PROTOCOL.md`, `docs/agents/issue-tracker.md`, and `tests/test_packaging_ci.py`.

## Risks and stale assumptions

1. Python command mismatch. README, CI, protocol docs, and CLAUDE examples use `python`, but this local shell only has `python3`. Future local workers copying the documented commands will fail before reaching tests.
2. Tracker-contract mismatch. `docs/SYMPHONY_RUN_PROTOCOL.md` says "Linear issue" while `docs/agents/issue-tracker.md` says GitHub Issues. A worker starting from repo docs alone could update/comment the wrong tracker or fail handoff.
3. Code index is both required and stale/missing. `CLAUDE.md` says lookup before planning `tensor_logic/` changes, but `python3 tools/code_index.py --status` exits 1 and `--lookup`/`--dump` would create ignored `tools/index.json`. That side effect is easy to miss in isolated worktrees.
4. Code-index tests mutate local ignored state. `tests/test_code_index.py::test_cli_status_exits_0_when_fresh` rebuilds `tools/index.json` in the real repo root. It is ignored, but it violates the clean-read expectation for validation-only workers.
5. Full dependency state is under-specified. `pyproject.toml` has dependencies, but there is no lockfile or `.python-version`; CI uses Python 3.11 while this worktree has Python 3.12.8.
6. The repo is dual-use: research notebook/script archive and shared package. `README.md` says it is a learning project, but also a shared library dependency. Handoffs should identify whether a task is product-library behavior, experiment result reproduction, or documentation only.
7. Generated experiment results are committed while large artifacts are intentionally ignored. `experiments/*_data/results*.json`, JSONL data, and PNGs are tracked; weights/checkpoints are ignored. Work orders must specify whether a result file is expected to change.

## Next safe tasks

### Task 1: Align worker command docs with local Python reality

Problem: `python` is not available here, but the repo's handoff docs rely on it.

Scope:

- Update docs only unless tests force tiny assertion changes.
- Candidate files: `README.md`, `CLAUDE.md`, `docs/SYMPHONY_RUN_PROTOCOL.md`, `web_workbench/README.md`, and `tests/test_packaging_ci.py`.

Acceptance criteria:

- Docs either use `python3` consistently for local worker commands, or explicitly define a preflight such as `python --version || python3 --version` and tell workers which executable to substitute.
- `web_workbench/README.md` matches `web_workbench/server.py`, which uses `sys.executable`.
- Packaging/CI documentation tests assert the chosen contract.

Validation:

- `python3 -m pytest tests/test_packaging_ci.py -q` should pass.
- `git status --short` should show only the intended docs/test edits.

### Task 2: Make code-index usage side-effect explicit and validation-safe

Problem: `CLAUDE.md` requires code-index lookup before planning `tensor_logic/` changes, but the index is stale/missing and the CLI can write ignored state during lookup/status test flows.

Scope:

- Candidate files: `tools/code_index.py`, `tests/test_code_index.py`, `CLAUDE.md`, and possibly `docs/superpowers/plans/2026-04-28-code-index.md` if docs need historical clarification.

Acceptance criteria:

- There is a documented no-surprise command for workers to check the index before planning.
- Tests that exercise CLI rebuild/status do not leave `tools/index.json` in the real repo root unless the test explicitly cleans it up.
- `CLAUDE.md` tells workers what to do when `tools/index.json` is missing or stale.

Validation:

- `python3 -m pytest tests/test_code_index.py -q` should pass and leave `git status --short --ignored` without a new `!! tools/index.json` unless the task deliberately changes that contract.
- `python3 tools/code_index.py --status` should have a documented expected result after a rebuild or when no index is present.

### Task 3: Resolve GitHub-vs-Linear handoff wording

Problem: Repo-local docs disagree about whether execution starts from Linear or GitHub Issues.

Scope:

- Candidate files: `docs/SYMPHONY_RUN_PROTOCOL.md`, `docs/agents/issue-tracker.md`, `docs/agents/triage-labels.md`, and `tests/test_packaging_ci.py`.

Acceptance criteria:

- One tracker is clearly declared as source of truth for this repo.
- PR handoff instructions name the exact comment/update destination.
- Tests guard the final wording so future protocol edits do not reintroduce mixed tracker assumptions.

Validation:

- `python3 -m pytest tests/test_packaging_ci.py -q` should pass.
- Optional: `gh issue list --state open --limit 1` only if credentials are available and the task explicitly asks to verify GitHub access.

### Task 4: Add a concise testing handoff page for support/stability workers

Problem: README and protocol mention full validation and a support fast path, but there is no single `docs/TESTING_FOR_AGENTS.md` style page that distinguishes cheap deterministic tests from neural smoke runs and generated result files.

Scope:

- Candidate files: new `docs/TESTING_FOR_AGENTS.md`, `README.md`, `docs/SYMPHONY_RUN_PROTOCOL.md`, and `tests/test_packaging_ci.py`.

Acceptance criteria:

- The page lists install, cheap package/docs validation, deterministic support tests, neural quick smoke, and full test commands.
- It states which commands may write generated result files and which are read-only.
- README/protocol link to the page rather than duplicating all command nuance.

Validation:

- `python3 -m pytest tests/test_packaging_ci.py -q` should pass.
- `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -q` should pass in an installed dev environment.

## Validation candidates

- Required queue validation: `git status --short`. Expected: exit 0 and show this report as the only untracked/changed file.
- Observed cheap validation: `python3 -m pytest tests/test_packaging_ci.py -q`. Result: passed, `4 passed in 0.02s`.
- Environment check: `python --version`. Observed fail, exit 127. This is a handoff risk, not a product test failure.
- Environment check: `python3 --version`. Observed pass, `Python 3.12.8`.
- Code-index status: `python3 tools/code_index.py --status`. Observed fail, exit 1, because `tools/index.json` is missing or stale. Do not run `--lookup`/`--dump` casually in this audit lane because it can create the ignored cache file.
- Full clean-checkout install candidate: `python3 -m pip install -e ".[dev]"`. Expected pass if network/package cache is available; not run in this read-only audit.
- Full CI candidate: `python3 -m pytest tests/ -v`. Expected pass after dev install; not run because the queue validation command is `git status --short` and full suite may run many research tests.
- Support deterministic candidate: `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`. Expected pass after dev install; not run.
- Neural support smoke candidate: `python3 experiments/exp86_support_baselines.py --quick`. Expected pass if `torch` is installed; intentionally outside CI per README/protocol.

## Non-goals

- No product code changes.
- No generated experiment result updates.
- No dependency installation.
- No external services, Kaggle jobs, model downloads, deploys, pushes, PR creation, or tracker updates.
- No attempt to run the full test suite or regenerate the code index in this audit.
- No decision on whether Tensor Logic's research claims are scientifically sufficient; this handoff audit only maps local workflow risk.

## Unknowns

- Whether the intended worker command contract should prefer `python`, `python3`, or a repo-local virtualenv executable.
- Whether the repo should remain GitHub-Issue-first or inherit the portfolio's Linear-first convention.
- Whether `tools/index.json` should be committed, generated in preflight, or kept ignored but recreated by agents as needed.
- Whether Python 3.12 is officially supported, despite CI pinning 3.11.
- Whether sibling repos depend on the broad public API in `tensor_logic/__init__.py`; changing exports may have cross-repo blast radius.
- Whether full `python3 -m pytest tests/ -v` currently passes in this exact worktree without installing dev dependencies.

## Worker handoff

Changed files:

- `docs/overnight/tensor-logic-workflow-handoff.md`

Commit SHA:

- Current HEAD before this report: `c9e2cfce206bb279a4348189726908fcb436d3dc`
- No commit was created because the queue item only requested a local audit report and PR creation is out of scope.

PR URL:

- None. PR creation is explicitly out of scope for this overnight audit queue item.

Blockers:

- None for writing the report.
- `python` executable is missing locally, so documented `python ...` commands fail unless workers use `python3` or a venv alias.
- Code-index status is stale/missing; rebuilding would create ignored local state and was not done in this read-only audit.
