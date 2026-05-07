# tensor-logic validation-map audit

Queue item: `tensor-logic-validation-map`
Date: 2026-05-07
Repo: `tensor-logic`
Branch: `codex/goal-tensor-logic-validation-map`
HEAD at audit start: `c9e2cfce206bb279a4348189726908fcb436d3dc`

## Scope and decision log

This audit is validation-map only. I did not edit product code, generated data,
secrets, workflows, experiment scripts, package metadata, or external trackers.
The only intended repo mutation is this report at
`docs/overnight/tensor-logic-validation-map.md`.

Decision: use `python3` for local proof commands because this worktree's shell
does not expose a `python` executable. The repo and CI documentation still use
`python`; that is valid in GitHub Actions because `actions/setup-python` places
`python` on `PATH`, but it is not valid in this local Homebrew-style shell.

Decision: run experiment CLIs that write result JSON only with an explicit
temporary `--output` path outside the repo. This avoids mutating tracked
`experiments/*_data/results_quick.json` files during a read-only audit.

## Repo purpose and current state

Purpose, from `README.md`: this is a Python research/package repo for Tensor
Logic experiments and reusable `tensor_logic` reasoning utilities. It contains
the reusable package under `tensor_logic/`, runnable demos under `demos/`, a
large experiment arc under `experiments/`, support/stability work under
`experiments/exp84` through `experiments/exp95`, and regression tests under
`tests/`.

Branch and cleanliness observations:

- `git branch --show-current` -> `codex/goal-tensor-logic-validation-map`
- Initial `git status --short` -> no output, clean tracked state.
- `git status --short --ignored` after running tests showed ignored local state:
  `.pytest_cache/` and `tools/index.json`.
- `tools/index.json` is intentionally ignored by `.gitignore`, and
  `tests/test_code_index.py` asserts that ignore contract.
- Final required validation command is `git status --short`; after writing this
  report it showed only `?? docs/overnight/`, the new untracked overnight report
  directory.

## Local evidence

Validation-relevant files and observations:

- `CLAUDE.md` declares the project test suite as `pytest tests/ -v` and says
  agents should run `python tools/code_index.py --lookup <RelevantSymbol>` before
  implementation plans that touch `tensor_logic/`.
- `pyproject.toml` declares package name `tensor-logic`, Python `>=3.11`,
  runtime dependency `torch>=2.0`, dev extras `matplotlib>=3.7`, `numpy>=1.24`,
  and `pytest>=8.0`, with pytest `testpaths = ["tests"]`.
- `README.md` documents clean worker validation:
  `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`.
- `README.md` also documents a support/stability fast path:
  `python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`
  plus `python experiments/exp86_support_baselines.py --quick`.
- `.github/workflows/ci.yml` runs Python 3.11 on pull requests and pushes to
  `main`, installs `".[dev]"`, then runs `python -m pytest tests/ -v`.
- `docs/SYMPHONY_RUN_PROTOCOL.md` repeats the full validation command, the
  support/stability fast path, the `exp86 --quick` smoke command, and the rule
  that docs-only changes should state why no code test was needed.
- `docs/superpowers/plans/2026-05-05-support-stability.md` explains the
  validation contract for the support/stability lane: deterministic tests in
  fast CI, neural `--quick` smoke checks outside CI, and exact TL gates for the
  V1 evaluation.
- `tests/conftest.py` injects the repo root into `sys.path`, so local tests can
  import the package without requiring an editable install.
- `tests/test_packaging_ci.py` asserts that `pyproject.toml`, README validation
  docs, Symphony protocol docs, and `.github/workflows/ci.yml` stay aligned.
- `tests/test_code_index.py` exercises `tools/code_index.py`, including CLI
  lookup/status behavior, and intentionally rebuilds the ignored index in tests.
- `tests/test_exp84_support_data.py` and `tests/test_exp85_support_tl.py` are
  the cheap deterministic support/stability fast path.
- `tests/test_exp86_support_baselines.py` validates neural baseline tensorization,
  masking, loss reduction, and the `run_baselines(quick=True)` metric schema.
- `tests/test_exp87_support_eval.py` validates V1 eval split coverage and gate
  schema through direct function calls with a `tmp_path` result file.
- `tests/test_exp88_support_noisy_relations.py` through
  `tests/test_exp95_scored_object_hypotheses.py` cover later robustness,
  uncertainty, detector, object-hypothesis, and scored-hypothesis experiment
  layers using temporary output paths.
- `tests/test_tensor_logic_core.py` is the largest test file and covers package
  core behavior, CLI/HTTP parity, proof/negative-proof JSON semantics, repo
  ingestion, proof tree rendering, includes, REPL helpers, and TL file examples.
- `tests/test_web_workbench.py` covers the web workbench query and why-not
  integration without starting a long-running server.
- `tools/code_index.py` writes `tools/index.json` when stale; that file is
  ignored and should not be treated as source.
- `experiments/exp87_support_eval.py` and `experiments/exp88` through `exp95`
  write default results under tracked `experiments/*_data/` directories unless
  `--output` is supplied.
- `web_workbench/server.py` has a local server entrypoint, but current tests call
  helper functions directly; no browser/server validation is required for this
  docs-only audit.

## Commands run

Repository and structure:

| Command | Result |
| --- | --- |
| `llm-tldr tree .` | Passed. Confirmed package, tests, docs, demos, experiments, web workbench, and data-output directories. |
| `git branch --show-current` | Passed. Branch is `codex/goal-tensor-logic-validation-map`. |
| `git status --short` | Passed. Initial tracked state clean. |
| `git rev-parse HEAD` | Passed. `c9e2cfce206bb279a4348189726908fcb436d3dc`. |
| `rg --files .github` | Passed. Found `.github/workflows/ci.yml`. |
| `git status --short --ignored` | Passed. Showed ignored `.pytest_cache/` and `tools/index.json`. |

Local interpreter and dependency surface:

| Command | Result |
| --- | --- |
| `python -m pytest --collect-only -q` | Failed before pytest: `/opt/homebrew/bin/bash: line 1: python: command not found`. |
| `python -m pytest tests/test_packaging_ci.py -v` | Failed before pytest for the same missing `python` executable. |
| `which python3` | Passed. `/usr/local/bin/python3`. |
| `python3 --version` | Passed. `Python 3.12.8`. |
| `python3 -m pytest --version` | Passed. `pytest 9.0.3`. |
| `uv --version` | Passed. `uv 0.11.5 (Homebrew 2026-04-08 aarch64-apple-darwin)`. |
| `python3 -m pip check` | Passed with pip cache ownership warning; output included `No broken requirements found.` |

Pytest validation:

| Command | Result |
| --- | --- |
| `python3 -m pytest --collect-only -q` | Passed. `198 tests collected in 21.85s`. |
| `python3 -m pytest tests/test_packaging_ci.py -v` | Passed. `4 passed in 0.02s`. |
| `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v` | Passed. `16 passed in 0.08s`. |
| `python3 -m pytest tests/test_tensor_logic_core.py tests/test_reason.py tests/test_web_workbench.py -q` | Passed. `66 passed in 15.37s`. |
| `python3 -m pytest tests/test_exp86_support_baselines.py -v` | Passed. `5 passed in 1.83s`. |
| `python3 -m pytest tests/ -v` | Passed. `198 passed in 33.32s`. |

Runtime smoke checks:

| Command | Result |
| --- | --- |
| `python3 tools/code_index.py --status` | Passed. Reported `fresh: .../tools/index.json`. |
| `python3 experiments/exp86_support_baselines.py --quick` | Passed. Printed deterministic JSON with `quick: true`, `max_objects: 8`, MLP ID accuracy `0.8947`, DeepSets ID accuracy `0.8947`, and OOD accuracies around `0.77`. |
| `python3 -m tensor_logic run examples/code_dependencies.tl` | Passed. Printed `depends_on(worker, models) = True` and a proof chain through `worker -> api -> db -> models`. |
| `python3 experiments/exp87_support_eval.py --quick --output /private/tmp/tensor-logic-exp87-quick.json` | Passed. Printed `thesis: "passed"` and `gates.v1_passed: true`; wrote outside the repo. |

## Validation surface map

Primary CI/full validation:

- Command in docs and CI: `python -m pip install -e ".[dev]"` then
  `python -m pytest tests/ -v`.
- Local equivalent that passed in this worktree: `python3 -m pytest tests/ -v`.
- Coverage: 198 tests across package core, proof engine, CLI/HTTP parity,
  web workbench helpers, code index, optimization helpers, FAFSA experiment,
  support/stability experiments, neural baselines, and scored object hypotheses.
- Local caveat: `tests/conftest.py` makes package imports work from source, so
  the local full-suite pass does not prove the editable install step by itself.
  `tests/test_packaging_ci.py` does check package metadata and CI command text.

Support/stability cheap path:

- Command in README/protocol: `python -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`.
- Local equivalent that passed: `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v`.
- Coverage: deterministic generator invariants plus TL stability/retraction
  behavior against hand-built and generated support scenes.
- Expected use: cheapest proof for changes limited to `experiments/exp84` or
  `experiments/exp85`.

Neural baseline smoke path:

- Command in README/protocol: `python experiments/exp86_support_baselines.py --quick`.
- Local equivalent that passed: `python3 experiments/exp86_support_baselines.py --quick`.
- Coverage: quick MLP and DeepSets training/eval on generated object tables.
- Expected use: required for changes touching `experiments/exp86_support_baselines.py`.
- Caveat: this script prints JSON and does not write a repo result file.

V1 evaluation CLI path:

- Candidate command: `python3 experiments/exp87_support_eval.py --quick --output /private/tmp/tensor-logic-exp87-quick.json`.
- Result: passed with `v1_passed: true`.
- Expected use: proof that the end-to-end support/stability gate still computes
  without mutating tracked quick-result files.
- Caveat: default command without `--output` writes to
  `experiments/exp87_support_data/results_quick.json`, which is tracked in git.

Robustness/evaluation result writers:

- `experiments/exp88_support_noisy_relations.py`
- `experiments/exp89_support_primitive_confidence.py`
- `experiments/exp90_support_repair_sweep.py`
- `experiments/exp91_interval_support_uncertainty.py`
- `experiments/exp92_pixel_abstain_recover.py`
- `experiments/exp93_detector_calibration_stress.py`
- `experiments/exp94_object_hypothesis_layer.py`
- `experiments/exp95_scored_object_hypotheses.py`

These scripts expose `--quick` and `--output` according to grep/read evidence.
Their tests exercise the run functions with `tmp_path` outputs, but this audit
did not run every CLI. For read-only/nightly audits, prefer:

```bash
python3 experiments/<script>.py --quick --output /private/tmp/<script>-quick.json
```

Package/core smoke path:

- `python3 -m tensor_logic run examples/code_dependencies.tl` passed and proves
  the installed-source CLI path can parse/run a TL file and print proof output.
- Targeted package tests in `tests/test_tensor_logic_core.py`, `tests/test_reason.py`,
  and `tests/test_web_workbench.py` passed as a 66-test core/web subset.

Code index validation:

- `python3 tools/code_index.py --status` passed and reported a fresh ignored
  `tools/index.json`.
- `tests/test_code_index.py` covers rebuild, status, lookup success, and lookup
  miss behavior.
- Risk: this is an operational helper, not part of package metadata or CI
  artifacts. Its ignored output can appear in dirty-state inspections only when
  `--ignored` is used.

Missing validation surfaces:

- No configured lint command was found in `pyproject.toml`, `.github/workflows/ci.yml`,
  README, CLAUDE, or Symphony protocol.
- No configured type-check command was found.
- No coverage threshold or coverage command was found.
- No tox/nox/pre-commit/Makefile/justfile validation wrapper was found.

## Risks and stale assumptions

1. Local docs use `python`, but this worker shell only has `python3`.
   Local agents following README literally will fail before pytest. CI likely
   still works because `actions/setup-python` provides `python`, but the repo
   lacks a cross-environment wrapper that normalizes this.

2. The full-suite local pass does not prove the documented install step.
   Because `tests/conftest.py` inserts the repo root into `sys.path`, tests can
   pass without `python -m pip install -e ".[dev]"`. The CI job does install,
   and `tests/test_packaging_ci.py` checks the install command is present, but
   this audit did not mutate the environment by running an editable install.

3. Many experiment CLIs write tracked result files by default.
   `exp87` through `exp95` have committed `results.json` and `results_quick.json`
   files under `experiments/*_data/`. Running a default quick command during an
   audit or implementation can create noisy tracked diffs even when the code is
   unchanged. Supplying `--output` avoids this.

4. There is no lint/type/format gate.
   The test suite is green, but validation has no automated guard for style,
   import hygiene, static typing regressions, dead code, or formatting drift.
   That may be acceptable for a research repo, but it should be explicit.

5. CI uses Python 3.11, local validation used Python 3.12.8.
   `pyproject.toml` permits `>=3.11`, so this is allowed, but it means the
   local pass is not an exact CI-version reproduction. Potential version-sensitive
   torch/numpy behavior remains an unknown without running Python 3.11 locally.

6. Neural quick checks are deterministic enough for smoke validation, but they
   are still training loops.
   The `exp86 --quick` and `exp87 --quick` commands passed quickly here, but
   they depend on torch CPU behavior and seeded training. They are more expensive
   and potentially more brittle than pure deterministic unit tests.

7. The code index helper writes ignored local state.
   `tools/code_index.py --status` is read-only when fresh, but `--lookup` and
   tests may rebuild `tools/index.json`. That is intentional and ignored, yet it
   can confuse workers who inspect only filesystem state without checking
   `.gitignore`.

## Next safe work

### Task 1: Add a repo-local validation wrapper

Acceptance criteria:

- Add a small `Makefile`, `justfile`, or script that resolves `python` vs
  `python3` once and exposes `test`, `test-fast`, `test-exp86-quick`, and
  `test-exp87-quick-temp` commands.
- Wrapper commands must not write tracked experiment result files by default.
- README and `docs/SYMPHONY_RUN_PROTOCOL.md` point workers at the wrapper while
  preserving raw commands for CI.

Validation:

```bash
<wrapper> test-fast
<wrapper> test-exp86-quick
<wrapper> test-exp87-quick-temp
git status --short
```

Expected status: wrapper commands pass; `git status --short` shows only intended
wrapper/docs changes.

### Task 2: Add CLI smoke tests for result-writing experiment scripts

Acceptance criteria:

- Add parametrized pytest coverage that invokes `exp87` through `exp95` CLIs
  with `--quick --output <tmp_path>/results.json` where practical.
- Assert each CLI exits 0, writes the requested output path, and does not write
  to the default tracked `experiments/*_data/` path during the test.
- Keep runtime bounded; split slow scripts or mark them if necessary.

Validation:

```bash
python3 -m pytest tests/test_experiment_cli_outputs.py -v
python3 -m pytest tests/ -v
git status --short
```

Expected status: new CLI tests and full suite pass; no tracked result JSON files
change during the tests.

### Task 3: Make the install contract observable locally

Acceptance criteria:

- Add a packaging smoke test or docs step that proves a clean editable install
  can import `tensor_logic` without relying on `tests/conftest.py`.
- Prefer a temporary virtual environment or `uv run`-based smoke that does not
  leave repo-local env artifacts except ignored caches.
- Record the expected Python version and interpreter command in README/protocol.

Validation:

```bash
python3 -m pytest tests/test_packaging_ci.py -v
<new install smoke command>
git status --short
```

Expected status: packaging tests pass; install smoke imports package and runs a
minimal CLI command; no product files change.

### Task 4: Decide and encode lint/type policy

Acceptance criteria:

- Either add a lightweight lint/format/type gate, or document explicitly that
  this research repo currently has no lint/type gate and tests are the only
  enforced validation.
- If a tool is added, add it to `pyproject.toml`, CI, README, and
  `tests/test_packaging_ci.py` so the validation contract remains aligned.
- Keep the first tool scoped and boring; avoid broad style churn in the same PR.

Validation candidates:

```bash
python3 -m pytest tests/test_packaging_ci.py -v
python3 -m pytest tests/ -v
<lint/type command if added>
```

Expected status: packaging contract and full suite pass; lint/type status is
either enforced or explicitly non-goal.

## Validation command candidates

| Situation | Command | Expected status |
| --- | --- | --- |
| Required queue validation | `git status --short` | Passed after report creation, exit 0, output `?? docs/overnight/`. |
| CI-equivalent in GitHub Actions | `python -m pip install -e ".[dev]" && python -m pytest tests/ -v` | Expected pass in CI Python 3.11; not locally runnable here because `python` is missing. |
| Local full suite | `python3 -m pytest tests/ -v` | Passed locally, `198 passed in 33.32s`. |
| Local collection check | `python3 -m pytest --collect-only -q` | Passed locally, `198 tests collected in 21.85s`. |
| Packaging/docs alignment | `python3 -m pytest tests/test_packaging_ci.py -v` | Passed locally, `4 passed in 0.02s`. |
| Support/stability deterministic fast path | `python3 -m pytest tests/test_exp84_support_data.py tests/test_exp85_support_tl.py -v` | Passed locally, `16 passed in 0.08s`. |
| Neural baseline unit tests | `python3 -m pytest tests/test_exp86_support_baselines.py -v` | Passed locally, `5 passed in 1.83s`. |
| Neural baseline CLI smoke | `python3 experiments/exp86_support_baselines.py --quick` | Passed locally; prints JSON only. |
| V1 evaluation CLI smoke without repo mutation | `python3 experiments/exp87_support_eval.py --quick --output /private/tmp/tensor-logic-exp87-quick.json` | Passed locally; `v1_passed: true`. |
| Package CLI smoke | `python3 -m tensor_logic run examples/code_dependencies.tl` | Passed locally; query true and proof printed. |
| Code index freshness | `python3 tools/code_index.py --status` | Passed locally; reported fresh ignored index. |
| Dependency consistency | `python3 -m pip check` | Passed locally with pip cache warning; no broken requirements. |

## Non-goals

- No product-code edits.
- No changes to package metadata, CI workflow, tests, demos, experiments, or
  generated result JSON files.
- No external service calls, deploys, pushes, PR creation, or tracker updates.
- No full remote/CI validation.
- No claim that Python 3.11 CI was reproduced locally; local interpreter was
  Python 3.12.8.
- No benchmark accuracy claim beyond the smoke results printed by the commands
  listed above.

## Unknowns

- Whether a fresh Python 3.11 virtualenv can install `".[dev]"` without network
  or platform issues in this exact workspace.
- Whether GitHub Actions is currently green on remote `main`; this audit did
  not query GitHub or external services.
- Whether all older `experiments/exp1` through `exp83` standalone scripts still
  run. The test suite covers selected experiment functions, not every historical
  script entrypoint.
- Whether lint/type validation should be added or intentionally omitted for the
  research workflow.
- Whether committed `experiments/*_data/results_quick.json` files are meant to
  be regenerated by humans only, or can be regenerated by automated workers with
  a stable protocol.

## Handoff

Changed files:

- `docs/overnight/tensor-logic-validation-map.md`

Commit SHA:

- No new commit created by this audit. Base/current HEAD is
  `c9e2cfce206bb279a4348189726908fcb436d3dc`.

PR URL:

- None. PR creation is out of scope for this overnight read-only audit.

Required validation:

- `git status --short` -> exit 0, output `?? docs/overnight/`.

Blockers:

- None for the report.
- Local command caveat: raw `python ...` validation commands fail in this shell
  because `python` is not on `PATH`; use `python3` locally or run under CI.
