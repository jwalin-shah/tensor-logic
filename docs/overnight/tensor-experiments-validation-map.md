# tensor-experiments validation-map audit

Date: 2026-05-07
Queue item: `tensor-experiments-validation-map`
Repo path: `/Users/jwalinshah/projects/agent-stack/.agent-stack-worktrees/2026-05-07-overnight-marathon/tensor-experiments-validation-map`
Branch: `codex/goal-tensor-experiments-validation-map`
HEAD observed before report: `5565791a8c888511bbbff1107d2d357164f88baa`

## Purpose and State

`tensor-experiments` is a Python research repository around tensor-logic reasoning, demos, and reusable `tensor_logic` package code. `README.md` frames it as a learning/research project and notes that the extracted `tensor_logic` package is also a shared dependency for sibling projects such as `fafsa-engine`.

Initial state was clean:

- `git status --short --branch` produced `## codex/goal-tensor-experiments-validation-map`.
- `git rev-parse --abbrev-ref HEAD` produced `codex/goal-tensor-experiments-validation-map`.
- `git rev-parse HEAD` produced `5565791a8c888511bbbff1107d2d357164f88baa`.
- Required queue validation command `git status --short` exited 0 with no output before the report write.

After running tests and code-index validation, ignored local state existed:

- `git status --short --ignored` showed `!! .pytest_cache/`.
- `git status --short --ignored` showed `!! tools/index.json`.
- `.gitignore` explicitly ignores `.pytest_cache/`, `.ruff_cache/`, `.uv/`, `tools/index.json`, build outputs, Python bytecode, virtualenvs, model weights, and local editor files.

## Declared Validation Contracts

The repo has one canonical automated validation path and several narrower proof commands.

- `pyproject.toml` declares package name `tensor-logic`, Python `>=3.11`, runtime dependency `torch>=2.0`, and dev extras `matplotlib>=3.7`, `numpy>=1.24`, and `pytest>=8.0`.
- `pyproject.toml` configures pytest with `testpaths = ["tests"]`.
- `README.md` says Symphony workers should run `python -m pip install -e ".[dev]"` and `python -m pytest tests/ -v`.
- `CLAUDE.md` gives the local test suite as `pytest tests/ -v`.
- `.github/workflows/ci.yml` runs on pull requests and pushes to `main`, sets up Python 3.11, installs `python -m pip install -e ".[dev]"`, and runs `python -m pytest tests/ -v`.
- `tests/test_packaging_ci.py` enforces the packaging/CI/README contract by asserting the pyproject dependency names, workflow triggers, install command, and test command.
- `docs/SYMPHONY_RUN_PROTOCOL.md` says issue workers should validate with the smallest direct command and record PR URL, commit SHA, validation command, and blockers.
- `docs/RUN_PROTOCOL.md` covers remote/Kaggle experiment runs and requires manifests, output pointers, and avoiding committed weights.

## Commands Run

Validation and discovery commands run during this audit:

- `llm-tldr tree .` completed and showed the main surfaces: `tensor_logic/`, `tests/`, `experiments/`, `demos/`, `phase_training/`, `web_workbench/`, `tools/`, `docs/`, and `notes/`.
- `llm-tldr search "pytest|ruff|mypy|lint|python -m|unittest|tox|nox|uv|pip" .` found the README/CI pytest contract, demo `uv run` commands, `.gitignore` cache entries, and many docs plan validation commands.
- `python --version` failed locally with `/opt/homebrew/bin/bash: line 1: python: command not found`.
- `python -m pytest tests/ -v` failed locally with `/opt/homebrew/bin/bash: line 1: python: command not found`.
- `python -m pip show torch pytest numpy matplotlib` failed locally with `/opt/homebrew/bin/bash: line 1: python: command not found`.
- `python3 --version` passed with `Python 3.12.8`.
- `python3 -m pytest tests/ -v` passed: `140 passed in 31.83s`.
- `python3 -m pip show torch pytest numpy matplotlib` found local installs: torch 2.11.0, pytest 9.0.3, numpy 1.26.4, and matplotlib installed under the Python 3.12 framework.
- `uv --version` passed with `uv 0.11.5 (Homebrew 2026-04-08 aarch64-apple-darwin)`.
- `python3 tools/code_index.py --status` passed with `fresh: .../tools/index.json`.
- `rg --files -g 'requirements*.txt' -g 'uv.lock' -g 'poetry.lock' -g 'Pipfile*' -g 'tox.ini' -g 'noxfile.py' -g 'Makefile' -g 'setup.cfg' -g '.pre-commit-config.yaml' -g 'mypy.ini' -g '.flake8'` exited 1 with no matches.

## Test Coverage Map

The current suite is useful and fast enough for every ordinary repo change, but it is not a full experiment reproducibility suite.

- `tests/test_tensor_logic_core.py` is the broadest core package suite. It covers named-index joins, recursive fixpoints, TL file parsing, CLI query/prove/run behavior, positive and negative proof JSON, provenance/ranking, repo graph helpers, HTTP helper parity, source-backed facts, stratified negation, include directives, and error semantics.
- `tests/test_code_index.py` covers `tools/code_index.py` extraction, staleness checks, CLI lookup/status behavior, and asserts `tools/index.json` is gitignored.
- `tests/test_packaging_ci.py` is a meta-test for the worker validation contract in `pyproject.toml`, `.github/workflows/ci.yml`, and `README.md`.
- `tests/test_exp79.py` covers the LeWM/TL experiment helpers from `experiments/exp79_lewm_tl.py`: synthetic frame generation, geometric relations, encoder/predictor shapes, JEPA loss/backward stability, relation probe shape/metrics, and TL retraction wiring.
- `tests/test_exp80.py` covers FAFSA SAI arithmetic invariants in `experiments/exp80_fafsa_kb.py` and synthetic family cases from `experiments/exp80_validate_synthetic.py`.
- `tests/test_exp81.py` covers rule-induction optimizer helpers from `experiments/exp81_optimize_rule_induction.py`: artifact parsing, miss explanations, evaluator scoring, ASI population, and proposer JSON shape.
- `tests/test_optimize.py` covers Pareto frontier logic and the optimize loop in `tensor_logic/optimize.py`.
- `tests/test_reason.py` covers observation-to-TL fact conversion and `tensor_logic.reason` query evaluation.
- `tests/test_web_workbench.py` covers workbench request plumbing and JSON parity with `tensor_logic.http_api`.
- `tests/test_proof_recursion.py` covers positive and negative recursive proof behavior on a small graph.

## Validation Candidate Matrix

| Command | Observed or expected status | Notes |
|---|---:|---|
| `git status --short` | Pass, exits 0 | Queue-required command. It was empty before report write; after this report it should show only `?? docs/overnight/tensor-experiments-validation-map.md` unless committed by the runner. |
| `python -m pip install -e ".[dev]"` | Local fail expected in this shell | `python` is not on PATH locally. CI setup-python should provide it. |
| `python -m pytest tests/ -v` | Local fail observed | Fails only because `python` is not on PATH. This is a local worker portability issue. |
| `python3 -m pytest tests/ -v` | Pass observed | `140 passed in 31.83s` on Python 3.12.8. |
| `python3 -m pytest tests/test_packaging_ci.py -v` | Pass expected | These three tests passed as part of the full suite and should be the cheapest guard for validation contract edits. |
| `python3 -m pytest tests/test_code_index.py -v` | Pass expected with ignored artifact | Full suite passed. This path may create or refresh ignored `tools/index.json`. |
| `python3 tools/code_index.py --status` | Pass observed | Reported a fresh ignored `tools/index.json`. Useful before work touching `tensor_logic/`. |
| `uv run --with torch python demos/transitive_closure.py` | Not run, expected smoke candidate | README lists demo commands, but they are not CI-gated and may use network/package resolution depending on uv cache state. |
| `python3 -m build` | Unknown/not declared | No build command, lockfile, or build validation target was found. |
| `ruff`, `mypy`, `black`, `tox`, `nox`, or pre-commit | Not available as repo contracts | No matching config files were found by the lock/tooling search command. |

## Risks and Stale Assumptions

1. The documented local validation command assumes `python` exists. This worktree has `python3` but no `python`, so a local worker following `README.md` exactly gets a false negative even though `python3 -m pytest tests/ -v` passes.

2. The dependency declaration is intentionally small relative to the experiment archive. `pyproject.toml` declares only `torch` at runtime and `matplotlib`, `numpy`, `pytest` for dev. Search evidence shows heavier scripts reference optional or external surfaces such as `transformers`/model loading in `experiments/exp60d_sft.py` and `experiments/exp77_schema_rule_construction.py`, `scipy` in `experiments/exp83_slot_attention.py`, `pip download` in `experiments/exp53_real_imports.py` and `experiments/exp54_big_imports.py`, and external FAFSA/ISIR validation data in `experiments/validate_exp80_isir.py`.

3. CI validates package code and selected experiment helpers, not the full experiment claims. This is a good default for cheap proof, but README headline claims and many `experiments/exp*.py` scripts are not continuously reproduced.

4. There is no declared lint, type, coverage, wheel-build, or import-all validation surface. That keeps validation cheap, but it means syntax/style/import drift outside tested modules can survive until a specific experiment script is run.

5. `tools/code_index.py --lookup` and related tests can create `tools/index.json`. The file is correctly ignored, and `tests/test_code_index.py` asserts that, but local workers should expect ignored state after validation.

6. Remote/long-running experiment discipline is documented in `docs/RUN_PROTOCOL.md`, but it is not enforced by tests. Scripts that write manifests, adapter pointers, or failure dumps can regress without the cheap suite noticing.

## Next Safe Work

1. Make validation docs portable across local shells.
   Acceptance criteria: `README.md`, `CLAUDE.md`, and `.github/workflows/ci.yml` still agree on the canonical CI command, and README also documents a local `python3` fallback when `python` is absent.
   Validation: `python3 -m pytest tests/test_packaging_ci.py -v` should pass after updating `tests/test_packaging_ci.py` if it intentionally checks the fallback text.

2. Add a repo-local `docs/VALIDATION.md` matrix.
   Acceptance criteria: document cheap CI validation, local fallback commands, code-index commands, demo smoke commands, heavyweight/remote experiment commands, expected artifacts, and expected pass/fail assumptions.
   Validation: `python3 -m pytest tests/test_packaging_ci.py -v` plus `python3 -m pytest tests/test_code_index.py -v`.

3. Split optional experiment dependencies into named extras or explicit docs.
   Acceptance criteria: heavyweight scripts that require `transformers`, `peft`, `datasets`, `accelerate`, `scipy`, or external downloads are either mapped to optional extras in `pyproject.toml` or listed in `docs/VALIDATION.md` as non-CI dependencies.
   Validation: `python3 -m pytest tests/ -v` should still pass, and a targeted import smoke should prove the base package remains installable without heavyweight extras.

4. Add a lightweight build/import proof.
   Acceptance criteria: add one documented command that proves the installable package imports and its CLI starts, without running experiments or external services.
   Candidate validation: `python3 -m pytest tests/test_packaging_ci.py -v` and `python3 -m tensor_logic --help`.

5. Add explicit slow/remote markers before expanding test coverage.
   Acceptance criteria: any new tests for demos or experiments that can train models, download packages, hit external data, or require GPU are marked or placed outside the default `tests/` path.
   Validation: default `python3 -m pytest tests/ -v` remains under one minute on a normal laptop.

## Non-Goals

- Did not change product code, experiments, tests, package metadata, CI, docs outside this report, generated data, secrets, external services, deploys, pushes, or PR state.
- Did not run remote/Kaggle/hosted jobs.
- Did not run heavyweight experiment scripts, demo training loops, model downloads, package-download experiments, or external FAFSA validation.
- Did not create or modify Linear/GitHub tracker state.
- Did not attempt to clean ignored `.pytest_cache/` or `tools/index.json`; they are validation byproducts and are already ignored.

## Unknowns

- Whether the intended local runner always provides a `python` shim or expects agents to use `python3`.
- Whether CI should remain Python 3.11 only while local validation passed on Python 3.12.8.
- Whether the experiment archive should be treated as historical evidence, runnable artifacts, or both.
- Whether heavyweight optional dependencies should be encoded as install extras, per-script comments, or a separate validation document.
- Whether headline README claims should get reproducibility smoke tests, recorded result fixtures, or remain manually audited research notes.

## Handoff Notes

Changed file: `docs/overnight/tensor-experiments-validation-map.md`.

No PR was created, no external tracker was updated, and no product code was edited. The useful validation evidence from this audit is that the full suite passes under `python3`, the documented `python` command is not runnable in this local shell, and ignored validation artifacts are expected after code-index/test execution.
