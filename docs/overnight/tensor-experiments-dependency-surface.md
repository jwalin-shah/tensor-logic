# tensor-experiments dependency-surface audit

Queue item: `tensor-experiments-dependency-surface`  
Date: 2026-05-07  
Branch: `codex/goal-tensor-experiments-dependency-surface`  
Starting HEAD: `5565791`

## Scope and method

This was a read-only dependency-surface audit. I did not change product code,
generated data, secrets, external services, deploys, pushes, or PR state. The
only intended tracked change is this report.

Commands and observations used as local evidence:

- `git status --short --branch` reported `## codex/goal-tensor-experiments-dependency-surface` with no tracked dirty files at the start.
- `llm-tldr tree .` showed a Python research repo with `tensor_logic/`, `experiments/`, `demos/`, `phase_training/`, `tests/`, `tools/`, `web_workbench/`, and committed experiment data.
- `rg --files -g 'requirements*.txt' -g 'poetry.lock' -g 'uv.lock' -g 'Pipfile*' -g 'environment*.yml' -g 'conda*.yml' -g '.python-version' -g '.env*' -g 'setup.py' -g 'setup.cfg' -g 'tox.ini' -g 'noxfile.py' -g '.github/**'` found `.github/workflows/ci.yml` but no lockfile, requirements file, env file, setup file, tox file, or nox file.
- `python3 --version` returned `Python 3.12.8`; CI pins Python `3.11`.
- `python3 -m pytest tests/test_packaging_ci.py -q` passed: `3 passed in 0.01s`.
- `python3 -m pytest tests/test_web_workbench.py -q` passed: `2 passed in 4.57s`.
- `python3 -m pytest tests/ -q` passed: `140 passed in 34.41s`.
- `python3 tools/code_index.py --status` initially returned exit 1 with `stale: index.json is missing or older than source files`; after the full test suite it returned exit 0 with `fresh: .../tools/index.json`, because tests generated the ignored index.
- `git status --short --ignored` after tests showed ignored local artifacts: `.pytest_cache/` and `tools/index.json`.

## Repo purpose

`tensor-experiments` is a Python 3.11+ tensor-logic research repo. The reusable
package is `tensor_logic/`; the rest of the repo is a mix of demos, experiment
scripts, phase-training prototypes, run protocols, notes, and tracked result
artifacts. `README.md` explicitly says the repo now also serves as the shared
`tensor_logic` library dependency for sibling projects such as `fafsa-engine`
and future `officeqa-*` projects.

## Declared dependency surface

The declared install surface is intentionally small:

- `pyproject.toml` declares build requirements `setuptools>=69` and `wheel`.
- `pyproject.toml` declares runtime `torch>=2.0`.
- `pyproject.toml` declares dev extras `matplotlib>=3.7`, `numpy>=1.24`, and `pytest>=8.0`.
- `pyproject.toml` uses `setuptools` package discovery for `tensor_logic*` only, so `demos/`, `experiments/`, `phase_training/`, `tools/`, and `web_workbench/` are repo scripts, not installed packages.
- `.github/workflows/ci.yml` installs `python -m pip install -e ".[dev]"` and runs `python -m pytest tests/ -v`.
- `README.md` documents the same worker validation commands.
- `tests/test_packaging_ci.py` asserts the current packaging contract: `torch` runtime, `pytest`/`numpy`/`matplotlib` dev, package include `tensor_logic*`, CI validation, and README validation docs.

There is no lockfile or constraints file in this worktree. Dependency versions
are lower-bounded only, so current reproducibility depends on PyPI state and the
local/CI Python version.

## Observed imports

An AST import scan across `tensor_logic`, `demos`, `experiments`,
`phase_training`, `tests`, `web_workbench`, and `tools` found:

- `torch`: 101 files.
- `numpy`: 4 files.
- `matplotlib`: 2 files.
- `transformers`: 5 files.
- `peft`: 1 file.
- `datasets`: 1 file.
- `mlx_lm`: 1 file.
- `scipy`: 1 file.
- `tqdm`: 1 file.

Non-stdlib modules not declared in `pyproject.toml` and not obvious local imports:

- `transformers`: `experiments/exp32_attention_compose.py`, `experiments/exp36_code_dependency.py`, `experiments/exp60d_sft.py`, `experiments/exp77_schema_rule_construction.py`, `experiments/exp78_rule_induction.py`.
- `peft`, `datasets`, `tqdm`: `experiments/exp60d_sft.py`.
- `mlx_lm`: `experiments/exp78_rule_induction.py`.
- `scipy`: `experiments/exp83_slot_attention.py`.

This split is mostly experiment-only, but it is not encoded as optional extras,
so future agents cannot install the right surface from package metadata alone.

## Runtime entrypoints and scripts

Important local entrypoints:

- `tensor_logic/__main__.py` provides CLI commands: `run`, `query`, `prove`, `ingest-python`, `repl`, `repo-graph`, and `serve`.
- `tensor_logic/http_api.py` uses only stdlib HTTP server primitives and exposes JSON POST handlers for ingest/run/query/prove operations.
- `web_workbench/server.py` serves static files and shells out to `[sys.executable, "-m", "tensor_logic"]` using a temporary `.tl` file.
- `web_workbench/README.md` documents `python web_workbench/server.py --host 127.0.0.1 --port 8080`.
- `tensor_logic/reason.py` is a subprocess entrypoint for `python -m tensor_logic.reason --query ... --facts-file ...`, but the current proposer is a deterministic keyword fallback and does not call an LLM despite accepting `--model`.
- `tools/code_index.py` is a stdlib AST indexer that writes `tools/index.json`; `.gitignore` ignores that file.
- `experiments/exp60d_sft.py` is a LoRA SFT runner with explicit install guidance: `pip install torch transformers peft datasets accelerate`.
- `experiments/exp78_rule_induction.py` is TL-only by default and uses `transformers` or `mlx_lm` only behind `--lm`.
- `experiments/exp53_real_imports.py` and `experiments/exp54_big_imports.py` run `python -m pip download --no-deps --no-binary :all:` against PyPI and unpack third-party archives into temp dirs.
- `experiments/exp79_lewm_tl.py` and `experiments/exp83_slot_attention.py` write experiment outputs under `experiments/exp79_data/` and `experiments/exp83_slot_data/`.

## Generated artifacts, caches, and local-only state

Tracked artifacts already in git:

- `experiments/exp60_data/train.jsonl`, `eval.jsonl`, and `eval_hard.jsonl`.
- `experiments/exp76_data/train.jsonl`, `eval.jsonl`, `eval_hard.jsonl`, and `train_paraphrased.jsonl`.
- `experiments/exp78_data/results.json`.
- `experiments/exp79_data/results.json` and `complexity_curve.png`.
- `experiments/exp80_spot_check_cases.json`.
- `experiments/exp83_slot_data/results.json` and `complexity_curve.png`.

Ignored/generated local state:

- `.gitignore` ignores `.pytest_cache/`, `.ruff_cache/`, `.venv/`, `.uv/`, `*.pt`, `*.pth`, `/.cocoindex_code/`, `tools/index.json`, `dist/`, `build/`, and `*.egg-info/`.
- `tests/test_code_index.py` includes a CLI test that runs `python tools/code_index.py --rebuild`; running the full suite generates ignored `tools/index.json`.
- `tools/code_index.py --status` exits 1 on a clean checkout until the ignored index has been generated. `--lookup` and `ensure_fresh()` rebuild it automatically.
- `experiments/exp79_lewm_tl.py` saves `encoder.pt` and `probe.pt`; those are ignored by `*.pt` and are local-only state.
- `experiments/exp60d_sft.py` defaults `--out` to `experiments/exp60_data/lora_adapter` and writes adapter files plus `manifest.json`. `.gitignore` does not currently ignore `*.safetensors`, adapter directories, or tokenizer/config JSON created by `save_pretrained()`.
- `docs/RUN_PROTOCOL.md` says weights should not be committed and that local git should contain result pointers/metrics, not adapters.

## Environment variables and credentials

The code surface I inspected does not require repo-local `.env` files, and the
lock/env-file search found none. Current credential/API-key references are in
planning docs, not active package code:

- `docs/superpowers/specs/2026-04-28-fafsa-engine-design.md` and
  `docs/superpowers/plans/2026-04-28-fafsa-engine.md` mention `FAFSA_LLM`,
  `FAFSA_LLM_MODEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and Ollama URLs for
  a sibling project design.
- `experiments/exp78_rule_induction.py` and `experiments/exp60d_sft.py` can
  trigger Hugging Face model downloads through `from_pretrained()`, but no
  token variable is hard-coded.
- `docs/RUN_PROTOCOL.md` describes Kaggle persistence and adapter datasets; no
  Kaggle credential is stored in the repo.

## Risks and stale assumptions

1. Optional experiment dependencies are not installable by contract. `README.md`
   and CI validate only `.[dev]`, while active experiments require
   `transformers`, `peft`, `datasets`, `accelerate`, `tqdm`, `mlx_lm`, and
   `scipy`. A worker can pass CI and still be unable to reproduce exp60, exp78
   LM mode, or exp83.

2. There is no lockfile or constraints file. `torch>=2.0`, `numpy>=1.24`,
   `matplotlib>=3.7`, and `pytest>=8.0` are lower bounds only. CI uses Python
   3.11, this local run used Python 3.12.8, and model-loading stacks are known
   to be sensitive to dependency drift.

3. Adapter outputs are under-ignored. `docs/RUN_PROTOCOL.md` says not to commit
   LoRA weights, but `.gitignore` ignores `*.pt`/`*.pth` and not
   `*.safetensors` or `experiments/*_data/lora_adapter/`. Running
   `experiments/exp60d_sft.py` could leave large or sensitive model artifacts
   visible to `git status`.

4. External-download experiments are not hermetic. `experiments/exp53_real_imports.py`
   and `experiments/exp54_big_imports.py` shell out to `pip download` and unpack
   third-party archives from PyPI. This couples results to network/PyPI state
   and gives future workers a larger external-service and archive-handling
   surface than the core tests exercise.

5. `tools/code_index.py --status` has clean-checkout ambiguity. A fresh checkout
   reports "stale" with exit 1 because `tools/index.json` is ignored. That is
   expected for generated local state, but can look like a failure unless the
   worker knows to run `--lookup` or `--rebuild`.

6. The web workbench subprocess has no timeout. `web_workbench/server.py` invokes
   `python -m tensor_logic` synchronously. A hung or expensive command can tie
   up a request thread; tests cover happy-path behavior, not timeout policy.

7. README demo claims depend on `uv` and a local runtime not declared in
   `pyproject.toml`. The README uses `uv run --with torch ...` for demos, but
   `uv` itself is not part of the repo contract.

8. `experiments/exp83_slot_attention.py` exposes `--skip-train` but documents it
   as "not supported in this PoC yet"; the CLI surface looks more stable than
   the implementation.

## Validation commands

Commands run during this audit:

- `python3 -m pytest tests/test_packaging_ci.py -q` -> passed, expected pass for packaging/docs/CI contract.
- `python3 -m pytest tests/test_web_workbench.py -q` -> passed, expected pass for local workbench command wrapper.
- `python3 -m pytest tests/ -q` -> passed, expected pass for full local suite; generated ignored `.pytest_cache/` and `tools/index.json`.
- `python3 tools/code_index.py --status` -> expected fail on clean checkout until `tools/index.json` exists; expected pass after `--rebuild` or after the full suite generates it.

Validation candidates for future workers:

- `git status --short` -> expected pass/exit 0; should show only intentional report/doc changes during an audit or be clean after commit.
- `python -m pip install -e ".[dev]"` -> expected pass in a network-enabled clean environment; not run here because the local suite already had dependencies and this queue item did not authorize external package changes.
- `python -m pytest tests/ -v` -> expected pass; CI canonical command.
- `python3 tools/code_index.py --lookup Program` -> expected pass and may generate ignored `tools/index.json`.
- `python3 tools/code_index.py --status` -> expected fail on fresh checkout if `tools/index.json` is missing, expected pass after lookup/rebuild.
- `python experiments/exp78_rule_induction.py --max-len 3` -> expected pass without LM extras.
- `python experiments/exp78_rule_induction.py --lm` -> expected to require `transformers` or use fallback; may download models if dependencies are installed.
- `python experiments/exp83_slot_attention.py --help` -> expected pass; full run expected to require undeclared `scipy` and nontrivial CPU/MPS time.
- `python experiments/exp60d_sft.py --skip-train --n-eval 1` -> expected to require undeclared `transformers`/`peft` stack and a model/cache unless guarded.

## Next safe work

1. Add explicit optional extras for experiment families.
   - Scope: `pyproject.toml`, `README.md`, `tests/test_packaging_ci.py`.
   - Acceptance: define extras such as `lm`, `sft`, `slot`, and maybe `all-experiments`; README maps exp60/78/83 to extras; packaging test asserts the new contract.
   - Validation: `python3 -m pytest tests/test_packaging_ci.py -q`.

2. Harden output ignore rules for model artifacts.
   - Scope: `.gitignore`, `docs/RUN_PROTOCOL.md`, maybe a small packaging/ignore test.
   - Acceptance: `*.safetensors`, adapter output dirs, tokenizer/model generated files, and known local run outputs are ignored or routed to documented artifact pointers.
   - Validation: `python3 -m pytest tests/test_packaging_ci.py tests/test_code_index.py::test_index_json_is_gitignored -q` plus `git check-ignore` examples for representative adapter files.

3. Make external package-download experiments reproducible/offline-friendly.
   - Scope: `experiments/exp53_real_imports.py`, `experiments/exp54_big_imports.py`, tests around helper functions.
   - Acceptance: add `--package-dir` or fixture input mode, record package versions/source archive names, and add safe archive extraction guards.
   - Validation: unit tests with local temp archives; no network required.

4. Clarify or change code-index status semantics.
   - Scope: `tools/code_index.py`, `CLAUDE.md`, `tests/test_code_index.py`.
   - Acceptance: either document that `--status` exits 1 on fresh checkout, or add a `--status --ensure` mode that rebuilds before checking.
   - Validation: `python3 -m pytest tests/test_code_index.py -q`.

5. Add timeout handling to web workbench subprocesses.
   - Scope: `web_workbench/server.py`, `tests/test_web_workbench.py`.
   - Acceptance: `subprocess.run(..., timeout=<small configurable default>)` returns structured timeout JSON and always unlinks the temp `.tl` file.
   - Validation: `python3 -m pytest tests/test_web_workbench.py -q`.

## Non-goals

- No dependency upgrades or lockfile generation were performed.
- No experiment was launched, trained, downloaded, or uploaded.
- No model weights, adapter artifacts, Kaggle datasets, Hugging Face downloads,
  or PyPI downloads were created intentionally.
- No product/library code was modified.
- No external tracker, PR, deployment, or remote job was changed.

## Unknowns

- Whether sibling repos consume `tensor_logic` through editable installs,
  path dependencies, or copied source.
- Whether the intended dependency policy is "minimal core only" or "installable
  reproducible experiment families."
- Whether `uv` is assumed globally available for all workers.
- Whether old committed experiment result files are authoritative outputs or
  just historical snapshots.
- Whether `.cocoindex_code/` is still used in practice; it is ignored and was
  not present in this worktree listing.
- Whether current external model IDs in exp60/77/78 are pinned enough for
  reproducible morning review.

## Handoff

- Changed tracked file: `docs/overnight/tensor-experiments-dependency-surface.md`.
- Product code changed: no.
- Commit: not created; current HEAD remains `5565791`.
- Push/PR: no, out of scope for this queue item.
- Blockers: writing the report succeeded, but local commit creation was blocked
  by sandbox permissions. `git add docs/overnight/tensor-experiments-dependency-surface.md`
  failed because git could not create
  `/Users/jwalinshah/projects/tensor/experiments/.git/worktrees/tensor-experiments-dependency-surface/index.lock`.
  Ignored local artifacts from validation remained after tests (`.pytest_cache/`,
  `tools/index.json`) because the sandbox policy rejected `rm -rf`; they do not
  appear in `git status --short`.
